"""FFT-accelerated non-parametric hypothesis tests for SPLISOSM."""

# ruff: noqa: E402

from __future__ import annotations

import os
import warnings
from typing import Any, Literal, Optional, Union

# Suppress deprecations from SpatialData stack before importing it.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)

try:
    import spatialdata as sd
except ImportError:
    sd = None

import numpy as np
import pandas as pd
import scipy.fft
import scipy.sparse as sp
import torch
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.stats import ttest_ind, combine_pvalues

from splisosm.kernel_gpr import (
    FFTKernelOp,
    FFTKernelGPR,
    _DEFAULT_GPR_CONFIGS,
    linear_hsic_test,
)
from splisosm.likelihood import liu_sf
from splisosm.utils import counts_to_ratios, false_discovery_control

__all__ = ["FFTKernel", "FFTKernelOp", "FFTKernelGPR", "SplisosmFFT"]


class FFTKernel:
    """FFT-based spatial kernel on a periodic 2D raster grid.

    This implementation currently supports only a CAR-style spatial kernel
    equivalent to a periodic, neighborhood graph-based autoregressive model.

    Parameters
    ----------
    shape
        Grid shape ``(ny, nx)``.
    spacing
        Physical spacing ``(dy, dx)`` between neighboring raster cells.
    rho
        Spatial autocorrelation coefficient in CAR kernel.
    neighbor_degree
        Neighbor ring degree for graph construction.
        ``1`` uses nearest neighbors in the periodic metric.
    workers
        Number of workers used by ``scipy.fft.fft2``.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        spacing: tuple[float, float] = (1.0, 1.0),
        rho: float = 0.99,
        neighbor_degree: int = 1,
        workers: int | None = None,
    ) -> None:
        if len(shape) != 2:
            raise ValueError("`shape` must be a tuple of length 2: (ny, nx).")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("Grid dimensions must be positive.")
        if neighbor_degree < 1:
            raise ValueError("`neighbor_degree` must be >= 1.")

        self.ny, self.nx = int(shape[0]), int(shape[1])
        self.dy, self.dx = float(spacing[0]), float(spacing[1])
        self.n_grid = self.ny * self.nx
        self.neighbor_degree = int(neighbor_degree)
        self.rho = min(float(rho), 0.99)
        self.workers = workers

        self._min_dist_sq = self._precompute_square_torus_distances()
        self._spectrum_2d = self._compute_car_spectrum()
        self.spectrum = self._spectrum_2d.ravel()

    def _precompute_square_torus_distances(self) -> np.ndarray:
        """Compute squared torus distances from origin for periodic grid."""
        y = np.arange(self.ny, dtype=float) * self.dy
        x = np.arange(self.nx, dtype=float) * self.dx
        y = np.minimum(y, (self.ny * self.dy) - y)
        x = np.minimum(x, (self.nx * self.dx) - x)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return yy**2 + xx**2

    def _compute_car_spectrum(self) -> np.ndarray:
        """Compute eigenvalues of the periodic CAR kernel via FFT."""
        unique_d2 = np.unique(self._min_dist_sq)
        if self.neighbor_degree < len(unique_d2):
            cutoff_sq = unique_d2[self.neighbor_degree]
        else:
            cutoff_sq = unique_d2[-1]

        # Graph adjacency image centered at (0, 0) in periodic metric.
        w_img = (self._min_dist_sq <= cutoff_sq).astype(float)
        w_img[0, 0] = 0.0
        degree = float(np.sum(w_img))

        if degree <= 0.0:
            return np.ones((self.ny, self.nx), dtype=float)

        lam_w = np.real(scipy.fft.fft2(w_img, workers=self.workers)) / degree
        return 1.0 / (1.0 - self.rho * lam_w)

    def power_spectral_density_1d(
        self, bins: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the 1D power spectral density (radial profile).

        Parameters
        ----------
        bins
            Number of bins for the 1D radial frequency.

        Returns
        -------
        freq_bins : np.ndarray
            The center frequencies of the valid bins.
        psd_1d : np.ndarray
            The average power (eigenvalue) in each frequency bin.
        """
        if bins < 1:
            raise ValueError("`bins` must be >= 1.")

        # Compute 2D frequencies
        fy = scipy.fft.fftfreq(self.ny, d=self.dy)
        fx = scipy.fft.fftfreq(self.nx, d=self.dx)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")

        # Compute radial frequency (norm)
        F_r = np.sqrt(FY**2 + FX**2)
        f_r_flat = F_r.ravel()
        spectrum_flat = self._spectrum_2d.ravel()

        # Bin the radial frequencies to get the average power per bin
        bin_edges = np.linspace(0, f_r_flat.max(), bins + 1)

        # Sum of spectrum in each bin
        psd_sum, _ = np.histogram(f_r_flat, bins=bin_edges, weights=spectrum_flat)
        counts, _ = np.histogram(f_r_flat, bins=bin_edges)

        # Avoid division by zero for empty bins
        valid = counts > 0
        psd_1d = np.zeros(bins)
        psd_1d[valid] = psd_sum[valid] / counts[valid]

        # Calculate bin centers for plotting
        freq_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Return only the valid (non-empty) bins
        return freq_bins[valid], psd_1d[valid]

    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """Compute ``x^T K x`` in ``O(N log N)`` via FFT.

        Parameters
        ----------
        x
            Input with shape ``(ny, nx)`` or ``(ny, nx, m)``.

        Returns
        -------
        float or np.ndarray
            Scalar for 2D input, or shape ``(m,)`` for 3D input.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            x = x[..., None]
        if x.ndim != 3:
            raise ValueError("`x` must have shape (ny, nx) or (ny, nx, m).")

        ny, nx, _ = x.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Input shape ({ny}, {nx}) does not match kernel shape ({self.ny}, {self.nx})."
            )

        x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=self.workers)
        power = np.abs(x_hat) ** 2
        weighted = np.sum(power * self._spectrum_2d[:, :, None], axis=(0, 1))
        q = weighted / (self.n_grid)
        return float(q[0]) if q.size == 1 else q

    def eigenvalues(self, k: int | None = None) -> np.ndarray:
        """Return kernel eigenvalues.

        Parameters
        ----------
        k
            Number of leading eigenvalues to return. If ``None``, return all.

        Returns
        -------
        np.ndarray
            Eigenvalues in descending order.
        """
        evals = np.sort(self.spectrum)[::-1]
        if k is None:
            return evals
        return evals[:k]

    def apply_residual_op(self, x: np.ndarray, epsilon: float) -> np.ndarray:
        """Apply the kernel regression residual operator ``R = epsilon * (K + epsilon * I)**(-1)``.

        Computed in O(N log N) via FFT as::

            R @ v = IFFT2( epsilon / (lambda + epsilon) * FFT2(v) )

        Parameters
        ----------
        x
            Input with shape ``(ny, nx)`` or ``(ny, nx, m)``.
        epsilon
            Regularization / noise level.

        Returns
        -------
        np.ndarray
            Residuals of the same shape as ``x``.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 2
        if scalar:
            x = x[..., None]
        if x.ndim != 3:
            raise ValueError("`x` must have shape (ny, nx) or (ny, nx, m).")

        ny, nx, _ = x.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Input shape ({ny}, {nx}) does not match kernel shape ({self.ny}, {self.nx})."
            )

        x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=self.workers)
        scale = epsilon / (self._spectrum_2d[:, :, None] + epsilon)
        result = np.real(
            scipy.fft.ifft2(scale * x_hat, axes=(0, 1), workers=self.workers)
        )
        return result[..., 0] if scalar else result

    def trace(self) -> float:
        """Return ``trace(K)``."""
        return float(np.sum(self.spectrum))

    def square_trace(self) -> float:
        """Return ``trace(K^2)``."""
        return float(np.sum(self.spectrum**2))


def _du_worker_fft(
    raster_layer: Any,
    iso_names: list[str],
    gpr_iso: "FFTKernelGPR",
    z_res_list: list[np.ndarray],
    ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"],
    spacing: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DU HSIC statistics and p-values for one gene against all factors.

    Uses ``FFTKernelGPR`` to residualize the isoform ratios (per-gene epsilon
    search), then measures residual linear association with the pre-residualized
    covariates via the linear HSIC statistic.

    Parameters
    ----------
    raster_layer : xarray.DataArray
        SpatialData raster image with channel dimension first ``(c, ny, nx)``.
    iso_names : list of str
        Channel names (isoforms) to select from ``raster_layer``.
    gpr_iso : FFTKernelGPR
        ``FFTKernelGPR`` instance used for per-gene isoform residualization.
        Performs an O(N log N) epsilon search followed by residualization.
    z_res_list : list of np.ndarray
        Per-factor spatially-residualized covariate grids, each shape ``(ny, nx)``.
        Pre-computed once before the per-gene loop.
    ratio_transformation : str
        Compositional transformation for isoform ratios.
    spacing : tuple of float, optional
        Physical spacing ``(dy, dx)`` between grid cells. Passed to
        ``fit_residuals_cube`` to set the kernel length-scale units.

    Returns
    -------
    stats : np.ndarray, shape (n_covariates,)
    pvals : np.ndarray, shape (n_covariates,)
    """
    n_factors = len(z_res_list)
    stats = np.zeros(n_factors)
    pvals = np.ones(n_factors)

    # SpatialData raster channel order is (c, y, x), convert to (y, x, c)
    data = raster_layer.sel(c=iso_names).values
    counts_cube = np.moveaxis(np.asarray(data, dtype=float), 0, -1)

    counts_flat = counts_cube.reshape(
        counts_cube.shape[0] * counts_cube.shape[1], counts_cube.shape[2]
    )
    ratios = counts_to_ratios(
        torch.from_numpy(counts_flat).float(),
        transformation=ratio_transformation,
        nan_filling="none",
    )
    y_cube = ratios.numpy().reshape(counts_cube.shape[0], counts_cube.shape[1], -1)

    # Spatially residualize isoform ratios via FFT + per-gene epsilon search
    y_res_cube, _ = gpr_iso.fit_residuals_cube(
        y_cube, spacing=spacing
    )  # (ny, nx, n_isos)
    n_grid = gpr_iso.n
    y_res = y_res_cube.reshape(n_grid, -1)  # (n_grid, n_isos)

    # Pre-compute gram matrix eigenvalues for y (shared across factors)
    gram_y = y_res.T @ y_res
    if not np.isfinite(gram_y).all():
        return stats, pvals
    try:
        lambda_y = np.linalg.eigvalsh(gram_y)
    except np.linalg.LinAlgError:
        return stats, pvals
    lambda_y = lambda_y[lambda_y > 1e-8]

    for _f, z_res_cube in enumerate(z_res_list):
        z_flat = z_res_cube.ravel()[:, None]  # (n_grid, 1)

        # HSIC = ||y_res^T z_res||_F^2
        cross = y_res.T @ z_flat  # (n_isos, 1)
        hsic_scaled = float(np.sum(cross**2))
        stats[_f] = hsic_scaled / (n_grid - 1) ** 2

        lambda_z = np.linalg.eigvalsh(z_flat.T @ z_flat)
        lambda_z = lambda_z[lambda_z > 1e-8]

        if lambda_z.size == 0 or lambda_y.size == 0:
            pvals[_f] = 1.0
            continue

        lambda_mix = (lambda_z[:, None] * lambda_y[None, :]).reshape(-1)
        pvals[_f] = float(liu_sf(hsic_scaled * n_grid, lambda_mix))

    return stats, pvals


def _du_worker_spot(
    raster_layer: Any,
    iso_names: list[str],
    rows: np.ndarray,
    cols: np.ndarray,
    z_list: list[np.ndarray],
    ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"],
    method: Literal["hsic", "t-fisher", "t-tippett"],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DU statistics for one gene at spot level (no spatial residualization).

    Used for the unconditional HSIC test (``method='hsic'``) and the
    T-test variants (``method='t-fisher'``, ``method='t-tippett'``).

    Parameters
    ----------
    raster_layer : xarray.DataArray
        SpatialData raster image ``(c, ny, nx)``.
    iso_names : list of str
        Channel names (isoforms) to select.
    rows : np.ndarray
        Grid row indices for each observation (0-based, offset applied).
    cols : np.ndarray
        Grid column indices for each observation (0-based, offset applied).
    z_list : list of np.ndarray
        Per-factor covariate vectors, each shape ``(n_obs,)``.
    ratio_transformation : str
        Compositional transformation for isoform ratios.
    method : str
        ``'hsic'``, ``'t-fisher'``, or ``'t-tippett'``.

    Returns
    -------
    stats : np.ndarray, shape (n_covariates,)
    pvals : np.ndarray, shape (n_covariates,)
    """
    n_factors = len(z_list)
    stats = np.zeros(n_factors)
    pvals = np.ones(n_factors)

    # Extract counts at spot positions: (n_iso, ny, nx) → (n_obs, n_iso)
    data = raster_layer.sel(c=iso_names).values  # (n_iso, ny, nx)
    counts_spots = np.asarray(data, dtype=float)[:, rows, cols].T  # (n_obs, n_iso)

    ratios = counts_to_ratios(
        torch.from_numpy(counts_spots).float(),
        transformation=ratio_transformation,
        nan_filling="none",
    )  # torch.Tensor, (n_obs, n_iso)

    if method == "hsic":
        for _f, z_vec in enumerate(z_list):
            z = torch.from_numpy(z_vec).float().unsqueeze(1)  # (n_obs, 1)
            _hsic, _pval = linear_hsic_test(z, ratios, centering=True)
            stats[_f] = _hsic
            pvals[_f] = _pval
    else:
        # T-test per isoform, combined p-values
        combine_method = method.split("-")[1]  # 'fisher' or 'tippett'
        y_np = ratios.numpy()
        n_isos = y_np.shape[1]
        for _f, z_vec in enumerate(z_list):
            groups = np.unique(z_vec[~np.isnan(z_vec)])
            if len(groups) != 2:
                continue
            iso_pvals = []
            iso_stats_list = []
            for _i in range(n_isos):
                y0 = y_np[z_vec == groups[0], _i]
                y1 = y_np[z_vec == groups[1], _i]
                y0 = y0[~np.isnan(y0)]
                y1 = y1[~np.isnan(y1)]
                if len(y0) < 2 or len(y1) < 2:
                    iso_pvals.append(1.0)
                    iso_stats_list.append(0.0)
                    continue
                t, p = ttest_ind(y0, y1)
                iso_pvals.append(float(p) if np.isfinite(p) else 1.0)
                iso_stats_list.append(float(t) if np.isfinite(t) else 0.0)
            if iso_pvals:
                combined_stat, combined_pval = combine_pvalues(
                    iso_pvals, method=combine_method
                )
                stats[_f] = float(combined_stat) if np.isfinite(combined_stat) else 0.0
                pvals[_f] = float(combined_pval) if np.isfinite(combined_pval) else 1.0

    return stats, pvals


def _sv_worker_fft(
    raster_layer: Any,
    iso_names: list[str],
    kernel: FFTKernel,
    kernel_eigvals: np.ndarray,
    method: Literal["hsic-ir", "hsic-ic", "hsic-gc"],
    ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"],
) -> tuple[float, float]:
    """Compute one gene-level FFT-HSIC statistic and p-value from rasterized channels."""
    # SpatialData raster channel order is (c, y, x), convert to (y, x, c)
    data = raster_layer.sel(c=iso_names).values
    counts_cube = np.moveaxis(np.asarray(data, dtype=float), 0, -1)

    if method == "hsic-ic":
        y_cube = counts_cube
    elif method == "hsic-gc":
        y_cube = counts_cube.sum(axis=2, keepdims=True)
    else:
        counts_flat = counts_cube.reshape(kernel.n_grid, counts_cube.shape[2])
        ratios = counts_to_ratios(
            torch.from_numpy(counts_flat).float(),
            transformation=ratio_transformation,
            nan_filling="none",
        )
        y_cube = ratios.numpy().reshape(kernel.ny, kernel.nx, -1)

    # manually center the data and replace NaN with zeros (i.e., mean padding)
    y_cube = y_cube - np.nanmean(y_cube, axis=(0, 1), keepdims=True)
    y_cube = np.nan_to_num(y_cube, nan=0.0, posinf=0.0, neginf=0.0)

    q = np.atleast_1d(np.asarray(kernel.xtKx(y_cube), dtype=float))
    hsic_scaled = float(np.sum(q))
    stat = hsic_scaled / ((kernel.n_grid - 1) ** 2)

    y_flat = y_cube.reshape(kernel.n_grid, -1)
    gram = y_flat.T @ y_flat

    if not np.isfinite(gram).all():
        return stat, 1.0

    try:
        lambda_y = np.linalg.eigvalsh(gram)
    except np.linalg.LinAlgError:
        return stat, 1.0

    lambda_y = lambda_y[lambda_y > 1e-8]

    if lambda_y.size == 0 or kernel_eigvals.size == 0:
        return stat, 1.0

    lambda_mix = (kernel_eigvals[:, None] * lambda_y[None, :]).reshape(-1)
    pvalue = float(liu_sf(hsic_scaled * kernel.n_grid, lambda_mix))
    return stat, pvalue


class SplisosmFFT:
    """FFT-accelerated SPLISOSM model for rasterized spatial isoform testing.

    The class follows the non-parametric SPLISOSM workflow but consumes a
    SpatialData table directly and rasterizes per-gene isoform counts on demand.

    Examples
    --------
    >>> from splisosm import SplisosmFFT
    >>> model = SplisosmFFT(rho=0.9, neighbor_degree=1)
    >>> model.setup_data(
    ...     sdata=sdata,
    ...     bins="ID_square_016um",
    ...     table_name="square_016um",
    ...     col_key="array_col",
    ...     row_key="array_row",
    ...     layer="counts",
    ...     group_iso_by="gene_ids",
    ...     gene_names="gene_name",
    ...     min_counts=10,
    ...     min_bin_pct=0.0,
    ... )
    >>> model.test_spatial_variability(method="hsic-ir")
    >>> sv_results = model.get_formatted_test_results(test_type="sv")
    """

    n_genes: int
    """Number of genes after filtering."""

    n_spots: int
    """Number of observed spots (bins)."""

    n_grid: int
    """Number of raster grid bins (including padding). n_grid = n_y * n_x"""

    n_isos: list[int]
    """List of numbers of isoforms per gene after filtering."""

    gene_names: list[str]
    """List of gene names corresponding to the genes in the model after filtering."""

    kernel: FFTKernel | None
    """FFTKernel instance used for spatial kernel computations."""

    sdata: Any | None
    """SpatialData object containing the input data."""

    sv_test_results: dict
    """Dictionary to store the spatial variability test results after running test_spatial_variability().
    It contains the following keys:
    
    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes,), the test statistic for each gene.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes,), the p-value for each gene.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes,), the BH adjusted p-value for each gene.
    """

    du_test_results: dict
    """Dictionary to store the differential usage test results after running test_differential_usage(). 
    It contains the following keys:
    
    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes, n_covariates), the test statistic for each gene and covariate.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes, n_covariates), the p-value for each gene and covariate.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes, n_covariates), the BH adjusted p-value for each gene and covariate. Each column/covariate is adjusted separately.
    """

    def __init__(
        self,
        rho: float = 0.99,
        neighbor_degree: int = 1,
        spacing: tuple[float, float] = (1.0, 1.0),
        workers: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        rho
            Spatial autocorrelation coefficient for CAR kernel.
        neighbor_degree
            Neighbor ring degree for CAR graph construction.
        spacing
            Raster spacing ``(dy, dx)``.
        workers
            Number of FFT workers.
        """
        self._rho = float(rho)
        self._neighbor_degree = int(neighbor_degree)
        self._spacing = spacing
        self._workers = workers

        self.n_genes = 0
        self.n_spots = 0
        self.n_grid = 0
        self.n_isos = []
        self.gene_names: list[str] = []

        self.sdata: Any | None = None
        self._adata: AnnData | None = None
        self._bins_name: str | None = None
        self._table_name: str | None = None
        self._row_key: str | None = None
        self._col_key: str | None = None
        self._counts_layer: str = "counts"
        self._group_iso_by: str = "gene_symbol"
        self._raster_key: str | None = None
        self._raster_layer: Any | None = None
        self._gene_iso_names: list[list[str]] = []

        self.kernel: FFTKernel | None = None
        self._kernel_eigvals: np.ndarray | None = None

        self.sv_test_results: dict[str, Any] = {}
        self.du_test_results: dict[str, Any] = {}

        self._gene_summary: Optional[pd.DataFrame] = None
        self._isoform_summary: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        """Return string representation of configured model state."""
        sv_status = (
            f"Completed ({self.sv_test_results['method']})"
            if len(self.sv_test_results) > 0
            else "NA"
        )
        du_status = (
            f"Completed ({self.du_test_results['method']})"
            if len(self.du_test_results) > 0
            else "NA"
        )
        return (
            "=== FFT SPLISOSM model for spatial isoform testings\n"
            f"- Number of genes: {self.n_genes}\n"
            f"- Number of observed spots: {self.n_spots}\n"
            f"- Number of raster cells: {self.n_grid}\n"
            f"- Average number of isoforms per gene: {np.mean(self.n_isos) if self.n_isos else None}\n"
            "=== Test results\n"
            f"- Spatial variability test: {sv_status}\n"
            f"- Differential usage test: {du_status}"
        )

    def _rasterize_bins(
        self,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        return_region_as_labels: bool = False,
    ) -> str:
        """Rasterize bins through SpatialData and cache image key.

        Parameters
        ----------
        bins
            Name of bin geometry element in SpatialData.
        table_name
            Name of the table annotating bins.
        col_key
            Column index key in ``adata.obs``.
        row_key
            Row index key in ``adata.obs``.
        value_key
            Optional obs key used by SpatialData rasterization.
        return_region_as_labels
            Whether to rasterize labels instead of aggregated values.

        Returns
        -------
        str
            Key used to store rasterized image in SpatialData.
        """
        if sd is None:
            raise ImportError(
                "spatialdata is required for SplisosmFFT. Please install it via 'pip install spatialdata'."
            )

        if self.sdata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")

        if table_name not in self.sdata.tables:
            raise ValueError(f"Table `{table_name}` not found in `sdata.tables`.")

        adata = self.sdata.tables[table_name]
        if hasattr(adata, "X") and not isinstance(adata.X, np.ndarray):
            if getattr(adata.X, "format", None) != "csc":
                adata.X = adata.X.tocsc()

        img_key = f"rasterized_{table_name}_{self._counts_layer}"
        rasterized = sd.rasterize_bins(
            self.sdata,
            bins=bins,
            table_name=table_name,
            col_key=col_key,
            row_key=row_key,
            value_key=value_key,
            return_region_as_labels=return_region_as_labels,
        )
        self.sdata[img_key] = rasterized
        return img_key

    def setup_data(
        self,
        sdata: Any,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        layer: str = "counts",
        group_iso_by: str = "gene_symbol",
        gene_names: Optional[str] = None,
        min_counts: int = 10,
        min_bin_pct: float = 0.0,
    ) -> None:
        """Setup SpatialData-backed isoform data for FFT-based testing.

        (bins, table_name, col_key, row_key) are passed to
        :func:`spatialdata.rasterize_bins` to rasterize isoform counts.

        Parameters
        ----------
        sdata
            SpatialData-like object with ``tables`` mapping.
        bins
            Name of the SpatialData bin geometry for rasterization.
        table_name
            Key of the table in ``sdata.tables``.
        col_key
            Column index key in ``adata.obs`` for rasterization.
        row_key
            Row index key in ``adata.obs`` for rasterization.
        layer
            AnnData layer that stores isoform count matrix.
        group_iso_by
            Column in ``adata.var`` used to group isoforms by gene. The
            unique values of this column define the gene-level groups.
        gene_names
            Optional column name in ``adata.var`` whose values are used as
            display gene names in results. If ``None``, the values of
            ``group_iso_by`` are used directly.
        min_counts
            Minimum total count (summed across all spots) required for an
            isoform to be retained. Isoforms below this threshold are
            excluded before gene grouping. Genes with fewer than two
            remaining isoforms after filtering are also excluded.
        min_bin_pct
            Minimum percentage of bins in which an isoform must be expressed
            (count greater than zero) to be retained. Values in ``[0, 1]`` are
            interpreted as fractions of bins, and values in ``(1, 100]`` are
            interpreted as percentages.

        Raises
        ------
        ValueError
            If required table/layer/metadata is missing.
        """
        if sd is None:
            raise ImportError(
                "spatialdata is required for SplisosmFFT. Please install it via 'pip install spatialdata'."
            )

        if not hasattr(sdata, "tables"):
            raise ValueError("`sdata` must provide a `tables` attribute.")
        if table_name not in sdata.tables:
            raise ValueError(f"Table `{table_name}` was not found in `sdata.tables`.")

        adata = sdata.tables[table_name]
        if not isinstance(adata, AnnData):
            raise ValueError(f"`sdata.tables[{table_name}]` must be an AnnData object.")
        if layer not in adata.layers:
            raise ValueError(f"Layer `{layer}` was not found in the AnnData table.")
        if min_counts < 0:
            raise ValueError("`min_counts` must be >= 0.")
        if min_bin_pct < 0 or min_bin_pct > 100:
            raise ValueError(
                "`min_bin_pct` must be between 0 and 1, or between 0 and 100."
            )
        if group_iso_by not in adata.var.columns:
            raise ValueError(f"`{group_iso_by}` was not found in `adata.var`.")
        if row_key not in adata.obs.columns:
            raise ValueError(f"`{row_key}` was not found in `adata.obs`.")
        if col_key not in adata.obs.columns:
            raise ValueError(f"`{col_key}` was not found in `adata.obs`.")
        if gene_names is not None and gene_names not in adata.var.columns:
            raise ValueError(
                f"`gene_names` column '{gene_names}' was not found in `adata.var`."
            )

        # Ensure selected layer is used by SpatialData rasterization.
        adata.X = adata.layers[layer]

        min_bin_frac = float(min_bin_pct)
        if min_bin_frac > 1.0:
            min_bin_frac /= 100.0

        # Per-isoform total counts for low-expression filtering.
        layer_counts = adata.layers[layer]
        total_counts = np.asarray(layer_counts.sum(axis=0)).ravel()
        if sp.issparse(layer_counts):
            bin_pct = np.asarray((layer_counts > 0).sum(axis=0)).ravel() / adata.n_obs
        else:
            bin_pct = (
                np.count_nonzero(np.asarray(layer_counts) > 0, axis=0) / adata.n_obs
            )

        var_df = adata.var.copy()
        var_df["__iso_name__"] = adata.var_names.astype(str)
        var_df["__total_counts__"] = total_counts
        var_df["__bin_pct__"] = bin_pct
        grouped = var_df.groupby(group_iso_by, observed=True, sort=False)

        grouped_iso_names: list[list[str]] = []
        grouped_gene_names: list[str] = []
        for gene, group in grouped:
            # Filter isoforms below the minimum count and prevalence thresholds.
            group = group[
                (group["__total_counts__"] >= min_counts)
                & (group["__bin_pct__"] >= min_bin_frac)
            ]
            # Genes with fewer than two remaining isoforms cannot have ratios.
            if group.shape[0] < 2:
                continue
            display_name = (
                str(group[gene_names].iloc[0]) if gene_names is not None else str(gene)
            )
            grouped_gene_names.append(display_name)
            grouped_iso_names.append(group["__iso_name__"].tolist())

        if len(grouped_iso_names) == 0:
            raise ValueError("No genes remained after grouping/filtering isoforms.")

        self.sdata = sdata
        self._adata = adata
        self._bins_name = bins
        self._table_name = table_name
        self._gene_summary = None
        self._isoform_summary = None
        self._col_key = col_key
        self._row_key = row_key
        self._counts_layer = layer
        self._group_iso_by = group_iso_by
        self._gene_iso_names = grouped_iso_names

        self._raster_key = self._rasterize_bins(
            bins=bins,
            table_name=table_name,
            col_key=col_key,
            row_key=row_key,
        )
        self._raster_layer = self.sdata[self._raster_key]
        _, ny, nx = self._raster_layer.shape

        self.n_spots = int(adata.n_obs)
        self.n_grid = int(ny * nx)
        self.n_genes = len(grouped_iso_names)
        self.n_isos = [len(v) for v in grouped_iso_names]
        self.gene_names = grouped_gene_names

        self.kernel = FFTKernel(
            shape=(ny, nx),
            spacing=self._spacing,
            rho=self._rho,
            neighbor_degree=self._neighbor_degree,
            workers=self._workers,
        )
        eigvals = self.kernel.eigenvalues()
        self._kernel_eigvals = eigvals[eigvals > 1e-8]

    def test_spatial_variability(
        self,
        method: Literal["hsic-ir", "hsic-ic", "hsic-gc"] = "hsic-ir",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        n_jobs: int = -1,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial variability using FFT-accelerated HSIC.

        Parameters
        ----------
        method
            One of ``"hsic-ir"`` (isoform ratios), ``"hsic-ic"`` (isoform counts),
            or ``"hsic-gc"`` (gene counts).
        ratio_transformation
            Ratio transform used when ``method="hsic-ir"``.
        n_jobs
            Number of joblib workers. ``-1`` uses all available CPUs.
        return_results
            If True, return result dictionary.
        print_progress
            Whether to show a progress bar.

        Returns
        -------
        dict or None
            Result dictionary when ``return_results=True``; otherwise ``None``.
        """
        if self._adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid ratio transformation. Must be one of {valid_transformations}."
            )

        if (
            self.kernel is None
            or self._kernel_eigvals is None
            or self._raster_layer is None
        ):
            raise RuntimeError(
                "Kernel/raster data is not initialized. Call setup_data() first."
            )

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        iterator = self._gene_iso_names
        if print_progress:
            iterator = tqdm(iterator, desc=f"SV ({method})")

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_sv_worker_fft)(
                self._raster_layer,
                iso_names,
                self.kernel,
                self._kernel_eigvals,
                method,
                ratio_transformation,
            )
            for iso_names in iterator
        )

        stats = np.asarray([r[0] for r in results], dtype=float)
        pvals_np = np.asarray([r[1] for r in results], dtype=float)

        self.sv_test_results = {
            "statistic": stats,
            "pvalue": pvals_np,
            "pvalue_adj": false_discovery_control(pvals_np),
            "method": method,
            "use_perm_null": False,
        }

        if return_results:
            return self.sv_test_results
        return None

    def test_differential_usage(
        self,
        design_matrix: Union[np.ndarray, "pd.DataFrame"],
        method: Literal["hsic", "hsic-gp", "t-fisher", "t-tippett"] = "hsic-gp",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        n_jobs: int = -1,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test for differential isoform usage.

        Four methods are supported:

        * ``"hsic-gp"`` (default): conditional HSIC test.  Both covariates and
          isoform ratios are residualized against the FFT spatial kernel using
          ``FFTKernelGPR`` (O(N log N) per gene, with per-gene epsilon search).
        * ``"hsic"``: unconditional HSIC test (multivariate RV coefficient) -
          no spatial conditioning.  Computed at spot level.
        * ``"t-fisher"``, ``"t-tippett"``: per-isoform two-sample t-tests whose
          p-values are combined gene-wise via Fisher's or Tippett's method.
          Requires binary (0/1) factors.  Computed at spot level.

        Parameters
        ----------
        design_matrix : np.ndarray or pd.DataFrame, shape (n_obs, n_covariates)
            Covariates of shape ``(n_obs, n_covariates)``. Must be aligned with
            ``adata.obs`` (same row order). Accepts a numpy array or a pandas
            DataFrame whose columns name the factors.
        method : str, optional
            Association test method. One of ``"hsic"``, ``"hsic-gp"``,
            ``"t-fisher"``, ``"t-tippett"``.
        ratio_transformation : str, optional
            Compositional transformation for isoform ratios. One of
            ``"none"``, ``"clr"``, ``"ilr"``, ``"alr"``, ``"radial"``.
        n_jobs : int, optional
            Number of joblib workers. ``-1`` uses all available CPUs.
        return_results : bool, optional
            If True, return the result dictionary.
        print_progress : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        results : dict or None
            Result dictionary when ``return_results=True``; otherwise ``None``.
            Keys: ``"statistic"`` (n_genes, n_covariates), ``"pvalue"``
            (n_genes, n_covariates), ``"pvalue_adj"`` (n_genes, n_covariates),
            ``"method"`` (str), ``"factor_names"`` (list[str]).

        Raises
        ------
        RuntimeError
            If ``setup_data()`` has not been called before this method.
        ValueError
            If ``method`` is not one of the supported methods, if
            ``ratio_transformation`` is not one of the supported values, or
            if ``design_matrix`` shape does not match the number of observations.
        """
        if self._adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if self.kernel is None or self._raster_layer is None:
            raise RuntimeError(
                "Kernel/raster data is not initialized. Call setup_data() first."
            )

        valid_methods = ["hsic", "hsic-gp", "t-fisher", "t-tippett"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")

        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid ratio_transformation. Must be one of {valid_transformations}."
            )

        import pandas as _pd

        if isinstance(design_matrix, _pd.DataFrame):
            factor_names = list(design_matrix.columns)
            design_np = design_matrix.values.astype(float)
        else:
            design_np = np.asarray(design_matrix, dtype=float)
            factor_names = [f"factor_{i}" for i in range(design_np.shape[1])]

        if design_np.ndim != 2:
            raise ValueError("`design_matrix` must be 2-D (n_obs, n_factors).")
        if design_np.shape[0] != self._adata.n_obs:
            raise ValueError(
                f"`design_matrix` has {design_np.shape[0]} rows but adata has "
                f"{self._adata.n_obs} observations."
            )

        n_factors = design_np.shape[1]
        ny, nx = self.kernel.ny, self.kernel.nx

        adata = self._adata
        rows_raw = np.asarray(adata.obs[self._row_key], dtype=int)
        cols_raw = np.asarray(adata.obs[self._col_key], dtype=int)
        row_offset = int(rows_raw.min())
        col_offset = int(cols_raw.min())
        rows = rows_raw - row_offset  # 0-based grid indices
        cols = cols_raw - col_offset

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        iterator = self._gene_iso_names
        prog_desc = f"DU ({method})"

        if method == "hsic-gp":
            # Build GPR using same configs as SklearnKernelGPR (RBF kernel)
            cov_cfg = {**_DEFAULT_GPR_CONFIGS["covariate"]}
            iso_cfg = {**_DEFAULT_GPR_CONFIGS["isoform"]}
            spacing = (self.kernel.dy, self.kernel.dx)

            gpr_cov = FFTKernelGPR(
                constant_value=cov_cfg["constant_value"],
                constant_value_bounds=cov_cfg["constant_value_bounds"],
                length_scale=cov_cfg["length_scale"],
                length_scale_bounds=cov_cfg["length_scale_bounds"],
            )
            gpr_iso = FFTKernelGPR(
                constant_value=iso_cfg["constant_value"],
                constant_value_bounds=iso_cfg["constant_value_bounds"],
                length_scale=iso_cfg["length_scale"],
                length_scale_bounds=iso_cfg["length_scale_bounds"],
            )

            # Rasterize and residualize covariates once (per factor)
            covariate_grid = np.zeros((ny, nx, n_factors), dtype=float)
            for _f in range(n_factors):
                covariate_grid[rows, cols, _f] = design_np[:, _f]

            z_res_list: list[np.ndarray] = []
            for _f in range(n_factors):
                z_cube = covariate_grid[:, :, _f]  # (ny, nx)
                z_res_cube, _ = gpr_cov.fit_residuals_cube(
                    z_cube, spacing=spacing
                )  # (ny, nx)
                # Normalise for numerical stability
                z_res_cube = z_res_cube - z_res_cube.mean()
                std = z_res_cube.std()
                if std > 0:
                    z_res_cube = z_res_cube / std
                z_res_list.append(z_res_cube)

            if print_progress:
                iterator = tqdm(iterator, desc=prog_desc)

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_du_worker_fft)(
                    self._raster_layer,
                    iso_names,
                    gpr_iso,
                    z_res_list,
                    ratio_transformation,
                    spacing,
                )
                for iso_names in iterator
            )

        else:  # 'hsic', 't-fisher', 't-tippett' - spot-level methods
            z_list = [design_np[:, _f] for _f in range(n_factors)]

            if print_progress:
                iterator = tqdm(iterator, desc=prog_desc)

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_du_worker_spot)(
                    self._raster_layer,
                    iso_names,
                    rows,
                    cols,
                    z_list,
                    ratio_transformation,
                    method,
                )
                for iso_names in iterator
            )

        stats = np.array([r[0] for r in results], dtype=float)  # (n_genes, n_factors)
        pvals = np.array([r[1] for r in results], dtype=float)  # (n_genes, n_factors)

        # Adjust p-values per factor (column-wise BH correction)
        pvals_adj = np.column_stack(
            [false_discovery_control(pvals[:, _f]) for _f in range(n_factors)]
        )

        self.du_test_results = {
            "statistic": stats,
            "pvalue": pvals,
            "pvalue_adj": pvals_adj,
            "method": method,
            "factor_names": factor_names,
        }

        if return_results:
            return self.du_test_results
        return None

    def get_formatted_test_results(
        self, test_type: Literal["sv", "du"]
    ) -> pd.DataFrame:
        """Get formatted test results as a pandas DataFrame.

        Parameters
        ----------
        test_type
            Test type: ``"sv"`` for spatial variability or ``"du"`` for
            differential usage.

        Returns
        -------
        pandas.DataFrame
            Formatted result table.
        """
        if test_type not in {"sv", "du"}:
            raise ValueError("Invalid test type. Must be one of 'sv' or 'du'.")

        if test_type == "sv":
            if len(self.sv_test_results) == 0:
                raise ValueError(
                    "No spatial variability results found. Run test_spatial_variability() first."
                )
            return pd.DataFrame(
                {
                    "gene": self.gene_names,
                    "statistic": self.sv_test_results["statistic"],
                    "pvalue": self.sv_test_results["pvalue"],
                    "pvalue_adj": self.sv_test_results["pvalue_adj"],
                }
            )

        if len(self.du_test_results) == 0:
            raise ValueError(
                "No differential usage results found. Run test_differential_usage() first."
            )

        du = self.du_test_results
        factor_names = du.get("factor_names", [])
        rows = []
        for _g, gene in enumerate(self.gene_names):
            for _f, factor in enumerate(factor_names):
                rows.append(
                    {
                        "gene": gene,
                        "factor": factor,
                        "statistic": du["statistic"][_g, _f],
                        "pvalue": du["pvalue"][_g, _f],
                        "pvalue_adj": du["pvalue_adj"][_g, _f],
                    }
                )
        return pd.DataFrame(rows)

    def _compute_feature_summaries(self, print_progress: bool = True) -> None:
        """Compute and cache both gene-level and isoform-level summaries."""
        if self._adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if self._gene_summary is not None and self._isoform_summary is not None:
            return

        adata = self._adata
        n_bins = int(adata.n_obs)
        var_names_index = adata.var_names
        iso_counts = adata.layers[self._counts_layer]
        is_sparse = sp.issparse(iso_counts)

        if is_sparse:
            if not sp.isspmatrix_csc(iso_counts):
                iso_counts = iso_counts.tocsc()
        else:
            iso_counts = np.asarray(iso_counts, dtype=float)

        gene_rows: list[dict[str, Any]] = []
        iso_rows: list[dict[str, Any]] = []
        all_iso_names: list[str] = []

        iterator = zip(self.gene_names, self._gene_iso_names)
        if print_progress:
            iterator = tqdm(iterator, desc="Genes", total=len(self.gene_names))

        for gene_name, iso_names in iterator:
            iso_idx = var_names_index.get_indexer(iso_names)
            if np.any(iso_idx < 0):
                raise ValueError(
                    f"Failed to locate one or more filtered isoforms for gene '{gene_name}'."
                )

            if is_sparse:
                gene_counts = iso_counts[:, iso_idx]
                iso_total = np.asarray(gene_counts.sum(axis=0), dtype=float).ravel()
                iso_sumsq = np.asarray(
                    gene_counts.power(2).sum(axis=0), dtype=float
                ).ravel()
                iso_nnz = np.diff(gene_counts.indptr).astype(float)
                row_sums = np.asarray(gene_counts.sum(axis=1), dtype=float).ravel()
            else:
                gene_counts = np.asarray(iso_counts[:, iso_idx], dtype=float)
                iso_total = gene_counts.sum(axis=0)
                iso_sumsq = np.square(gene_counts).sum(axis=0)
                iso_nnz = np.count_nonzero(gene_counts, axis=0).astype(float)
                row_sums = gene_counts.sum(axis=1)

            gene_total = float(iso_total.sum())
            valid_rows = np.flatnonzero(row_sums > 0.0)
            n_valid = int(valid_rows.size)

            iso_count_avg = iso_total / n_bins
            iso_count_var = np.maximum(
                (iso_sumsq / n_bins) - np.square(iso_count_avg), 0.0
            )
            iso_count_std = np.sqrt(iso_count_var)
            iso_pct_bin_on = iso_nnz / n_bins

            if gene_total > 0.0:
                ratio_total = iso_total / gene_total
            else:
                ratio_total = np.zeros(len(iso_names), dtype=float)

            if n_valid > 0:
                if is_sparse:
                    ratio_counts = gene_counts.tocsr()[valid_rows]
                    ratio_counts = ratio_counts.multiply(
                        (1.0 / row_sums[valid_rows])[:, None]
                    )
                    ratio_sum = np.asarray(
                        ratio_counts.sum(axis=0), dtype=float
                    ).ravel()
                    ratio_sumsq = np.asarray(
                        ratio_counts.power(2).sum(axis=0), dtype=float
                    ).ravel()
                else:
                    ratio_counts = gene_counts[valid_rows] / row_sums[valid_rows, None]
                    ratio_sum = ratio_counts.sum(axis=0)
                    ratio_sumsq = np.square(ratio_counts).sum(axis=0)

                ratio_avg = ratio_sum / n_valid
                ratio_var = np.maximum(
                    (ratio_sumsq / n_valid) - np.square(ratio_avg), 0.0
                )
                ratio_std = np.sqrt(ratio_var)
            else:
                ratio_avg = np.zeros(len(iso_names), dtype=float)
                ratio_std = np.zeros(len(iso_names), dtype=float)

            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -(np.log(ratio_total) * ratio_total)
                entropy = float(np.nan_to_num(entropy).sum())

            gene_count_avg = float(gene_total / n_bins)
            gene_count_sumsq = float(np.square(row_sums).sum())
            gene_count_var = max((gene_count_sumsq / n_bins) - (gene_count_avg**2), 0.0)

            gene_rows.append(
                {
                    "gene": gene_name,
                    "n_isos": len(iso_names),
                    "perplexity": float(np.exp(entropy)),
                    "pct_bin_on": float(n_valid / n_bins),
                    "count_avg": gene_count_avg,
                    "count_std": float(np.sqrt(gene_count_var)),
                }
            )

            all_iso_names.extend(iso_names)
            for (
                iso_name,
                pct_bin_on,
                count_total,
                count_avg,
                count_std,
                iso_ratio_total,
                iso_ratio_avg,
                iso_ratio_std,
            ) in zip(
                iso_names,
                iso_pct_bin_on,
                iso_total,
                iso_count_avg,
                iso_count_std,
                ratio_total,
                ratio_avg,
                ratio_std,
            ):
                iso_rows.append(
                    {
                        "isoform": iso_name,
                        "pct_bin_on": float(pct_bin_on),
                        "count_total": float(count_total),
                        "count_avg": float(count_avg),
                        "count_std": float(count_std),
                        "ratio_total": float(iso_ratio_total),
                        "ratio_avg": float(iso_ratio_avg),
                        "ratio_std": float(iso_ratio_std),
                    }
                )

        self._gene_summary = pd.DataFrame(gene_rows).set_index("gene")

        var_df = adata.var.loc[all_iso_names].copy()
        stats_df = pd.DataFrame(iso_rows).set_index("isoform")
        self._isoform_summary = pd.concat([var_df, stats_df], axis=1)

    def extract_feature_summary(
        self,
        level: Literal["gene", "isoform"] = "gene",
        print_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute filtered feature-level summary statistics.

        Gene-level statistics are aggregated across all isoforms that passed
        the filters applied in :meth:`setup_data`.  Isoform-level statistics
        are computed per isoform and augmented onto the corresponding rows of
        ``adata.var``.

        Results are cached: repeated calls with the same ``level`` return the
        cached :class:`pandas.DataFrame` without recomputation.

        Parameters
        ----------
        level
            Summary granularity.
            ``'gene'``: one row per gene.
            ``'isoform'``: one row per isoform that passed filtering.
        print_progress
            Whether to show a progress bar.

        Returns
        -------
        pandas.DataFrame
            For ``level='gene'``, the index is the gene display name and the
            columns are:

            - ``'n_isos'``: int. Number of isoforms retained after filtering.
            - ``'perplexity'``: float. Effective number of isoforms based on
              the marginal isoform usage entropy.
            - ``'pct_bin_on'``: float. Fraction of bins with non-zero total
              gene counts.
            - ``'count_avg'``: float. Mean per-spot total count for the gene.
            - ``'count_std'``: float. Std of per-spot total count for the gene.

            For ``level='isoform'``, the index is the isoform name (matching
            ``adata.var_names``) and the columns are the original ``adata.var``
            columns plus:

            - ``'pct_bin_on'``: float. Fraction of bins with count > 0.
            - ``'count_total'``: float. Total counts across all bins.
            - ``'count_avg'``: float. Mean count per bin.
            - ``'count_std'``: float. Std of count per bin.
            - ``'ratio_total'``: float. Fraction of total gene counts
              attributable to this isoform.
            - ``'ratio_avg'``: float. Mean per-bin isoform usage ratio
              (computed over bins with non-zero gene coverage).
            - ``'ratio_std'``: float. Std of per-bin isoform usage ratio
              (computed over bins with non-zero gene coverage).

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        ValueError
            If ``level`` is not ``'gene'`` or ``'isoform'``.
        """
        if self._adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if level not in {"gene", "isoform"}:
            raise ValueError("`level` must be one of 'gene' or 'isoform'.")

        self._compute_feature_summaries(print_progress=print_progress)

        if level == "gene":
            return self._gene_summary
        return self._isoform_summary
