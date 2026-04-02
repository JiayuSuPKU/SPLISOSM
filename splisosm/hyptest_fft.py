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
from tqdm.auto import tqdm

from splisosm.kernel_gpr import (
    FFTKernelOp,
    FFTKernelGPR,
    _DEFAULT_GPR_CONFIGS,
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
    method: str,
    gpr_iso_cfg: Optional[dict],
    z_list: list[np.ndarray],
    ratio_transformation: str,
    spacing: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DU statistics and p-values for one gene against all factors.

    Dispatches to one of four test strategies based on ``method``:

    - ``"hsic-gp"``: spatially residualize covariates (and optionally isoform
      ratios) with ``FFTKernelGPR``, then compute linear HSIC.
    - ``"hsic"``: linear HSIC on raw (centered) isoform ratios and covariates
      without any spatial residualization.
    - ``"t-fisher"``: per-isoform two-sample t-test (binary covariates only)
      combined with Fisher's method (chi-squared statistic, df = 2 × n_isoforms).
    - ``"t-tippett"``: per-isoform two-sample t-test (binary covariates only)
      combined with Tippett's corrected minimum p-value method.

    Parameters
    ----------
    raster_layer : array-like
        Lazy raster layer with ``.sel(c=...)`` API; channels are loaded
        on-the-fly for this gene only.
    iso_names : list of str
        Isoform channel names to select from the raster.
    method : str
        Test method: ``"hsic-gp"``, ``"hsic"``, ``"t-fisher"``, or
        ``"t-tippett"``.
    gpr_iso_cfg : dict or None
        ``FFTKernelGPR`` kwargs for isoform residualization (``"hsic-gp"``
        with ``residualize="both"`` only). ``None`` otherwise.
    z_list : list of np.ndarray
        Per-factor covariate grids, each shape ``(ny, nx)``.  For
        ``"hsic-gp"`` these are GPR-residualized and standardized; for
        ``"hsic"`` they are centered raw grids; for ``"t-fisher"`` and
        ``"t-tippett"`` they are raw grids (NaN-masked, not centered) with
        exactly two distinct non-NaN values (validated by the caller).
    ratio_transformation : str
        Compositional transformation applied before ratio computation.
    spacing : tuple of float
        Physical ``(dy, dx)`` spacing in kernel length-scale units.

    Returns
    -------
    stats : np.ndarray, shape (n_covariates,)
    pvals : np.ndarray, shape (n_covariates,)
    """
    from scipy.stats import chi2 as _chi2_dist

    n_factors = len(z_list)
    stats = np.zeros(n_factors)
    pvals = np.ones(n_factors)

    # ── Load isoform ratios on-the-fly ────────────────────────────────────
    data = raster_layer.sel(c=iso_names).values  # (n_iso, ny, nx)
    counts_cube = np.moveaxis(np.asarray(data, dtype=float), 0, -1)  # (ny, nx, n_iso)
    ny_g, nx_g = counts_cube.shape[:2]
    n_grid = ny_g * nx_g
    n_iso = counts_cube.shape[2]

    counts_flat = counts_cube.reshape(n_grid, n_iso)
    ratios = counts_to_ratios(
        torch.from_numpy(counts_flat).float(),
        transformation=ratio_transformation,
        nan_filling="none",
        fill_before_transform=False,
    )
    y_cube = ratios.numpy().reshape(ny_g, nx_g, -1)

    # ── HSIC-based methods (hsic-gp and hsic) ────────────────────────────
    if method in ("hsic-gp", "hsic"):
        if method == "hsic-gp" and gpr_iso_cfg is not None:
            gpr_iso = FFTKernelGPR(**gpr_iso_cfg)
            y_res_cube, _ = gpr_iso.fit_residuals_cube(y_cube, spacing=spacing)
        else:
            y_res_cube = y_cube - np.nanmean(y_cube, axis=(0, 1), keepdims=True)
            y_res_cube = np.nan_to_num(y_res_cube, nan=0.0)

        y_res = y_res_cube.reshape(n_grid, -1)  # (n_grid, n_iso)

        gram_y = y_res.T @ y_res
        if not np.isfinite(gram_y).all():
            return stats, pvals
        try:
            lambda_y = np.linalg.eigvalsh(gram_y)
        except np.linalg.LinAlgError:
            return stats, pvals
        lambda_y = lambda_y[lambda_y > 1e-8]

        for _f, z_cube in enumerate(z_list):
            z_flat = z_cube.ravel()[:, None]
            cross = y_res.T @ z_flat
            hsic_scaled = float(np.sum(cross**2))
            stats[_f] = hsic_scaled / (n_grid - 1) ** 2
            lambda_z = np.linalg.eigvalsh(z_flat.T @ z_flat)
            lambda_z = lambda_z[lambda_z > 1e-8]
            if lambda_z.size == 0 or lambda_y.size == 0:
                continue
            lambda_mix = (lambda_z[:, None] * lambda_y[None, :]).reshape(-1)
            pvals[_f] = float(liu_sf(hsic_scaled * n_grid, lambda_mix))

    # ── Per-isoform two-sample t-test (binary covariates only) ───────────
    else:
        from scipy.stats import ttest_ind as _ttest_ind

        # Use raw isoform ratios (not centered) for two-group comparison
        y_flat = y_cube.reshape(n_grid, -1)  # (n_grid, n_iso)

        for _f, z_cube_f in enumerate(z_list):
            z_flat = z_cube_f.ravel()  # (n_grid,) — raw binary values, NaN for missing

            # Identify the two binary groups from valid (non-NaN) pixels.
            # Binariness has already been validated by the caller.
            valid_mask = np.isfinite(z_flat)
            unique_vals = np.unique(z_flat[valid_mask])
            threshold = (unique_vals[0] + unique_vals[1]) / 2.0
            g0_mask = valid_mask & (z_flat < threshold)
            g1_mask = valid_mask & (z_flat >= threshold)

            pvals_iso = np.ones(n_iso)

            for k in range(n_iso):
                y_k = y_flat[:, k]
                y0 = y_k[g0_mask & np.isfinite(y_k)]
                y1 = y_k[g1_mask & np.isfinite(y_k)]

                if len(y0) < 3 or len(y1) < 3:
                    continue
                t_stat, p = _ttest_ind(y0, y1)
                if np.isfinite(t_stat) and np.isfinite(p):
                    pvals_iso[k] = float(p)

            if method == "t-fisher":
                chi2_stat = float(-2.0 * np.sum(np.log(pvals_iso + 1e-300)))
                stats[_f] = chi2_stat
                pvals[_f] = float(_chi2_dist.sf(chi2_stat, df=2 * n_iso))
            else:  # t-tippett: corrected minimum p-value
                min_p = float(np.min(pvals_iso))
                stats[_f] = float(-np.log10(min_p + 1e-300))
                pvals[_f] = float(1.0 - (1.0 - min_p) ** n_iso)

    return stats, pvals


def _sv_worker_fft(
    raster_layer: Any,
    iso_names: list[str],
    kernel: FFTKernel,
    kernel_eigvals: np.ndarray,
    method: Literal["hsic-ir", "hsic-ic", "hsic-gc"],
    ratio_transformation: str,
) -> tuple[float, float]:
    """Compute one gene-level FFT-HSIC statistic and p-value from rasterized channels."""
    data = raster_layer.sel(c=iso_names).values  # (c, ny, nx)
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
            fill_before_transform=False,
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

    Spatial variability test:

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

    Differential usage test:

    >>> model = SplisosmFFT(rho=0.9, neighbor_degree=1)
    >>> model.setup_data(
    ...     sdata=sdata,
    ...     bins="ID_square_016um",
    ...     table_name="square_016um_svp",
    ...     design_mtx="square_016um_rbp_sve",
    ...     col_key="array_col",
    ...     row_key="array_row",
    ...     layer="counts",
    ...     group_iso_by="gene_ids",
    ...     gene_names="gene_name",
    ...     min_counts=10,
    ...     min_bin_pct=0.0,
    ... )
    >>> model.test_differential_usage(method="hsic-gp", residualize="cov_only")
    >>> du_results = model.get_formatted_test_results("du")
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

    n_factors: int
    """Number of covariates to test for differential usage."""

    covariate_names: list[str]
    """List of covariate names corresponding to columns of the design matrix."""

    design_mtx: Optional[Any]
    """Rasterized design matrix stored as an :class:`anndata.AnnData` table
    inside :attr:`sdata`; ``None`` if no covariates were provided to
    :meth:`setup_data`.  Use :attr:`covariate_names` and :attr:`n_factors`
    to inspect the covariate layout without accessing this object directly."""

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
    - ``'statistic'``: numpy.ndarray of shape (n_genes, n_factors), the test statistic for each gene and covariate.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes, n_factors), the p-value for each gene and covariate.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes, n_factors), the BH adjusted p-value for each gene and covariate. Each column/covariate is adjusted separately.
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
        self._gene_iso_names: list[list[str]] = []

        self.kernel: FFTKernel | None = None
        self._kernel_eigvals: np.ndarray | None = None
        self._raster_key: str | None = None
        self._raster_layer: Any | None = None

        self.sv_test_results: dict[str, Any] = {}
        self.du_test_results: dict[str, Any] = {}

        self._gene_summary: Optional[pd.DataFrame] = None
        self._isoform_summary: Optional[pd.DataFrame] = None

        self.design_mtx: Optional[Any] = None
        self._design_table_name: Optional[str] = None
        self.covariate_names: list[str] = []
        self.n_factors: int = 0

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
            f"- Number of covariates: {self.n_factors}\n"
            f"- Average number of isoforms per gene: {np.mean(self.n_isos) if self.n_isos else None}\n"
            "=== Test results\n"
            f"- Spatial variability test: {sv_status}\n"
            f"- Differential usage test: {du_status}"
        )

    def _rasterize_bins(self, bins, table_name, col_key, row_key) -> str:
        """Rasterize bins and cache image key in sdata."""
        if sd is None:
            raise ImportError("spatialdata is required.")
        if self.sdata is None:
            raise RuntimeError("Call setup_data() first.")

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
        filter_single_iso_genes: bool = True,
        design_mtx: Optional[Union[np.ndarray, "pd.DataFrame", str, list[str]]] = None,
        covariate_names: Optional[list[str]] = None,
    ) -> None:
        """Setup SpatialData-backed isoform data for FFT-based testing.

        (bins, table_name, col_key, row_key) are passed to
        :func:`spatialdata.rasterize_bins` to rasterize isoform counts.

        Parameters
        ----------
        sdata : spatialdata.SpatialData
            SpatialData-like object with ``tables`` mapping.
        bins : str
            Name of the SpatialData bin geometry for rasterization.
        table_name : str
            Key of the table in ``sdata.tables``.
        col_key : str
            Column index key in ``adata.obs`` for rasterization.
        row_key : str
            Row index key in ``adata.obs`` for rasterization.
        layer : str, optional
            AnnData layer that stores isoform count matrix.
        group_iso_by : str, optional
            Column in ``adata.var`` used to group isoforms by gene. The
            unique values of this column define the gene-level groups.
        gene_names : str or None, optional
            Optional column name in ``adata.var`` whose values are used as
            display gene names in results. If ``None``, the values of
            ``group_iso_by`` are used directly.
        min_counts : int, optional
            Minimum total count (summed across all spots) required for an
            isoform to be retained. Isoforms below this threshold are
            excluded before gene grouping. Genes with fewer than two
            remaining isoforms after filtering are also excluded.
        min_bin_pct : float, optional
            Minimum percentage of bins in which an isoform must be expressed
            (count greater than zero) to be retained. Values in ``[0, 1]`` are
            interpreted as fractions of bins, and values in ``(1, 100]`` are
            interpreted as percentages.
        filter_single_iso_genes : bool, optional
            If ``True`` (default), genes with fewer than two isoforms passing
            QC filters are removed — they cannot contribute to within-gene ratio
            tests.  Set to ``False`` to keep single-isoform genes, e.g. when
            testing **gene-level expression variability** with
            ``test_spatial_variability(method="hsic-gc")``.
        design_mtx : str, list[str], np.ndarray, scipy.sparse matrix, pd.DataFrame, or None
            Design matrix specification.  Three input modes:

            1. **Table name** (``str`` matching a key in ``sdata.tables``): Use the
               existing AnnData table's ``X`` as the design matrix.  Must have the
               same number of observations as the isoform table.
            2. **Obs column names** (``str`` or ``list[str]`` not matching a table):
               Extract the named columns from the isoform table's ``adata.obs``.
               Categorical columns are one-hot encoded automatically.
            3. **Pre-computed matrix** (ndarray, sparse, or DataFrame of shape
               ``(n_obs, n_factors)``): Used as-is.

            In cases 2 and 3, the design matrix will be stored as a new AnnData table inside
            ``sdata``. The matrix is also rasterized via :func:`spatialdata.rasterize_bins`
            when :meth:`test_differential_usage` is called.
        covariate_names : list[str] or None, optional
            Factor names. Will override inferred names.
            If None, inferred from ``design_mtx`` column names when possible;
            otherwise auto-generated as ``["factor_0", ...]``.

        Raises
        ------
        ValueError
            If required table/layer/metadata is missing.

        See Also
        --------
        :func:`splisosm.hyptest_np.SplisosmNP.setup_data` : AnnData-based setup for data with general geometry.

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
            # Genes with fewer than two isoforms cannot have within-gene ratios
            # unless the caller explicitly keeps single-isoform genes.
            if filter_single_iso_genes and group.shape[0] < 2:
                continue
            if group.shape[0] < 1:
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

        # --- Process design_mtx: store as properly-structured AnnData table ---
        import scipy.sparse as _sp
        from splisosm.utils import _process_design_mtx

        design_key = f"_splisosm_design_{table_name}"
        if design_mtx is not None:
            # Mode A: str matching an existing sdata table — no duplication
            if isinstance(design_mtx, str) and design_mtx in sdata.tables:
                ref_table = sdata.tables[design_mtx]
                if ref_table.n_obs != adata.n_obs:
                    raise ValueError(
                        f"Design table '{design_mtx}' has {ref_table.n_obs} observations "
                        f"but the isoform table has {adata.n_obs}."
                    )
                if covariate_names is None:
                    covariate_names = list(ref_table.var_names)
                elif len(covariate_names) != ref_table.n_vars:
                    raise ValueError(
                        f"covariate_names length ({len(covariate_names)}) must match "
                        f"design table n_vars ({ref_table.n_vars})."
                    )
                self._design_table_name = design_mtx
                self.design_mtx = ref_table
                n_factors = ref_table.n_vars
            else:
                # Mode B/C: obs column names or pre-computed matrix
                dm_np, covariate_names = _process_design_mtx(
                    adata, design_mtx, covariate_names
                )

                # Use sparse X when density < 50%
                density = (
                    np.count_nonzero(dm_np) / dm_np.size if dm_np.size > 0 else 1.0
                )
                x_data = _sp.csr_matrix(dm_np) if density < 0.5 else dm_np

                # Build obs with spatial indexing columns + spatialdata link columns
                obs_cols = [row_key, col_key]
                attrs = adata.uns.get("spatialdata_attrs", {})
                region_key_col = attrs.get("region_key", "region")
                instance_key_col = attrs.get("instance_key", "instance_id")
                for _col in [region_key_col, instance_key_col]:
                    if _col in adata.obs.columns:
                        obs_cols.append(_col)
                obs_for_design = adata.obs[obs_cols].copy()

                # Add region/instance columns if absent (required by TableModel)
                if region_key_col not in obs_for_design.columns:
                    obs_for_design[region_key_col] = pd.Categorical(
                        [bins] * len(obs_for_design)
                    )
                if instance_key_col not in obs_for_design.columns:
                    obs_for_design[instance_key_col] = np.arange(
                        len(obs_for_design), dtype=int
                    )

                design_adata = AnnData(X=x_data, obs=obs_for_design)
                design_adata.var_names = list(covariate_names)

                # Parse with TableModel to establish spatialdata link
                try:
                    from spatialdata.models import TableModel

                    design_adata = TableModel.parse(
                        design_adata,
                        region=bins,
                        region_key=region_key_col,
                        instance_key=instance_key_col,
                    )
                except Exception:
                    pass  # Store without metadata if TableModel unavailable

                sdata[design_key] = design_adata
                self._design_table_name = design_key
                self.design_mtx = design_adata
                n_factors = dm_np.shape[1]

            self.covariate_names = list(covariate_names)
            self.n_factors = n_factors
        else:
            self._design_table_name = None
            self.design_mtx = None
            self.covariate_names = list(covariate_names) if covariate_names else []
            self.n_factors = 0

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
        method : {"hsic-ir", "hsic-ic", "hsic-gc"}, optional
            One of ``"hsic-ir"`` (isoform ratios), ``"hsic-ic"`` (isoform counts),
            or ``"hsic-gc"`` (gene counts).
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Ratio transform used when ``method="hsic-ir"``.
        n_jobs : int, optional
            Number of joblib workers. ``-1`` uses all available CPUs.
        return_results : bool, optional
            If True, return result dictionary.
        print_progress : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        dict or None
            Result dictionary when ``return_results=True``; otherwise ``None``.

        See Also
        --------
        :func:`splisosm.hyptest_np.SplisosmNP.test_spatial_variability`: Non-FFT version of this function for comparison.
        """
        if self._adata is None or self.kernel is None or self._raster_layer is None:
            raise RuntimeError("Data not initialized. Call setup_data() first.")

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid ratio transformation. Must be one of {valid_transformations}."
            )

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        iterator = tqdm(
            self._gene_iso_names,
            desc=f"SV [{method}]",
            total=len(self._gene_iso_names),
            disable=not print_progress,
        )

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
        method: Literal["hsic", "hsic-gp", "t-fisher", "t-tippett"] = "hsic-gp",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        gpr_configs: Optional[dict[str, Any]] = None,
        residualize: Literal["cov_only", "both"] = "cov_only",
        n_jobs: int = -1,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test for differential isoform usage against spatial covariate expression.

        Before running this function, the design matrix must be set up using :func:`setup_data`.
        Each column of the design matrix corresponds to a covariate to test for differential
        association with the isoform usage ratios of each gene.
        Test statistics and p-values are computed per (gene, covariate) pair separately.

        Four test strategies are supported, all operating on rasterized grid data
        to avoid densifying the full isoform or covariate matrix in memory:

        - ``"hsic-gp"`` *(default)*: spatially residualize covariates (and
          optionally isoform ratios) with ``FFTKernelGPR``, then compute linear
          HSIC.  Controlled by ``residualize``.
        - ``"hsic"``: linear HSIC between raw centered isoform ratios and raw
          centered covariates—no spatial residualization.
        - ``"t-fisher"``: per-isoform two-sample t-tests (**binary covariates
          only**) combined by Fisher's method (chi-squared, df = 2 × n_isoforms).
        - ``"t-tippett"``: per-isoform two-sample t-tests (**binary covariates
          only**) combined by Tippett's corrected minimum p-value.

        Regardless of method, covariates are processed in chunks of at most 100
        at a time and isoform data is loaded on-the-fly per gene so that neither
        the full covariate grid nor the full isoform matrix is held in memory
        simultaneously.

        Parameters
        ----------
        method : str, optional
            Method for association testing:

            * ``"hsic"``: Unconditional HSIC test (multivariate RV coefficient).
              For continuous factors, equivalent to the multivariate Pearson correlation
              test.  For binary factors, equivalent to the two-sample Hotelling T**2 test.
            * ``"hsic-gp"``: Conditional HSIC test.  Spatial effects are removed via
              Gaussian process regression before computing the HSIC statistic.

            Or one of the T-tests (binary factors only):

            * ``"t-fisher"``, ``"t-tippett"``: two-sample t-test per isoform
              (binary covariates only — exactly two distinct non-NaN values required);
              p-values are combined gene-wise via Fisher's chi-squared or
              Tippett's corrected minimum method.
        ratio_transformation : str, optional
            Compositional transformation for isoform ratios.
            One of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, ``'radial'``
            :cite:`park2022kernel`.  See :func:`splisosm.utils.counts_to_ratios`.
        gpr_configs : dict, optional
            Nested configuration dict for the GPR objects, with optional keys
            ``'covariate'`` and/or ``'isoform'``.  Each sub-dict is forwarded to
            :func:`splisosm.kernel_gpr.make_kernel_gpr`.  Unspecified keys use the
            defaults from :data:`splisosm.kernel_gpr._DEFAULT_GPR_CONFIGS`::

                {
                    "covariate": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                    },
                    "isoform": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                    },
                }

        residualize : {"cov_only", "both"}, optional
            Controls which signals are spatially residualized when
            ``method="hsic-gp"``:

            * ``"cov_only"`` (default): residualize covariates only; test
              HSIC(Z_res, Y_raw).  Fastest; calibration matches ``"both"``
              when covariate GPR captures most spatial confounding.
            * ``"both"``: residualize both covariates and isoform ratios.

        n_jobs : int, optional
            Number of parallel jobs. ``-1`` uses all available CPUs.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        return_results : bool, optional
            Whether to return the test statistics and p-values.
            If False, the results are stored in ``self.du_test_results``.

        Returns
        -------
        results : dict or None
            If ``return_results`` is True, returns dict with test statistics and
            p-values. Otherwise, returns None and stores results in
            ``self.du_test_results``.

        Raises
        ------
        RuntimeError
            If ``setup_data()`` or the ``design_mtx`` argument has not been set.
        ValueError
            If ``method``, ``residualize``, or ``ratio_transformation`` is invalid.

        See Also
        --------
        :func:`splisosm.hyptest_np.SplisosmNP.test_differential_usage` : Non-FFT version of this function for comparison.
        """
        if self._adata is None or self.kernel is None or self._raster_layer is None:
            raise RuntimeError("Data not initialized. Call setup_data() first.")
        if self._design_table_name is None:
            raise RuntimeError(
                "No design matrix found. Pass `design_mtx` to setup_data() before "
                "calling test_differential_usage()."
            )

        valid_methods = ["hsic", "hsic-gp", "t-fisher", "t-tippett"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}.")
        valid_residualize = ["cov_only", "both"]
        if residualize not in valid_residualize:
            raise ValueError(f"residualize must be one of {valid_residualize}.")
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                f"ratio_transformation must be one of {valid_transformations}."
            )

        factor_names = self.covariate_names or [
            f"factor_{i}" for i in range(self.n_factors)
        ]
        n_factors = self.n_factors
        spacing = (self.kernel.dy, self.kernel.dx)

        # GPR is only needed for "hsic-gp"
        use_gpr_cov = method == "hsic-gp"
        gpr_cov = None
        gpr_iso_cfg = None
        if use_gpr_cov:
            _cov_cfg = {**_DEFAULT_GPR_CONFIGS["covariate"]}
            _iso_cfg = {**_DEFAULT_GPR_CONFIGS["isoform"]}
            if gpr_configs is not None:
                if "covariate" in gpr_configs:
                    _cov_cfg.update(gpr_configs["covariate"])
                if "isoform" in gpr_configs:
                    _iso_cfg.update(gpr_configs["isoform"])
            gpr_cov = FFTKernelGPR(
                constant_value=_cov_cfg["constant_value"],
                constant_value_bounds=_cov_cfg["constant_value_bounds"],
                length_scale=_cov_cfg["length_scale"],
                length_scale_bounds=_cov_cfg["length_scale_bounds"],
            )
            if residualize == "both":
                gpr_iso_cfg = dict(
                    constant_value=_iso_cfg["constant_value"],
                    constant_value_bounds=_iso_cfg["constant_value_bounds"],
                    length_scale=_iso_cfg["length_scale"],
                    length_scale_bounds=_iso_cfg["length_scale_bounds"],
                )

        # Lazy-rasterize design table once; channels loaded per-chunk below
        dm_raster = sd.rasterize_bins(
            self.sdata,
            self._bins_name,
            self._design_table_name,
            self._col_key,
            self._row_key,
        )

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        n_genes = len(self._gene_iso_names)
        stats = np.zeros((n_genes, n_factors))
        pvals = np.ones((n_genes, n_factors))

        # Process factors in chunks (≤100) to bound covariate-residual memory
        _COV_CHUNK = 100
        n_chunks = max(1, (n_factors + _COV_CHUNK - 1) // _COV_CHUNK)
        show_gene_bar = print_progress and n_chunks == 1
        show_chunk_bar = print_progress and n_chunks > 1

        chunk_range = tqdm(
            range(0, n_factors, _COV_CHUNK),
            desc="Covariates",
            total=n_chunks,
            disable=not show_chunk_bar,
        )

        for f_start in chunk_range:
            f_end = min(f_start + _COV_CHUNK, n_factors)
            chunk_names = factor_names[f_start:f_end]
            chunk_size = f_end - f_start

            # Load this covariate chunk from lazy raster: (ny, nx, chunk_size)
            chunk_vals = np.moveaxis(
                np.asarray(dm_raster.sel(c=chunk_names).values, dtype=float), 0, -1
            )
            z_chunk: list[np.ndarray] = []
            for _i in range(chunk_size):
                z_cube = chunk_vals[:, :, _i]
                if use_gpr_cov:
                    # GPR-residualize and standardize for "hsic-gp"
                    z_res, _ = gpr_cov.fit_residuals_cube(z_cube, spacing=spacing)
                    z_res = z_res - z_res.mean()
                    _std = z_res.std()
                    if _std > 0:
                        z_res /= _std
                elif method == "hsic":
                    # Center for "hsic" (HSIC requires zero-mean inputs)
                    z_res = z_cube - np.nanmean(z_cube)
                    z_res = np.nan_to_num(z_res, nan=0.0)
                else:
                    # "t-fisher" / "t-tippett": validate binary covariate then
                    # pass raw values so that group membership is preserved.
                    z_valid_vals = z_cube[np.isfinite(z_cube)]
                    unique_vals = np.unique(z_valid_vals)
                    if len(unique_vals) > 2:
                        raise ValueError(
                            f"More than two groups detected for factor "
                            f"'{chunk_names[_i]}'. Only binary covariates "
                            "(exactly two distinct values) are supported for "
                            f"'{method}'."
                        )
                    z_res = z_cube.copy()
                z_chunk.append(z_res)

            gene_iter = tqdm(
                self._gene_iso_names,
                desc=f"DU [{method}]",
                total=len(self._gene_iso_names),
                disable=not show_gene_bar,
            )

            chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_du_worker_fft)(
                    self._raster_layer,
                    iso_names,
                    method,
                    gpr_iso_cfg,
                    z_chunk,
                    ratio_transformation,
                    spacing,
                )
                for iso_names in gene_iter
            )

            for _g, (s, p) in enumerate(chunk_results):
                stats[_g, f_start:f_end] = s
                pvals[_g, f_start:f_end] = p

        pvals_adj = np.column_stack(
            [false_discovery_control(pvals[:, _f]) for _f in range(n_factors)]
        )

        self.du_test_results = {
            "statistic": stats,
            "pvalue": pvals,
            "pvalue_adj": pvals_adj,
            "method": method,
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
        test_type : {"sv", "du"}
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
        covariate_names = self.covariate_names or [
            f"factor_{i}" for i in range(self.n_factors)
        ]
        rows = []
        for _g, gene in enumerate(self.gene_names):
            for _f, covariate in enumerate(covariate_names):
                rows.append(
                    {
                        "gene": gene,
                        "covariate": covariate,
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

        iterator = tqdm(
            zip(self.gene_names, self._gene_iso_names),
            desc="Genes",
            total=len(self.gene_names),
            disable=not print_progress,
        )

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
