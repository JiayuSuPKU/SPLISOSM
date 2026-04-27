"""FFT-accelerated non-parametric hypothesis tests for SPLISOSM."""

# ruff: noqa: E402

from __future__ import annotations

import os
import warnings
from typing import Any, Literal, Optional, Union

# Suppress deprecations from SpatialData stack before optional imports.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm

from splisosm.hyptest._base import _FeatureSummaryMixin, _ResultsMixin
from splisosm.utils._chunking import pack_gene_chunks, resolve_chunk_size
from splisosm.kernel import FFTKernel
from splisosm.utils.hsic import (
    _cumulants_from_eigenvalues,
    _feature_cumulants_from_data,
    _hsic_liu_pvalue,
    liu_sf,
)
from splisosm.gpr.config import _DEFAULT_GPR_CONFIGS
from splisosm.gpr.fft import FFTKernelGPR
from splisosm.utils.preprocessing import (
    counts_to_ratios,
)
from splisosm.utils.stats import (
    false_discovery_control,
)

__all__ = ["SplisosmFFT"]


def _require_spatialdata() -> Any:
    """Import spatialdata lazily for FFT setup/rasterization paths."""
    try:
        import spatialdata as sd
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required for SplisosmFFT. "
            "Please install it via 'pip install spatialdata'."
        ) from exc
    return sd


def _load_fft_ratio_cube(
    raster_layer: Any,
    iso_names: list[str],
    ratio_transformation: str,
) -> tuple[np.ndarray, int, int]:
    """Load one gene's rasterized counts and return a ratio cube."""
    data = raster_layer.sel(c=iso_names).values
    counts_cube = np.moveaxis(np.asarray(data, dtype=float), 0, -1)
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
    return ratios.numpy().reshape(ny_g, nx_g, -1), n_grid, n_iso


def _du_hsic_worker_fft(
    y_cube: np.ndarray,
    method: str,
    gpr_iso_cfg: Optional[dict],
    z_list: list[np.ndarray],
    spacing: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT DU statistics for HSIC and HSIC-GP."""
    n_factors = len(z_list)
    stats = np.zeros(n_factors)
    pvals = np.ones(n_factors)
    n_grid = int(np.prod(y_cube.shape[:2]))

    if method == "hsic-gp" and gpr_iso_cfg is not None:
        gpr_iso = FFTKernelGPR(**gpr_iso_cfg)
        y_res_cube, _ = gpr_iso.fit_residuals_cube(y_cube, spacing=spacing)
    else:
        y_res_cube = y_cube - np.nanmean(y_cube, axis=(0, 1), keepdims=True)
        y_res_cube = np.nan_to_num(y_res_cube, nan=0.0)

    y_res = y_res_cube.reshape(n_grid, -1)
    gram_y = y_res.T @ y_res
    if not np.isfinite(gram_y).all():
        return stats, pvals
    try:
        lambda_y = np.linalg.eigvalsh(gram_y)
    except np.linalg.LinAlgError:
        return stats, pvals
    lambda_y = lambda_y[lambda_y > 1e-8]

    for factor_idx, z_cube in enumerate(z_list):
        z_flat = z_cube.ravel()[:, None]
        cross = y_res.T @ z_flat
        hsic_scaled = float(np.sum(cross**2))
        stats[factor_idx] = hsic_scaled / (n_grid - 1) ** 2
        lambda_z = np.linalg.eigvalsh(z_flat.T @ z_flat)
        lambda_z = lambda_z[lambda_z > 1e-8]
        if lambda_z.size == 0 or lambda_y.size == 0:
            continue
        lambda_mix = (lambda_z[:, None] * lambda_y[None, :]).reshape(-1)
        pvals[factor_idx] = float(liu_sf(hsic_scaled * n_grid, lambda_mix))

    return stats, pvals


def _du_ttest_worker_fft(
    y_cube: np.ndarray,
    method: str,
    z_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT DU statistics for binary-covariate t-test methods."""
    from scipy.stats import chi2 as _chi2_dist
    from scipy.stats import ttest_ind as _ttest_ind

    n_factors = len(z_list)
    stats = np.zeros(n_factors)
    pvals = np.ones(n_factors)
    y_flat = y_cube.reshape(-1, y_cube.shape[2])
    n_iso = y_cube.shape[2]

    for factor_idx, z_cube in enumerate(z_list):
        z_flat = z_cube.ravel()
        valid_mask = np.isfinite(z_flat)
        unique_vals = np.unique(z_flat[valid_mask])
        threshold = (unique_vals[0] + unique_vals[1]) / 2.0
        g0_mask = valid_mask & (z_flat < threshold)
        g1_mask = valid_mask & (z_flat >= threshold)
        pvals_iso = np.ones(n_iso)

        for iso_idx in range(n_iso):
            y_k = y_flat[:, iso_idx]
            y0 = y_k[g0_mask & np.isfinite(y_k)]
            y1 = y_k[g1_mask & np.isfinite(y_k)]
            if len(y0) < 3 or len(y1) < 3:
                continue
            t_stat, pval = _ttest_ind(y0, y1)
            if np.isfinite(t_stat) and np.isfinite(pval):
                pvals_iso[iso_idx] = float(pval)

        if method == "t-fisher":
            chi2_stat = float(-2.0 * np.sum(np.log(pvals_iso + 1e-300)))
            stats[factor_idx] = chi2_stat
            pvals[factor_idx] = float(_chi2_dist.sf(chi2_stat, df=2 * n_iso))
        else:
            min_p = float(np.min(pvals_iso))
            stats[factor_idx] = float(-np.log10(min_p + 1e-300))
            pvals[factor_idx] = float(1.0 - (1.0 - min_p) ** n_iso)

    return stats, pvals


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
    n_factors = len(z_list)
    stats = np.zeros(n_factors)
    pvals = np.ones(n_factors)

    # Single-isoform gene: no differential usage is possible.
    if len(iso_names) <= 1:
        return stats, pvals

    y_cube, _, _ = _load_fft_ratio_cube(raster_layer, iso_names, ratio_transformation)
    if method in ("hsic-gp", "hsic"):
        return _du_hsic_worker_fft(y_cube, method, gpr_iso_cfg, z_list, spacing)
    return _du_ttest_worker_fft(y_cube, method, z_list)


def _sv_response_width_fft(
    iso_names: list[str],
    method: Literal["hsic-ir", "hsic-ic", "hsic-gc"],
    ratio_transformation: str,
) -> int:
    """Return response-channel width contributed by one FFT gene."""
    if method == "hsic-gc":
        return 1
    n_iso = len(iso_names)
    if method == "hsic-ir" and ratio_transformation in {"ilr", "alr"}:
        return max(1, n_iso - 1)
    return max(1, n_iso)


def _sv_chunk_worker_fft(
    raster_layer: Any,
    iso_names_chunk: list[list[str]],
    kernel: FFTKernel,
    kernel_cumulants: dict[int, float],
    method: Literal["hsic-ir", "hsic-ic", "hsic-gc"],
    ratio_transformation: str,
) -> list[tuple[float, float]]:
    """Compute FFT-HSIC statistics for one response-channel chunk."""
    flat_iso_names = [iso for iso_names in iso_names_chunk for iso in iso_names]
    data = raster_layer.sel(c=flat_iso_names).values  # (c, ny, nx)
    counts_all = np.moveaxis(np.asarray(data, dtype=float), 0, -1)

    y_blocks: list[np.ndarray] = []
    slices: list[slice | None] = []
    start = 0
    channel_start = 0
    results: list[tuple[float, float] | None] = [None] * len(iso_names_chunk)

    for pos, iso_names in enumerate(iso_names_chunk):
        n_iso = len(iso_names)
        counts_cube = counts_all[:, :, channel_start : channel_start + n_iso]
        channel_start += n_iso

        if method == "hsic-ir" and n_iso <= 1:
            results[pos] = (0.0, 1.0)
            slices.append(None)
            continue

        if method == "hsic-ic":
            y_cube = counts_cube
        elif method == "hsic-gc":
            y_cube = counts_cube.sum(axis=2, keepdims=True)
        else:
            counts_flat = counts_cube.reshape(kernel.n_grid, n_iso)
            ratios = counts_to_ratios(
                torch.from_numpy(counts_flat).float(),
                transformation=ratio_transformation,
                nan_filling="none",
                fill_before_transform=False,
            )
            y_cube = ratios.numpy().reshape(kernel.ny, kernel.nx, -1)

        y_cube = y_cube - np.nanmean(y_cube, axis=(0, 1), keepdims=True)
        y_cube = np.nan_to_num(y_cube, nan=0.0, posinf=0.0, neginf=0.0)
        stop = start + y_cube.shape[2]
        y_blocks.append(y_cube)
        slices.append(slice(start, stop))
        start = stop

    if y_blocks:
        y_all = np.concatenate(y_blocks, axis=2)
        q_cols = np.atleast_1d(np.asarray(kernel.xtKx(y_all), dtype=float))
        y_flat = y_all.reshape(kernel.n_grid, -1)

        for pos, sl in enumerate(slices):
            if sl is None:
                continue
            hsic_scaled = float(np.sum(q_cols[sl]))
            stat = hsic_scaled / ((kernel.n_grid - 1) ** 2)
            y_gene = y_flat[:, sl]
            if not np.isfinite(y_gene).all():
                results[pos] = (stat, 1.0)
                continue

            feature_cumulants = _feature_cumulants_from_data(y_gene)
            if feature_cumulants[2] <= 0.0 or kernel_cumulants[2] <= 0.0:
                results[pos] = (stat, 1.0)
                continue

            pvalue = _hsic_liu_pvalue(
                hsic_scaled,
                kernel_cumulants,
                feature_cumulants,
                kernel.n_grid,
            )
            results[pos] = (stat, pvalue)

    return [r if r is not None else (0.0, 1.0) for r in results]


class SplisosmFFT(_ResultsMixin, _FeatureSummaryMixin):
    """FFT-accelerated SPLISOSM model for rasterized spatial isoform testing.

    ``SplisosmFFT`` follows the same non-parametric workflow as
    :class:`splisosm.SplisosmNP`, but it operates on regular raster grids
    stored in ``SpatialData`` and applies spatial kernels with FFTs.

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

    # -- Public attributes (populated by :meth:`setup_data`) ------------------

    n_genes: int
    """Number of genes after filtering."""

    n_spots: int
    """Number of observed spots (bins with non-zero data)."""

    n_grid: int
    """Total raster grid cells (``ny * nx``, including zero-padded positions)."""

    n_isos_per_gene: list[int]
    """Number of isoforms per gene (list of length :attr:`n_genes`)."""

    gene_names: list[str]
    """Gene display names (length :attr:`n_genes`)."""

    sp_kernel: FFTKernel | None
    """:class:`~splisosm.kernel.FFTKernel` for FFT-accelerated spatial operations."""

    sdata: Any | None
    """Source ``SpatialData`` object; ``None`` before :meth:`setup_data`."""

    n_factors: int
    """Number of covariates for differential usage testing."""

    covariate_names: list[str]
    """Covariate display names (length :attr:`n_factors`)."""

    design_mtx: Optional[Any]
    """Design matrix stored as an AnnData table inside :attr:`sdata`.
    ``None`` if no covariates."""

    def __init__(
        self,
        rho: float = 0.99,
        neighbor_degree: int = 1,
        spacing: tuple[float, float] = (1.0, 1.0),
        workers: int | None = None,
    ) -> None:
        """Initialize the FFT model.

        Parameters
        ----------
        rho : float, optional
            Spatial autocorrelation coefficient for the CAR kernel.
        neighbor_degree : int, optional
            Neighbor-ring degree for CAR graph construction.
        spacing : tuple of float, optional
            Raster spacing ``(dy, dx)`` in kernel length-scale units.
        workers : int or None, optional
            Number of ``scipy.fft`` worker threads. ``None`` lets SciPy choose.
        """
        self._rho = float(rho)
        self._neighbor_degree = int(neighbor_degree)
        self._spacing = spacing
        self._workers = workers

        self.n_genes = 0
        self.n_spots = 0
        self.n_grid = 0
        self.n_isos_per_gene = []
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

        self.sp_kernel: FFTKernel | None = None
        self._kernel_eigvals: np.ndarray | None = None
        self._kernel_cumulants: dict[int, float] | None = None
        self._raster_key: str | None = None
        self._raster_layer: Any | None = None

        self._sv_test_results: dict[str, Any] = {}
        self._du_test_results: dict[str, Any] = {}

        self._gene_summary: Optional[pd.DataFrame] = None
        self._isoform_summary: Optional[pd.DataFrame] = None

        self.design_mtx: Optional[Any] = None
        self._design_table_name: Optional[str] = None
        self.covariate_names: list[str] = []
        self.n_factors: int = 0

    def __str__(self) -> str:
        """Return string representation of configured model state."""
        sv_status = (
            f"Completed ({self._sv_test_results['method']})"
            if len(self._sv_test_results) > 0
            else "N/A"
        )
        du_status = (
            f"Completed ({self._du_test_results['method']})"
            if len(self._du_test_results) > 0
            else "N/A"
        )
        _avg_iso = (
            f"{np.mean(self.n_isos_per_gene):.1f}" if self.n_isos_per_gene else "N/A"
        )
        return (
            f"=== SplisosmFFT\n"
            f"- Number of genes: {self.n_genes}\n"
            f"- Number of spots: {self.n_spots}\n"
            f"- Number of spots after rasterization: {self.n_grid}\n"
            f"- Number of covariates: {self.n_factors}\n"
            f"- Average isoforms per gene: {_avg_iso}\n"
            "=== Model configurations\n"
            f"- Neighborhood degree: {self._neighbor_degree}\n"
            f"- Spatial autocorrelation rho: {self._rho}\n"
            "=== Test results\n"
            f"- Spatial variability: {sv_status}\n"
            f"- Differential usage: {du_status}"
        )

    __repr__ = __str__

    def _rasterize_bins(self, bins, table_name, col_key, row_key) -> str:
        """Rasterize bins and cache image key in sdata."""
        if self.sdata is None:
            raise RuntimeError("Call setup_data() first.")
        sd = _require_spatialdata()

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

    def _validate_setup_inputs(
        self,
        sdata: Any,
        table_name: str,
        layer: str,
        group_iso_by: str,
        row_key: str,
        col_key: str,
        gene_names: Optional[str],
        min_counts: int,
        min_bin_pct: float,
    ) -> AnnData:
        """Validate SpatialData setup inputs and return the source AnnData table."""
        _require_spatialdata()
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
        missing_var = [
            col
            for col in [group_iso_by, gene_names]
            if col is not None and col not in adata.var.columns
        ]
        if missing_var:
            missing = missing_var[0]
            message = (
                f"`gene_names` column '{missing}' was not found in `adata.var`."
                if missing == gene_names
                else f"`{missing}` was not found in `adata.var`."
            )
            raise ValueError(message)

        missing_obs = [
            col for col in [row_key, col_key] if col not in adata.obs.columns
        ]
        if missing_obs:
            raise ValueError(f"`{missing_obs[0]}` was not found in `adata.obs`.")
        return adata

    def _group_fft_isoforms(
        self,
        adata: AnnData,
        layer: str,
        group_iso_by: str,
        gene_names: Optional[str],
        min_counts: int,
        min_bin_pct: float,
        filter_single_iso_genes: bool,
    ) -> tuple[list[list[str]], list[str]]:
        """Filter and group raster channels by gene."""
        min_bin_frac = float(min_bin_pct)
        if min_bin_frac > 1.0:
            min_bin_frac /= 100.0

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

        grouped_iso_names: list[list[str]] = []
        grouped_gene_names: list[str] = []
        for gene, group in var_df.groupby(group_iso_by, observed=True, sort=False):
            group = group[
                (group["__total_counts__"] >= min_counts)
                & (group["__bin_pct__"] >= min_bin_frac)
            ]
            if filter_single_iso_genes and group.shape[0] < 2:
                continue
            if group.shape[0] < 1:
                continue
            grouped_gene_names.append(
                str(group[gene_names].iloc[0]) if gene_names is not None else str(gene)
            )
            grouped_iso_names.append(group["__iso_name__"].tolist())

        if len(grouped_iso_names) == 0:
            raise ValueError("No genes remained after grouping/filtering isoforms.")
        return grouped_iso_names, grouped_gene_names

    def _initialize_fft_kernel(self, ny: int, nx: int) -> None:
        """Initialize the FFT spatial kernel and cached cumulants."""
        self.sp_kernel = FFTKernel(
            shape=(ny, nx),
            spacing=self._spacing,
            rho=self._rho,
            neighbor_degree=self._neighbor_degree,
            workers=self._workers,
            centering=True,
        )
        eigvals = self.sp_kernel.eigenvalues()
        self._kernel_eigvals = eigvals[eigvals > 1e-8]
        self._kernel_cumulants = _cumulants_from_eigenvalues(eigvals)

    def _design_obs_for_table(
        self,
        adata: AnnData,
        bins: str,
        row_key: str,
        col_key: str,
    ) -> pd.DataFrame:
        """Return obs metadata needed for rasterizing a generated design table."""
        obs_cols = [row_key, col_key]
        attrs = adata.uns.get("spatialdata_attrs", {})
        region_key_col = attrs.get("region_key", "region")
        instance_key_col = attrs.get("instance_key", "instance_id")
        for col in [region_key_col, instance_key_col]:
            if col in adata.obs.columns:
                obs_cols.append(col)
        obs_for_design = adata.obs[obs_cols].copy()
        if region_key_col not in obs_for_design.columns:
            obs_for_design[region_key_col] = pd.Categorical(
                [bins] * len(obs_for_design)
            )
        if instance_key_col not in obs_for_design.columns:
            obs_for_design[instance_key_col] = np.arange(len(obs_for_design), dtype=int)
        return obs_for_design

    def _setup_design_table(
        self,
        sdata: Any,
        adata: AnnData,
        bins: str,
        table_name: str,
        row_key: str,
        col_key: str,
        design_mtx: Optional[Union[np.ndarray, "pd.DataFrame", str, list[str]]],
        covariate_names: Optional[list[str]],
    ) -> None:
        """Resolve DU design input and store/rasterize metadata for later use."""
        if design_mtx is None:
            self._design_table_name = None
            self.design_mtx = None
            self.covariate_names = list(covariate_names) if covariate_names else []
            self.n_factors = 0
            return

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
            self.covariate_names = list(covariate_names)
            self.n_factors = ref_table.n_vars
            return

        from scipy import sparse as _sp
        from splisosm.utils.preprocessing import _process_design_mtx

        dm_np, covariate_names = _process_design_mtx(adata, design_mtx, covariate_names)
        if _sp.issparse(dm_np):
            density = dm_np.nnz / (dm_np.shape[0] * dm_np.shape[1])
            x_data = dm_np.tocsr() if density < 0.5 else dm_np.toarray()
        else:
            density = np.count_nonzero(dm_np) / dm_np.size if dm_np.size > 0 else 1.0
            x_data = _sp.csr_matrix(dm_np) if density < 0.5 else dm_np

        design_adata = AnnData(
            X=x_data,
            obs=self._design_obs_for_table(adata, bins, row_key, col_key),
        )
        design_adata.var_names = list(covariate_names)

        attrs = adata.uns.get("spatialdata_attrs", {})
        region_key_col = attrs.get("region_key", "region")
        instance_key_col = attrs.get("instance_key", "instance_id")
        try:
            from spatialdata.models import TableModel

            design_adata = TableModel.parse(
                design_adata,
                region=bins,
                region_key=region_key_col,
                instance_key=instance_key_col,
            )
        except Exception:
            pass

        design_key = f"_splisosm_design_{table_name}"
        sdata[design_key] = design_adata
        self._design_table_name = design_key
        self.design_mtx = design_adata
        self.covariate_names = list(covariate_names)
        self.n_factors = dm_np.shape[1]

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
        """Set up SpatialData-backed isoform data for FFT-based testing.

        ``bins``, ``table_name``, ``col_key``, and ``row_key`` are passed to
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
            Column-coordinate key in ``adata.obs`` for rasterization.
        row_key : str
            Row-coordinate key in ``adata.obs`` for rasterization.
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
        design_mtx : str, list[str], array, sparse matrix, DataFrame, or None
            Design matrix specification. Three input modes:

            1. **Table name** (``str`` matching a key in ``sdata.tables``): Use the
               existing AnnData table's ``X`` as the design matrix.  Must have the
               same number of observations as the isoform table.
            2. **Obs column names** (``str`` or ``list[str]`` not matching a table):
               Extract the named columns from the isoform table's ``adata.obs``.
               Categorical columns are one-hot encoded automatically.
            3. **Pre-computed matrix** (array, sparse matrix, or DataFrame of
               shape ``(n_spots, n_factors)``): Used as-is.

            In cases 2 and 3, the design matrix is stored as a new AnnData
            table inside ``sdata`` and rasterized when
            :meth:`test_differential_usage` is called.
        covariate_names : list[str] or None, optional
            Factor names. These override inferred names. If ``None``, names
            are inferred from ``design_mtx`` when possible; otherwise they are
            generated as ``["factor_0", ...]``.

        Raises
        ------
        ValueError
            If required table/layer/metadata is missing.

        See Also
        --------
        :meth:`splisosm.SplisosmNP.setup_data`
            AnnData-based setup for data with general geometry.

        """
        adata = self._validate_setup_inputs(
            sdata=sdata,
            table_name=table_name,
            layer=layer,
            group_iso_by=group_iso_by,
            row_key=row_key,
            col_key=col_key,
            gene_names=gene_names,
            min_counts=min_counts,
            min_bin_pct=min_bin_pct,
        )
        adata.X = adata.layers[layer]
        grouped_iso_names, grouped_gene_names = self._group_fft_isoforms(
            adata=adata,
            layer=layer,
            group_iso_by=group_iso_by,
            gene_names=gene_names,
            min_counts=min_counts,
            min_bin_pct=min_bin_pct,
            filter_single_iso_genes=filter_single_iso_genes,
        )

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
        self.n_isos_per_gene = [len(v) for v in grouped_iso_names]
        self.gene_names = grouped_gene_names

        self._initialize_fft_kernel(ny, nx)
        self._setup_design_table(
            sdata=sdata,
            adata=adata,
            bins=bins,
            table_name=table_name,
            row_key=row_key,
            col_key=col_key,
            design_mtx=design_mtx,
            covariate_names=covariate_names,
        )

    def _feature_summary_adata(self) -> AnnData:
        """Return the filtered AnnData table used for feature summaries."""
        if self._adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        kept_isos = [iso for gene_isos in self._gene_iso_names for iso in gene_isos]
        return self._adata[:, kept_isos]

    def _require_ready_for_fft_tests(self) -> None:
        """Ensure FFT setup has produced the kernel and raster layer."""
        if self._adata is None or self.sp_kernel is None or self._raster_layer is None:
            raise RuntimeError("Data not initialized. Call setup_data() first.")

    def _set_fft_worker_budget(self, n_jobs: int) -> int:
        """Resolve joblib workers and coordinate scipy.fft worker threads."""
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        self.sp_kernel.workers = max(1, (os.cpu_count() or 1) // n_jobs)
        return n_jobs

    def _validate_sv_args(
        self,
        method: str,
        ratio_transformation: str,
    ) -> None:
        """Validate FFT spatial-variability arguments."""
        self._require_ready_for_fft_tests()
        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid ratio transformation. Must be one of {valid_transformations}."
            )

    def _run_fft_sv(
        self,
        method: str,
        ratio_transformation: str,
        chunk_size: int | Literal["auto"],
        n_jobs: int,
        print_progress: bool,
    ) -> dict[str, Any]:
        """Run chunked FFT spatial variability tests."""
        n_jobs = self._set_fft_worker_budget(n_jobs)
        column_cap = resolve_chunk_size(
            chunk_size,
            n_observations=self.n_grid,
            backend="fft",
            n_jobs=n_jobs,
            dtype_bytes=8,
        )
        widths = [
            _sv_response_width_fft(iso_names, method, ratio_transformation)
            for iso_names in self._gene_iso_names
        ]
        gene_chunks = pack_gene_chunks(widths, column_cap)

        chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_sv_chunk_worker_fft)(
                self._raster_layer,
                [self._gene_iso_names[i] for i in chunk],
                self.sp_kernel,
                self._kernel_cumulants,
                method,
                ratio_transformation,
            )
            for chunk in tqdm(
                gene_chunks,
                desc=f"SV [{method}]",
                total=len(gene_chunks),
                disable=not print_progress,
            )
        )
        results = [res for chunk in chunk_results for res in chunk]
        pvals = np.asarray([r[1] for r in results], dtype=float)
        return {
            "statistic": np.asarray([r[0] for r in results], dtype=float),
            "pvalue": pvals,
            "pvalue_adj": false_discovery_control(pvals),
            "method": method,
            "null_method": "liu",
            "use_perm_null": False,
            "chunk_size": column_cap,
        }

    def _validate_du_args(
        self,
        method: str,
        residualize: str,
        ratio_transformation: str,
    ) -> None:
        """Validate FFT differential-usage arguments."""
        self._require_ready_for_fft_tests()
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

    def _factor_names(self) -> list[str]:
        """Return user-facing covariate names, filling defaults if needed."""
        if self.covariate_names is not None and len(self.covariate_names) > 0:
            return self.covariate_names
        return [f"factor_{i}" for i in range(self.n_factors)]

    def _resolve_fft_gpr_configs(
        self,
        method: str,
        residualize: str,
        gpr_configs: Optional[dict[str, Any]],
    ) -> tuple[Optional[FFTKernelGPR], Optional[dict[str, Any]]]:
        """Create FFT GPR objects/configs needed by DU testing."""
        if method != "hsic-gp":
            return None, None

        cov_cfg = {**_DEFAULT_GPR_CONFIGS["covariate"]}
        iso_cfg = {**_DEFAULT_GPR_CONFIGS["isoform"]}
        if gpr_configs is not None:
            if "covariate" in gpr_configs:
                cov_cfg.update(gpr_configs["covariate"])
            if "isoform" in gpr_configs:
                iso_cfg.update(gpr_configs["isoform"])

        gpr_cov = FFTKernelGPR(
            constant_value=cov_cfg["constant_value"],
            constant_value_bounds=cov_cfg["constant_value_bounds"],
            length_scale=cov_cfg["length_scale"],
            length_scale_bounds=cov_cfg["length_scale_bounds"],
        )
        if residualize != "both":
            return gpr_cov, None
        return gpr_cov, {
            "constant_value": iso_cfg["constant_value"],
            "constant_value_bounds": iso_cfg["constant_value_bounds"],
            "length_scale": iso_cfg["length_scale"],
            "length_scale_bounds": iso_cfg["length_scale_bounds"],
        }

    def _prepare_fft_covariate_chunk(
        self,
        chunk_vals: np.ndarray,
        chunk_names: list[str],
        method: str,
        gpr_cov: Optional[FFTKernelGPR],
        spacing: tuple[float, float],
    ) -> list[np.ndarray]:
        """Prepare one rasterized covariate chunk for the selected DU method."""
        z_chunk: list[np.ndarray] = []
        for idx, chunk_name in enumerate(chunk_names):
            z_cube = chunk_vals[:, :, idx]
            if method == "hsic-gp":
                z_res, _ = gpr_cov.fit_residuals_cube(z_cube, spacing=spacing)
                z_res = z_res - z_res.mean()
                std = z_res.std()
                if std > 0:
                    z_res /= std
            elif method == "hsic":
                z_res = z_cube - np.nanmean(z_cube)
                z_res = np.nan_to_num(z_res, nan=0.0)
            else:
                unique_vals = np.unique(z_cube[np.isfinite(z_cube)])
                if len(unique_vals) > 2:
                    raise ValueError(
                        f"More than two groups detected for factor "
                        f"'{chunk_name}'. Only binary covariates "
                        "(exactly two distinct values) are supported for "
                        f"'{method}'."
                    )
                z_res = z_cube.copy()
            z_chunk.append(z_res)
        return z_chunk

    def _run_fft_du(
        self,
        method: str,
        ratio_transformation: str,
        gpr_configs: Optional[dict[str, Any]],
        residualize: str,
        n_jobs: int,
        print_progress: bool,
    ) -> dict[str, Any]:
        """Run FFT differential-usage tests in covariate chunks."""
        factor_names = self._factor_names()
        n_factors = self.n_factors
        spacing = (self.sp_kernel.dy, self.sp_kernel.dx)
        gpr_cov, gpr_iso_cfg = self._resolve_fft_gpr_configs(
            method, residualize, gpr_configs
        )

        sd = _require_spatialdata()
        dm_raster = sd.rasterize_bins(
            self.sdata,
            self._bins_name,
            self._design_table_name,
            self._col_key,
            self._row_key,
        )

        n_jobs = self._set_fft_worker_budget(n_jobs)
        fft_workers = self.sp_kernel.workers
        if gpr_cov is not None:
            gpr_cov._workers = fft_workers
        if gpr_iso_cfg is not None:
            gpr_iso_cfg = {**gpr_iso_cfg, "workers": fft_workers}

        stats = np.zeros((len(self._gene_iso_names), n_factors))
        pvals = np.ones((len(self._gene_iso_names), n_factors))
        cov_chunk_size = 100
        n_chunks = max(1, (n_factors + cov_chunk_size - 1) // cov_chunk_size)
        show_gene_bar = print_progress and n_chunks == 1
        show_chunk_bar = print_progress and n_chunks > 1

        for f_start in tqdm(
            range(0, n_factors, cov_chunk_size),
            desc="Covariates",
            total=n_chunks,
            disable=not show_chunk_bar,
        ):
            f_end = min(f_start + cov_chunk_size, n_factors)
            chunk_names = factor_names[f_start:f_end]
            chunk_vals = np.moveaxis(
                np.asarray(dm_raster.sel(c=chunk_names).values, dtype=float), 0, -1
            )
            z_chunk = self._prepare_fft_covariate_chunk(
                chunk_vals, chunk_names, method, gpr_cov, spacing
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
                for iso_names in tqdm(
                    self._gene_iso_names,
                    desc=f"DU [{method}]",
                    total=len(self._gene_iso_names),
                    disable=not show_gene_bar,
                )
            )
            for gene_idx, (stat_row, pval_row) in enumerate(chunk_results):
                stats[gene_idx, f_start:f_end] = stat_row
                pvals[gene_idx, f_start:f_end] = pval_row

        return {
            "statistic": stats,
            "pvalue": pvals,
            "pvalue_adj": np.column_stack(
                [false_discovery_control(pvals[:, f]) for f in range(n_factors)]
            ),
            "method": method,
        }

    def test_spatial_variability(
        self,
        method: Literal["hsic-ir", "hsic-ic", "hsic-gc"] = "hsic-ir",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        chunk_size: int | Literal["auto"] = "auto",
        n_jobs: int = -1,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test each gene for spatial variability using FFT-accelerated HSIC.

        Parameters
        ----------
        method : {"hsic-ir", "hsic-ic", "hsic-gc"}, optional
            One of ``"hsic-ir"`` (isoform ratios), ``"hsic-ic"`` (isoform counts),
            or ``"hsic-gc"`` (gene counts).
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Ratio transform used when ``method="hsic-ir"``.
        chunk_size : int or {"auto"}, optional
            Maximum number of response channels to process in one FFT kernel
            call. ``"auto"`` (default) estimates a memory-safe cap using a
            2 GiB live-memory budget per worker and caps the result at 32
            channels for per-feature runtime.  Genes are never split across
            chunks; a single gene with more channels than the cap is processed
            as a singleton chunk.
        n_jobs : int, optional
            Number of joblib workers. ``-1`` uses all available CPUs.
        return_results : bool, optional
            If ``True``, return the result dictionary.
        print_progress : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        dict or None
            Result dictionary when ``return_results=True``; otherwise ``None``.

        See Also
        --------
        :meth:`splisosm.SplisosmNP.test_spatial_variability`
            Non-FFT implementation for arbitrary spatial geometries.
        """
        self._validate_sv_args(method, ratio_transformation)
        self._sv_test_results = self._run_fft_sv(
            method=method,
            ratio_transformation=ratio_transformation,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            print_progress=print_progress,
        )

        if return_results:
            return self._sv_test_results
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
        """Test each gene for differential isoform usage on a raster grid.

        Call :meth:`setup_data` with ``design_mtx`` before running this
        method. Each design-matrix column is tested against each gene's
        isoform usage ratios, producing one statistic and p-value per
        ``(gene, covariate)`` pair.

        Four test strategies are supported, all operating on rasterized grid data
        to avoid densifying the full isoform or covariate matrix in memory:

        - ``"hsic-gp"`` *(default)*: spatially residualize covariates (and
          optionally isoform ratios) with :class:`splisosm.gpr.FFTKernelGPR`,
          then compute linear HSIC. Controlled by ``residualize``.
        - ``"hsic"``: linear HSIC between raw centered isoform ratios and raw
          centered covariates, with no spatial residualization.
        - ``"t-fisher"``: per-isoform two-sample t-tests (**binary covariates
          only**) combined by Fisher's method (chi-square with
          ``df = 2 * n_isoforms``).
        - ``"t-tippett"``: per-isoform two-sample t-tests (**binary covariates
          only**) combined by Tippett's corrected minimum p-value.

        Regardless of method, covariates are processed in chunks of at most 100
        at a time and isoform data is loaded on-the-fly per gene so that neither
        the full covariate grid nor the full isoform matrix is held in memory
        simultaneously.

        Parameters
        ----------
        method : {"hsic", "hsic-gp", "t-fisher", "t-tippett"}, optional
            Method for association testing:

            * ``"hsic"``: Unconditional HSIC test (multivariate RV coefficient).
              For continuous factors, equivalent to the multivariate Pearson correlation
              test.  For binary factors, equivalent to the two-sample Hotelling T**2 test.
            * ``"hsic-gp"``: Conditional HSIC test.  Spatial effects are removed via
              Gaussian process regression before computing the HSIC statistic.

            * ``"t-fisher"``, ``"t-tippett"``: two-sample t-test per isoform
              (binary covariates only; exactly two distinct non-NaN values required);
              p-values are combined gene-wise via Fisher's chi-squared or
              Tippett's corrected minimum method.
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Compositional transformation for isoform ratios.
            One of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, ``'radial'``
            :cite:`park2022kernel`.  See :func:`splisosm.utils.preprocessing.counts_to_ratios`.
        gpr_configs : dict, optional
            Nested configuration dict for the GPR objects, with optional keys
            ``'covariate'`` and/or ``'isoform'``.  Each sub-dict is forwarded to
            the FFT GPR path.  Unspecified keys use the common GP defaults from
            SPLISOSM's backend configuration: ``constant_value``,
            ``constant_value_bounds``, ``length_scale``, and
            ``length_scale_bounds``. Large-n sklearn/gpytorch and NUFFT-only keys
            in the shared defaults are ignored by this FFT path. See
            :class:`splisosm.gpr.FFTKernelGPR` for backend-specific options.

        residualize : {"cov_only", "both"}, optional
            Controls which signals are spatially residualized when
            ``method="hsic-gp"``:

            * ``"cov_only"`` (default): residualize covariates only; test
              ``HSIC(Z_res, Y)``. Fastest; calibration matches ``"both"``
              when covariate GPR captures most spatial confounding.
            * ``"both"``: residualize both covariates and isoform ratios.

        n_jobs : int, optional
            Number of parallel jobs. ``-1`` uses all available CPUs.
        print_progress : bool, optional
            Whether to show the progress bar. Default ``True``.
        return_results : bool, optional
            If ``True``, return the test statistics and p-values. Otherwise,
            store results in ``self._du_test_results``.

        Returns
        -------
        results : dict or None
            If ``return_results`` is ``True``, return a dictionary with test
            statistics and p-values. Otherwise, return ``None`` and store results in
            ``self._du_test_results``.

        Raises
        ------
        RuntimeError
            If ``setup_data()`` or the ``design_mtx`` argument has not been set.
        ValueError
            If ``method``, ``residualize``, or ``ratio_transformation`` is invalid.

        See Also
        --------
        :meth:`splisosm.SplisosmNP.test_differential_usage`
            Non-FFT implementation for arbitrary spatial geometries.
        """
        self._validate_du_args(method, residualize, ratio_transformation)
        self._du_test_results = self._run_fft_du(
            method=method,
            ratio_transformation=ratio_transformation,
            gpr_configs=gpr_configs,
            residualize=residualize,
            n_jobs=n_jobs,
            print_progress=print_progress,
        )
        if return_results:
            return self._du_test_results
        return None
