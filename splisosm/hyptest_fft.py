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
import scipy.sparse as sp
import torch
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm

from splisosm.kernel import FFTKernel  # noqa: F401 — re-export for backward compat
from splisosm.kernel_gpr import (
    FFTKernelOp,
    FFTKernelGPR,
    _DEFAULT_GPR_CONFIGS,
)
from splisosm.likelihood import liu_sf
from splisosm.utils import (
    compute_feature_summaries,
    counts_to_ratios,
    false_discovery_control,
)

__all__ = ["FFTKernel", "FFTKernelOp", "FFTKernelGPR", "SplisosmFFT"]


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

    # Single-isoform gene: no differential usage is possible.
    if len(iso_names) <= 1:
        return stats, pvals

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

    # Single-isoform gene: HSIC-IR has no ratio variation → pval=1.
    # HSIC-IC and HSIC-GC test count-level SV and proceed normally.
    if method == "hsic-ir" and counts_cube.shape[2] <= 1:
        return 0.0, 1.0

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
            else "NA"
        )
        du_status = (
            f"Completed ({self._du_test_results['method']})"
            if len(self._du_test_results) > 0
            else "NA"
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
        self.n_isos_per_gene = [len(v) for v in grouped_iso_names]
        self.gene_names = grouped_gene_names

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
        if self._adata is None or self.sp_kernel is None or self._raster_layer is None:
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

        # Auto-coordinate FFT workers with joblib n_jobs to prevent thread
        # oversubscription: scipy.fft.fft2 spawns `workers` threads internally,
        # so total threads = n_jobs × workers without this cap.
        self.sp_kernel.workers = max(1, (os.cpu_count() or 1) // n_jobs)

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
                self.sp_kernel,
                self._kernel_eigvals,
                method,
                ratio_transformation,
            )
            for iso_names in iterator
        )

        stats = np.asarray([r[0] for r in results], dtype=float)
        pvals_np = np.asarray([r[1] for r in results], dtype=float)

        self._sv_test_results = {
            "statistic": stats,
            "pvalue": pvals_np,
            "pvalue_adj": false_discovery_control(pvals_np),
            "method": method,
            "use_perm_null": False,
        }

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
            If False, the results are stored in ``self._du_test_results``.

        Returns
        -------
        results : dict or None
            If ``return_results`` is True, returns dict with test statistics and
            p-values. Otherwise, returns None and stores results in
            ``self._du_test_results``.

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
        if self._adata is None or self.sp_kernel is None or self._raster_layer is None:
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

        factor_names = (
            self.covariate_names
            if self.covariate_names is not None and len(self.covariate_names) > 0
            else [f"factor_{i}" for i in range(self.n_factors)]
        )
        n_factors = self.n_factors
        spacing = (self.sp_kernel.dy, self.sp_kernel.dx)

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

        # Auto-coordinate FFT workers with joblib n_jobs to prevent thread
        # oversubscription (same rationale as in test_spatial_variability).
        _fft_workers = max(1, (os.cpu_count() or 1) // n_jobs)
        self.sp_kernel.workers = _fft_workers
        if gpr_cov is not None:
            gpr_cov._workers = _fft_workers
        if gpr_iso_cfg is not None:
            gpr_iso_cfg = {**gpr_iso_cfg, "workers": _fft_workers}

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

        self._du_test_results = {
            "statistic": stats,
            "pvalue": pvals,
            "pvalue_adj": pvals_adj,
            "method": method,
        }
        if return_results:
            return self._du_test_results
        return None

    def get_formatted_test_results(
        self,
        test_type: Literal["sv", "du"],
        with_gene_summary: bool = False,
    ) -> pd.DataFrame:
        """Get formatted test results as a pandas DataFrame.

        Parameters
        ----------
        test_type : {"sv", "du"}
            Test type: ``"sv"`` for spatial variability or ``"du"`` for
            differential usage.
        with_gene_summary : bool, optional
            If ``True``, append gene-level summary statistics from
            :meth:`extract_feature_summary`.

        Returns
        -------
        pandas.DataFrame
            Formatted result table.
        """
        if test_type not in {"sv", "du"}:
            raise ValueError("Invalid test type. Must be one of 'sv' or 'du'.")

        if test_type == "sv":
            if len(self._sv_test_results) == 0:
                raise ValueError(
                    "No spatial variability results found. Run test_spatial_variability() first."
                )
            df = pd.DataFrame(
                {
                    "gene": self.gene_names,
                    "statistic": self._sv_test_results["statistic"],
                    "pvalue": self._sv_test_results["pvalue"],
                    "pvalue_adj": self._sv_test_results["pvalue_adj"],
                }
            )
        else:
            if len(self._du_test_results) == 0:
                raise ValueError(
                    "No differential usage results found. Run test_differential_usage() first."
                )
            covariate_names = (
                self.covariate_names
                if self.covariate_names is not None and len(self.covariate_names) > 0
                else [f"factor_{i}" for i in range(self.n_factors)]
            )
            df = pd.DataFrame(
                {
                    "gene": np.repeat(self.gene_names, self.n_factors),
                    "covariate": np.tile(covariate_names, self.n_genes),
                    "statistic": self._du_test_results["statistic"].reshape(-1),
                    "pvalue": self._du_test_results["pvalue"].reshape(-1),
                    "pvalue_adj": self._du_test_results["pvalue_adj"].reshape(-1),
                }
            )

        if with_gene_summary:
            gene_df = self.extract_feature_summary(level="gene", print_progress=False)
            df = df.merge(gene_df, left_on="gene", right_index=True, how="left")

        return df

    def _compute_feature_summaries(self, print_progress: bool = True) -> None:
        """Compute and cache both gene-level and isoform-level summaries."""
        if self._adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if self._gene_summary is not None and self._isoform_summary is not None:
            return
        # self._adata is the full unfiltered table; restrict to filtered isoforms
        kept_isos = [iso for gene_isos in self._gene_iso_names for iso in gene_isos]
        filtered_adata = self._adata[:, kept_isos]
        self._gene_summary, self._isoform_summary = compute_feature_summaries(
            filtered_adata,
            self.gene_names,
            layer=self._counts_layer,
            group_iso_by=self._group_iso_by,
            print_progress=print_progress,
        )

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
            raise RuntimeError("Call setup_data() first.")
        if level not in {"gene", "isoform"}:
            raise ValueError("`level` must be one of 'gene' or 'isoform'.")

        self._compute_feature_summaries(print_progress=print_progress)

        if level == "gene":
            return self._gene_summary
        return self._isoform_summary
