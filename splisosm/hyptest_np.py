"""Non-parametric hypothesis tests for spatial isoform usage."""

from __future__ import annotations

import os
import warnings
import re
from typing import Any, Optional, Union, Literal
from joblib import Parallel, delayed
from scipy.stats import (
    ttest_ind,
    combine_pvalues,
    norm as _norm_dist,
    chi2 as _chi2_dist,
)
import numpy as np
import scipy.sparse
import pandas as pd
import torch
from tqdm import tqdm
from anndata import AnnData

from splisosm.utils import (
    compute_feature_summaries,
    counts_to_ratios,
    false_discovery_control,
    prepare_inputs_from_anndata,
    run_sparkx,
)
from splisosm.kernel import IdentityKernel, SpatialCovKernel
from splisosm.likelihood import liu_sf
from splisosm.kernel_gpr import (
    linear_hsic_test,
    fit_kernel_gpr,
    make_kernel_gpr,
    _DEFAULT_GPR_CONFIGS,
)

__all__ = [
    "linear_hsic_test",
    "fit_kernel_gpr",
    "SplisosmNP",
]


def _calc_ttest_differential_usage(
    data, groups, combine_pval=True, combine_method="tippett"
):
    """Calculate the two-sample t-test statistic for differential usage.

    The t-test is applied to each isoform independently and combined if combine_pval is True.

    Parameters
    ----------
    data : torch.Tensor
        Shape (n_spots, n_isos), the observed isoform ratios for a given gene.
    groups : torch.Tensor or scipy.sparse.spmatrix
        Shape ``(n_spots,)`` or ``(n_spots, 1)``, the binary group labels for
        each spot.  When a scipy sparse matrix is provided (e.g. a column from
        a one-hot-encoded design matrix), the nonzero rows are taken as group 1
        and zero rows as group 0, avoiding full densification of the column.
    combine_pval : bool, optional
        Whether to combine p-values across isoforms.
    combine_method : str, optional
        The method to combine p-values. See scipy.stats.combine_pvalues() for more details.

    Returns
    -------
    stats : float or numpy scalar
        Combined test statistic (scalar when combine_pval=True).
    pval : float or numpy scalar
        P-value.
    """
    if scipy.sparse.issparse(groups):
        # Sparse column (n, 1): nonzero rows belong to group 1, zeros to group 0.
        # This avoids creating a full dense boolean mask.
        groups_csr = groups.tocsr()
        g1_idx = groups_csr.nonzero()[0]  # row indices where value is nonzero
        n = groups_csr.shape[0]
        g0_mask = np.ones(n, dtype=bool)
        g0_mask[g1_idx] = False
        t1 = data[g1_idx]  # (k, n_isos)
        t2 = data[g0_mask]  # (n - k, n_isos)
    else:
        # Dense path: groups is a 1-D tensor or array.
        _g = torch.unique(
            groups
            if isinstance(groups, torch.Tensor)
            else torch.from_numpy(np.asarray(groups))
        )
        if len(_g) > 2:
            raise ValueError(
                "More than two groups detected. Only two are allowed for the two-sample t-test."
            )
        t1 = data[groups == _g[0], :]  # (k, n_isos)
        t2 = data[groups == _g[1], :]  # (n_spots - k, n_isos)

    stats, pval = ttest_ind(t1, t2, axis=0, nan_policy="omit")

    # combine p-values across isoforms
    if combine_pval:
        stats, pval = combine_pvalues(pval, method=combine_method)

    return stats, pval


def _sv_gene_worker_np(
    counts: "torch.Tensor",
    method: str,
    ratio_transformation: str,
    nan_filling: str,
    null_method: str,
    n_spots: int,
    K_sp,
    lambda_sp: "Optional[torch.Tensor]",
    k_eff: int,
    trK: "Optional[torch.Tensor]",
    trK2: "Optional[torch.Tensor]",
    n_nulls: int,
    perm_batch_size: int,
) -> "tuple[float, float]":
    """Process a single gene for :meth:`SplisosmNP.test_spatial_variability`.

    Shared objects (``K_sp``, ``lambda_sp``, ``trK``, ``trK2``) are
    passed by reference and only read, making this safe for
    ``joblib.Parallel(prefer="threads")``.

    Returns
    -------
    tuple[float, float]
        ``(hsic_norm, pval)`` for this gene.
    """
    if counts.is_sparse:
        counts = counts.to_dense()

    # Single-isoform gene with HSIC-IR: ratios are constant (all 1.0), so there is
    # no usage variation to detect.  Return stat=0, pval=1.
    # HSIC-IC and HSIC-GC still produce meaningful results for single-isoform genes
    # (they test count-level spatial variability, equivalent to gene-level SV).
    if method == "hsic-ir" and counts.shape[1] <= 1:
        return (0.0, 1.0)

    lambda_sp_eff = lambda_sp
    k_eff_eff = k_eff
    # By default, null-distribution inputs come from the global (already
    # double-centred) K_sp.  The `hsic-ir + nan_filling='none'` branch below
    # overrides these with per-gene values derived from K_sp_gene so that the
    # statistic and the null reference the *same* (centred) kernel submatrix.
    K_sp_null = K_sp
    trK_eff = trK
    trK2_eff = trK2

    if method == "hsic-ir" and nan_filling == "none":
        # per-gene spatial kernel: drop NaN spots and recompute K
        y = counts_to_ratios(
            counts,
            transformation=ratio_transformation,
            nan_filling="none",
            fill_before_transform=False,
        )
        is_nan = torch.isnan(y).any(1)
        y = y[~is_nan]
        n_spots_eff = y.shape[0]
        K_sp_gene = K_sp.realization()[~is_nan, :][:, ~is_nan]
        K_sp_gene = K_sp_gene - K_sp_gene.mean(dim=0, keepdim=True)
        K_sp_gene = K_sp_gene - K_sp_gene.mean(dim=1, keepdim=True)

        hsic_scaled = torch.trace(y.T @ K_sp_gene @ y)

        # The passed-in null inputs (lambda_sp, trK, trK2) describe the global
        # K_sp; with per-gene NaN filtering they no longer match the statistic.
        # Recompute them from the per-gene centred submatrix.
        if null_method == "eig":
            lambda_sp_eff = torch.linalg.eigvalsh(K_sp_gene)
            lambda_sp_eff = lambda_sp_eff[lambda_sp_eff > 1e-5]
            k_eff_eff = len(lambda_sp_eff)
        elif null_method in ("trace", "welch"):
            trK_eff = torch.trace(K_sp_gene)
            trK2_eff = K_sp_gene.pow(2).sum()
        else:  # null_method == "perm"
            # Use the per-gene centred kernel directly; `K_sp.xtKx` would apply
            # the global (wrong-shape) kernel to the NaN-filtered y_batch.
            K_sp_null = K_sp_gene

    else:  # global spatial kernel shared across all genes
        if method == "hsic-ic":
            y = counts - counts.mean(0, keepdim=True)
        elif method == "hsic-gc":
            y = counts.sum(1, keepdim=True)
            y = y - y.mean()
        else:  # hsic-ir with mean nan_filling (global kernel case)
            y = counts_to_ratios(
                counts,
                transformation=ratio_transformation,
                nan_filling="mean",
                fill_before_transform=False,
            )
            y = y - y.mean(0, keepdim=True)

        n_spots_eff = n_spots

        if null_method == "eig":
            hsic_scaled = torch.trace(K_sp.xtKx_approx(y, k=k_eff_eff))
        else:
            hsic_scaled = torch.trace(K_sp.xtKx_exact(y))

    hsic_norm = float(hsic_scaled / (n_spots_eff - 1) ** 2)

    if null_method == "eig":
        try:
            lambda_y = torch.linalg.eigvalsh(y.T @ y)
        except torch._C._LinAlgError:
            lambda_y = torch.linalg.eigvalsh(y.T @ y + 1e-6 * torch.eye(y.shape[1]))
        lambda_y = lambda_y[lambda_y > 1e-5]
        lambda_spy = (lambda_sp_eff.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1)
        pval = liu_sf((hsic_scaled * n_spots_eff).numpy(), lambda_spy.numpy())

    elif null_method in ("trace", "welch"):
        S = y.T @ y
        trS = torch.trace(S).item()
        trS2 = torch.trace(S @ S).item()
        n1 = n_spots_eff - 1
        mean_null = trK_eff.item() * trS / n1
        var_null = 2.0 * trK2_eff.item() * trS2 / (n1**2)
        if null_method == "trace":
            # moment-matching normal (CLT) approximation
            z = (hsic_scaled.item() - mean_null) / (var_null**0.5 + 1e-12)
            pval = float(_norm_dist.sf(z))
        else:
            # Welch-Satterthwaite: match the first two moments of the chi-squared
            # mixture null to a scaled chi-squared g * chi2(h), with
            #   g = Var/(2 * E),  h = 2 * E^2 / Var
            # Falls back to the CLT z-test if the variance is non-positive.
            if var_null > 0 and mean_null > 0:
                scale_g = var_null / (2.0 * mean_null)
                df_h = 2.0 * mean_null**2 / var_null
                pval = float(_chi2_dist.sf(hsic_scaled.item() / scale_g, df=df_h))
            else:
                z = (hsic_scaled.item() - mean_null) / (var_null**0.5 + 1e-12)
                pval = float(_norm_dist.sf(z))

    else:  # null_method == "perm"
        p_isos = y.shape[1]
        null_stats = []
        for chunk_start in range(0, n_nulls, perm_batch_size):
            B = min(perm_batch_size, n_nulls - chunk_start)
            y_batch = torch.cat(
                [y[torch.randperm(n_spots_eff)] for _ in range(B)], dim=1
            )
            if isinstance(K_sp_null, torch.Tensor):
                R = y_batch.T @ K_sp_null @ y_batch
            else:
                R = K_sp_null.xtKx(y_batch)
            null_stats.append(torch.diagonal(R).reshape(B, p_isos).sum(dim=1))
        null_m = torch.cat(null_stats)
        pval = float((null_m > hsic_scaled).sum() / n_nulls)

    return hsic_norm, pval


def _du_hsic_gene_worker_np(
    counts: "torch.Tensor",
    z_list: list,
    ratio_transformation: str,
    nan_filling: str,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Per-gene worker for SplisosmNP DU test (method='hsic')."""
    if counts.is_sparse:
        counts = counts.to_dense()
    # Single-isoform gene: no differential usage is possible.
    if counts.shape[1] <= 1:
        n_factors = len(z_list)
        return torch.zeros(n_factors), torch.ones(n_factors)
    y = counts_to_ratios(
        counts,
        transformation=ratio_transformation,
        nan_filling=nan_filling,
        fill_before_transform=False,
    )
    n_factors = len(z_list)
    hsic_row = torch.empty(n_factors)
    pvals_row = torch.empty(n_factors)
    for _f, z in enumerate(z_list):
        hsic_row[_f], pvals_row[_f] = linear_hsic_test(z, y, centering=True)
    return hsic_row, pvals_row


def _du_hsic_gp_gene_worker_np(
    counts: "torch.Tensor",
    gpr_iso,
    x: "torch.Tensor",
    z_res_list: list,
    ratio_transformation: str,
    nan_filling: str,
    residualize: str,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Per-gene worker for SplisosmNP DU test (method='hsic-gp')."""
    if counts.is_sparse:
        counts = counts.to_dense()
    if counts.shape[1] <= 1:
        n_factors = len(z_res_list)
        return torch.zeros(n_factors), torch.ones(n_factors)
    y = counts_to_ratios(
        counts,
        transformation=ratio_transformation,
        nan_filling=nan_filling,
        fill_before_transform=False,
    )
    if residualize == "both" and gpr_iso is not None:
        y = gpr_iso.fit_residuals(x, y)
    n_factors = len(z_res_list)
    hsic_row = torch.empty(n_factors)
    pvals_row = torch.empty(n_factors)
    for _f, z_res in enumerate(z_res_list):
        hsic_row[_f], pvals_row[_f] = linear_hsic_test(z_res, y, centering=True)
    return hsic_row, pvals_row


def _du_ttest_gene_worker_np(
    counts: "torch.Tensor",
    groups_list: list,
    ratio_transformation: str,
    nan_filling: str,
    combine_method: str,
) -> "tuple[np.ndarray, np.ndarray]":
    """Per-gene worker for SplisosmNP DU test (method='t-fisher'/'t-tippett')."""
    if counts.is_sparse:
        counts = counts.to_dense()
    if counts.shape[1] <= 1:
        n_factors = len(groups_list)
        return np.zeros(n_factors), np.ones(n_factors)
    ratios = counts_to_ratios(
        counts,
        transformation=ratio_transformation,
        nan_filling=nan_filling,
        fill_before_transform=False,
    )
    n_factors = len(groups_list)
    stats_row = np.empty(n_factors)
    pvals_row = np.empty(n_factors)
    for _ind, groups in enumerate(groups_list):
        stats_row[_ind], pvals_row[_ind] = _calc_ttest_differential_usage(
            ratios, groups, combine_pval=True, combine_method=combine_method
        )
    return stats_row, pvals_row


class SplisosmNP:
    """Non-parametric spatial isoform statistical model.

    Examples
    --------
    Spatial variability test:

    >>> from splisosm import SplisosmNP
    >>> # adata : AnnData of shape (n_spots, n_isoforms)
    >>> #   adata.layers["counts"]    — raw isoform counts
    >>> #   adata.var["gene_symbol"]  — column grouping isoforms by gene
    >>> #   adata.obsm["spatial"]     — (n_spots, 2) spatial coordinates
    >>> model = SplisosmNP()
    >>> model.setup_data(adata, layer="counts", group_iso_by="gene_symbol")
    >>> model.test_spatial_variability(method="hsic-ir")
    >>> sv_results = model.get_formatted_test_results("sv")

    Differential usage test:

    >>> model = SplisosmNP()
    >>> model.setup_data(
    ...     adata, layer="counts", group_iso_by="gene_symbol",
    ...     design_mtx="covariate",  # obs column name, or (n_spots, n_factors) array
    ... )
    >>> model.test_differential_usage(method="hsic-gp", residualize="cov_only")
    >>> du_results = model.get_formatted_test_results("du")
    """

    # -- Public attributes (populated by :meth:`setup_data`) ------------------

    n_genes: int
    """Number of genes after filtering."""

    n_spots: int
    """Number of spatial spots/cells."""

    n_isos_per_gene: list[int]
    """Number of isoforms per gene (list of length :attr:`n_genes`)."""

    n_factors: int
    """Number of covariates for differential usage testing."""

    gene_names: list[str]
    """Gene display names (length :attr:`n_genes`)."""

    covariate_names: list[str]
    """Covariate display names (length :attr:`n_factors`)."""

    adata: Optional[AnnData]
    """Source :class:`~anndata.AnnData` object; ``None`` before :meth:`setup_data`."""

    sp_kernel: Any
    """Spatial kernel (:class:`~splisosm.kernel.SpatialCovKernel` or
    :class:`~splisosm.kernel.IdentityKernel`).  Set by :meth:`setup_data`."""

    design_mtx: Optional[torch.Tensor]
    """Design matrix ``(n_spots, n_factors)``; ``None`` if no covariates."""

    def __init__(
        self,
        k_neighbors: int = 4,
        rho: float = 0.99,
        standardize_cov: bool = True,
    ) -> None:
        """Initialise the model.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbours used to build the spatial adjacency
            graph for the CAR kernel (default 4).
        rho : float, optional
            Spatial autocorrelation strength in the CAR model (default 0.99).
            Values close to 1 give a smoother spatial kernel.
        standardize_cov : bool, optional
            Whether to standardise the spatial covariance matrix so that its
            diagonal entries are 1 (default ``True``).
        """
        # Spatial kernel hyperparameters (private config — shown in __str__)
        self._k_neighbors: int = k_neighbors
        self._rho: float = rho
        self._standardize_cov: bool = standardize_cov

        # Populated by setup_data()
        self.n_genes: int | None = None
        self.n_spots: int | None = None
        self.n_isos_per_gene: list[int] | None = None
        self.n_factors: int | None = None
        self.adata: AnnData | None = None
        self._setup_input_mode: str | None = None
        self._skip_kernel_construction: bool = False
        self._kernel_source: str | None = None

        # Feature summary cache (populated by _compute_feature_summaries)
        self._filtered_adata: AnnData | None = None
        self._gene_summary: pd.DataFrame | None = None
        self._isoform_summary: pd.DataFrame | None = None

        # Test results (private — access via get_formatted_test_results)
        self._sv_test_results: dict = {}
        self._du_test_results: dict = {}

    def __str__(self) -> str:
        """Return string representation of the model."""
        _sv_status = (
            f"Completed ({self._sv_test_results['method']})"
            if len(self._sv_test_results) > 0
            else "N/A"
        )
        _du_status = (
            f"Completed ({self._du_test_results['method']})"
            if len(self._du_test_results) > 0
            else "N/A"
        )
        _avg_iso = (
            f"{np.mean(self.n_isos_per_gene):.1f}"
            if self.n_isos_per_gene is not None
            else "N/A"
        )
        _k_neighors = (
            self._k_neighbors if "spatial_key" in self._kernel_source else "N/A"
        )
        return (
            "=== SplisosmNP\n"
            + f"- Number of genes: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Average isoforms per gene: {_avg_iso}\n"
            + "=== Model configurations\n"
            + f"- Spatial kernel source: {self._kernel_source}\n"
            + f"- k_neighbors: {_k_neighors}, rho: {self._rho}\n"
            + f"- Standardize spatial covariance: {self._standardize_cov}\n"
            + "=== Test results\n"
            + f"- Spatial variability: {_sv_status}\n"
            + f"- Differential usage: {_du_status}"
        )

    __repr__ = __str__

    @property
    def filtered_adata(self) -> AnnData:
        """The filtered AnnData of shape (:attr:`n_spots`, sum(:attr:`n_isos_per_gene`)).

        This is the data used internally after :meth:`setup_data`.
        It is a copy of the input :attr:`adata`, subsetted to the retained spots and isoforms after filtering.

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        """
        if self._filtered_adata is None:
            raise RuntimeError("Data not initialised. Call setup_data() first.")
        return self._filtered_adata

    def setup_data(
        self,
        adata: AnnData,
        *,
        spatial_key: str = "spatial",
        adj_key: Optional[str] = None,
        layer: str = "counts",
        group_iso_by: str = "gene_symbol",
        gene_names: Optional[str] = None,
        design_mtx: Optional[
            Union[torch.Tensor, np.ndarray, pd.DataFrame, str, list[str]]
        ] = None,
        covariate_names: Optional[list[str]] = None,
        min_counts: int = 10,
        min_bin_pct: float = 0.0,
        filter_single_iso_genes: bool = True,
        min_component_size: int = 1,
        skip_spatial_kernel: bool = False,
    ) -> None:
        """Setup isoform-level spatial data for hypothesis testing.

        Extracts isoform count tensors from an AnnData object, optionally
        filters disconnected graph components, builds a spatial covariance
        kernel, and resolves the design matrix.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix.  Counts are read from
            ``adata.layers[layer]`` grouped by ``group_iso_by``, and
            spatial coordinates from ``adata.obsm[spatial_key]``.
            See :func:`splisosm.utils.prepare_inputs_from_anndata` for
            full preprocessing details.
        spatial_key : str, optional
            Key in ``adata.obsm`` for spatial coordinates (default
            ``"spatial"``).  Optional when ``adj_key`` is provided: if the key
            is missing from ``adata.obsm`` the spatial kernel is built from
            the adjacency alone and coordinate-free SV tests
            (``"hsic-ir"`` / ``"hsic-ic"`` / ``"hsic-gc"``) and DU tests still
            run.  ``method="spark-x"`` (SV) and ``method="hsic-gp"`` (DU)
            require raw coordinates and raise a clear error at call time
            when they are absent.
        adj_key : str or None, optional
            Key in ``adata.obsp`` for a pre-built adjacency matrix.
            When provided, it overrides the k-NN graph construction
            from coordinates and be used directly to build the spatial kernel.
            Also makes ``spatial_key`` optional (see above).
            The adjacency matrix is symmetrized internally.
        layer : str, optional
            Layer in ``adata.layers`` that stores isoform counts (default
            ``"counts"``).
        group_iso_by : str, optional
            Column in ``adata.var`` used to group isoforms by gene
            (default ``"gene_symbol"``).
        gene_names : str or None, optional
            Column name in ``adata.var`` used as display names for genes.
            If ``None``, the values of ``group_iso_by`` are used.
        design_mtx : tensor, array, DataFrame, str, or list of str, optional
            Design matrix for differential-usage tests.  Accepts an
            array/tensor/DataFrame of shape ``(n_spots, n_factors)``, a
            single obs-column name (str), or a list of obs-column names.
            Categorical obs columns are one-hot encoded automatically.

            When a **scipy sparse matrix** is passed directly, it is stored as
            scipy CSR internally and all differential-usage methods handle it
            without densifying: ``"hsic"`` uses a sparse matrix-multiply path
            in :func:`linear_hsic_test`; ``"t-fisher"`` and ``"t-tippett"``
            extract group indices directly from the sparse non-zero structure.
            ``"hsic-gp"`` densifies each column via :meth:`_get_design_col`
            before GPR fitting (GPR residuals are always dense).

            All other input types (obs column names, array, tensor, DataFrame)
            are converted to a dense torch float32 tensor.
        covariate_names : list of str or None, optional
            Explicit covariate names.  When ``design_mtx`` is given as
            column name(s) and this is ``None``, the column names are used
            automatically; otherwise auto-generated as ``factor_1``, etc.
        min_counts : int, optional
            Minimum total isoform count across spots required to retain an
            isoform (default 10).
        min_bin_pct : float, optional
            Minimum fraction/percentage of spots where an isoform must be
            expressed (default 0.0).
        filter_single_iso_genes : bool, optional
            Whether to remove genes with fewer than two retained isoforms
            (default ``True``).
        min_component_size : int, optional
            Minimum number of spots a connected component must contain to
            be retained.  Spots in smaller components are removed from all
            data structures before the spatial kernel is built.  Default 1
            disables filtering.  A ``UserWarning`` is issued when spots are
            removed.
        skip_spatial_kernel : bool, optional
            If ``True``, skip construction of the CAR spatial kernel and
            store an :class:`~splisosm.kernel.IdentityKernel` placeholder as
            ``self.sp_kernel`` instead.  Use this when only
            :meth:`test_differential_usage` is needed (it fits custom GPR
            to handle spatial autocorrelation).
            Calling :meth:`test_spatial_variability` on a model set up with
            ``skip_spatial_kernel=True`` will raise a ``RuntimeError``.
            Default ``False``.

        Raises
        ------
        ValueError
            If input arguments are invalid or required fields are missing.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("`adata` must be an AnnData object.")

        (
            data,
            coordinates,
            resolved_gene_names,
            resolved_design,
            resolved_cov_names,
            adj_matrix,
            filtered_adata,
        ) = prepare_inputs_from_anndata(
            adata=adata,
            layer=layer,
            group_iso_by=group_iso_by,
            spatial_key=spatial_key,
            min_counts=min_counts,
            min_bin_pct=min_bin_pct,
            filter_single_iso_genes=filter_single_iso_genes,
            gene_names=gene_names,
            design_mtx=design_mtx,
            covariate_names=covariate_names,
            min_component_size=min_component_size,
            adj_key=adj_key,
            k_neighbors=self._k_neighbors,
            return_filtered_anndata=True,
        )

        self.adata = adata
        self._setup_input_mode = "anndata"
        self._counts_layer = layer
        self._group_iso_by = group_iso_by
        self._filtered_adata = filtered_adata
        self._gene_summary = None
        self._isoform_summary = None

        self.n_genes = len(data)
        # Derive n_spots from the filtered anndata
        self.n_spots = filtered_adata.shape[0]
        self.n_isos_per_gene = [g.shape[1] for g in data]
        self.gene_names = resolved_gene_names

        # Convert to float tensors
        self._data = [g.float() for g in data]
        self._coordinates = coordinates

        # Build spatial kernel from the adjacency returned by prepare_inputs_from_anndata.
        # adj_matrix is not None when:
        # (1) min_component_size > 1, or,
        # (2) adj_key is provided
        self._skip_kernel_construction = skip_spatial_kernel
        if skip_spatial_kernel:
            self.sp_kernel = IdentityKernel(self.n_spots, centering=True)
            self._kernel_source = "identity (skip_spatial_kernel=True)"
        elif adj_matrix is not None:
            self.sp_kernel = SpatialCovKernel(
                coords=None,
                adj_matrix=adj_matrix,
                rho=self._rho,
                standardize_cov=self._standardize_cov,
                centering=True,
            )
            self._kernel_source = (
                f"adj_key='{adj_key}'"
                if adj_key is not None
                else f"spatial_key='{spatial_key}' (component-filtered)"
            )
        else:
            self.sp_kernel = SpatialCovKernel(
                coords=coordinates,
                adj_matrix=None,
                k_neighbors=self._k_neighbors,
                rho=self._rho,
                standardize_cov=self._standardize_cov,
                centering=True,
            )
            self._kernel_source = f"spatial_key='{spatial_key}'"

        # Process design matrix from _process_design_mtx.
        # resolved_design is a numpy float32 array, a scipy sparse CSR matrix, or None.
        # Sparse design matrices are kept as scipy CSR to avoid densifying large
        # one-hot-encoded covariate tables; columns are extracted one at a time during
        # hypothesis testing.
        if resolved_design is not None:
            n_factors = resolved_design.shape[1]

            # Check for constant/zero-variance covariates without densifying
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                if scipy.sparse.issparse(resolved_design):
                    _means = np.asarray(resolved_design.mean(axis=0)).ravel()
                    _sq_means = np.asarray(
                        resolved_design.power(2).mean(axis=0)
                    ).ravel()
                    _stds = np.sqrt(np.maximum(_sq_means - _means**2, 0.0))
                    zero_var_indices = np.where(_stds < 1e-5)[0]
                else:
                    design_mtx_t = torch.from_numpy(
                        np.asarray(resolved_design, dtype=np.float32)
                    )
                    if design_mtx_t.dim() == 1:
                        design_mtx_t = design_mtx_t.unsqueeze(1)
                    _stds_t = design_mtx_t.std(dim=0)
                    zero_var_indices = torch.where(_stds_t < 1e-5)[0].numpy()
                for idx in zero_var_indices:
                    _cname = (
                        resolved_cov_names[int(idx)]
                        if resolved_cov_names is not None
                        else str(int(idx))
                    )
                    warnings.warn(
                        f"Covariate '{_cname}' has near-zero variance "
                        "(std < 1e-5). Consider removing it.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Store: sparse CSR when the input was sparse; dense torch tensor otherwise.
            if scipy.sparse.issparse(resolved_design):
                self.design_mtx = resolved_design.tocsr()
            else:
                self.design_mtx = design_mtx_t  # already constructed above
            self.n_factors = n_factors
            self.covariate_names = resolved_cov_names
        else:
            self.design_mtx = None
            self.n_factors = 0
            self.covariate_names = None

    def _get_design_col(self, factor_idx: int) -> torch.Tensor:
        """Extract one covariate column as a dense (n_spots, 1) float32 tensor.

        Works for both torch-tensor and scipy-sparse design matrices so that
        the bulk of the design matrix is never fully densified.
        """
        if scipy.sparse.issparse(self.design_mtx):
            col = np.asarray(self.design_mtx.getcol(factor_idx).todense()).ravel()
            return torch.from_numpy(col.astype(np.float32)).unsqueeze(1)
        return self.design_mtx[:, factor_idx].clone().float().unsqueeze(1)

    def _compute_feature_summaries(self, print_progress: bool = True) -> None:
        """Compute and cache both gene-level and isoform-level summaries."""
        if self._filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if self._gene_summary is not None and self._isoform_summary is not None:
            return
        self._gene_summary, self._isoform_summary = compute_feature_summaries(
            self._filtered_adata,
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
            - ``'pct_bin_on'``: float. Fraction of spots with non-zero total
              gene counts.
            - ``'count_avg'``: float. Mean per-spot total count for the gene.
            - ``'count_std'``: float. Std of per-spot total count for the gene.

            For ``level='isoform'``, the index is the isoform name (matching
            ``adata.var_names``) and the columns are the original ``adata.var``
            columns plus:

            - ``'pct_bin_on'``: float. Fraction of spots with count > 0.
            - ``'count_total'``: float. Total counts across all spots.
            - ``'count_avg'``: float. Mean count per spot.
            - ``'count_std'``: float. Std of count per spot.
            - ``'ratio_total'``: float. Fraction of total gene counts
              attributable to this isoform.
            - ``'ratio_avg'``: float. Mean per-spot isoform usage ratio
              (computed over spots with non-zero gene coverage).
            - ``'ratio_std'``: float. Std of per-spot isoform usage ratio
              (computed over spots with non-zero gene coverage).

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        ValueError
            If ``level`` is not ``'gene'`` or ``'isoform'``.
        """
        if self._filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if level not in {"gene", "isoform"}:
            raise ValueError("`level` must be one of 'gene' or 'isoform'.")

        self._compute_feature_summaries(print_progress=print_progress)

        if level == "gene":
            return self._gene_summary
        return self._isoform_summary

    def get_formatted_test_results(
        self,
        test_type: Literal["sv", "du"],
        with_gene_summary: bool = False,
    ) -> pd.DataFrame:
        """Get formatted test results as a pandas DataFrame.

        Parameters
        ----------
        test_type : {"sv", "du"}
            Which results to retrieve: ``"sv"`` for spatial variability or
            ``"du"`` for differential usage.
        with_gene_summary : bool, optional
            If ``True``, append gene-level summary statistics from
            :meth:`extract_feature_summary` (columns: ``n_isos``,
            ``perplexity``, ``pct_bin_on``, ``count_avg``, ``count_std``).

        Returns
        -------
        pandas.DataFrame
            Formatted test results.
        """
        if test_type not in {"sv", "du"}:
            raise ValueError("test_type must be 'sv' or 'du'.")
        if test_type == "sv":
            if len(self._sv_test_results) == 0:
                raise ValueError(
                    "No spatial variability results. Run test_spatial_variability() first."
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
                    "No differential usage results. Run test_differential_usage() first."
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

    def test_spatial_variability(
        self,
        method: Literal["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"] = "hsic-ir",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        null_method: Literal["eig", "trace", "welch", "perm"] = "eig",
        null_configs: Optional[dict[str, Any]] = None,
        n_jobs: int = -1,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial variability.

        Kernel-based multivariate hypothesis testing for spatial variability in

        - gene-level total counts (``"hsic-gc"`` or ``"spark-x"`` :cite:`zhu2021spark`)
        - isoform usage ratios (``"hsic-ir"``)
        - isoform counts (``"hsic-ic"``)

        Test statistics and p-values are computed per gene for each gene separately.

        Parameters
        ----------
        method : {"hsic-ir", "hsic-ic", "hsic-gc", "spark-x"}, optional
            Test target: ``"hsic-ir"`` (isoform usage ratios), ``"hsic-ic"``
            (isoform counts), ``"hsic-gc"`` (gene-level counts), or
            ``"spark-x"`` (SPARK-X :cite:`zhu2021spark`).
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Compositional transformation applied to isoform ratios when
            ``method="hsic-ir"``.  See :func:`splisosm.utils.counts_to_ratios`
            and :cite:`park2022kernel` for details.
        nan_filling : {"mean", "none"}, optional
            Strategy for NaN values in isoform ratios.
            See :func:`splisosm.utils.counts_to_ratios` for details.
        null_method : {"eig", "trace", "welch", "perm"}, optional
            Method for computing the null distribution of the test statistic:

            * ``"eig"`` (default): asymptotic chi-square mixture using kernel
              eigenvalues; Liu's method :cite:`liu2009new`.  Supports optional
              ``null_configs["approx_rank"]`` (int) to use only the top-k
              eigenvalues. By default, approx_rank = np.ceil(np.sqrt(n_spots) * 4)
              for large datasets (n_spots > 5000). Set it to None to use
              all eigenvalues, which can be slow for large n_spots.
            * ``"trace"``: moment-matching normal (CLT) approximation using
              tr(K') and tr(K'²) of the (centred) spatial kernel.  Fastest,
              but tail probabilities can be inaccurate when the effective
              degrees of freedom of the chi-squared mixture is small.
            * ``"welch"``: Welch-Satterthwaite moment matching.  Uses the same
              tr(K') and tr(K'²) as ``"trace"`` but approximates the null by a
              scaled chi-squared ``g * chi2(h)`` with
              ``g = Var/(2*E)`` and ``h = 2*E^2/Var``.  Comparable cost to
              ``"trace"`` with more accurate right-tail p-values, typically
              closer to the ``"eig"`` (Liu) reference.
            * ``"perm"``: permutation-based null distribution.  Supports
              optional ``null_configs["n_perms_per_gene"]`` (default 1000),
              and ``null_configs["perm_batch_size"]`` (default 50, larger values
              lead to more memory usage) for batch-wise null statistic computation.
        null_configs : dict or None, optional
            Extra keyword arguments for the chosen ``null_method``.
        n_jobs : int, optional
            Number of parallel workers for the per-gene loop.  ``-1`` uses all
            available CPUs.  Each worker densifies one sparse count tensor
            (~4–40 MB at 100 K–1 M spots × 10 isoforms) so choose ``n_jobs``
            to fit within available RAM.  Default ``-1``.
        return_results : bool, optional
            If ``True``, return the result dict.  Otherwise store results in
            :attr:`sv_test_results` and return ``None``.
        print_progress : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise, returns None and stores results in self._sv_test_results.

        Notes
        -----
        To run the SPARK-X test, the R-package `SPARK` must be installed and accessible from Python via `rpy2`.
        """

        if self._skip_kernel_construction:
            raise RuntimeError(
                "setup_data was called with skip_spatial_kernel=True; the spatial "
                "kernel is a placeholder IdentityKernel. Re-run setup_data with "
                "skip_spatial_kernel=False to enable spatial variability testing."
            )

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"]
        valid_null_methods = ["eig", "trace", "welch", "perm"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["mean", "none"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            null_method in valid_null_methods
        ), f"Invalid null method. Must be one of {valid_null_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid ratio transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid NaN filling method. Must be one of {valid_nan_filling}."

        if nan_filling == "none" and method == "hsic-ir":
            warnings.warn(
                "nan_filling='none' with method='hsic-ir' triggers a per-gene spatial "
                "kernel materialization and eigendecomposition (O(n_spots³) per gene). "
                "For large datasets this is very slow. "
                "Consider nan_filling='mean' to reuse a single pre-computed kernel.",
                UserWarning,
                stacklevel=2,
            )

        if method == "spark-x":  # run the gene-level SPARK-X test
            if self._coordinates is None:
                raise ValueError(
                    "method='spark-x' requires raw spatial coordinates, but "
                    "setup_data was called on an AnnData without "
                    "`obsm[spatial_key]`. Re-run setup_data with spatial "
                    "coordinates, or choose the kernel-based 'hsic-gc'."
                )
            # prepare the data in gene-level counts
            counts_g = torch.concat(
                [_counts.sum(1, keepdim=True) for _counts in self._data], axis=1
            )  # tensor(n_spots, n_genes)
            self._sv_test_results = run_sparkx(
                counts_g.numpy(), self._coordinates.numpy()
            )
        else:
            if n_jobs == -1:
                n_jobs = os.cpu_count() or 1

            # use a global spatial kernel unless nan_filling is 'none'
            n_spots = self.n_spots
            K_sp = self.sp_kernel  # the Kernel class object was already centered

            # pre-compute null distribution inputs (once, before per-gene loop)
            lambda_sp = None
            k_eff = 0
            trK = None
            trK2 = None
            n_nulls = 0
            _perm_batch_size = 1
            configs = null_configs or {}
            if null_method == "eig":
                _rank = (
                    np.ceil(np.sqrt(self.n_spots) * 4).astype(int)
                    if self.n_spots > 5000
                    else self.n_spots
                )
                approx_rank = configs.get("approx_rank", _rank)
                if approx_rank is None and self.n_spots > 5000:
                    warnings.warn(
                        "Computing all eigenvalues for null distribution can be slow for large n_spots. "
                        "Consider setting a small value for null_configs['approx_rank'] to use low-rank approximation.",
                        UserWarning,
                        stacklevel=2,
                    )
                lambda_sp = self.sp_kernel.eigenvalues(k=approx_rank)
                lambda_sp = lambda_sp[lambda_sp > 1e-5]
                k_eff = len(lambda_sp)
            elif null_method in ("trace", "welch"):
                trK = self.sp_kernel.trace()
                trK2 = self.sp_kernel.square_trace()
            elif null_method == "perm":
                n_nulls = int(configs.get("n_perms_per_gene", 1000))
                _perm_batch_size = int(configs.get("perm_batch_size", 50))

            # parallel per-gene SV test; shared objects are read-only
            _sv_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_sv_gene_worker_np)(
                    counts,
                    method,
                    ratio_transformation,
                    nan_filling,
                    null_method,
                    n_spots,
                    K_sp,
                    lambda_sp if null_method == "eig" else None,
                    k_eff if null_method == "eig" else 0,
                    trK if null_method in ("trace", "welch") else None,
                    trK2 if null_method in ("trace", "welch") else None,
                    n_nulls if null_method == "perm" else 0,
                    _perm_batch_size if null_method == "perm" else 1,
                )
                for counts in tqdm(
                    self._data,
                    desc=f"SV [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            )
            hsic_arr = np.array([r[0] for r in _sv_results], dtype=float)
            pvals_arr = np.array([r[1] for r in _sv_results], dtype=float)

            # store results after the loop (NP-12 fix: no longer rebuilt per gene)
            self._sv_test_results = {
                "statistic": hsic_arr,
                "pvalue": pvals_arr,
                "method": method,
                "null_method": null_method,
            }

            # calculate adjusted p-values
            self._sv_test_results["pvalue_adj"] = false_discovery_control(
                self._sv_test_results["pvalue"]
            )

        # return results
        if return_results:
            return self._sv_test_results

    def test_differential_usage(
        self,
        method: Literal["hsic", "hsic-gp", "t-fisher", "t-tippett"] = "hsic-gp",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        gpr_backend: Literal["sklearn", "gpytorch"] = "sklearn",
        gpr_configs: Optional[dict[str, Any]] = None,
        residualize: Literal["cov_only", "both"] = "cov_only",
        n_jobs: int = -1,
        print_progress: bool = True,
        return_results: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial isoform differential usage.

        Before running this function, the design matrix must be set up using :func:`setup_data`.
        Each column of the design matrix corresponds to a covariate to test for differential
        association with the isoform usage ratios of each gene.
        Test statistics and p-values are computed per (gene, covariate) pair separately.

        Two types of association tests are supported:

        - Unconditional (``"hsic"``, ``"t-fisher"``, ``"t-tippett"``): test the
          unconditional association between isoform usage ratios and the covariate.
        - Conditional (``"hsic-gp"``): test the association conditioned on spatial
          coordinates via Gaussian process regression.  See :cite:`zhang2012kernel`
          for more details.

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

            * ``"t-fisher"``, ``"t-tippett"``: each isoform is tested independently
              and p-values are combined gene-wise via Fisher's or Tippett's method.
        ratio_transformation : str, optional
            Compositional transformation for isoform ratios.
            One of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, ``'radial'``
            :cite:`park2022kernel`.  See :func:`splisosm.utils.counts_to_ratios`.
        nan_filling : str, optional
            How to fill NaN values in isoform ratios.  One of ``'mean'`` or ``'none'``.
            See :func:`splisosm.utils.counts_to_ratios`.
        gpr_backend : str, optional
            GPR backend to use for ``method='hsic-gp'``.
            One of ``'sklearn'`` (default) or ``'gpytorch'``.
            For FFT-accelerated spatial GP on regular grids use
            :class:`~splisosm.hyptest_fft.SplisosmFFT` instead.
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
                        "n_inducing": 5000,
                    },
                    "isoform": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                        "n_inducing": 5000,
                    },
                }

            ``"n_inducing"`` *(int or None)* controls the scale of spatial GP
            fitting for each backend:

            * **sklearn** — maximum number of observations used for
              hyperparameter fitting.  Full exact GP when ``n_obs ≤ n_inducing``
              (or ``None``); a randomly sub-sampled **subset-of-data** of
              ``n_inducing`` points otherwise (**not** the same inducing-point
              approximation as gpytorch).  Default: ``5000``.  Set to ``None``
              to use all observations (warns when ``n_obs > 10_000``).
            * **gpytorch** — FITC sparse-GP inducing-point approximation with
              ``n_inducing`` points; set to ``None`` for exact GP.
              Default: ``5000``.

        residualize : {"cov_only", "both"}, optional
            Controls which signals are spatially residualized when
            ``method="hsic-gp"``:

            * ``"cov_only"`` (default): residualize covariates only; test
              HSIC(Z_res, Y_raw).  Fastest; calibration matches ``"both"``
              when covariate GPR captures most spatial confounding.
            * ``"both"``: residualize both covariates and isoform ratios.
        n_jobs : int, optional
            Number of parallel workers for the per-gene loop.  ``-1`` uses all
            available CPUs.  Each worker densifies one sparse count tensor
            (~4–40 MB at 100 K–1 M spots × 10 isoforms).  When
            ``gpr_backend="gpytorch"`` and ``device != "cpu"``, the GPU is not
            thread-safe; parallelism is automatically disabled.  Default ``-1``.
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
        """
        if self.design_mtx is None:
            raise ValueError(
                "Cannot find the design matrix. Perhaps you forgot to set it up using setup_data()."
            )

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["hsic", "hsic-gp", "t-fisher", "t-tippett"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["none", "mean"]
        valid_residualize = ["cov_only", "both"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid nan_filling. Must be one of {valid_nan_filling}."
        assert (
            residualize in valid_residualize
        ), f"Invalid residualize. Must be one of {valid_residualize}."

        n_genes = self.n_genes
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if method == "hsic":  # unconditional HSIC test (multivariate RV coefficient)
            # Pre-compute covariates: keep sparse when design_mtx is scipy sparse
            # so that linear_hsic_test can use the memory-efficient sparse-X path.
            z_list = []
            for _ind in range(n_factors):
                if scipy.sparse.issparse(self.design_mtx):
                    z = self.design_mtx.getcol(_ind)  # scipy sparse (n_spots, 1)
                    _mean = float(z.mean())
                    _sq_mean = float(z.multiply(z).mean())
                    _std = float(np.sqrt(max(_sq_mean - _mean**2, 0.0)))
                else:
                    z = self._get_design_col(_ind)  # dense (n_spots, 1) tensor
                    _std = float(z.std())
                assert (
                    _std > 1e-5
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z_list.append(z)

            hsic_all = torch.empty(n_genes, n_factors)
            pvals_all = torch.empty(n_genes, n_factors)

            _du_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_du_hsic_gene_worker_np)(
                    counts, z_list, ratio_transformation, nan_filling
                )
                for counts in tqdm(
                    self._data,
                    desc=f"DU [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            )
            for _g, (h_row, p_row) in enumerate(_du_results):
                hsic_all[_g] = h_row
                pvals_all[_g] = p_row

            self._du_test_results = {
                "statistic": hsic_all.numpy(),
                "pvalue": pvals_all.numpy(),
                "method": method,
            }

        elif method == "hsic-gp":  # conditional HSIC via GP regression residuals
            if self._coordinates is None:
                raise ValueError(
                    "method='hsic-gp' fits a Gaussian process on raw spatial "
                    "coordinates, but setup_data was called on an AnnData "
                    "without `obsm[spatial_key]`. Re-run setup_data with "
                    "spatial coordinates, or use an unconditional DU method "
                    "(e.g. method='hsic', 't-fisher', 't-tippett')."
                )
            # Build GPR configs (merge user overrides over defaults)
            cov_config = {**_DEFAULT_GPR_CONFIGS["covariate"]}
            iso_config = {**_DEFAULT_GPR_CONFIGS["isoform"]}
            if gpr_configs is not None:
                if gpr_configs.keys() - {"covariate", "isoform"}:
                    raise ValueError(
                        "gpr_configs must have a nested structure. Use keys "
                        "'covariate' and/or 'isoform' for the respective GPR configurations."
                    )
                if "covariate" in gpr_configs:
                    cov_config.update(gpr_configs["covariate"])
                if "isoform" in gpr_configs:
                    iso_config.update(gpr_configs["isoform"])

            # Normalize spatial coordinates once
            x = torch.as_tensor(
                self._coordinates, dtype=torch.float64
            ).clone()  # (n_spots, n_dims)
            x = (x - x.mean(0)) / x.std(0).clamp(min=1e-8)

            # Fit GPR for covariates and get residuals (n_factors small tensors, never sparse)
            gpr_cov = make_kernel_gpr(gpr_backend, **cov_config)
            z_res_list = []
            for _ind in tqdm(
                range(n_factors),
                desc="Covariates",
                total=n_factors,
                disable=not print_progress,
            ):
                z = self._get_design_col(_ind).squeeze(1)
                assert (
                    z.std() > 0
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z = (z - z.mean()) / z.std()
                z_res_list.append(gpr_cov.fit_residuals(x, z.unsqueeze(1)))

            # Optionally fit GPR for isoform ratios to residualize spatial effects in the response as well.
            gpr_iso = None
            if residualize == "both":
                gpr_iso = make_kernel_gpr(gpr_backend, **iso_config)
                # Warm up the shared eigendecomposition for backends that support it
                # (e.g. sklearn with fixed signal bounds) so the first gene does not
                # pay the cost of a redundant full GP fit.
                if (
                    hasattr(gpr_iso, "precompute_shared_kernel")
                    and gpr_iso.signal_bounds_fixed
                ):
                    gpr_iso.precompute_shared_kernel(x)

            # GPU guard: CUDA operations are not thread-safe across threads
            _iso_device = iso_config.get("device", "cpu")
            if gpr_backend == "gpytorch" and _iso_device != "cpu":
                _effective_n_jobs = 1  # sequential fallback for GPU backend
            else:
                _effective_n_jobs = n_jobs

            # --- Main loop: densify and process one gene at a time ---
            hsic_all = torch.empty(n_genes, n_factors)
            pvals_all = torch.empty(n_genes, n_factors)

            _gpr_iso_for_worker = gpr_iso if residualize == "both" else None
            _du_gp_results = Parallel(n_jobs=_effective_n_jobs, prefer="threads")(
                delayed(_du_hsic_gp_gene_worker_np)(
                    counts,
                    _gpr_iso_for_worker,
                    x,
                    z_res_list,
                    ratio_transformation,
                    nan_filling,
                    residualize,
                )
                for counts in tqdm(
                    self._data,
                    desc=f"DU [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            )
            for _g, (h_row, p_row) in enumerate(_du_gp_results):
                hsic_all[_g] = h_row
                pvals_all[_g] = p_row

            self._du_test_results = {
                "statistic": hsic_all.numpy(),
                "pvalue": pvals_all.numpy(),
                "method": method,
            }

        else:  # two-sample t-test
            # method to combine p-values across isoforms, either 'fisher' or 'tippett'
            combine_method = re.findall(r"^t-(.+)", method)[0]

            stats_all = np.empty((n_genes, n_factors))
            pvals_all = np.empty((n_genes, n_factors))

            # Pre-extract group columns once (sparse or dense) to avoid repeated
            # column lookups inside the gene loop.
            _design_is_sparse = scipy.sparse.issparse(self.design_mtx)
            groups_list = [
                (
                    self.design_mtx.getcol(_ind)  # scipy sparse (n, 1)
                    if _design_is_sparse
                    else self.design_mtx[:, _ind]
                )  # dense 1-D tensor
                for _ind in range(n_factors)
            ]

            # Parallel per-gene t-test
            _du_tt_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_du_ttest_gene_worker_np)(
                    counts,
                    groups_list,
                    ratio_transformation,
                    nan_filling,
                    combine_method,
                )
                for counts in tqdm(
                    self._data,
                    desc=f"DU [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            )
            for _g, (s_row, p_row) in enumerate(_du_tt_results):
                stats_all[_g] = s_row
                pvals_all[_g] = p_row

            self._du_test_results = {
                "statistic": stats_all,  # (n_genes, n_factors)
                "pvalue": pvals_all,  # (n_genes, n_factors)
                "method": method,
            }

        # calculate adjusted p-values (independently for each factor)
        self._du_test_results["pvalue_adj"] = false_discovery_control(
            self._du_test_results["pvalue"], axis=0
        )

        # return the results if needed
        if return_results:
            return self._du_test_results
