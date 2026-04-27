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
)
import numpy as np
import scipy.sparse
import pandas as pd
import torch
from tqdm import tqdm
from anndata import AnnData

from splisosm.hyptest._base import _FeatureSummaryMixin, _ResultsMixin
from splisosm.utils.preprocessing import (
    counts_to_ratios,
    prepare_inputs_from_anndata,
)
from splisosm.utils.stats import (
    false_discovery_control,
    run_sparkx,
)
from splisosm.utils._chunking import pack_gene_chunks, resolve_chunk_size
from splisosm.kernel import IdentityKernel, SpatialCovKernel, _MaskedSpatialKernel
from splisosm.utils.hsic import (
    _feature_cumulants_from_data,
    _hutchinson_cumulants,
    _hsic_liu_pvalue,
    _hsic_welch_pvalue,
    _kernel_cumulants_for_null,
    _normalize_hsic_null_method,
    linear_hsic_test,
)
from splisosm.gpr.config import _DEFAULT_GPR_CONFIGS
from splisosm.gpr.factory import make_kernel_gpr

__all__ = ["SplisosmNP"]


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


def _torch_sparse_to_scipy_csc(tensor: torch.Tensor) -> scipy.sparse.csc_matrix:
    """Convert a 2-D torch sparse COO tensor to scipy CSC without densifying."""
    tensor = tensor.coalesce().detach().cpu()
    idx = tensor.indices().numpy()
    vals = tensor.values().numpy()
    return scipy.sparse.coo_matrix(
        (vals, (idx[0], idx[1])),
        shape=tuple(tensor.shape),
    ).tocsc()


def _tensor_to_numpy_dense(x: torch.Tensor) -> np.ndarray:
    """Return a CPU float64 dense numpy view/copy of a torch tensor."""
    if x.is_sparse:
        x = x.to_dense()
    return x.detach().cpu().numpy().astype(np.float64, copy=False)


def _response_width_np(
    counts: torch.Tensor,
    method: str,
    ratio_transformation: str,
) -> int:
    """Return response-column width contributed by one gene."""
    if method == "hsic-gc":
        return 1
    n_cols = int(counts.shape[1])
    if method == "hsic-ir" and ratio_transformation in {"ilr", "alr"}:
        return max(1, n_cols - 1)
    return max(1, n_cols)


def _sparse_counts_to_ratios_centered(
    counts: scipy.sparse.spmatrix,
    transformation: str,
    nan_filling: str,
) -> tuple[scipy.sparse.csc_matrix, np.ndarray | None]:
    """Return a centred sparse ratio matrix from sparse isoform counts.

    The calculation mirrors :func:`splisosm.utils.preprocessing.counts_to_ratios` with
    ``fill_before_transform=False``, then centres the returned ratio matrix.
    For ``nan_filling="mean"``, zero-coverage rows are filled with the
    expressed-row column mean and become exactly zero after centring.  For
    ``nan_filling="none"``, zero-coverage rows are left as structural zeros in
    the sparse output and returned as a boolean NaN-row mask.
    """
    if transformation not in {"none", "clr", "ilr", "alr", "radial"}:
        raise ValueError(f"Unsupported ratio transformation: {transformation!r}.")
    if nan_filling not in {"mean", "none"}:
        raise ValueError("`nan_filling` must be either 'mean' or 'none'.")

    counts_csr = counts.tocsr(copy=False)
    n_spots, n_features = counts_csr.shape
    row_sums = np.asarray(counts_csr.sum(axis=1)).ravel().astype(np.float64, copy=False)
    expressed = row_sums > 0.0
    nan_mask = ~expressed

    out_features = n_features - 1 if transformation in {"ilr", "alr"} else n_features
    if not expressed.any():
        sparse = scipy.sparse.csc_matrix((n_spots, max(0, out_features)), dtype=float)
        return sparse, nan_mask if nan_filling == "none" else None

    expressed_rows = np.flatnonzero(expressed)
    counts_expr = counts_csr[expressed_rows].toarray().astype(np.float64, copy=False)

    if transformation == "none":
        values_expr = counts_expr / row_sums[expressed_rows, None]
    elif transformation in {"clr", "ilr", "alr"}:
        try:
            from skbio.stats.composition import alr, clr, ilr
        except ImportError:
            warnings.warn(
                f"Please install scikit-bio to use ratio transformation='{transformation}'. Switching to 'none'.",
                UserWarning,
                stacklevel=2,
            )
            values_expr = counts_expr / row_sums[expressed_rows, None]
        else:
            # Match counts_to_ratios(..., fill_before_transform=False): the
            # pseudocount uses the global column mean, including zero rows.
            global_mean = np.asarray(counts_csr.sum(axis=0)).ravel() / float(n_spots)
            values_expr = (1 - 1e-2) * counts_expr + 1e-2 * global_mean[None, :]
            values_expr = values_expr / values_expr.sum(axis=1, keepdims=True)
            if transformation == "clr":
                values_expr = clr(values_expr)
            elif transformation == "ilr":
                values_expr = ilr(values_expr)
            else:
                values_expr = alr(values_expr)
    elif transformation == "radial":
        values_expr = counts_expr / row_sums[expressed_rows, None]
        values_expr = values_expr / np.linalg.norm(values_expr, axis=1, keepdims=True)

    values_expr = np.asarray(values_expr, dtype=np.float64)
    centered_expr = values_expr - values_expr.mean(axis=0, keepdims=True)
    sparse_expr = scipy.sparse.coo_matrix(centered_expr)
    sparse = scipy.sparse.coo_matrix(
        (
            sparse_expr.data,
            (expressed_rows[sparse_expr.row], sparse_expr.col),
        ),
        shape=(n_spots, centered_expr.shape[1]),
    ).tocsc()
    sparse.eliminate_zeros()
    return sparse, nan_mask if nan_filling == "none" else None


def _prepare_np_response(
    counts: torch.Tensor,
    method: str,
    ratio_transformation: str,
) -> tuple[np.ndarray | scipy.sparse.csc_matrix, bool]:
    """Build one gene's response matrix and whether it is already centred."""
    if method == "hsic-ic":
        if counts.is_sparse:
            sparse = _torch_sparse_to_scipy_csc(counts)
            return sparse, False
        return _tensor_to_numpy_dense(counts), False

    if method == "hsic-gc":
        if counts.is_sparse:
            sparse_counts = _torch_sparse_to_scipy_csc(counts)
            sparse = sparse_counts.sum(axis=1)
            sparse = scipy.sparse.csc_matrix(sparse)
            return sparse, False
        dense = _tensor_to_numpy_dense(counts).sum(axis=1, keepdims=True)
        return dense, False

    if counts.is_sparse:
        sparse_counts = _torch_sparse_to_scipy_csc(counts)
        sparse, _ = _sparse_counts_to_ratios_centered(
            sparse_counts,
            transformation=ratio_transformation,
            nan_filling="mean",
        )
        return sparse, True

    dense_counts = counts.to_dense() if counts.is_sparse else counts
    y = counts_to_ratios(
        dense_counts,
        transformation=ratio_transformation,
        nan_filling="mean",
        fill_before_transform=False,
    )
    return _tensor_to_numpy_dense(y), False


def _quadratic_columns_exact(
    K_sp: Any,
    response: np.ndarray | scipy.sparse.spmatrix,
) -> np.ndarray:
    """Return diagonal of ``X.T @ H K H @ X`` for uncentered responses."""
    if scipy.sparse.issparse(response):
        mat = K_sp.xtKx_exact(response)
    else:
        tensor = torch.from_numpy(np.asarray(response, dtype=np.float32))
        mat = K_sp.xtKx_exact(tensor)
    return torch.diagonal(mat).detach().cpu().numpy().astype(float)


def _quadratic_columns_approx(
    K_sp: Any,
    response: np.ndarray | scipy.sparse.spmatrix,
    k: int,
) -> np.ndarray:
    """Return diagonal column quadratic forms under a rank-k spatial kernel."""
    if scipy.sparse.issparse(response):
        mat = K_sp.xtKx_approx(response, k=k)
    else:
        tensor = torch.from_numpy(np.asarray(response, dtype=np.float32))
        mat = K_sp.xtKx_approx(tensor, k=k)
    return torch.diagonal(mat).detach().cpu().numpy().astype(float)


def _sv_chunk_worker_np(
    counts_chunk: list[torch.Tensor],
    method: str,
    ratio_transformation: str,
    nan_filling: str,
    null_method: str,
    n_spots: int,
    K_sp: Any,
    kernel_cumulants: Optional[dict[int, float]],
    kernel_approx_rank: Optional[int],
    n_nulls: int,
    perm_batch_size: int,
    n_probes: int,
    rng_seed: int,
) -> list[tuple[float, float]]:
    """Process a response-column chunk for NP spatial variability."""
    if null_method == "perm" or (method == "hsic-ir" and nan_filling == "none"):
        return [
            _sv_gene_worker_np(
                counts,
                method,
                ratio_transformation,
                nan_filling,
                null_method,
                n_spots,
                K_sp,
                kernel_cumulants,
                kernel_approx_rank,
                n_nulls,
                perm_batch_size,
                n_probes,
                rng_seed,
            )
            for counts in counts_chunk
        ]

    results: list[tuple[float, float] | None] = [None] * len(counts_chunk)
    response_blocks: list[np.ndarray | scipy.sparse.csc_matrix] = []
    centered_flags: list[bool] = []
    active_positions: list[int] = []
    slices: list[slice] = []
    start = 0

    for pos, counts in enumerate(counts_chunk):
        if method == "hsic-ir" and counts.shape[1] <= 1:
            results[pos] = (0.0, 1.0)
            continue
        response, is_centered = _prepare_np_response(
            counts, method, ratio_transformation
        )
        stop = start + response.shape[1]
        response_blocks.append(response)
        centered_flags.append(is_centered)
        active_positions.append(pos)
        slices.append(slice(start, stop))
        start = stop

    if not response_blocks:
        return [(0.0, 1.0) if r is None else r for r in results]

    if all(scipy.sparse.issparse(block) for block in response_blocks):
        response_all = scipy.sparse.hstack(response_blocks, format="csc")
    else:
        response_all = np.concatenate(
            [
                block.toarray() if scipy.sparse.issparse(block) else block
                for block in response_blocks
            ],
            axis=1,
        )

    is_response_all_sparse = scipy.sparse.issparse(response_all)
    response_all_dense = (
        None if is_response_all_sparse else np.asarray(response_all, dtype=np.float64)
    )

    if null_method == "liu" and kernel_approx_rank is not None:
        q_cols = _quadratic_columns_approx(K_sp, response_all, k=kernel_approx_rank)
    else:
        q_cols = _quadratic_columns_exact(K_sp, response_all)

    for pos, sl, is_centered in zip(active_positions, slices, centered_flags):
        y_gene = (
            response_all[:, sl] if is_response_all_sparse else response_all_dense[:, sl]
        )
        is_finite = (
            np.isfinite(y_gene.data).all()
            if scipy.sparse.issparse(y_gene)
            else np.isfinite(y_gene).all()
        )
        if not is_finite:
            results[pos] = (float("nan"), 1.0)
            continue

        hsic_scaled = float(np.sum(q_cols[sl]))
        hsic_norm = float(hsic_scaled / (n_spots - 1) ** 2)
        feature_cumulants = _feature_cumulants_from_data(
            y_gene,
            centered=is_centered,
        )

        if feature_cumulants[2] <= 0.0 or (kernel_cumulants or {}).get(2, 0.0) <= 0.0:
            pval = 1.0
        elif null_method == "liu":
            pval = _hsic_liu_pvalue(
                hsic_scaled,
                kernel_cumulants,
                feature_cumulants,
                n_spots,
            )
        else:
            pval = _hsic_welch_pvalue(
                hsic_scaled,
                kernel_cumulants,
                feature_cumulants,
                n_spots,
            )
        results[pos] = (hsic_norm, pval)

    return [r if r is not None else (0.0, 1.0) for r in results]


def _sv_gene_worker_np(
    counts: "torch.Tensor",
    method: str,
    ratio_transformation: str,
    nan_filling: str,
    null_method: str,
    n_spots: int,
    K_sp,
    kernel_cumulants: "Optional[dict[int, float]]",
    kernel_approx_rank: "Optional[int]",
    n_nulls: int,
    perm_batch_size: int,
    n_probes: int = 60,
    rng_seed: int = 0,
) -> "tuple[float, float]":
    """Process a single gene for :meth:`SplisosmNP.test_spatial_variability`.

    Shared objects (``K_sp`` and ``kernel_cumulants``) are passed by
    reference and only read, making this safe for
    ``joblib.Parallel(prefer="threads")``.

    Returns
    -------
    tuple[float, float]
        ``(hsic_norm, pval)`` for this gene.
    """
    # Single-isoform gene with HSIC-IR: ratios are constant (all 1.0), so there is
    # no usage variation to detect.  Return stat=0, pval=1.
    # HSIC-IC and HSIC-GC still produce meaningful results for single-isoform genes
    # (they test count-level spatial variability, equivalent to gene-level SV).
    if method == "hsic-ir" and counts.shape[1] <= 1:
        return (0.0, 1.0)

    kernel_cumulants_eff = kernel_cumulants
    kernel_approx_rank_eff = kernel_approx_rank
    # By default, null-distribution inputs come from the global (already
    # double-centred) K_sp.  The `hsic-ir + nan_filling='none'` branch below
    # overrides these with per-gene values derived from K_sp_gene so that the
    # statistic and the null reference the *same* (centred) kernel submatrix.
    K_sp_null = K_sp

    if method == "hsic-ir" and nan_filling == "none":
        if counts.is_sparse:
            sparse_counts = _torch_sparse_to_scipy_csc(counts)
            y_sparse, is_nan = _sparse_counts_to_ratios_centered(
                sparse_counts,
                transformation=ratio_transformation,
                nan_filling="none",
            )
            keep_mask = ~is_nan
            y = y_sparse[keep_mask]
            y_is_centered = True
        else:
            # per-gene spatial kernel: drop NaN spots and recompute K
            y = counts_to_ratios(
                counts,
                transformation=ratio_transformation,
                nan_filling="none",
                fill_before_transform=False,
            )
            is_nan = torch.isnan(y).any(1)
            keep_mask = (~is_nan).detach().cpu().numpy()
            y = y[~is_nan]
            y = y - y.mean(0, keepdim=True)
            y_is_centered = True
        n_spots_eff = y.shape[0]
        if n_spots_eff <= 1:
            return (0.0, 1.0)
        K_sp_gene = _MaskedSpatialKernel(K_sp, keep_mask)

        hsic_scaled = torch.trace(K_sp_gene.xtKx_exact(y))

        # The passed-in null inputs describe the global
        # K_sp; with per-gene NaN filtering they no longer match the statistic.
        # Recompute them from the per-gene masked kernel.
        if null_method in ("liu", "welch"):
            kernel_cumulants_eff = _hutchinson_cumulants(
                K_sp_gene,
                n_probes=n_probes,
                rng_seed=rng_seed,
                max_power=4 if null_method == "liu" else 2,
            )
            kernel_approx_rank_eff = None
        else:
            K_sp_null = K_sp_gene

    else:  # global spatial kernel shared across all genes
        if counts.is_sparse:
            sparse_counts = _torch_sparse_to_scipy_csc(counts)
            if method == "hsic-ic":
                y = sparse_counts
                y_is_centered = False
            elif method == "hsic-gc":
                y = scipy.sparse.csc_matrix(sparse_counts.sum(axis=1))
                y_is_centered = False
            else:
                y, _ = _sparse_counts_to_ratios_centered(
                    sparse_counts,
                    transformation=ratio_transformation,
                    nan_filling="mean",
                )
                y_is_centered = True
        else:
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
            y_is_centered = True

        n_spots_eff = n_spots

        if null_method == "liu" and kernel_approx_rank_eff is not None:
            hsic_scaled = torch.trace(K_sp.xtKx_approx(y, k=kernel_approx_rank_eff))
        else:
            hsic_scaled = torch.trace(K_sp.xtKx_exact(y))

    hsic_norm = float(hsic_scaled / (n_spots_eff - 1) ** 2)

    if null_method == "liu":
        feature_cumulants = _feature_cumulants_from_data(y, centered=y_is_centered)
        pval = _hsic_liu_pvalue(
            float(hsic_scaled),
            kernel_cumulants_eff,
            feature_cumulants,
            n_spots_eff,
        )

    elif null_method == "welch":
        feature_cumulants = _feature_cumulants_from_data(y, centered=y_is_centered)
        pval = _hsic_welch_pvalue(
            float(hsic_scaled),
            kernel_cumulants_eff,
            feature_cumulants,
            n_spots_eff,
        )

    else:  # null_method == "perm"
        p_isos = y.shape[1]
        null_stats = []
        for chunk_start in range(0, n_nulls, perm_batch_size):
            B = min(perm_batch_size, n_nulls - chunk_start)
            if scipy.sparse.issparse(y):
                y_batch = scipy.sparse.hstack(
                    [y[np.random.permutation(n_spots_eff), :] for _ in range(B)],
                    format="csc",
                )
            else:
                y_batch = torch.cat(
                    [y[torch.randperm(n_spots_eff)] for _ in range(B)], dim=1
                )
            if isinstance(K_sp_null, torch.Tensor):
                if scipy.sparse.issparse(y_batch):
                    y_batch = torch.from_numpy(y_batch.toarray()).float()
                R = y_batch.T @ K_sp_null @ y_batch
            else:
                R = K_sp_null.xtKx(y_batch)
            null_stats.append(torch.diagonal(R).reshape(B, p_isos).sum(dim=1))
        null_m = torch.cat(null_stats)
        pval = float((1 + (null_m >= hsic_scaled).sum()) / (n_nulls + 1))

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


class SplisosmNP(_ResultsMixin, _FeatureSummaryMixin):
    """Non-parametric SPLISOSM model for arbitrary spatial geometries.

    ``SplisosmNP`` works directly on spot- or cell-level coordinates and a
    sparse CAR spatial kernel. Use :meth:`setup_data` to load an
    :class:`~anndata.AnnData` object, then run spatial-variability (SV) or
    differential-usage (DU) tests.

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
            diagonal entries are 1 (default ``True``).  This can reduce
            leverage from spatial graph outliers, but it slows setup.
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

    def _store_prepared_anndata(
        self,
        adata: AnnData,
        layer: str,
        group_iso_by: str,
        data: list[torch.Tensor],
        coordinates: Optional[torch.Tensor],
        gene_names: list[str],
        filtered_adata: AnnData,
    ) -> None:
        """Store prepared AnnData inputs and reset derived caches."""
        self.adata = adata
        self._setup_input_mode = "anndata"
        self._counts_layer = layer
        self._group_iso_by = group_iso_by
        self._filtered_adata = filtered_adata
        self._gene_summary = None
        self._isoform_summary = None

        self.n_genes = len(data)
        self.n_spots = filtered_adata.shape[0]
        self.n_isos_per_gene = [g.shape[1] for g in data]
        self.gene_names = gene_names
        self._data = [g.float() for g in data]
        self._coordinates = coordinates

    def _build_setup_spatial_kernel(
        self,
        coordinates: Optional[torch.Tensor],
        adj_matrix: Optional[scipy.sparse.spmatrix],
        adj_key: Optional[str],
        spatial_key: str,
        skip_spatial_kernel: bool,
    ) -> None:
        """Build or skip the setup spatial kernel."""
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

    def _store_design_matrix(
        self,
        resolved_design: Optional[Any],
        resolved_cov_names: Optional[list[str]],
    ) -> None:
        """Store a prepared DU design matrix and warn on near-constant factors."""
        if resolved_design is None:
            self.design_mtx = None
            self.n_factors = 0
            self.covariate_names = None
            return

        n_factors = resolved_design.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if scipy.sparse.issparse(resolved_design):
                means = np.asarray(resolved_design.mean(axis=0)).ravel()
                sq_means = np.asarray(resolved_design.power(2).mean(axis=0)).ravel()
                zero_var_indices = np.where(
                    np.sqrt(np.maximum(sq_means - means**2, 0.0)) < 1e-5
                )[0]
            else:
                design_mtx_t = torch.from_numpy(
                    np.asarray(resolved_design, dtype=np.float32)
                )
                if design_mtx_t.dim() == 1:
                    design_mtx_t = design_mtx_t.unsqueeze(1)
                zero_var_indices = torch.where(design_mtx_t.std(dim=0) < 1e-5)[
                    0
                ].numpy()
            for idx in zero_var_indices:
                cov_name = (
                    resolved_cov_names[int(idx)]
                    if resolved_cov_names is not None
                    else str(int(idx))
                )
                warnings.warn(
                    f"Covariate '{cov_name}' has near-zero variance "
                    "(std < 1e-5). Consider removing it.",
                    UserWarning,
                    stacklevel=2,
                )

        self.design_mtx = (
            resolved_design.tocsr()
            if scipy.sparse.issparse(resolved_design)
            else design_mtx_t
        )
        self.n_factors = n_factors
        self.covariate_names = resolved_cov_names

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
        """Set up isoform-level spatial data for hypothesis testing.

        This method extracts isoform count tensors, optionally filters small
        disconnected graph components, builds the spatial kernel, and resolves
        the design matrix for DU tests.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix.  Counts are read from
            ``adata.layers[layer]`` grouped by ``group_iso_by``, and
            spatial coordinates from ``adata.obsm[spatial_key]``.
            See :func:`splisosm.utils.preprocessing.prepare_inputs_from_anndata` for
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
            from coordinates and is used directly to build the spatial kernel.
            It also makes ``spatial_key`` optional (see above). The adjacency
            matrix is symmetrized internally.
        layer : str, optional
            Layer in ``adata.layers`` that stores isoform counts (default
            ``"counts"``).
        group_iso_by : str, optional
            Column in ``adata.var`` used to group isoforms by gene
            (default ``"gene_symbol"``).
        gene_names : str or None, optional
            Column name in ``adata.var`` used as display names for genes.
            If ``None``, the values of ``group_iso_by`` are used.
        design_mtx : tensor, array, sparse matrix, DataFrame, str, or list of str, optional
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
            Minimum spot prevalence required to retain an isoform. Values in
            ``[0, 1]`` are fractions; values in ``(1, 100]`` are percentages.
            Default ``0.0``.
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
            :meth:`test_differential_usage` is needed; ``method="hsic-gp"``
            fits its own GP for spatial residualization.
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

        self._store_prepared_anndata(
            adata=adata,
            layer=layer,
            group_iso_by=group_iso_by,
            data=data,
            coordinates=coordinates,
            gene_names=resolved_gene_names,
            filtered_adata=filtered_adata,
        )
        self._build_setup_spatial_kernel(
            coordinates=coordinates,
            adj_matrix=adj_matrix,
            adj_key=adj_key,
            spatial_key=spatial_key,
            skip_spatial_kernel=skip_spatial_kernel,
        )
        self._store_design_matrix(resolved_design, resolved_cov_names)

    def _get_design_col(self, factor_idx: int) -> torch.Tensor:
        """Extract one covariate column as a dense (n_spots, 1) float32 tensor.

        Works for both torch-tensor and scipy-sparse design matrices so that
        the bulk of the design matrix is never fully densified.
        """
        if scipy.sparse.issparse(self.design_mtx):
            col = np.asarray(self.design_mtx.getcol(factor_idx).todense()).ravel()
            return torch.from_numpy(col.astype(np.float32)).unsqueeze(1)
        return self.design_mtx[:, factor_idx].clone().float().unsqueeze(1)

    def _validate_sv_args(
        self,
        method: str,
        ratio_transformation: str,
        nan_filling: str,
        null_method: str,
    ) -> str:
        """Validate SV arguments and return the normalized null method."""
        if self._skip_kernel_construction:
            raise RuntimeError(
                "setup_data was called with skip_spatial_kernel=True; the spatial "
                "kernel is a placeholder IdentityKernel. Re-run setup_data with "
                "skip_spatial_kernel=False to enable spatial variability testing."
            )

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["mean", "none"]
        null_method = _normalize_hsic_null_method(null_method, allow_perm=True)
        valid_null_methods = ["liu", "welch", "perm"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        if null_method not in valid_null_methods:
            raise ValueError(
                f"Invalid null method. Must be one of {valid_null_methods}."
            )
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                "Invalid ratio transformation. "
                f"Must be one of {valid_transformations}."
            )
        if nan_filling not in valid_nan_filling:
            raise ValueError(
                f"Invalid NaN filling method. Must be one of {valid_nan_filling}."
            )
        if nan_filling == "none" and method == "hsic-ir":
            warnings.warn(
                "nan_filling='none' with method='hsic-ir' uses a per-gene "
                "masked spatial kernel. This avoids dense spatial-kernel "
                "materialization but is slower than nan_filling='mean' because "
                "kernel cumulants are recomputed for each valid-spot subset.",
                UserWarning,
                stacklevel=2,
            )
        return null_method

    def _run_sparkx_sv(self) -> dict[str, Any]:
        """Run the SPARK-X gene-count SV path."""
        if self._coordinates is None:
            raise ValueError(
                "method='spark-x' requires raw spatial coordinates, but "
                "setup_data was called on an AnnData without "
                "`obsm[spatial_key]`. Re-run setup_data with spatial "
                "coordinates, or choose the kernel-based 'hsic-gc'."
            )
        counts_g = torch.concat(
            [_counts.sum(1, keepdim=True) for _counts in self._data], axis=1
        )
        return run_sparkx(counts_g.numpy(), self._coordinates.numpy())

    def _run_hsic_sv(
        self,
        method: str,
        ratio_transformation: str,
        nan_filling: str,
        null_method: str,
        null_configs: Optional[dict[str, Any]],
        chunk_size: int | Literal["auto"],
        n_jobs: int,
        print_progress: bool,
    ) -> dict[str, Any]:
        """Run chunked HSIC-based NP spatial variability tests."""
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        column_cap = resolve_chunk_size(
            chunk_size,
            n_observations=self.n_spots,
            backend="np",
            n_jobs=n_jobs,
            dtype_bytes=8,
        )

        n_spots = self.n_spots
        configs = null_configs or {}
        kernel_cumulants = None
        kernel_approx_rank = None
        n_nulls = 0
        perm_batch_size = 1
        if null_method in ("liu", "welch"):
            kernel_cumulants, kernel_approx_rank = _kernel_cumulants_for_null(
                self.sp_kernel,
                null_method=null_method,
                n_spots=n_spots,
                null_configs=configs,
                dense_threshold=getattr(SpatialCovKernel, "DENSE_THRESHOLD", 5000),
            )
        elif null_method == "perm":
            n_nulls = int(configs.get("n_perms_per_gene", 1000))
            perm_batch_size = int(configs.get("perm_batch_size", 50))

        widths = [
            _response_width_np(counts, method, ratio_transformation)
            for counts in self._data
        ]
        gene_chunks = pack_gene_chunks(widths, column_cap)
        n_probes = int(configs.get("n_probes", 60))
        rng_seed = int(configs.get("rng_seed", 0))

        chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_sv_chunk_worker_np)(
                [self._data[i] for i in chunk],
                method,
                ratio_transformation,
                nan_filling,
                null_method,
                n_spots,
                self.sp_kernel,
                (kernel_cumulants if null_method in ("liu", "welch") else None),
                kernel_approx_rank if null_method == "liu" else None,
                n_nulls if null_method == "perm" else 0,
                perm_batch_size if null_method == "perm" else 1,
                n_probes,
                rng_seed,
            )
            for chunk in tqdm(
                gene_chunks,
                desc=f"SV [{method}]",
                total=len(gene_chunks),
                disable=not print_progress,
            )
        )
        sv_results = [res for chunk in chunk_results for res in chunk]
        pvals = np.array([r[1] for r in sv_results], dtype=float)
        return {
            "statistic": np.array([r[0] for r in sv_results], dtype=float),
            "pvalue": pvals,
            "method": method,
            "null_method": null_method,
            "chunk_size": column_cap,
            "pvalue_adj": false_discovery_control(pvals),
        }

    def _validate_du_args(
        self,
        method: str,
        ratio_transformation: str,
        nan_filling: str,
        residualize: str,
    ) -> tuple[int, int]:
        """Validate DU arguments and return design matrix shape."""
        if self.design_mtx is None:
            raise ValueError(
                "Cannot find the design matrix. Perhaps you forgot to set it up using setup_data()."
            )

        n_spots, n_factors = self.design_mtx.shape
        valid_methods = ["hsic", "hsic-gp", "t-fisher", "t-tippett"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["none", "mean"]
        valid_residualize = ["cov_only", "both"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        if ratio_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid transformation. Must be one of {valid_transformations}."
            )
        if nan_filling not in valid_nan_filling:
            raise ValueError(
                f"Invalid nan_filling. Must be one of {valid_nan_filling}."
            )
        if residualize not in valid_residualize:
            raise ValueError(
                f"Invalid residualize. Must be one of {valid_residualize}."
            )
        return n_spots, n_factors

    def _prepare_hsic_covariates(self, n_factors: int) -> list[Any]:
        """Prepare covariate columns for unconditional HSIC without densifying sparse input."""
        z_list = []
        for factor_idx in range(n_factors):
            if scipy.sparse.issparse(self.design_mtx):
                z = self.design_mtx.getcol(factor_idx)
                mean = float(z.mean())
                sq_mean = float(z.multiply(z).mean())
                std = float(np.sqrt(max(sq_mean - mean**2, 0.0)))
            else:
                z = self._get_design_col(factor_idx)
                std = float(z.std())
            if std <= 1e-5:
                raise ValueError(
                    "The factor of interest "
                    f"{self.covariate_names[factor_idx]} has zero variance."
                )
            z_list.append(z)
        return z_list

    def _resolve_gpr_configs(
        self, gpr_configs: Optional[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge DU GPR user overrides onto backend defaults."""
        cov_config = {**_DEFAULT_GPR_CONFIGS["covariate"]}
        iso_config = {**_DEFAULT_GPR_CONFIGS["isoform"]}
        if gpr_configs is None:
            return cov_config, iso_config
        if gpr_configs.keys() - {"covariate", "isoform"}:
            raise ValueError(
                "gpr_configs must have a nested structure. Use keys "
                "'covariate' and/or 'isoform' for the respective GPR configurations."
            )
        if "covariate" in gpr_configs:
            cov_config.update(gpr_configs["covariate"])
        if "isoform" in gpr_configs:
            iso_config.update(gpr_configs["isoform"])
        return cov_config, iso_config

    def _normalized_coordinates(self) -> torch.Tensor:
        """Return z-scored spatial coordinates for GP residualization."""
        if self._coordinates is None:
            raise ValueError(
                "method='hsic-gp' fits a Gaussian process on raw spatial "
                "coordinates, but setup_data was called on an AnnData "
                "without `obsm[spatial_key]`. Re-run setup_data with "
                "spatial coordinates, or use an unconditional DU method "
                "(e.g. method='hsic', 't-fisher', 't-tippett')."
            )
        x = torch.as_tensor(self._coordinates, dtype=torch.float64).clone()
        return (x - x.mean(0)) / x.std(0).clamp(min=1e-8)

    def _fit_covariate_gp_residuals(
        self,
        x: torch.Tensor,
        n_factors: int,
        gpr_backend: str,
        cov_config: dict[str, Any],
        print_progress: bool,
    ) -> list[torch.Tensor]:
        """Fit one spatial GP per covariate and return standardized residuals."""
        gpr_cov = make_kernel_gpr(gpr_backend, **cov_config)
        z_res_list = []
        for factor_idx in tqdm(
            range(n_factors),
            desc="Covariates",
            total=n_factors,
            disable=not print_progress,
        ):
            z = self._get_design_col(factor_idx).squeeze(1)
            if z.std() <= 0:
                raise ValueError(
                    "The factor of interest "
                    f"{self.covariate_names[factor_idx]} has zero variance."
                )
            z = (z - z.mean()) / z.std()
            z_res_list.append(gpr_cov.fit_residuals(x, z.unsqueeze(1)))
        return z_res_list

    def _empty_du_arrays(
        self, n_factors: int, *, torch_output: bool
    ) -> tuple[Any, Any]:
        """Allocate DU result arrays with the expected shape."""
        if torch_output:
            return torch.empty(self.n_genes, n_factors), torch.empty(
                self.n_genes, n_factors
            )
        return np.empty((self.n_genes, n_factors)), np.empty((self.n_genes, n_factors))

    def _run_du_hsic(
        self,
        n_factors: int,
        ratio_transformation: str,
        nan_filling: str,
        n_jobs: int,
        print_progress: bool,
    ) -> dict[str, Any]:
        """Run unconditional HSIC DU tests."""
        z_list = self._prepare_hsic_covariates(n_factors)
        hsic_all, pvals_all = self._empty_du_arrays(n_factors, torch_output=True)
        du_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_du_hsic_gene_worker_np)(
                counts, z_list, ratio_transformation, nan_filling
            )
            for counts in tqdm(
                self._data,
                desc="DU [hsic]",
                total=self.n_genes,
                disable=not print_progress,
            )
        )
        for gene_idx, (h_row, p_row) in enumerate(du_results):
            hsic_all[gene_idx] = h_row
            pvals_all[gene_idx] = p_row
        return {
            "statistic": hsic_all.numpy(),
            "pvalue": pvals_all.numpy(),
            "method": "hsic",
        }

    def _run_du_hsic_gp(
        self,
        n_factors: int,
        ratio_transformation: str,
        nan_filling: str,
        gpr_backend: str,
        gpr_configs: Optional[dict[str, Any]],
        residualize: str,
        n_jobs: int,
        print_progress: bool,
    ) -> dict[str, Any]:
        """Run conditional HSIC DU tests with spatial GP residualization."""
        cov_config, iso_config = self._resolve_gpr_configs(gpr_configs)
        x = self._normalized_coordinates()
        z_res_list = self._fit_covariate_gp_residuals(
            x, n_factors, gpr_backend, cov_config, print_progress
        )

        gpr_iso = None
        if residualize == "both":
            gpr_iso = make_kernel_gpr(gpr_backend, **iso_config)
            if (
                hasattr(gpr_iso, "precompute_shared_kernel")
                and gpr_iso.signal_bounds_fixed
            ):
                gpr_iso.precompute_shared_kernel(x)

        effective_n_jobs = (
            1
            if gpr_backend == "gpytorch" and iso_config.get("device", "cpu") != "cpu"
            else n_jobs
        )
        hsic_all, pvals_all = self._empty_du_arrays(n_factors, torch_output=True)
        du_results = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
            delayed(_du_hsic_gp_gene_worker_np)(
                counts,
                gpr_iso if residualize == "both" else None,
                x,
                z_res_list,
                ratio_transformation,
                nan_filling,
                residualize,
            )
            for counts in tqdm(
                self._data,
                desc="DU [hsic-gp]",
                total=self.n_genes,
                disable=not print_progress,
            )
        )
        for gene_idx, (h_row, p_row) in enumerate(du_results):
            hsic_all[gene_idx] = h_row
            pvals_all[gene_idx] = p_row
        return {
            "statistic": hsic_all.numpy(),
            "pvalue": pvals_all.numpy(),
            "method": "hsic-gp",
        }

    def _run_du_ttest(
        self,
        method: str,
        n_factors: int,
        ratio_transformation: str,
        nan_filling: str,
        n_jobs: int,
        print_progress: bool,
    ) -> dict[str, Any]:
        """Run two-sample t-test based DU tests."""
        combine_method = re.findall(r"^t-(.+)", method)[0]
        stats_all, pvals_all = self._empty_du_arrays(n_factors, torch_output=False)
        design_is_sparse = scipy.sparse.issparse(self.design_mtx)
        groups_list = [
            (
                self.design_mtx.getcol(factor_idx)
                if design_is_sparse
                else self.design_mtx[:, factor_idx]
            )
            for factor_idx in range(n_factors)
        ]
        du_results = Parallel(n_jobs=n_jobs, prefer="threads")(
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
        for gene_idx, (s_row, p_row) in enumerate(du_results):
            stats_all[gene_idx] = s_row
            pvals_all[gene_idx] = p_row
        return {
            "statistic": stats_all,
            "pvalue": pvals_all,
            "method": method,
        }

    def test_spatial_variability(
        self,
        method: Literal["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"] = "hsic-ir",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        null_method: Literal["liu", "welch", "perm", "eig", "clt", "trace"] = "liu",
        null_configs: Optional[dict[str, Any]] = None,
        chunk_size: int | Literal["auto"] = "auto",
        n_jobs: int = -1,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test each gene for spatial variability.

        The HSIC-based methods test association between a centered spatial
        kernel and one gene-level response at a time:

        - ``"hsic-ir"``: isoform usage ratios.
        - ``"hsic-ic"``: isoform counts.
        - ``"hsic-gc"``: gene-level total counts.
        - ``"spark-x"``: SPARK-X gene-count test :cite:`zhu2021spark`.

        Parameters
        ----------
        method : {"hsic-ir", "hsic-ic", "hsic-gc", "spark-x"}, optional
            Test target: ``"hsic-ir"`` (isoform usage ratios), ``"hsic-ic"``
            (isoform counts), ``"hsic-gc"`` (gene-level counts), or
            ``"spark-x"`` (SPARK-X :cite:`zhu2021spark`).
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Compositional transformation applied to isoform ratios when
            ``method="hsic-ir"``.  See :func:`splisosm.utils.preprocessing.counts_to_ratios`
            and :cite:`park2022kernel` for details.
        nan_filling : {"mean", "none"}, optional
            Strategy for NaN values in isoform ratios.
            See :func:`splisosm.utils.preprocessing.counts_to_ratios` for details.
        null_method : {"liu", "welch", "perm"}, optional
            Method for computing the null distribution of the test statistic:

            * ``"liu"`` (default): asymptotic chi-square mixture using
              cumulants with Liu's method :cite:`liu2009new`. Exact
              eigenvalue cumulants are used when cheap; large implicit
              kernels use Hutchinson Rademacher trace estimates controlled
              by ``null_configs["n_probes"]``.
            * ``"welch"``: Welch-Satterthwaite moment matching. It uses the
              first two null cumulants and approximates the null by a scaled
              chi-square variable ``g * chi2(h)``, where
              ``g = Var / (2 * E)`` and ``h = 2 * E**2 / Var``.
            * ``"perm"``: permutation null distribution. Supports
              optional ``null_configs["n_perms_per_gene"]`` (default 1000),
              and ``null_configs["perm_batch_size"]`` (default 50) for
              batched null-statistic computation.

            ``"eig"`` is accepted as a deprecated alias for ``"liu"``.
            ``"clt"`` and ``"trace"`` are accepted as deprecated aliases for
            ``"welch"``.
        null_configs : dict or None, optional
            Extra keyword arguments for the chosen ``null_method``.
            For ``null_method="liu"``, SPLISOSM evaluates Liu's
            approximation from cumulants instead of materializing all pairwise
            eigenvalue products. Supported keys include:

            * ``"n_probes"``: int. Estimate spatial kernel cumulants with this
              many Hutchinson Rademacher probes when exact eigenvalue or trace
              cumulants are unavailable. The same budget is used by ``"welch"``
              when the spatial kernel does not expose exact ``tr(K)`` and
              ``tr(K**2)`` (for example implicit CAR kernels). Large implicit
              kernels use 60 probes by default.
            * ``"approx_rank"``: int or None. Advanced diagnostic override to
              use the top-k spatial eigenvalues and a rank-consistent
              statistic. This is normally unnecessary for SV tests because the
              default Liu path uses direct cumulant estimates.
        chunk_size : int or {"auto"}, optional
            Maximum number of response columns to process per worker task.
            ``"auto"`` (default) estimates a memory-safe column cap using a
            2 GiB live-memory budget per worker and then caps the result at
            32 columns for per-feature runtime.  Genes are never split across
            chunks; a single gene with more response columns than the cap is
            processed as a singleton chunk.  For ``method="hsic-gc"``, each
            gene contributes one response column.
        n_jobs : int, optional
            Number of parallel workers for the per-gene loop.  ``-1`` uses all
            available CPUs.  With ``chunk_size="auto"``, each worker targets
            roughly a 2 GiB live-memory budget.  Default ``-1``.
        return_results : bool, optional
            If ``True``, return the result dict.  Otherwise store results in
            ``self._sv_test_results`` and return ``None``.
        print_progress : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        dict or None
            If ``return_results`` is ``True``, return a result dictionary with
            test statistics and p-values. Otherwise, store results in
            ``self._sv_test_results`` and return ``None``.

        Notes
        -----
        ``method="spark-x"`` requires the R package ``SPARK`` and Python
        package ``rpy2``.
        """

        null_method = self._validate_sv_args(
            method, ratio_transformation, nan_filling, null_method
        )
        if method == "spark-x":
            self._sv_test_results = self._run_sparkx_sv()
        else:
            self._sv_test_results = self._run_hsic_sv(
                method=method,
                ratio_transformation=ratio_transformation,
                nan_filling=nan_filling,
                null_method=null_method,
                null_configs=null_configs,
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
        nan_filling: Literal["mean", "none"] = "mean",
        gpr_backend: Literal["sklearn", "gpytorch", "nufft", "finufft"] = "sklearn",
        gpr_configs: Optional[dict[str, Any]] = None,
        residualize: Literal["cov_only", "both"] = "cov_only",
        n_jobs: int = -1,
        print_progress: bool = True,
        return_results: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Test each gene for differential isoform usage.

        Call :meth:`setup_data` with ``design_mtx`` before running this
        method. Each design-matrix column is tested against each gene's
        isoform usage ratios, producing one statistic and p-value per
        ``(gene, covariate)`` pair.

        Two types of association tests are supported:

        - Unconditional (``"hsic"``, ``"t-fisher"``, ``"t-tippett"``): test the
          association between isoform usage ratios and the covariate.
        - Conditional (``"hsic-gp"``): test the association conditioned on spatial
          coordinates by residualizing with spatial GP regression.
          See :cite:`zhang2012kernel`.

        Parameters
        ----------
        method : {"hsic", "hsic-gp", "t-fisher", "t-tippett"}, optional
            Method for association testing:

            * ``"hsic"``: Unconditional HSIC test (multivariate RV coefficient).
              For continuous factors, equivalent to the multivariate Pearson correlation
              test.  For binary factors, equivalent to the two-sample Hotelling T**2 test.
            * ``"hsic-gp"``: Conditional HSIC test.  Spatial effects are removed via
              Gaussian process regression before computing the HSIC statistic.

            * ``"t-fisher"``, ``"t-tippett"``: each isoform is tested independently
              for binary covariates only; p-values are combined gene-wise via
              Fisher's or Tippett's method.
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Compositional transformation for isoform ratios.
            One of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, ``'radial'``
            :cite:`park2022kernel`.  See :func:`splisosm.utils.preprocessing.counts_to_ratios`.
        nan_filling : {"mean", "none"}, optional
            How to fill NaN values in isoform ratios.  One of ``'mean'`` or ``'none'``.
            See :func:`splisosm.utils.preprocessing.counts_to_ratios`.
        gpr_backend : {"sklearn", "gpytorch", "nufft", "finufft"}, optional
            GPR backend to use for ``method='hsic-gp'``.
            One of ``'sklearn'`` (default), ``'gpytorch'``, ``'nufft'``, or
            ``'finufft'``.  The NUFFT aliases use FINUFFT for irregular 2-D
            coordinates with an implicit periodic RBF kernel.
            For FFT-accelerated spatial GP on regular grids use
            :class:`~splisosm.hyptest.fft.SplisosmFFT` instead.
        gpr_configs : dict, optional
            Nested configuration dict for the GPR objects, with optional keys
            ``"covariate"`` and/or ``"isoform"``. Each sub-dict configures the
            selected ``gpr_backend``. Unspecified keys use SPLISOSM defaults.
            The shared defaults include ``constant_value=1.0``,
            ``constant_value_bounds=(1e-3, 1e3)``, ``length_scale=1.0``,
            and ``length_scale_bounds="fixed"``. Backend-irrelevant known keys
            are ignored.

            ``"n_inducing"`` *(int or None)* controls the scale of spatial GP
            fitting for the dense/sparse GP backends:

            * **sklearn** — maximum number of observations used for
              hyperparameter fitting. Full exact GP when
              ``n_spots <= n_inducing``
              (or ``None``); a randomly sub-sampled **subset-of-data** of
              ``n_inducing`` points otherwise (**not** the same inducing-point
              approximation as gpytorch).  Default: ``5000``.  Set to ``None``
              to use all observations (warns when ``n_spots > 10_000``).
            * **gpytorch** — FITC sparse-GP inducing-point approximation with
              ``n_inducing`` points; set to ``None`` for exact GP.
              Default: ``5000``.

            **nufft / finufft** has additional NUFFT-specific options
            for the RBF-GP fitting. See :class:`splisosm.gpr.NUFFTKernelGPR`
            for the full backend signature. Common options include:

            * ``max_auto_modes`` - cap the size of the automatically inferred grid;
            * ``lml_approx_rank`` - approximate the irregular-coordinate GP likelihoods
              using only the leading eigenvalues and eigensummaries.
              Ignored when the input grid is already regular (full spectrum available via FFT).
              It costs ``O(n_spots * lml_approx_rank)`` memory and uses a
              trace/trace(K^2)-corrected tail for the omitted spectrum.  In
              practice, ranks around ``32``-``64`` often beat same-time sklearn
              subset fits by a wide margin; increase the rank when memory
              permits and hyperparameter accuracy is important.

        residualize : {"cov_only", "both"}, optional
            Controls which signals are spatially residualized when
            ``method="hsic-gp"``:

            * ``"cov_only"`` (default): residualize covariates only; test
              ``HSIC(Z_res, Y)``. Fastest; calibration matches ``"both"``
              when covariate GPR captures most spatial confounding.
            * ``"both"``: residualize both covariates and isoform ratios.
        n_jobs : int, optional
            Number of parallel workers for the per-gene loop.  ``-1`` uses all
            available CPUs.  Each worker densifies one sparse count tensor
            (about 4-40 MB for 100K-1M spots by 10 isoforms). When
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
            If ``return_results`` is ``True``, return a dictionary with test
            statistics and p-values. Otherwise, return ``None`` and store results in
            ``self._du_test_results``.
        """
        _, n_factors = self._validate_du_args(
            method, ratio_transformation, nan_filling, residualize
        )
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if method == "hsic":
            self._du_test_results = self._run_du_hsic(
                n_factors=n_factors,
                ratio_transformation=ratio_transformation,
                nan_filling=nan_filling,
                n_jobs=n_jobs,
                print_progress=print_progress,
            )
        elif method == "hsic-gp":
            self._du_test_results = self._run_du_hsic_gp(
                n_factors=n_factors,
                ratio_transformation=ratio_transformation,
                nan_filling=nan_filling,
                gpr_backend=gpr_backend,
                gpr_configs=gpr_configs,
                residualize=residualize,
                n_jobs=n_jobs,
                print_progress=print_progress,
            )
        else:
            self._du_test_results = self._run_du_ttest(
                method=method,
                n_factors=n_factors,
                ratio_transformation=ratio_transformation,
                nan_filling=nan_filling,
                n_jobs=n_jobs,
                print_progress=print_progress,
            )

        self._du_test_results["pvalue_adj"] = false_discovery_control(
            self._du_test_results["pvalue"], axis=0
        )
        if return_results:
            return self._du_test_results
        return None
