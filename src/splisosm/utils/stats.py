"""Statistical helper functions for SPLISOSM tests."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Literal
from scipy.sparse.csgraph import connected_components as _connected_components

import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse
import torch

from splisosm.utils._chunking import (
    _resolve_n_jobs,
    pack_gene_chunks,
    resolve_chunk_size,
)

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "false_discovery_control",
    "run_hsic_gc",
    "run_sparkx",
]


# From scipy v1.13.1
# https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_morestats.py#L4737
def false_discovery_control(
    ps: ArrayLike, *, axis: Optional[int] = 0, method: Literal["bh", "by"] = "bh"
) -> np.ndarray:
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg :cite:`benjamini1995controlling` (Eq. 1), ``'by'`` is for Benjamini-Yekutieli
        :cite:`benjamini2001control` (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adjusted : numpy.ndarray
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    Notes
    -----
    From `scipy.stats.false_discovery_control` in SciPy v1.13.1.
    See https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_morestats.py#L4737.
    """
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    # Handle NaNs
    if np.isnan(ps).any():
        warnings.warn(
            "NaNs encountered in p-values. These will be ignored.",
            UserWarning,
            stacklevel=2,
        )
        # ignore NaNs in the p-values
        ps_in_range = np.issubdtype(ps.dtype, np.number) and np.all(
            ps[~np.isnan(ps)] == np.clip(ps[~np.isnan(ps)], 0, 1)
        )
    else:
        ps_in_range = np.issubdtype(ps.dtype, np.number) and np.all(
            ps == np.clip(ps, 0, 1)
        )

    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {"bh", "by"}
    if method.lower() not in methods:
        raise ValueError(
            f"Unrecognized `method` '{method}'." f"Method must be one of {methods}."
        )
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m + 1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == "by":
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.fmin.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)


def run_sparkx(
    counts_gene: np.ndarray | torch.Tensor,
    coordinates: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """Wrapper for running the SPARK-X test for spatial gene expression variability.

    It runs the R-package SPARK :cite:`zhu2021spark` via rpy2.

    Parameters
    ----------
    counts_gene
        Shape (n_spots, n_genes), the observed gene counts.
    coordinates
        Shape (n_spots, 2), the spatial coordinates.

    Returns
    -------
    dict
        Results of the SPARK-X spatial variability test with keys:

        - ``'statistic'``: np.ndarray of shape (n_genes,). Mean SPARK-X statistics.
        - ``'pvalue'``: np.ndarray of shape (n_genes,). Combined p-values.
        - ``'pvalue_adj'``: np.ndarray of shape (n_genes,). Adjusted combined p-values.
        - ``'method'``: str. Method name "spark-x".
    """
    if scipy.sparse.issparse(counts_gene):
        raise ValueError("run_sparkx does not support sparse input for counts_gene.")
    if isinstance(counts_gene, torch.Tensor) and counts_gene.is_sparse:
        raise ValueError("run_sparkx does not support sparse input for counts_gene.")

    # load packages neccessary for running SPARK-X
    import rpy2.robjects as ro
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    spark = importr("SPARK")

    # prepare robject inputs
    if isinstance(counts_gene, torch.Tensor):
        counts_gene = counts_gene.numpy()
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.numpy()

    with (ro.default_converter + numpy2ri.converter).context():
        coords_r = ro.conversion.get_conversion().py2rpy(coordinates)  # (n_spots, 2)
        counts_r = ro.conversion.get_conversion().py2rpy(
            counts_gene.T
        )  # (n_genes, n_spots)

    counts_r.colnames = ro.vectors.StrVector(r["rownames"](coords_r))

    # run SPARK-X and extract outputs
    sparkx_res = spark.sparkx(counts_r, coords_r)
    with (ro.default_converter + numpy2ri.converter).context():
        sv_sparkx = {
            "statistic": ro.conversion.get_conversion()
            .rpy2py(sparkx_res.rx2("stats"))
            .mean(1),
            "pvalue": ro.conversion.get_conversion().rpy2py(
                sparkx_res.rx2("res_mtest")
            )["combinedPval"],
            "pvalue_adj": ro.conversion.get_conversion().rpy2py(
                sparkx_res.rx2("res_mtest")
            )["adjustedPval"],
            "method": "spark-x",
        }

    return sv_sparkx


def _run_hsic_gc_chunk_worker(
    counts_gene: np.ndarray | scipy.sparse.spmatrix,
    chunk: list[int],
    *,
    is_scipy_sparse: bool,
    K_sp: Any,
    null_method: Literal["liu", "welch"],
    kernel_cumulants: dict[int, float],
    kernel_approx_rank: Optional[int],
    n_spots: int,
) -> list[tuple[int, float, float]]:
    """Compute HSIC-GC statistics and p-values for one gene-column chunk."""
    from splisosm.utils.hsic import (
        _feature_cumulants_from_data,
        _hsic_liu_pvalue,
        _hsic_welch_pvalue,
    )

    cols = np.asarray(chunk, dtype=int)
    if is_scipy_sparse:
        response_block = counts_gene[:, cols]
        q_input = response_block
    else:
        response_block = np.asarray(counts_gene[:, cols], dtype=np.float64)
        q_input = torch.from_numpy(response_block.astype(np.float32, copy=False))

    # When Liu uses an explicit low-rank diagnostic override, keep the observed
    # statistic on the same rank-k scale as the null cumulants.
    if null_method == "liu" and kernel_approx_rank is not None:
        q_mat = K_sp.xtKx_approx(q_input, k=kernel_approx_rank)
    else:
        q_mat = K_sp.xtKx_exact(q_input)

    if isinstance(q_mat, torch.Tensor):
        q_cols = torch.diagonal(q_mat).detach().cpu().numpy().astype(float)
    elif scipy.sparse.issparse(q_mat):
        q_cols = np.asarray(q_mat.diagonal(), dtype=float)
    else:
        q_cols = np.diag(np.asarray(q_mat, dtype=float))

    results: list[tuple[int, float, float]] = []
    for local_idx, gene_idx in enumerate(cols):
        hsic_scaled = float(q_cols[local_idx])
        counts_col = response_block[:, local_idx : local_idx + 1]
        feature_cumulants = _feature_cumulants_from_data(counts_col, centered=False)

        if null_method == "liu":
            pval = _hsic_liu_pvalue(
                hsic_scaled,
                kernel_cumulants,
                feature_cumulants,
                n_spots,
            )
        else:  # "welch"
            pval = _hsic_welch_pvalue(
                hsic_scaled,
                kernel_cumulants,
                feature_cumulants,
                n_spots,
            )

        results.append((int(gene_idx), hsic_scaled / (n_spots - 1) ** 2, pval))
    return results


def _coerce_hsic_gc_adata_counts(
    adata: AnnData,
    layer: Optional[str],
    min_counts: int,
    min_bin_pct: float,
) -> torch.Tensor:
    """Load AnnData gene counts as a torch dense or sparse tensor."""
    raw = adata.X if layer is None else adata.layers[layer]

    if scipy.sparse.issparse(raw):
        coo = raw.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
        )
        values = torch.from_numpy(coo.data)
        with torch.sparse.check_sparse_tensor_invariants(False):
            counts_gene = torch.sparse_coo_tensor(
                indices,
                values,
                torch.Size(coo.shape),
            )
    else:
        counts_gene = torch.as_tensor(np.asarray(raw, dtype=np.float32))

    if min_counts <= 0 and min_bin_pct <= 0:
        return counts_gene

    dense_counts = counts_gene.to_dense() if counts_gene.is_sparse else counts_gene
    gene_totals = dense_counts.sum(0)
    gene_bin_pcts = (dense_counts > 0).float().mean(0)
    gene_mask = (gene_totals >= min_counts) & (gene_bin_pcts >= min_bin_pct)
    return dense_counts[:, gene_mask]


def _resolve_hsic_gc_adata_coordinates(
    adata: AnnData,
    spatial_key: str,
    adj_key: Optional[str],
) -> tuple[torch.Tensor | None, scipy.sparse.spmatrix | None]:
    """Load coordinates and optional prebuilt adjacency from AnnData."""
    if spatial_key in adata.obsm:
        coordinates = torch.as_tensor(
            np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        )
    elif adj_key is not None and adj_key in adata.obsp:
        coordinates = None
    else:
        raise ValueError(
            f"Neither `adata.obsm['{spatial_key}']` nor "
            f"`adata.obsp['{adj_key}']` is available. Provide spatial "
            "coordinates via `spatial_key` and/or a pre-built adjacency "
            "via `adj_key`."
        )

    adj_prebuilt = adata.obsp[adj_key] if adj_key is not None else None
    return coordinates, adj_prebuilt


def _filter_hsic_gc_components(
    counts_gene: Any,
    coordinates: Any,
    adj_prebuilt: scipy.sparse.spmatrix | None,
    *,
    min_component_size: int,
    spatial_kernel_kwargs: dict[str, Any],
) -> tuple[Any, Any, scipy.sparse.spmatrix | None]:
    """Remove small graph components before HSIC-GC kernel construction."""
    if min_component_size <= 1:
        return counts_gene, coordinates, adj_prebuilt

    from splisosm.kernel import _build_adj_from_coords

    if adj_prebuilt is not None:
        adj_for_comp = scipy.sparse.csc_matrix(adj_prebuilt)
    elif coordinates is not None:
        coords_t = (
            coordinates
            if isinstance(coordinates, torch.Tensor)
            else torch.as_tensor(np.asarray(coordinates), dtype=torch.float32)
        )
        adj_for_comp = _build_adj_from_coords(
            coords_t,
            k_neighbors=spatial_kernel_kwargs.get("k_neighbors", 4),
            mutual_neighbors=True,
        ).tocsc()
    else:
        raise ValueError(
            "`min_component_size > 1` without `adj_key` requires spatial "
            "coordinates via `spatial_key`."
        )

    _, labels = _connected_components(adj_for_comp, directed=False)
    comp_sizes = np.bincount(labels)
    keep_mask = comp_sizes[labels] >= min_component_size
    n_removed = int((~keep_mask).sum())
    if n_removed == 0:
        return counts_gene, coordinates, adj_prebuilt

    warnings.warn(
        f"Removed {n_removed} spot(s) belonging to graph components with fewer "
        f"than {min_component_size} member(s). {int(keep_mask.sum())} spot(s) remain.",
        UserWarning,
        stacklevel=3,
    )
    if scipy.sparse.issparse(counts_gene):
        counts_gene = counts_gene[keep_mask]
    elif isinstance(counts_gene, torch.Tensor):
        counts_gene = counts_gene.to_dense() if counts_gene.is_sparse else counts_gene
        counts_gene = counts_gene[keep_mask]
    else:
        counts_gene = np.asarray(counts_gene)[keep_mask]

    if coordinates is not None:
        coordinates = (
            coordinates[keep_mask]
            if isinstance(coordinates, torch.Tensor)
            else np.asarray(coordinates)[keep_mask]
        )

    return counts_gene, coordinates, adj_for_comp[keep_mask][:, keep_mask].tocsc()


def _prepare_hsic_gc_inputs(
    counts_gene: Any,
    coordinates: Any,
    *,
    adata: AnnData | None,
    layer: Optional[str],
    spatial_key: str,
    adj_key: Optional[str],
    min_counts: int,
    min_bin_pct: float,
    min_component_size: int,
    spatial_kernel_kwargs: dict[str, Any],
) -> tuple[Any, Any, scipy.sparse.spmatrix | None]:
    """Resolve matrix/AnnData HSIC-GC inputs into count, coordinate, adjacency triples."""
    if adata is not None:
        if counts_gene is not None or coordinates is not None:
            raise ValueError(
                "When `adata` is provided, `counts_gene` and `coordinates` "
                "must not be provided."
            )
        counts_gene = _coerce_hsic_gc_adata_counts(
            adata,
            layer,
            min_counts,
            min_bin_pct,
        )
        coordinates, adj_prebuilt = _resolve_hsic_gc_adata_coordinates(
            adata,
            spatial_key,
            adj_key,
        )
    elif counts_gene is None or coordinates is None:
        raise ValueError(
            "Either `adata` or both `counts_gene` and `coordinates` must be provided."
        )
    else:
        adj_prebuilt = None

    return _filter_hsic_gc_components(
        counts_gene,
        coordinates,
        adj_prebuilt,
        min_component_size=min_component_size,
        spatial_kernel_kwargs=spatial_kernel_kwargs,
    )


def _build_hsic_gc_kernel(
    coordinates: Any,
    adj_prebuilt: scipy.sparse.spmatrix | None,
    spatial_kernel_kwargs: dict[str, Any],
) -> Any:
    """Construct the centered CAR kernel used by HSIC-GC."""
    from splisosm.kernel import SpatialCovKernel

    kernel_kwargs = {
        "k_neighbors": 4,
        "rho": 0.99,
        "standardize_cov": True,
        "centering": True,
    }
    if "centering" in spatial_kernel_kwargs:
        warnings.warn(
            "The 'centering' argument in spatial_kernel_kwargs will be ignored. "
            "It is always set to True for HSIC-GC.",
            UserWarning,
            stacklevel=3,
        )
    kernel_kwargs.update(
        {k: v for k, v in spatial_kernel_kwargs.items() if k != "centering"}
    )
    return SpatialCovKernel(
        coords=None if adj_prebuilt is not None else coordinates,
        adj_matrix=adj_prebuilt,
        **kernel_kwargs,
    )


def _normalize_hsic_gc_counts(
    counts_gene: Any,
) -> tuple[np.ndarray | scipy.sparse.csc_matrix, bool, int, int]:
    """Normalize HSIC-GC counts to dense NumPy or sparse CSC format."""
    if scipy.sparse.issparse(counts_gene):
        n_spots, n_genes = counts_gene.shape
        counts_csc = (
            counts_gene
            if scipy.sparse.isspmatrix_csc(counts_gene)
            else counts_gene.tocsc()
        )
        return counts_csc, True, n_spots, n_genes

    if isinstance(counts_gene, torch.Tensor) and counts_gene.is_sparse:
        if counts_gene.dtype not in (torch.float32, torch.float64):
            counts_gene = counts_gene.float()
        if not counts_gene.is_coalesced():
            counts_gene = counts_gene.coalesce()
        idx = counts_gene.indices().detach().cpu().numpy()
        vals = counts_gene.values().detach().cpu().numpy()
        counts_csc = scipy.sparse.coo_matrix(
            (vals, (idx[0], idx[1])),
            shape=tuple(counts_gene.shape),
        ).tocsc()
        n_spots, n_genes = counts_csc.shape
        return counts_csc, True, n_spots, n_genes

    if isinstance(counts_gene, torch.Tensor):
        counts_gene = counts_gene.detach().cpu().numpy()
    counts_arr = np.asarray(counts_gene, dtype=np.float64)
    n_spots, n_genes = counts_arr.shape
    return counts_arr, False, n_spots, n_genes


def _compute_hsic_gc_chunked_results(
    counts_gene: np.ndarray | scipy.sparse.csc_matrix,
    *,
    is_scipy_sparse: bool,
    K_sp: Any,
    null_method: Literal["liu", "welch"],
    kernel_cumulants: dict[int, float],
    kernel_approx_rank: Optional[int],
    n_spots: int,
    n_genes: int,
    chunk_size: int | Literal["auto"],
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Execute HSIC-GC statistics in response-column chunks."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    effective_n_jobs = _resolve_n_jobs(n_jobs)
    column_cap = resolve_chunk_size(
        chunk_size,
        n_observations=n_spots,
        backend="np",
        n_jobs=effective_n_jobs,
        dtype_bytes=8,
    )
    gene_chunks = pack_gene_chunks([1] * n_genes, column_cap)
    hsic_arr = np.empty(n_genes, dtype=float)
    pvals_arr = np.empty(n_genes, dtype=float)

    chunk_results = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
        delayed(_run_hsic_gc_chunk_worker)(
            counts_gene,
            chunk,
            is_scipy_sparse=is_scipy_sparse,
            K_sp=K_sp,
            null_method=null_method,
            kernel_cumulants=kernel_cumulants,
            kernel_approx_rank=kernel_approx_rank,
            n_spots=n_spots,
        )
        for chunk in tqdm(gene_chunks, desc="Genes", total=len(gene_chunks))
    )
    for chunk_result in chunk_results:
        for gene_idx, statistic, pvalue in chunk_result:
            hsic_arr[gene_idx] = statistic
            pvals_arr[gene_idx] = pvalue
    return hsic_arr, pvals_arr, column_cap


def run_hsic_gc(
    counts_gene: "np.ndarray | torch.Tensor | None" = None,
    coordinates: "np.ndarray | torch.Tensor | None" = None,
    null_method: Literal["liu", "welch", "eig", "clt", "trace"] = "liu",
    null_configs: Optional[dict[str, Any]] = None,
    chunk_size: int | Literal["auto"] = "auto",
    n_jobs: int = 1,
    min_component_size: int = 1,
    adata: "AnnData | None" = None,
    layer: "str | None" = None,
    spatial_key: str = "spatial",
    adj_key: "str | None" = None,
    min_counts: int = 0,
    min_bin_pct: float = 0.0,
    **spatial_kernel_kwargs: Any,
) -> dict[str, Any]:
    """Compute the HSIC-GC statistic for gene-level counts.

    This standalone helper runs the gene-count SV test used by
    ``method="hsic-gc"`` without constructing a ``SplisosmNP`` model.

    Parameters
    ----------
    counts_gene
        Shape ``(n_spots, n_genes)``. Gene counts.
    coordinates
        Shape ``(n_spots, n_dim)``. Spatial coordinates of spots.
    null_method : {"liu", "welch"}, optional
        Method for computing the null distribution of the test statistic:

        * ``"liu"`` (default): asymptotic chi-square mixture using kernel
          cumulants with Liu's method.  Use ``null_configs["n_probes"]``
          (int) to tune Hutchinson Rademacher probes for implicit CAR
          cumulants.  An advanced ``null_configs["approx_rank"]`` override
          is available for diagnostics, but is not needed for the default SV
          path.
        * ``"welch"``: Welch-Satterthwaite scaled chi-squared approximation
          using the first two HSIC null moments.  Typically more accurate
          in the right tail than the retired normal approximation at the same
          cost.

        ``"eig"`` is accepted as a deprecated alias for ``"liu"``.
        ``"clt"`` and ``"trace"`` are accepted as deprecated aliases for
        ``"welch"``.
    null_configs : dict or None, optional
        Extra keyword arguments for the chosen ``null_method``. Use
        ``"n_probes"`` to control the Hutchinson budget for both ``"liu"``
        cumulants and ``"welch"`` first-two-moment traces when the spatial
        kernel has no exact trace path; implicit CAR kernels default to 60
        probes. ``"n_probes"`` is stochastic trace control, not a low-rank
        approximation.
    chunk_size : int or {"auto"}, optional
        Maximum number of gene-count response columns to process in one
        spatial-kernel application. ``"auto"`` (default) estimates a
        memory-safe cap from a 2 GiB live-memory budget and caps the result
        at 32 columns for per-feature runtime. Genes are never split across
        chunks.
    n_jobs : int, optional
        Number of joblib workers for chunked gene-level HSIC computation.
        ``1`` (default) runs serially.  Use ``-1`` to use all available CPUs.
        Threads are preferred so the sparse spatial kernel and count matrix
        can be shared without large pickled copies.
    min_component_size : int, optional
        Minimum number of spots a connected component must contain to be
        retained.  Spots that belong to components smaller than this
        threshold are removed before the spatial kernel is built. Components
        are detected on the same k-NN graph used for the spatial kernel
        (controlled by ``k_neighbors``), unless ``adj_key`` supplies a graph.
        The default value of ``1`` disables filtering. A ``UserWarning`` is
        issued whenever spots are removed.
    adata : AnnData or None, optional
        If provided, use ``adata.X`` (when ``layer=None``) or
        ``adata.layers[layer]`` as ``counts_gene`` — a ``(n_spots, n_genes)``
        count matrix (dense or sparse) — and ``adata.obsm[spatial_key]`` for
        coordinates.  Mutually exclusive with ``counts_gene`` / ``coordinates``.
    layer : str or None, optional
        Layer key in ``adata.layers``.  When ``None`` (default), ``adata.X``
        is used.  Used only in AnnData mode.
    spatial_key : str, optional
        Key in ``adata.obsm`` for spatial coordinates.  Used only in
        AnnData mode and optional when ``adj_key`` is provided: if the key
        is missing from ``adata.obsm`` the kernel is built from the
        adjacency alone.
    adj_key : str or None, optional
        Key in ``adata.obsp`` for a pre-built adjacency matrix.  When
        provided in AnnData mode, the adjacency is loaded from
        ``adata.obsp[adj_key]`` and used for both component filtering and
        the spatial kernel construction.  In AnnData mode this also makes
        ``spatial_key`` optional.  Ignored in matrix mode.
    min_counts : int, optional
        Minimum total count to retain a gene in AnnData mode.  Default 0.
    min_bin_pct : float, optional
        Minimum fraction of spots expressing a gene (count > 0). Default
        ``0.0``.
    **spatial_kernel_kwargs
        Additional arguments forwarded to :class:`~splisosm.kernel.SpatialCovKernel`.
        For example, ``standardize_cov=True`` (default) standardises the CAR
        covariance diagonal to one, which can reduce graph-outlier leverage
        but slows setup.

    Returns
    -------
    dict
        Results with keys:

        - ``'statistic'``: np.ndarray of shape (n_genes,).
        - ``'pvalue'``: np.ndarray of shape (n_genes,).
        - ``'pvalue_adj'``: np.ndarray of shape (n_genes,).
        - ``'method'``: ``"hsic-gc"``.
        - ``'null_method'``: the value of *null_method*.
        - ``'n_spots'``: number of spots after component filtering.
    """

    from splisosm.utils.hsic import (
        _kernel_cumulants_for_null,
        _normalize_hsic_null_method,
    )

    spatial_kernel_kwargs = dict(spatial_kernel_kwargs)
    counts_gene, coordinates, adj_prebuilt = _prepare_hsic_gc_inputs(
        counts_gene,
        coordinates,
        adata=adata,
        layer=layer,
        spatial_key=spatial_key,
        adj_key=adj_key,
        min_counts=min_counts,
        min_bin_pct=min_bin_pct,
        min_component_size=min_component_size,
        spatial_kernel_kwargs=spatial_kernel_kwargs,
    )
    counts_gene, is_scipy_sparse, n_spots, n_genes = _normalize_hsic_gc_counts(
        counts_gene
    )

    null_method = _normalize_hsic_null_method(null_method, allow_perm=False)
    configs = null_configs or {}
    K_sp = _build_hsic_gc_kernel(coordinates, adj_prebuilt, spatial_kernel_kwargs)

    kernel_cumulants, kernel_approx_rank = _kernel_cumulants_for_null(
        K_sp,
        null_method=null_method,
        n_spots=n_spots,
        null_configs=configs,
        dense_threshold=getattr(type(K_sp), "DENSE_THRESHOLD", 5000),
    )

    hsic_arr, pvals_arr, column_cap = _compute_hsic_gc_chunked_results(
        counts_gene,
        is_scipy_sparse=is_scipy_sparse,
        K_sp=K_sp,
        null_method=null_method,
        kernel_cumulants=kernel_cumulants,
        kernel_approx_rank=kernel_approx_rank,
        n_spots=n_spots,
        n_genes=n_genes,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
    )

    sv_test_results = {
        "statistic": hsic_arr,
        "pvalue": pvals_arr,
        "method": "hsic-gc",
        "null_method": null_method,
        "n_spots": n_spots,
        "chunk_size": column_cap,
    }
    sv_test_results["pvalue_adj"] = false_discovery_control(sv_test_results["pvalue"])
    return sv_test_results
