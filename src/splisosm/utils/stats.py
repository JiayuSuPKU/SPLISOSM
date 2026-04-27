"""Statistical helper functions for SPLISOSM tests."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Literal
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.stats import ncx2

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse
import torch
from anndata import AnnData
from tqdm import tqdm
from splisosm.utils._chunking import (
    _resolve_n_jobs,
    pack_gene_chunks,
    resolve_chunk_size,
)
from splisosm.gpr import linear_hsic_test
from splisosm.utils._hsic_null import (
    _feature_cumulants_from_data,
    _hsic_liu_pvalue,
    _hsic_welch_pvalue,
    _kernel_cumulants_for_null,
    _normalize_hsic_null_method,
)

__all__ = [
    "false_discovery_control",
    "run_hsic_gc",
    "run_sparkx",
    "linear_hsic_test",
    "liu_sf",
    "liu_sf_from_cumulants",
]


_DELTA = 1e-10


def liu_sf(
    t: ArrayLike,
    lambs: ArrayLike,
    dofs: Optional[ArrayLike] = None,
    deltas: Optional[ArrayLike] = None,
    kurtosis: bool = False,
) -> np.ndarray:
    """Compute p-values for weighted sums of chi-squared variables using the Liu approximation.

    Let :math:`X = \\sum_{i=1}^{d} \\lambda_i * \\chi^2(h_i, \\delta_i)` be a linear combination of
    `d` chi-squared random variables, each with degree of freedom :math:`h_i` and noncentrality
    :math:`\\delta_i`, this function approximates :math:``Pr(X > t)`` using the
    Liu moment-matching approach :cite:`liu2009new`.

    Parameters
    ----------
    t
        Shape (n,), observed test statistic.
    lambs
        Shape (d,), weights of each chi-squared component.
    dofs
        Shape (d,), degrees of freedom for each component. Defaults to all ones.
    deltas
        Shape (d,), noncentrality parameters for each component. Defaults to all zeros.
    kurtosis
        If True, uses kurtosis matching proposed in :cite:`lee2012optimal`; otherwise uses the original
        skewness matching :cite:`liu2009new`.
    Returns
    -------
    numpy.ndarray
        P-values of shape (n,) computed as ``Pr(X > t)``.

    Notes
    -----
    From https://github.com/limix/chiscore/blob/master/chiscore/_liu.py
    """
    coeffs = _liu_prepare(
        lambs,
        dofs=dofs,
        deltas=deltas,
        kurtosis=kurtosis,
    )
    return _liu_apply(t, coeffs)


def liu_sf_from_cumulants(
    t: ArrayLike,
    cumulants: dict[int, float],
    kurtosis: bool = False,
) -> np.ndarray:
    """Compute Liu p-values from spectral cumulants.

    Parameters
    ----------
    t
        Observed test statistic value(s).
    cumulants
        Mapping ``{1: c1, 2: c2, 3: c3, 4: c4}``, where
        ``cp = trace(K**p)`` for the chi-squared mixture weights.
    kurtosis
        If True, uses the kurtosis-matching edge-case branch.
    Returns
    -------
    numpy.ndarray
        P-values computed as ``Pr(X > t)``.
    """
    return _liu_apply(
        t,
        _liu_prepare_from_cumulants(
            cumulants,
            kurtosis=kurtosis,
        ),
    )


def _liu_prepare(
    lambs: ArrayLike,
    dofs: Optional[ArrayLike] = None,
    deltas: Optional[ArrayLike] = None,
    kurtosis: bool = False,
) -> dict[str, float]:
    """Precompute Liu shifted-chi-squared coefficients from eigenvalues."""
    if dofs is None:
        dofs = np.ones_like(lambs)
    if deltas is None:
        deltas = np.zeros_like(lambs)

    lambs = np.asarray(lambs, float)
    dofs = np.asarray(dofs, float)
    deltas = np.asarray(deltas, float)

    lambs = {i: lambs**i for i in range(1, 5)}

    c = {
        i: float(np.sum(lambs[i] * dofs) + i * np.sum(lambs[i] * deltas))
        for i in range(1, 5)
    }
    return _liu_prepare_from_cumulants(c, kurtosis=kurtosis)


def _liu_prepare_from_cumulants(
    cumulants: dict[int, float],
    kurtosis: bool = False,
) -> dict[str, float]:
    """Precompute Liu shifted-chi-squared coefficients from cumulants."""
    c = {i: float(cumulants.get(i, 0.0)) for i in range(1, 5)}

    if c[2] <= _DELTA or not np.isfinite(c[2]):
        return {
            "mu_q": float(c[1]),
            "sigma_q": 0.0,
            "mu_x": 1.0,
            "sigma_x": np.sqrt(2.0),
            "dof_x": 1.0,
            "delta_x": 0.0,
        }

    s1 = c[3] / (np.sqrt(c[2]) ** 3 + _DELTA)
    s2 = c[4] / (c[2] ** 2 + _DELTA)

    s12 = s1**2
    if s12 > s2:
        denom = s1 - np.sqrt(s12 - s2)
        if abs(denom) < _DELTA:
            delta_x = 0.0
            dof_x = 1 / (s2 + _DELTA)
        else:
            a = 1 / denom
            delta_x = s1 * a**3 - a**2
            dof_x = a**2 - 2 * delta_x
    else:
        delta_x = 0
        if kurtosis:
            a = 1 / np.sqrt(s2)
            dof_x = 1 / s2
        else:
            a = 1 / (s1 + _DELTA)
            dof_x = 1 / (s12 + _DELTA)

    dof_x = max(float(dof_x), _DELTA)
    delta_x = max(float(delta_x), 0.0)

    var_q = 2.0 * c[2]

    mu_x = dof_x + delta_x
    sigma_x = np.sqrt(2 * (dof_x + 2 * delta_x))

    return {
        "mu_q": float(c[1]),
        "sigma_q": float(np.sqrt(max(var_q, 0.0))),
        "mu_x": float(mu_x),
        "sigma_x": float(sigma_x),
        "dof_x": float(dof_x),
        "delta_x": float(delta_x),
    }


def _liu_apply(t: ArrayLike, coeffs: dict[str, float]) -> np.ndarray:
    """Apply cached Liu coefficients to one or more statistics."""
    t = np.asarray(t, float)
    if coeffs["sigma_q"] <= _DELTA:
        return np.where(t > coeffs["mu_q"] + _DELTA, 0.0, 1.0)
    t_star = (t - coeffs["mu_q"]) / (coeffs["sigma_q"] + _DELTA)
    tfinal = t_star * coeffs["sigma_x"] + coeffs["mu_x"]
    return ncx2.sf(tfinal, coeffs["dof_x"], max(coeffs["delta_x"], 1e-9))


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

    # ── AnnData mode: load gene-level counts directly ────────────────────────
    _adj_prebuilt: "scipy.sparse.spmatrix | None" = None
    if adata is not None:
        if counts_gene is not None or coordinates is not None:
            raise ValueError(
                "When `adata` is provided, `counts_gene` and `coordinates` "
                "must not be provided."
            )

        # Load count matrix: adata.X or adata.layers[layer]
        _raw = adata.X if layer is None else adata.layers[layer]

        # Convert to torch (keep sparse as COO for now)
        if scipy.sparse.issparse(_raw):
            _coo = _raw.tocoo().astype(np.float32)
            _i = torch.from_numpy(
                np.vstack((_coo.row, _coo.col)).astype(np.int64, copy=False)
            )
            _v = torch.from_numpy(_coo.data)
            with torch.sparse.check_sparse_tensor_invariants(False):
                counts_gene = torch.sparse_coo_tensor(_i, _v, torch.Size(_coo.shape))
        else:
            counts_gene = torch.as_tensor(np.asarray(_raw, dtype=np.float32))

        # Apply gene-level filters (min_counts, min_bin_pct)
        if min_counts > 0 or min_bin_pct > 0:
            _dense = counts_gene.to_dense() if counts_gene.is_sparse else counts_gene
            _gene_totals = _dense.sum(0)  # (n_genes,)
            _gene_bin_pcts = (_dense > 0).float().mean(0)  # (n_genes,)
            _gene_mask = (_gene_totals >= min_counts) & (_gene_bin_pcts >= min_bin_pct)
            counts_gene = (
                counts_gene.to_dense()[:, _gene_mask]
                if counts_gene.is_sparse
                else counts_gene[:, _gene_mask]
            )

        # Load spatial coordinates (optional when a pre-built adjacency is
        # provided via `adj_key`: the kernel is built from the adjacency alone
        # and HSIC-GC does not need raw coordinates).
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

        # Load pre-built adjacency if provided
        if adj_key is not None:
            _adj_prebuilt = adata.obsp[adj_key]

        # Component filtering (same logic as matrix mode below)
        if min_component_size > 1:
            from splisosm.kernel import _build_adj_from_coords

            _k_adata = spatial_kernel_kwargs.get("k_neighbors", 4)
            if _adj_prebuilt is not None:
                _adj_for_comp = scipy.sparse.csc_matrix(_adj_prebuilt)
            elif coordinates is not None:
                _adj_for_comp = _build_adj_from_coords(
                    coordinates, k_neighbors=_k_adata, mutual_neighbors=True
                ).tocsc()
            else:
                raise ValueError(
                    "`min_component_size > 1` without `adj_key` requires "
                    "spatial coordinates via `spatial_key`."
                )
            _, _labels = _connected_components(_adj_for_comp, directed=False)
            _comp_sizes = np.bincount(_labels)
            _keep_mask = _comp_sizes[_labels] >= min_component_size
            _n_removed = int((~_keep_mask).sum())
            if _n_removed > 0:
                warnings.warn(
                    f"Removed {_n_removed} spot(s) belonging to graph components "
                    f"with fewer than {min_component_size} member(s). "
                    f"{int(_keep_mask.sum())} spot(s) remain.",
                    UserWarning,
                    stacklevel=2,
                )
                _dense_cg = (
                    counts_gene.to_dense() if counts_gene.is_sparse else counts_gene
                )
                counts_gene = _dense_cg[_keep_mask]
                if coordinates is not None:
                    coordinates = coordinates[_keep_mask]
                if _adj_prebuilt is not None:
                    _adj_prebuilt = scipy.sparse.csc_matrix(_adj_prebuilt)[_keep_mask][
                        :, _keep_mask
                    ]
                else:
                    _adj_prebuilt = _adj_for_comp[_keep_mask][:, _keep_mask]

        # Filtering already done above — skip component filtering below
        min_component_size = 1
    elif counts_gene is None or coordinates is None:
        raise ValueError(
            "Either `adata` or both `counts_gene` and `coordinates` must be provided."
        )
    else:
        # Matrix mode: apply component filtering if requested
        if min_component_size > 1:
            from splisosm.kernel import _build_adj_from_coords

            _k_mat = spatial_kernel_kwargs.get("k_neighbors", 4)
            _coords_t = (
                coordinates
                if isinstance(coordinates, torch.Tensor)
                else torch.as_tensor(np.asarray(coordinates), dtype=torch.float32)
            )
            _adj_mat = _build_adj_from_coords(
                _coords_t, k_neighbors=_k_mat, mutual_neighbors=True
            ).tocsc()
            _, _labels = _connected_components(_adj_mat, directed=False)
            _comp_sizes = np.bincount(_labels)
            _keep_mask = _comp_sizes[_labels] >= min_component_size
            _n_removed = int((~_keep_mask).sum())
            if _n_removed > 0:
                _n_remaining = int(_keep_mask.sum())
                warnings.warn(
                    f"Removed {_n_removed} spot(s) belonging to graph components "
                    f"with fewer than {min_component_size} member(s). "
                    f"{_n_remaining} spot(s) remain.",
                    UserWarning,
                    stacklevel=2,
                )
                if scipy.sparse.issparse(counts_gene):
                    counts_gene = counts_gene[_keep_mask]
                elif isinstance(counts_gene, torch.Tensor):
                    counts_gene = counts_gene[_keep_mask]
                else:
                    counts_gene = np.asarray(counts_gene)[_keep_mask]
                coordinates = (
                    coordinates[_keep_mask]
                    if isinstance(coordinates, torch.Tensor)
                    else np.asarray(coordinates)[_keep_mask]
                )
                _adj_prebuilt = _adj_mat[_keep_mask][:, _keep_mask].tocsc()

    from splisosm.kernel import (
        SpatialCovKernel,
    )

    n_spots = counts_gene.shape[0]
    n_genes = counts_gene.shape[1]
    configs = null_configs or {}

    null_method = _normalize_hsic_null_method(null_method, allow_perm=False)

    # Set default spatial kernel kwargs
    default_spatial_kernel_kwargs = {
        "k_neighbors": 4,
        "rho": 0.99,
        "standardize_cov": True,
        "centering": True,
    }
    if spatial_kernel_kwargs is not None:
        if "centering" in spatial_kernel_kwargs:
            warnings.warn(
                "The 'centering' argument in spatial_kernel_kwargs will be ignored. "
                "It is always set to True for HSIC-GC.",
                UserWarning,
                stacklevel=2,
            )
            spatial_kernel_kwargs.pop("centering")
        default_spatial_kernel_kwargs.update(spatial_kernel_kwargs)

    # If a pre-built adjacency is provided, we skip the k-NN graph construction
    if _adj_prebuilt is not None:
        K_sp = SpatialCovKernel(
            coords=None, adj_matrix=_adj_prebuilt, **default_spatial_kernel_kwargs
        )
    else:
        # Otherwise, build the spatial kernel from coordinates
        K_sp = SpatialCovKernel(
            coords=coordinates, adj_matrix=None, **default_spatial_kernel_kwargs
        )

    # Pre-compute null distribution inputs (once, before the gene loop)
    if null_method in ("liu", "welch"):
        kernel_cumulants, kernel_approx_rank = _kernel_cumulants_for_null(
            K_sp,
            null_method=null_method,
            n_spots=n_spots,
            null_configs=configs,
            dense_threshold=getattr(SpatialCovKernel, "DENSE_THRESHOLD", 5000),
        )
    else:
        raise ValueError(f"null_method must be 'liu' or 'welch', got {null_method!r}")

    # compute the HSIC-GC statistic in response-column chunks
    is_scipy_sparse = scipy.sparse.issparse(counts_gene)
    is_torch_sparse = isinstance(counts_gene, torch.Tensor) and counts_gene.is_sparse

    if is_scipy_sparse:
        n_spots, n_genes = counts_gene.shape
        if not scipy.sparse.isspmatrix_csc(counts_gene):
            counts_gene = counts_gene.tocsc()
    elif is_torch_sparse:
        n_spots, n_genes = counts_gene.shape
        if counts_gene.dtype != torch.float32 and counts_gene.dtype != torch.float64:
            counts_gene = counts_gene.float()
        if not counts_gene.is_coalesced():
            counts_gene = counts_gene.coalesce()
        idx = counts_gene.indices().detach().cpu().numpy()
        vals = counts_gene.values().detach().cpu().numpy()
        counts_gene = scipy.sparse.coo_matrix(
            (vals, (idx[0], idx[1])),
            shape=tuple(counts_gene.shape),
        )
        counts_gene = counts_gene.tocsc()
        is_scipy_sparse = True
    else:
        if isinstance(counts_gene, torch.Tensor):
            counts_gene = counts_gene.detach().cpu().numpy()
        counts_gene = np.asarray(counts_gene, dtype=np.float64)
        n_spots, n_genes = counts_gene.shape

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

    sv_test_results = {
        "statistic": hsic_arr,
        "pvalue": pvals_arr,
        "method": "hsic-gc",
        "null_method": null_method,
        "n_spots": n_spots,
        "chunk_size": column_cap,
    }

    # calculate adjusted p-values
    sv_test_results["pvalue_adj"] = false_discovery_control(sv_test_results["pvalue"])

    return sv_test_results
