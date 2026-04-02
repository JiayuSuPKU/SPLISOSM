"""General utilities for preprocessing and statistical helpers."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Literal
from scipy.stats import norm as _norm_dist
from scipy.sparse.csgraph import connected_components as _connected_components

import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse
from tqdm.auto import tqdm
import pandas as pd
import torch
from anndata import AnnData
from splisosm.likelihood import liu_sf
from splisosm.io import load_visium_sp_meta  # noqa: F401 — backward compat

__all__ = [
    "get_cov_sp",
    "counts_to_ratios",
    "false_discovery_control",
    "load_visium_sp_meta",
    "prepare_inputs_from_anndata",
    "add_ratio_layer",
    "extract_counts_n_ratios",
    "extract_gene_level_statistics",
    "run_sparkx",
    "run_hsic_gc",
]


def _index_rows_sparse_coo(t: torch.Tensor, keep_indices: np.ndarray) -> torch.Tensor:
    """Row-index a 2-D sparse COO tensor without densifying.

    Operates entirely on the stored indices and values so memory usage is
    proportional to the number of retained non-zeros, not to ``n × m``.

    Parameters
    ----------
    t : torch.Tensor
        Sparse COO tensor of shape ``(n, m)``.  Need not be coalesced.
    keep_indices : np.ndarray
        1-D integer array of row positions to retain, in ascending order
        (as returned by ``np.where(mask)[0]``).

    Returns
    -------
    torch.Tensor
        Sparse COO tensor of shape ``(len(keep_indices), m)`` with the same
        dtype as *t*.
    """
    t = t.coalesce()
    n = t.shape[0]

    # Build old_row → new_row lookup (-1 means the row is dropped)
    row_map = torch.full((n,), -1, dtype=torch.long)
    keep_t = torch.from_numpy(keep_indices.astype(np.int64))
    row_map[keep_t] = torch.arange(len(keep_indices), dtype=torch.long)

    row_idx = t.indices()[0]  # (nnz,)
    new_rows = row_map[row_idx]
    mask = new_rows >= 0  # entries that survive

    new_indices = torch.stack([new_rows[mask], t.indices()[1][mask]])
    new_values = t.values()[mask]
    new_size = torch.Size([len(keep_indices), t.shape[1]])

    return torch.sparse_coo_tensor(new_indices, new_values, new_size, dtype=t.dtype)


def get_cov_sp(
    coords: np.ndarray | torch.Tensor, k: int = 4, rho: float = 0.99
) -> torch.Tensor:
    """Return the dense standardised CAR spatial covariance matrix.

    Constructs a k-mutual-nearest-neighbour graph from *coords*, builds the
    CAR precision matrix M = I - rho * D^{-1/2} W_sym D^{-1/2}, and
    returns K = M⁻¹ with unit marginal variance.

    Parameters
    ----------
    coords
        Shape (n_spots, n_dims). Euclidean spatial coordinates of spots.
    k
        Number of mutual nearest neighbors.
    rho
        Spatial autocorrelation coefficient.

    Returns
    -------
    cov_sp : torch.Tensor
        Shape (n_spots, n_spots). Spatial covariance matrix with standardized variance (== 1).
    """
    from splisosm.kernel import SpatialCovKernel

    return SpatialCovKernel.from_coordinates(
        coords, k_neighbors=k, rho=rho, standardize_cov=True, centering=False
    ).realization()


def counts_to_ratios(
    counts: np.ndarray | torch.Tensor,
    transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
    nan_filling: Literal["mean", "none"] = "mean",
    fill_before_transform: Optional[bool] = None,
) -> torch.Tensor:
    """Convert isoform counts to proportions.

    Spots with zero total counts ("zero-coverage spots") are handled according to
    ``nan_filling`` and ``fill_before_transform``.  When ``nan_filling='mean'`` the
    zero-coverage rows are replaced with the per-isoform mean of the remaining rows.
    The timing of this fill relative to the ratio transformation is controlled by
    ``fill_before_transform``:

    * ``False`` (**new default**): zero-coverage rows are filled with the column-wise
      mean of the **transformed** values (i.e. after the ratio transform is applied).
      For log-ratio transforms (clr, ilr, alr) this means pseudocount-filled rows are
      replaced with the mean of the true transformed values.
    * ``True`` (**old behaviour**): zero-coverage rows are filled with the mean of the
      raw isoform ratios **before** the transformation is applied.  For log-ratio
      transforms the pseudocount-based rows are kept as-is (no explicit fill).

    .. note::
        **Behaviour change since v1.1.0:** the default filling now happens *after*
        transformation.  Code that relied on the previous before-transform fill
        should pass ``fill_before_transform=True`` explicitly.  Passing
        ``fill_before_transform=False`` adopts the new default without a warning.

    After conversion, the isoform ratios can be further transformed using log-ratio-based
    transformations (clr, ilr, alr) or radial transformation :cite:`park2022kernel`.

    Parameters
    ----------
    counts
        Shape (n_spots, n_isos). Isoform counts.
    transformation
        Transformation applied to the proportions. Can be one of the following:
        ``'none'``: no transformation, return isoform ratios.
        ``'clr'``: centered log-ratio transformation.
        ``'ilr'``: isometric log-ratio transformation.
        ``'alr'``: additive log-ratio transformation.
        ``'radial'``: radial transformation :cite:`park2022kernel`.
    nan_filling
        Method to fill all-zero rows.
        ``'mean'``: fill all-zero rows with the per-isoform mean across expressed spots
        (timing controlled by ``fill_before_transform``).
        ``'none'``: do not fill rows and return NaNs at all-zero rows.
    fill_before_transform
        Controls when zero-coverage rows are mean-filled relative to the ratio
        transformation.  Only relevant when ``nan_filling='mean'`` and
        ``transformation != 'none'``.
        ``False``: fill **after** transformation (new default).
        ``True``: fill **before** transformation (old behaviour).
        ``None`` (default): use the new default (``False``) and emit a
        :class:`FutureWarning` to inform callers of the behaviour change.

    Returns
    -------
    ratios : torch.Tensor
        Shape (n_spots, n_isos) or (n_spots, n_isos - 1) if ilr or alr transformation is used.

    Notes
    -----
    Log-ratio-based transformations (clr, ilr, alr) are implemented via ``scikit-bio``, with
    a pseudocount of 1% of the global mean per isoform to avoid zeros in the ratio.
    """
    assert transformation in ["none", "clr", "ilr", "alr", "radial"]
    if transformation in ["clr", "ilr", "alr"]:
        try:
            from skbio.stats.composition import (
                clr,
                ilr,
                alr,
            )  # for ratio transformation
        except ImportError:
            warnings.warn(
                f"Please install scikit-bio to use ratio transformation='{transformation}'. Switching to 'none'.",
                UserWarning,
                stacklevel=2,
            )
            transformation = "none"

    assert nan_filling in ["mean", "none"]

    # Resolve fill timing; warn when the new default differs from old behaviour.
    if fill_before_transform is None:
        if nan_filling == "mean" and transformation != "none":
            warnings.warn(
                "The default NaN-filling timing for `counts_to_ratios` has changed: "
                "zero-coverage spots are now filled with the per-isoform mean "
                "**after** the ratio transformation (previously filled before). "
                "Pass `fill_before_transform=False` to silence this warning and "
                "use the new default, or `fill_before_transform=True` to restore "
                "the previous behaviour.",
                FutureWarning,
                stacklevel=2,
            )
        _fill_before = False
    else:
        _fill_before = fill_before_transform

    if isinstance(counts, np.ndarray):
        counts = torch.from_numpy(counts).float()

    # identify zero rows to fill
    is_nan = counts.sum(1) == 0  # (n_spots,)

    # calculate isoform ratios
    if transformation in ["clr", "ilr", "alr"]:
        # add pseudocounts equal to 1% of the global mean per isoform to avoid zeros
        y = (1 - 1e-2) * counts + 1e-2 * counts.mean(0, keepdim=True)
        y = y / y.sum(1, keepdim=True)
        if nan_filling == "mean" and _fill_before:
            # old behaviour: replace pseudocount-filled rows with the mean of
            # the raw (pre-pseudocount) ratios of expressed spots
            _raw = counts / counts.sum(1, keepdim=True)
            if (~is_nan).any():
                y[is_nan] = _raw[~is_nan].mean(0, keepdim=True)
    else:
        y = counts / counts.sum(1, keepdim=True)  # NaN where is_nan
        if nan_filling == "mean" and _fill_before:
            if (~is_nan).any():
                y[is_nan] = y[~is_nan].mean(0, keepdim=True)

    # apply transformation
    if transformation == "clr":
        y = torch.from_numpy(clr(y)).float()  # (n_spots, n_isos)
    elif transformation == "ilr":
        y = torch.from_numpy(ilr(y)).float()  # (n_spots, n_isos - 1)
    elif transformation == "alr":
        y = torch.from_numpy(alr(y)).float()  # (n_spots, n_isos - 1)
    elif transformation == "radial":
        y = y / y.norm(dim=1, keepdim=True)  # NaN rows stay NaN

    # post-transform fill (new default) or explicit NaN restore
    if nan_filling == "mean" and not _fill_before:
        if is_nan.any() and (~is_nan).any():
            y[is_nan] = y[~is_nan].mean(0, keepdim=True)
    elif nan_filling == "none":
        y[is_nan] = torch.nan

    return y


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
        Benjamini-Hochberg :cite:`benjamini1995controlling` (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
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


def _process_design_mtx(
    adata: AnnData,
    design_mtx: Optional[Any],
    covariate_names: Optional[list[str]],
) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Process and resolve design matrix from AnnData.obs columns.

    Handles extraction, categorical encoding, sparse array conversion, and
    dimension validation for design matrices extracted from AnnData object.
    Automatically infers covariate names when not explicitly provided.

    Supports three input modes:

    1. Single column name (str): extracts one ``adata.obs`` column
    2. Multiple column names (list[str]): extracts and concatenates columns
    3. Pre-computed matrix (array/dataframe/tensor): used as-is

    For columns containing categorical data (dtype 'category' or object),
    one-hot encoding is applied. Numerical columns are converted as-is.

    Parameters
    ----------
    adata
        Annotated data matrix.
    design_mtx
        Design matrix specification or pre-computed matrix.

        - str: single column name in ``adata.obs``
        - list[str]: list of column names in ``adata.obs``
        - ndarray/tensor/sparse/DataFrame: pre-computed design matrix

    covariate_names
        Explicit covariate names. If provided, dimensions must match
        the resulting design matrix. If None, names are inferred as follows:

        - **From obs columns (str/list[str])**: column names are used, with
          categorical columns expanded to one-hot encoded names.
        - **From DataFrame**: column names are extracted.
        - **Otherwise**: default names ``factor_1``, ``factor_2``, etc.

    Returns
    -------
    resolved_design : numpy.ndarray | scipy.sparse.spmatrix | None
        Design matrix of shape ``(n_spots, n_factors)``, dtype float32.
        Returns ``None`` if ``design_mtx`` is ``None``.
    resolved_covariates : list[str] | None
        Inferred or explicit covariate names matching design matrix columns.
        Resolves as:

        1. User-provided ``covariate_names`` (if valid dimensions)
        2. Extracted from obs columns or DataFrame
        3. Auto-generated ``factor_1``, ``factor_2``, etc.

        Returns ``None`` if ``design_mtx`` is ``None``.

    Raises
    ------
    ValueError
        If columns are missing, covariate_names dimensions don't match,
        or other validation failures occur.
    """
    if design_mtx is None:
        return None, None

    # Extract columns from adata.obs
    if isinstance(design_mtx, str):
        # Single column
        col_names = [design_mtx]
        columns_to_extract = col_names
    elif (
        isinstance(design_mtx, list)
        and len(design_mtx) > 0
        and all(isinstance(x, str) for x in design_mtx)
    ):
        # Multiple columns
        col_names = list(design_mtx)
        columns_to_extract = col_names
    else:
        # Pre-computed matrix: validate and return
        columns_to_extract = None
        col_names = None

    if columns_to_extract is not None:
        # Validate columns exist
        missing = [col for col in columns_to_extract if col not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} were not found in `adata.obs` for design_mtx."
            )

        # Extract and process columns
        design_list = []
        resolved_cov_names = []

        for col in columns_to_extract:
            col_data = adata.obs[col]
            col_dtype = col_data.dtype

            # Check if categorical (either pandas Categorical or object/string type)
            is_categorical = (
                isinstance(col_dtype, pd.CategoricalDtype)
                or col_dtype == "object"
                or col_dtype.kind == "O"  # Object array
            )

            if is_categorical:
                # One-hot encode categorical column
                # Use pd.get_dummies to create binary indicators
                dummies = pd.get_dummies(col_data, prefix=col, drop_first=False)
                design_list.append(dummies.values.astype(np.float32))
                resolved_cov_names.extend(dummies.columns.tolist())
            else:
                # Numerical column: convert to float
                col_array = col_data.values.astype(np.float32).reshape(-1, 1)
                design_list.append(col_array)
                resolved_cov_names.append(col)

        # Concatenate all columns
        resolved_design = np.concatenate(design_list, axis=1)

    else:
        # Pre-computed matrix: convert to appropriate format
        if isinstance(design_mtx, pd.DataFrame):
            resolved_design = design_mtx.values.astype(np.float32)
            resolved_cov_names = list(design_mtx.columns)
        elif isinstance(design_mtx, torch.Tensor):
            resolved_design = design_mtx.cpu().numpy().astype(np.float32)
            resolved_cov_names = None
        elif scipy.sparse.issparse(design_mtx):
            # Convert sparse matrix to csr format
            resolved_design = design_mtx.tocsr().astype(np.float32)
            resolved_cov_names = None
        elif isinstance(design_mtx, np.ndarray):
            resolved_design = design_mtx.astype(np.float32)
            resolved_cov_names = None
        else:
            raise TypeError(
                f"Unsupported design_mtx type: {type(design_mtx)}. "
                "Expected array-like, DataFrame, tensor, or sparse matrix."
            )

    # Reshape 1-D arrays/tensors to 2-D column vectors
    if not scipy.sparse.issparse(resolved_design) and np.ndim(resolved_design) == 1:
        resolved_design = resolved_design.reshape(-1, 1)

    # Validate shapes
    n_spots = adata.n_obs
    if resolved_design.shape[0] != n_spots:
        raise ValueError(
            f"Design matrix row count ({resolved_design.shape[0]}) "
            f"must match number of spots ({n_spots})."
        )

    # Validate and set covariate names
    n_factors = resolved_design.shape[1]
    if covariate_names is not None:
        if len(covariate_names) != n_factors:
            raise ValueError(
                f"covariate_names length ({len(covariate_names)}) must match "
                f"design matrix columns ({n_factors})."
            )
        resolved_cov_names = covariate_names
    elif resolved_cov_names is None:
        # Generate default names
        resolved_cov_names = [f"factor_{i+1}" for i in range(n_factors)]

    return resolved_design, resolved_cov_names


def prepare_inputs_from_anndata(
    adata: AnnData,
    layer: str,
    group_iso_by: str,
    spatial_key: str,
    adj_key: Optional[str] = None,
    min_counts: int = 0,
    min_bin_pct: float = 0,
    filter_single_iso_genes: bool = False,
    gene_names: Optional[str] = None,
    design_mtx: Optional[Any] = None,
    covariate_names: Optional[list[str]] = None,
    min_component_size: int = 1,
    k_neighbors: int = 4,
    return_filtered_anndata: bool = False,
) -> tuple[
    list[torch.Tensor],
    torch.Tensor,
    list[str],
    Optional[Any],
    Optional[list[str]],
    "scipy.sparse.spmatrix | None",
    "AnnData | None",
]:
    """Extract and filter isoform count tensors from an AnnData object.

    Shared helper used by both :class:`splisosm.hyptest_np.SplisosmNP` and
    :class:`splisosm.hyptest_glmm.SplisosmGLMM` to prepare legacy-compatible
    tensors from an AnnData input.  Feature filtering, sparse/dense handling,
    coordinate extraction, and design-matrix resolution are all performed here.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Key in ``adata.layers`` containing raw isoform counts.
    group_iso_by
        Column in ``adata.var`` used to group isoforms by gene.
    spatial_key
        Key in ``adata.obsm`` for spatial coordinates.
    adj_key
        Key in ``adata.obsp`` for a pre-built adjacency matrix.
        When provided, it overrides the k-NN graph construction
        from coordinates and be used directly to build the spatial kernel.
        The adjacency matrix is symmetrized internally before returned.
    min_counts
        Minimum total isoform count across spots required to retain an isoform.
    min_bin_pct
        Minimum fraction/percentage of spots with non-zero expression for an
        isoform.  Values in ``[0, 1]`` are treated as fractions; values in
        ``(1, 100]`` are treated as percentages.
    filter_single_iso_genes
        Whether to discard genes with fewer than two retained isoforms.
    gene_names
        Column name in ``adata.var`` used as display names for grouped genes.
        If ``None``, the grouped gene IDs are used.
    design_mtx
        Design matrix for differential-usage tests.  Accepts a
        tensor/array/dataframe of shape ``(n_spots, n_factors)``, a single
        obs-column name (str), or a list of obs-column names.
    covariate_names
        Explicit covariate names.  When ``design_mtx`` is given as column
        name(s) and this is ``None``, the column names are used automatically.
    min_component_size
        Minimum number of spots a connected component must contain to be
        retained.  Spots in smaller components are removed from all arrays
        (counts, coordinates, design matrix).  Requires building a k-NN
        graph from coordinates (unless ``adj_key`` is provided).
        Default 1 disables filtering.  Emits ``UserWarning`` when spots
        are removed.
    k_neighbors
        Number of nearest neighbours for the k-NN graph when
        ``adj_key`` is ``None`` and component filtering is active.
        Default 4.
    return_filtered_anndata
        If ``True``, return a copy of ``adata`` restricted to the spots and
        isoforms that survived all filtering steps, with columns ordered to
        match ``counts_list``.  Default ``False`` returns ``None`` as the
        seventh element.

    Returns
    -------
    counts_list : list[torch.Tensor]
        Per-gene isoform count tensors, each of shape ``(n_spots, n_isos)``.
        Sparse ``adata.layers[layer]`` input yields sparse COO tensors.
    coordinates : torch.Tensor
        Shape ``(n_spots, 2)`` spatial coordinates, dtype float32.
    resolved_gene_names : list[str]
        Display names for each gene in ``counts_list``.
    resolved_design : np.ndarray or tensor or None
        Resolved design matrix, or the original object if it was already
        array-like; ``None`` when ``design_mtx`` is ``None``.
    resolved_covariates : list[str] or None
        Resolved covariate names, or ``None`` when ``design_mtx`` is ``None``.
    adj_matrix : scipy.sparse.spmatrix or None
        The (possibly filtered) adjacency matrix to use for building the
        spatial kernel.  This is:

        * ``adata.obsp[adj_key]`` (filtered to retained spots) when
          ``adj_key`` is provided, or
        * the k-NN adjacency built from coordinates (filtered) when
          ``min_component_size > 1`` and filtering removed spots, or
        * ``None`` otherwise (caller should build k-NN inside
          :class:`~splisosm.kernel.SpatialCovKernel`).
    filtered_adata : anndata.AnnData
        A copy of ``adata`` restricted to the spots and isoforms that survived
        all filtering steps (component filtering + min_counts / min_bin_pct /
        single-isoform-gene filtering).  Columns are ordered to match the
        concatenation order of isoforms across genes in ``counts_list``.
        This is useful for computing per-feature summary statistics without
        re-running the filtering logic.

    Raises
    ------
    ValueError
        If required fields are missing from ``adata``, no isoforms survive
        filtering, or argument values are out of range.
    """
    if not isinstance(adata, AnnData):
        raise ValueError("`adata` must be an AnnData object.")
    if layer not in adata.layers:
        raise ValueError(f"Layer `{layer}` was not found in `adata.layers`.")
    if group_iso_by not in adata.var.columns:
        raise ValueError(f"`{group_iso_by}` was not found in `adata.var`.")
    if min_counts < 0:
        raise ValueError("`min_counts` must be >= 0.")
    if min_bin_pct < 0 or min_bin_pct > 100:
        raise ValueError("`min_bin_pct` must be between 0 and 1, or between 0 and 100.")
    if isinstance(gene_names, list):
        raise ValueError(
            "In AnnData mode, `gene_names` must be a var-column name (str) or None."
        )
    if not isinstance(min_component_size, int) or min_component_size < 1:
        raise ValueError("`min_component_size` must be an integer >= 1.")
    if hasattr(design_mtx, "shape") and design_mtx.shape[0] != adata.n_obs:
        raise ValueError(
            f"Design matrix row count ({design_mtx.shape[0]}) "
            f"must match number of spots ({adata.n_obs})."
        )
    if spatial_key not in adata.obsm:
        raise ValueError(f"`{spatial_key}` was not found in `adata.obsm`.")

    # Extract spatial coordinates
    coordinates = adata.obsm[spatial_key]
    coordinates = torch.as_tensor(np.asarray(coordinates), dtype=torch.float32)
    if coordinates.dim() != 2:
        raise ValueError("Coordinates in `adata.obsm[spatial_key]` must be a 2D array.")

    # Load/build adjacency and filter disconnected components if needed
    _adj_out: Optional[scipy.sparse.spmatrix] = None

    # Load adj from adata.obsp if adj_key is provided
    if adj_key is not None:
        if adj_key not in adata.obsp:
            raise ValueError(f"`adj_key` '{adj_key}' was not found in `adata.obsp`.")
        _adj_raw = adata.obsp[adj_key]
        _adj_out = (
            _adj_raw.tocsc()
            if scipy.sparse.issparse(_adj_raw)
            else scipy.sparse.csc_matrix(_adj_raw)
        )

        # Symmetrize the adjacency matrix if it's not already symmetric
        if not (_adj_out != _adj_out.T).nnz == 0:
            _adj_out = (_adj_out + _adj_out.T) / 2
            warnings.warn(
                "Provided adjacency matrix is not symmetric; symmetrising by averaging with its transpose.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif min_component_size > 1:
        # Build k-NN graph from coordinates for component filtering
        from splisosm.kernel import _build_adj_from_coords

        _adj_out = _build_adj_from_coords(
            coordinates, k_neighbors=k_neighbors, mutual_neighbors=True
        ).tocsc()

    # Spot-level filtering
    if min_component_size > 1:
        # Count connected components
        _, _labels = _connected_components(_adj_out, directed=False)
        _comp_sizes = np.bincount(_labels)
        _keep_mask = _comp_sizes[_labels] >= min_component_size
        _n_removed = int((~_keep_mask).sum())

        # filter out spots in small components from adata, coords, adj, and design_mtx
        if _n_removed > 0:
            _n_remaining = int(_keep_mask.sum())
            if _n_remaining == 0:
                raise ValueError(
                    "No spots remained after filtering small graph components. "
                    "Lower `min_component_size` or check your coordinate/adjacency matrix."
                )
            adata = adata[
                _keep_mask, :
            ].copy()  # filter adata to keep only the retained spots
            coordinates = adata.obsm[spatial_key] if coordinates is not None else None
            _adj_out = _adj_out[_keep_mask][:, _keep_mask].tocsc()

            # Filter design matrix if provided as array-like
            if design_mtx is not None:
                if isinstance(design_mtx, pd.DataFrame):
                    design_mtx = design_mtx.iloc[_keep_mask].copy()
                elif isinstance(design_mtx, torch.Tensor):
                    design_mtx = design_mtx[_keep_mask].clone()
                elif scipy.sparse.issparse(design_mtx):
                    design_mtx = design_mtx.tocsr()[_keep_mask].copy()
                elif isinstance(design_mtx, np.ndarray):
                    design_mtx = design_mtx[_keep_mask, :].copy()

            warnings.warn(
                f"Removed {_n_removed} spot(s) belonging to graph components "
                f"with fewer than {min_component_size} member(s). "
                f"{_n_remaining} spot(s) remain.",
                UserWarning,
                stacklevel=2,
            )

    # Feature-level filtering
    iso_counts = adata.layers[layer]
    min_bin_frac = float(min_bin_pct)
    if min_bin_frac > 1.0:
        min_bin_frac /= 100.0

    is_sparse_input = scipy.sparse.issparse(iso_counts)

    # Normalise sparse format to CSR/CSC once, upfront — avoids repeated conversions.
    if is_sparse_input and not (
        scipy.sparse.isspmatrix_csr(iso_counts)
        or scipy.sparse.isspmatrix_csc(iso_counts)
    ):
        iso_counts = iso_counts.tocsr()

    # Compute per-isoform stats efficiently.
    total_counts = np.asarray(iso_counts.sum(axis=0)).ravel()
    if is_sparse_input:
        # getnnz reads directly from the sparse internal arrays — no intermediate
        # boolean matrix compared to (iso_counts > 0).sum(axis=0).
        bin_pct = iso_counts.getnnz(axis=0).astype(np.float64) / adata.n_obs
    else:
        counts_arr_for_stats = np.asarray(iso_counts)
        bin_pct = np.count_nonzero(counts_arr_for_stats, axis=0) / adata.n_obs

    # Copy only the columns needed from adata.var — avoids duplicating potentially
    # hundreds of annotation columns that are irrelevant here.
    needed_var_cols = [group_iso_by]
    if gene_names is not None:
        if gene_names not in adata.var.columns:
            raise ValueError(
                f"`gene_names` column `{gene_names}` was not found in `adata.var`."
            )
        if gene_names != group_iso_by:
            needed_var_cols.append(gene_names)

    # var_df index == adata.var_names (isoform names)
    var_df = adata.var[needed_var_cols].copy()
    var_df["__total_counts__"] = total_counts
    var_df["__bin_pct__"] = bin_pct

    # Remove isoforms that do not meet the count and bin percentage thresholds.
    var_df = var_df[
        (var_df["__total_counts__"] >= min_counts)
        & (var_df["__bin_pct__"] >= min_bin_frac)
    ]
    if var_df.shape[0] == 0:
        raise ValueError(
            "No features remained after applying `min_counts`/`min_bin_pct` filtering."
        )

    # Remove genes with fewer than 2 isoforms if requested.
    if filter_single_iso_genes:
        n_iso_per_gene = var_df.groupby(group_iso_by, observed=True).size()
        keep_genes = n_iso_per_gene[n_iso_per_gene >= 2].index
        var_df = var_df[var_df[group_iso_by].isin(keep_genes)]
        if var_df.shape[0] == 0:
            raise ValueError("No genes with >=2 isoforms remained after filtering.")

    # Prepare per-gene spot-by-isoform count tensors.
    # We also resolve gene display names here to avoid a second groupby call.
    iso_groups = list(var_df.groupby(group_iso_by, observed=True, sort=False))
    if len(iso_groups) == 0:
        raise ValueError("No genes remained after extracting isoform counts.")

    all_iso_names_flat: list[str] = []
    gene_slice_offsets: list[tuple[int, int]] = []
    grouped_genes: list[str] = []
    resolved_gene_names: list[str] = []
    pos = 0
    for gene_id, group in iso_groups:
        # group.index contains the isoform names for this gene (== adata.var_names subset)
        iso_names = group.index.astype(str).tolist()
        all_iso_names_flat.extend(iso_names)
        gene_slice_offsets.append((pos, pos + len(iso_names)))
        pos += len(iso_names)
        grouped_genes.append(str(gene_id))
        resolved_gene_names.append(
            str(group[gene_names].iloc[0]) if gene_names is not None else str(gene_id)
        )

    # Vectorised column-index lookup. This is much faster than a Python dict
    # comprehension loop for large var_df
    var_index = pd.Index(adata.var_names)
    all_iso_indices = var_index.get_indexer(all_iso_names_flat)
    if (all_iso_indices < 0).any():
        missing = [n for n, idx in zip(all_iso_names_flat, all_iso_indices) if idx < 0]
        raise ValueError(
            f"Could not locate {len(missing)} isoform name(s) in adata.var_names "
            f"(first few: {missing[:5]}). This indicates an internal inconsistency."
        )

    # Single upfront column slice + one format conversion
    if is_sparse_input:
        # Build a CSC sub-matrix (fast per-gene column access) in float32.
        sub_csc = iso_counts[:, all_iso_indices].tocsc().astype(np.float32, copy=False)
    else:
        # One contiguous float32 copy; per-gene views share this memory (no per-gene copy).
        sub_arr = np.ascontiguousarray(iso_counts[:, all_iso_indices], dtype=np.float32)
        sub_tensor = torch.from_numpy(sub_arr)  # zero-copy

    # Per-gene extraction: fast positional slices from the small sub-matrix
    counts_list: list[torch.Tensor] = []
    for (_, _group), (start, end) in zip(iso_groups, gene_slice_offsets):
        if is_sparse_input:
            _coo = sub_csc[:, start:end].tocoo()
            # torch.from_numpy avoids an extra copy vs torch.LongTensor/FloatTensor
            _i = torch.from_numpy(
                np.vstack((_coo.row, _coo.col)).astype(np.int64, copy=False)
            )
            _v = torch.from_numpy(_coo.data)  # already float32
            with torch.sparse.check_sparse_tensor_invariants(False):
                _counts = torch.sparse_coo_tensor(_i, _v, torch.Size(_coo.shape))
        else:
            _counts = sub_tensor[:, start:end]  # zero-copy view
        counts_list.append(_counts)

    # Process design matrix (handles column extraction, categorical encoding, etc.)
    resolved_design, resolved_covariates = _process_design_mtx(
        adata, design_mtx, covariate_names
    )

    # Optionally build filtered adata: spots retained after component filtering, isoforms
    # in the same order as all_iso_names_flat (matches counts_list gene/iso ordering).
    filtered_adata = (
        adata[:, all_iso_names_flat].copy() if return_filtered_anndata else None
    )

    return (
        counts_list,
        coordinates,
        resolved_gene_names,
        resolved_design,
        resolved_covariates,
        _adj_out,
        filtered_adata,
    )


def add_ratio_layer(
    adata: AnnData,
    layer: str,
    group_iso_by: str,
    ratio_layer_key: str,
    fill_nan_with_mean: bool = False,
) -> None:
    """Compute within-gene isoform usage ratios and store them as a new layer.

    For each spot and gene, every isoform's ratio is its count divided by the
    total count of that gene at that spot.  The result is written to
    ``adata.layers[ratio_layer_key]`` **in-place**.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Key in ``adata.layers`` containing raw isoform counts.
    group_iso_by
        Column in ``adata.var`` grouping isoforms to genes.
    ratio_layer_key
        Key under which to store the computed ratio matrix.  Must differ
        from *layer*.
    fill_nan_with_mean : bool, optional
        How to handle spots where the gene has zero total counts.

        * ``False`` (default): store as a **sparse CSR matrix** with the same
          sparsity structure as the count layer.  Spots with zero gene total
          have ratio 0 for all isoforms of that gene (structural zeros, not
          stored explicitly).
        * ``True``: store as a dense float32 matrix; spots with zero gene
          total are filled with the per-isoform mean ratio across expressed
          spots (or 0.0 if no spot expresses the isoform).

    Raises
    ------
    ValueError
        If *layer* is missing, *group_iso_by* is not a ``var`` column, or
        *ratio_layer_key* equals *layer*.
    """
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    if group_iso_by not in adata.var.columns:
        raise ValueError(f"'{group_iso_by}' not found in adata.var.")
    if ratio_layer_key == layer:
        raise ValueError("`ratio_layer_key` must differ from `layer`.")

    iso_counts = adata.layers[layer]
    is_sparse = scipy.sparse.issparse(iso_counts)
    n_isos = adata.n_vars

    # Map each isoform to its gene index (stable sort preserves var order)
    gene_labels = adata.var[group_iso_by].values
    _, gene_idx = np.unique(gene_labels, return_inverse=True)  # (n_isos,)
    n_genes = int(gene_idx.max()) + 1 if n_isos > 0 else 0

    # Summation matrix S[i, g] = 1 iff isoform i belongs to gene g
    # shape (n_isos, n_genes) — used to aggregate isoforms to gene totals
    S = scipy.sparse.csc_matrix(
        (np.ones(n_isos, dtype=np.float32), (np.arange(n_isos), gene_idx)),
        shape=(n_isos, n_genes),
    )

    if is_sparse:
        _csr = (
            iso_counts
            if scipy.sparse.isspmatrix_csr(iso_counts)
            else iso_counts.tocsr()
        )
        _csr = _csr.astype(np.float32, copy=False)
        gene_totals = (_csr @ S).toarray()  # (n_spots, n_genes)
    else:
        counts_dense = np.asarray(iso_counts, dtype=np.float32)
        # S.T: (n_genes, n_isos) sparse; counts_dense.T: (n_isos, n_spots) dense
        # scipy handles sparse @ dense → dense ndarray
        gene_totals = np.asarray((S.T @ counts_dense.T).T)  # (n_spots, n_genes)

    # Expand gene totals back to per-isoform denominators
    gene_sum_per_iso = gene_totals[:, gene_idx]  # (n_spots, n_isos)

    if fill_nan_with_mean:
        # Dense output: safe division, NaN → per-isoform column mean (or 0)
        counts_dense = _csr.toarray() if is_sparse else counts_dense
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = np.where(
                gene_sum_per_iso > 0, counts_dense / gene_sum_per_iso, np.nan
            )
        iso_means = np.nanmean(ratios, axis=0)  # (n_isos,)
        iso_means = np.where(np.isnan(iso_means), 0.0, iso_means)
        nan_locs = np.isnan(ratios)
        if nan_locs.any():
            ratios[nan_locs] = iso_means[np.where(nan_locs)[1]]
        adata.layers[ratio_layer_key] = ratios.astype(np.float32)
    else:
        # Sparse output: same sparsity as the count matrix.
        # ratio[s, i] = count[s, i] / gene_total[s, gene(i)], 0 where count == 0.
        # No division-by-zero risk: if count > 0 then gene total > 0.
        if is_sparse:
            _result = _csr.copy()
        else:
            _result = scipy.sparse.csr_matrix(counts_dense.astype(np.float32))
        _result.data /= gene_sum_per_iso[_result.nonzero()]
        adata.layers[ratio_layer_key] = _result


def extract_counts_n_ratios(
    adata: AnnData,
    layer: str = "counts",
    group_iso_by: str = "gene_symbol",
    return_sparse: bool = False,
    filter_single_iso_genes: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], Optional[np.ndarray]]:
    """Extract per-gene lists of isoform counts and ratios from anndata.

    .. deprecated:: 1.1.0
        Use :func:`add_ratio_layer` to compute ratios and store them in
        ``adata.layers``, then use
        :func:`prepare_inputs_from_anndata` with the ratio layer key to
        extract per-gene ratio tensors.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to extract isoform counts (adata.layers[layer]).
    group_iso_by
        Gene index in adata.var to group isoforms by.
    return_sparse
        Whether to return sparse torch tensors for counts_list.
        If True, `ratios_list` will be empty and `ratio_obs_merged` will be None.
    filter_single_iso_genes
        Whether to filter out genes with only one isoform.
        By default True for compatibility with splisosm models.

    Returns
    -------
    counts_list : list[torch.Tensor]
        Isoform counts per gene, each of shape (n_spots, n_isos).
    ratios_list : list[torch.Tensor]
        Isoform ratios per gene, each of shape (n_spots, n_isos).
    gene_name_list : list[str]
        Gene names.
    ratio_obs_merged : np.ndarray | None
        Observed isoform ratios, shape (n_spots, n_isos_total), or None if `return_sparse` is True.
    """
    warnings.warn(
        "`extract_counts_n_ratios` is deprecated. "
        "Use `add_ratio_layer` to compute ratios inplace, then call "
        "`prepare_inputs_from_anndata` with the ratio layer key.",
        DeprecationWarning,
        stacklevel=2,
    )
    # extract isoform counts
    iso_counts = adata.layers[layer]  # (n_spots, n_isos_total)

    # Check if input is sparse
    is_sparse_input = scipy.sparse.issparse(iso_counts)
    if (
        is_sparse_input
        and not scipy.sparse.isspmatrix_csc(iso_counts)
        and not scipy.sparse.isspmatrix_csr(iso_counts)
    ):
        # convert to csr for efficient slicing if it is other sparse format
        iso_counts = iso_counts.tocsr()

    counts_list = []  # isoform counts per gene, each of (n_spots, n_isos)
    ratios_list = []  # isoform ratios per gene, each of (n_spots, n_isos)
    gene_name_list = []  # of length n_genes
    iso_ind_list = []  # of length n_genes

    _gene_groups = adata.var.reset_index().groupby(group_iso_by, observed=True)
    for _gene, _group in tqdm(_gene_groups, desc="Genes", total=_gene_groups.ngroups):
        # filter single-isoform genes if needed
        if filter_single_iso_genes and _group.shape[0] < 2:
            continue

        # extract isoform name and index per gene
        gene_name_list.append(_gene)
        iso_indices = _group.index.tolist()
        iso_ind_list.append(iso_indices)

        # extract isoform counts and relative ratio
        if is_sparse_input and return_sparse:
            # since ratios are usually dense, we do not compute ratios for sparse input
            # ratios_list will be empty and ratio_obs_merged will be None

            # slice sparse matrix
            _counts_scipy = iso_counts[:, iso_indices]

            # convert to torch sparse coo tensor
            _counts_coo = _counts_scipy.tocoo()
            values = _counts_coo.data
            indices = np.vstack((_counts_coo.row, _counts_coo.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = _counts_coo.shape
            with torch.sparse.check_sparse_tensor_invariants(False):
                _counts = torch.sparse_coo_tensor(i, v, torch.Size(shape))
            counts_list.append(_counts)
        else:
            if is_sparse_input:
                # slice sparse matrix and convert to dense
                _counts_np = iso_counts[:, iso_indices].toarray()
            else:
                _counts_np = iso_counts[:, iso_indices]

            # compute counts and ratios as dense tensors
            _counts = torch.from_numpy(_counts_np).float()  # (n_spots, n_isos)
            _ratios = counts_to_ratios(
                _counts, transformation="none"
            )  # (n_spots, n_isos)

            counts_list.append(_counts)
            ratios_list.append(_ratios)

    if is_sparse_input and return_sparse:
        ratio_obs_merged = None
    else:
        # reshape and store the observed ratio in anndata
        ratio_obs_merged = torch.concat(
            ratios_list, axis=1
        ).numpy()  # (n_spots, n_isos_total)
        ratio_obs_merged = ratio_obs_merged[
            :, np.argsort(np.concatenate(iso_ind_list))
        ]  # (n_spots, n_isos_total)

    return counts_list, ratios_list, gene_name_list, ratio_obs_merged


def extract_gene_level_statistics(
    adata: AnnData, layer: str = "counts", group_iso_by: str = "gene_symbol"
) -> pd.DataFrame:
    """Extract gene-level metadata from isoform-level counts anndata.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to extract isoform counts (adata.layers[layer]).
    group_iso_by
        Gene index in adata.var to group isoforms by.

    Returns
    -------
    pandas.DataFrame
        Gene-level metadata with columns:

        - ``'n_iso'``: int. Number of isoforms per gene.
        - ``'pct_spot_on'``: float. Percentage of spots with non-zero counts.
        - ``'count_avg'``: float. Average counts per gene.
        - ``'count_std'``: float. Standard deviation of counts per gene.
        - ``'perplexity'``: float. Expression-based effective number of isoforms.
        - ``'major_ratio_avg'``: float. Average ratio of the major isoform.

    See Also
    --------
    :func:`splisosm.hyptest_np.SplisosmNP.extract_feature_summary`
    :func:`splisosm.hyptest_glmm.SplisosmGLMM.extract_feature_summary`

    """
    # extract isoform counts
    iso_counts = adata.layers[layer]  # (n_spots, n_isos_total)

    # Check if input is sparse
    is_sparse_input = scipy.sparse.issparse(iso_counts)
    if (
        is_sparse_input
        and not scipy.sparse.isspmatrix_csc(iso_counts)
        and not scipy.sparse.isspmatrix_csr(iso_counts)
    ):
        iso_counts = iso_counts.tocsr()

    df_list = []
    # loop through genes
    _gene_groups = adata.var.reset_index().groupby(group_iso_by, observed=True)
    for _gene, _group in tqdm(_gene_groups, desc="Genes", total=_gene_groups.ngroups):
        # extract isoform counts and relative ratio
        iso_indices = _group.index.tolist()
        _counts = iso_counts[:, iso_indices]

        if is_sparse_input:
            # Calculate statistics on sparse matrix without densifying
            _sum_per_iso = np.asarray(_counts.sum(0)).flatten()  # (n_isos,)
            _row_sums = np.asarray(_counts.sum(1)).flatten()  # (n_spots,)
            _total_sum = _sum_per_iso.sum()

            pct_spot_on = (_row_sums > 0).mean()
            count_avg = _row_sums.mean()
            count_std = _row_sums.std()
        else:
            if not isinstance(_counts, np.ndarray):
                _counts = np.asarray(_counts)

            _sum_per_iso = _counts.sum(0)
            _row_sums = _counts.sum(1)
            _total_sum = _counts.sum()

            pct_spot_on = (_row_sums > 0).mean()
            count_avg = _row_sums.mean()
            count_std = _row_sums.std()

        # Avoid division by zero
        if _total_sum == 0:
            _ratios_avg = np.zeros_like(_sum_per_iso)
        else:
            _ratios_avg = _sum_per_iso / _total_sum  # (n_isos,)

        # calculate and store gene-level statistics
        # handle zeros in log
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -(np.log(_ratios_avg) * _ratios_avg)
            entropy = np.nan_to_num(entropy).sum()

        df_list.append(
            {
                "gene": _gene,
                "n_iso": _group.shape[0],
                "pct_spot_on": pct_spot_on,
                "count_avg": count_avg,
                "count_std": count_std,
                "perplexity": np.exp(entropy),
                "major_ratio_avg": _ratios_avg.max() if _ratios_avg.size > 0 else 0.0,
            }
        )

    df_gene_meta = pd.DataFrame(df_list).set_index("gene")

    return df_gene_meta


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


def run_hsic_gc(
    counts_gene: "np.ndarray | torch.Tensor | None" = None,
    coordinates: "np.ndarray | torch.Tensor | None" = None,
    null_method: Literal["eig", "trace"] = "eig",
    null_configs: Optional[dict[str, Any]] = None,
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

    This function is designed to be a plugin replacement for SPARK-X.

    Parameters
    ----------
    counts_gene
        Shape ``(n_spots, n_genes)``. Gene counts.
    coordinates
        Shape ``(n_spots, n_dim)``. Spatial coordinates of spots.
    null_method : {"eig", "trace"}, optional
        Method for computing the null distribution of the test statistic:

        * ``"eig"`` (default): asymptotic chi-square mixture using kernel
          eigenvalues; Liu's method.  Supports optional
          ``null_configs["approx_rank"]`` (int) to restrict to the top-k
          eigenvalues.
        * ``"trace"``: moment-matching normal approximation using
          tr(K') and tr(K'²) of the centred spatial kernel and the scalar
          per-gene variance.
    null_configs : dict or None, optional
        Extra keyword arguments for the chosen ``null_method``.
    min_component_size : int, optional
        Minimum number of spots a connected component must contain to be
        retained.  Spots that belong to components smaller than this
        threshold are removed from all data structures (counts, coordinates,
        design matrix) before the spatial kernel is built. Components are detected
        on the same k-NN graph used for the spatial kernel (controlled by ``k_neighbors``).
        The default value of ``1`` disables filtering.
        A ``UserWarning`` is issued whenever spots are removed.
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
        AnnData mode.
    adj_key : str or None, optional
        Key in ``adata.obsp`` for a pre-built adjacency matrix.  When
        provided in AnnData mode, the adjacency is loaded from
        ``adata.obsp[adj_key]`` and used for both component filtering and
        the spatial kernel construction.  Ignored in matrix mode.
    min_counts : int, optional
        Minimum total count to retain a gene in AnnData mode.  Default 0.
    min_bin_pct : float, optional
        Minimum fraction of spots expressing a gene (count > 0).  Default 0.
    **spatial_kernel_kwargs
        Additional arguments forwarded to :class:`~splisosm.kernel.SpatialCovKernel`.

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

        # Load spatial coordinates
        coordinates = torch.as_tensor(
            np.asarray(adata.obsm[spatial_key], dtype=np.float32)
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
            else:
                _adj_for_comp = _build_adj_from_coords(
                    coordinates, k_neighbors=_k_adata, mutual_neighbors=True
                ).tocsc()
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
    if null_method == "eig":
        approx_rank = configs.get("approx_rank", None)
        # auto-cap rank for large datasets to keep eigendecomposition tractable
        if n_spots > 5000:
            max_rank = int(np.ceil(np.sqrt(n_spots) * 4))
            approx_rank = (
                min(approx_rank, max_rank) if approx_rank is not None else max_rank
            )
        elif approx_rank is not None:
            approx_rank = approx_rank if approx_rank < n_spots else None
        lambda_sp = K_sp.eigenvalues(k=approx_rank)
        lambda_sp = lambda_sp[lambda_sp > 1e-5]
        k_eff = len(lambda_sp)
        # Low-rank factor Q_k so that K ≈ Q_k Q_k^T; used per-gene to
        # produce a rank-consistent test stat for liu_sf.
        _Q_sp_raw = getattr(K_sp, "Q", None)
        _Q_sp = _Q_sp_raw[:, :k_eff] if _Q_sp_raw is not None else None
    elif null_method == "trace":
        trK = K_sp.trace()
        trK2 = K_sp.square_trace()
    else:
        raise ValueError(f"null_method must be 'eig' or 'trace', got {null_method!r}")

    # compute the HSIC-GC statistic per-gene
    is_scipy_sparse = scipy.sparse.issparse(counts_gene)
    is_torch_sparse = isinstance(counts_gene, torch.Tensor) and counts_gene.is_sparse

    if is_scipy_sparse:
        n_spots, n_genes = counts_gene.shape
        # Ensure efficient column slicing
        if not scipy.sparse.isspmatrix_csc(
            counts_gene
        ) and not scipy.sparse.isspmatrix_csr(counts_gene):
            counts_gene = counts_gene.tocsc()
    elif is_torch_sparse:
        n_spots, n_genes = counts_gene.shape
        if counts_gene.dtype != torch.float32 and counts_gene.dtype != torch.float64:
            counts_gene = counts_gene.float()
        if not counts_gene.is_coalesced():
            counts_gene = counts_gene.coalesce()
        # Pre-allocate selection vector for column extraction
        selection_vec = torch.zeros(
            n_genes, 1, device=counts_gene.device, dtype=counts_gene.dtype
        )
    else:
        if not isinstance(counts_gene, torch.Tensor):
            counts_gene = torch.from_numpy(counts_gene).float()  # (n_spots, n_genes)
        n_spots, n_genes = counts_gene.shape
        y_dense = counts_gene - counts_gene.mean(
            0, keepdim=True
        )  # center the counts, (n_spots, n_genes)

    hsic_list, pvals_list = [], []
    for i in tqdm(range(n_genes), desc="Genes", total=n_genes):
        if is_scipy_sparse:
            col = counts_gene[:, i].toarray()  # dense (n_spots, 1)
            counts = torch.from_numpy(col).float()
            counts = counts - counts.mean()
        elif is_torch_sparse:
            # Extract column i using Sparse x Dense matrix multiplication
            # (N, G) @ (G, 1) -> (N, 1)
            # This is efficient and avoids densifying the full matrix
            selection_vec.zero_()
            selection_vec[i, 0] = 1.0

            col = torch.sparse.mm(counts_gene, selection_vec)
            counts = col - col.mean()
        else:
            counts = y_dense[:, i : i + 1]  # (n_spots, 1)

        # Compute HSIC test statistic.
        # When `eig` null is used and a low-rank factor Q is available, use
        # the Q-based projection so the stat and the Liu null eigenvalues are
        # on the same (rank-k) scale — preventing the p=0 scale-mismatch bug.
        # Otherwise fall back to the exact full-kernel quadratic form.
        if null_method == "eig" and _Q_sp is not None:
            xtQ = counts.t() @ _Q_sp  # (1, k_eff)
            hsic_scaled = torch.trace(xtQ @ xtQ.t())
        else:
            hsic_scaled = torch.trace(K_sp.xtKx_exact(counts))

        if null_method == "eig":
            try:
                lambda_y = torch.linalg.eigvalsh(counts.T @ counts)  # (1,)
            except torch._C._LinAlgError:
                lambda_y = torch.linalg.eigvalsh(
                    counts.T @ counts + 1e-6 * torch.eye(counts.shape[1])
                )
            lambda_y = lambda_y[lambda_y > 1e-5]
            lambda_spy = (lambda_sp.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1)
            pval = liu_sf((hsic_scaled * n_spots).numpy(), lambda_spy.numpy())
        else:  # "trace"
            S = counts.T @ counts  # (1, 1)
            trS = torch.trace(S).item()
            trS2 = torch.trace(S @ S).item()
            n1 = n_spots - 1
            mean_null = trK.item() * trS / n1
            var_null = 2.0 * trK2.item() * trS2 / (n1**2)
            z = (hsic_scaled.item() - mean_null) / (var_null**0.5 + 1e-12)
            pval = float(_norm_dist.sf(z))

        hsic_list.append(hsic_scaled / (n_spots - 1) ** 2)  # HSIC statistic
        pvals_list.append(pval)

    sv_test_results = {
        "statistic": torch.tensor(hsic_list).numpy(),
        "pvalue": torch.tensor(pvals_list).numpy(),
        "method": "hsic-gc",
        "null_method": null_method,
        "n_spots": n_spots,
    }

    # calculate adjusted p-values
    sv_test_results["pvalue_adj"] = false_discovery_control(sv_test_results["pvalue"])

    return sv_test_results
