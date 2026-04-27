"""Data preparation helpers for SPLISOSM workflows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Literal
from scipy.sparse.csgraph import connected_components as _connected_components

import numpy as np
import scipy.sparse
from tqdm import tqdm
import pandas as pd
import torch
from anndata import AnnData
from splisosm.utils._chunking import auto_chunk_size

__all__ = [
    "get_cov_sp",
    "counts_to_ratios",
    "prepare_inputs_from_anndata",
    "add_ratio_layer",
    "extract_counts_n_ratios",
    "compute_feature_summaries",
    "extract_gene_level_statistics",
    "auto_chunk_size",
]


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
    valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
    if transformation not in valid_transformations:
        raise ValueError(
            f"Invalid ratio transformation. Must be one of {valid_transformations}."
        )
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

    valid_nan_filling = ["mean", "none"]
    if nan_filling not in valid_nan_filling:
        raise ValueError(
            f"Invalid NaN filling method. Must be one of {valid_nan_filling}."
        )

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
        y = torch.from_numpy(clr(y.detach().cpu().numpy())).float()  # (n_spots, n_isos)
    elif transformation == "ilr":
        y = torch.from_numpy(
            ilr(y.detach().cpu().numpy())
        ).float()  # (n_spots, n_isos - 1)
    elif transformation == "alr":
        y = torch.from_numpy(
            alr(y.detach().cpu().numpy())
        ).float()  # (n_spots, n_isos - 1)
    elif transformation == "radial":
        y = y / y.norm(dim=1, keepdim=True)  # NaN rows stay NaN

    # post-transform fill (new default) or explicit NaN restore
    if nan_filling == "mean" and not _fill_before:
        if is_nan.any() and (~is_nan).any():
            y[is_nan] = y[~is_nan].mean(0, keepdim=True)
    elif nan_filling == "none":
        y[is_nan] = torch.nan

    return y


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


@dataclass
class _SpatialInputs:
    adata: AnnData
    coordinates: Optional[torch.Tensor]
    adj_matrix: Optional[scipy.sparse.spmatrix]
    design_mtx: Optional[Any]


@dataclass
class _FeatureGroups:
    counts_list: list[torch.Tensor]
    gene_names: list[str]
    iso_names: list[str]


def _validate_prepare_inputs(
    adata: AnnData,
    layer: str,
    group_iso_by: str,
    spatial_key: str,
    adj_key: Optional[str],
    min_counts: int,
    min_bin_pct: float,
    gene_names: Optional[str],
    design_mtx: Optional[Any],
    min_component_size: int,
) -> None:
    """Validate public AnnData preprocessing arguments."""
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
    if spatial_key not in adata.obsm and not (
        adj_key is not None and adj_key in adata.obsp
    ):
        raise ValueError(
            f"Neither `adata.obsm['{spatial_key}']` nor `adata.obsp['{adj_key}']` "
            "is available. Provide spatial coordinates via `spatial_key` and/or "
            "a pre-built adjacency via `adj_key`."
        )


def _resolve_coordinates(
    adata: AnnData,
    spatial_key: str,
    adj_key: Optional[str],
) -> Optional[torch.Tensor]:
    """Return coordinates when present, allowing adjacency-only inputs."""
    if spatial_key in adata.obsm:
        coordinates = torch.as_tensor(
            np.asarray(adata.obsm[spatial_key]), dtype=torch.float32
        )
        if coordinates.dim() != 2:
            raise ValueError(
                "Coordinates in `adata.obsm[spatial_key]` must be a 2D array."
            )
        return coordinates
    if adj_key is not None and adj_key in adata.obsp:
        return None
    raise ValueError(
        f"Neither `adata.obsm['{spatial_key}']` nor `adata.obsp['{adj_key}']` "
        "is available. Provide spatial coordinates via `spatial_key` and/or "
        "a pre-built adjacency via `adj_key`."
    )


def _load_or_build_adjacency(
    adata: AnnData,
    coordinates: Optional[torch.Tensor],
    adj_key: Optional[str],
    min_component_size: int,
    k_neighbors: int,
) -> Optional[scipy.sparse.spmatrix]:
    """Load a supplied adjacency or build one for component filtering."""
    if adj_key is not None:
        if adj_key not in adata.obsp:
            raise ValueError(f"`adj_key` '{adj_key}' was not found in `adata.obsp`.")
        adj_raw = adata.obsp[adj_key]
        adj_out = (
            adj_raw.tocsc()
            if scipy.sparse.issparse(adj_raw)
            else scipy.sparse.csc_matrix(adj_raw)
        )
        if not (adj_out != adj_out.T).nnz == 0:
            adj_out = (adj_out + adj_out.T) / 2
            warnings.warn(
                "Provided adjacency matrix is not symmetric; symmetrising by averaging with its transpose.",
                RuntimeWarning,
                stacklevel=2,
            )
        return adj_out

    if min_component_size <= 1:
        return None
    if coordinates is None:
        raise ValueError(
            "`min_component_size > 1` without `adj_key` requires spatial "
            "coordinates via `spatial_key`.  Either provide `adj_key` or "
            "set `min_component_size=1`."
        )

    from splisosm.kernel import _build_adj_from_coords

    return _build_adj_from_coords(
        coordinates, k_neighbors=k_neighbors, mutual_neighbors=True
    ).tocsc()


def _filter_small_components(
    adata: AnnData,
    coordinates: Optional[torch.Tensor],
    adj_matrix: Optional[scipy.sparse.spmatrix],
    design_mtx: Optional[Any],
    spatial_key: str,
    min_component_size: int,
) -> _SpatialInputs:
    """Filter spots in small graph components from all aligned inputs."""
    if min_component_size <= 1:
        return _SpatialInputs(adata, coordinates, adj_matrix, design_mtx)
    if adj_matrix is None:
        raise ValueError("Internal error: component filtering requires an adjacency.")

    _, labels = _connected_components(adj_matrix, directed=False)
    comp_sizes = np.bincount(labels)
    keep_mask = comp_sizes[labels] >= min_component_size
    n_removed = int((~keep_mask).sum())
    if n_removed == 0:
        return _SpatialInputs(adata, coordinates, adj_matrix, design_mtx)

    n_remaining = int(keep_mask.sum())
    if n_remaining == 0:
        raise ValueError(
            "No spots remained after filtering small graph components. "
            "Lower `min_component_size` or check your coordinate/adjacency matrix."
        )

    adata = adata[keep_mask, :].copy()
    coordinates = (
        torch.as_tensor(np.asarray(adata.obsm[spatial_key]), dtype=torch.float32)
        if coordinates is not None and spatial_key in adata.obsm
        else None
    )
    adj_matrix = adj_matrix[keep_mask][:, keep_mask].tocsc()

    if design_mtx is not None:
        if isinstance(design_mtx, pd.DataFrame):
            design_mtx = design_mtx.iloc[keep_mask].copy()
        elif isinstance(design_mtx, torch.Tensor):
            design_mtx = design_mtx[keep_mask].clone()
        elif scipy.sparse.issparse(design_mtx):
            design_mtx = design_mtx.tocsr()[keep_mask].copy()
        elif isinstance(design_mtx, np.ndarray):
            design_mtx = design_mtx[keep_mask, :].copy()

    warnings.warn(
        f"Removed {n_removed} spot(s) belonging to graph components "
        f"with fewer than {min_component_size} member(s). "
        f"{n_remaining} spot(s) remain.",
        UserWarning,
        stacklevel=2,
    )
    return _SpatialInputs(adata, coordinates, adj_matrix, design_mtx)


def _filtered_feature_table(
    adata: AnnData,
    layer: str,
    group_iso_by: str,
    min_counts: int,
    min_bin_pct: float,
    filter_single_iso_genes: bool,
    gene_names: Optional[str],
) -> tuple[pd.DataFrame, Any, bool]:
    """Return filtered isoform metadata and normalized count storage."""
    iso_counts = adata.layers[layer]
    min_bin_frac = float(min_bin_pct)
    if min_bin_frac > 1.0:
        min_bin_frac /= 100.0

    is_sparse_input = scipy.sparse.issparse(iso_counts)
    if is_sparse_input and not (
        scipy.sparse.isspmatrix_csr(iso_counts)
        or scipy.sparse.isspmatrix_csc(iso_counts)
    ):
        iso_counts = iso_counts.tocsr()

    total_counts = np.asarray(iso_counts.sum(axis=0)).ravel()
    if is_sparse_input:
        bin_pct = iso_counts.getnnz(axis=0).astype(np.float64) / adata.n_obs
    else:
        counts_arr_for_stats = np.asarray(iso_counts)
        bin_pct = np.count_nonzero(counts_arr_for_stats, axis=0) / adata.n_obs

    needed_var_cols = [group_iso_by]
    if gene_names is not None:
        if gene_names not in adata.var.columns:
            raise ValueError(
                f"`gene_names` column `{gene_names}` was not found in `adata.var`."
            )
        if gene_names != group_iso_by:
            needed_var_cols.append(gene_names)

    var_df = adata.var[needed_var_cols].copy()
    var_df["__total_counts__"] = total_counts
    var_df["__bin_pct__"] = bin_pct
    var_df = var_df[
        (var_df["__total_counts__"] >= min_counts)
        & (var_df["__bin_pct__"] >= min_bin_frac)
    ]
    if var_df.shape[0] == 0:
        raise ValueError(
            "No features remained after applying `min_counts`/`min_bin_pct` filtering."
        )

    if filter_single_iso_genes:
        n_iso_per_gene = var_df.groupby(group_iso_by, observed=True).size()
        keep_genes = n_iso_per_gene[n_iso_per_gene >= 2].index
        var_df = var_df[var_df[group_iso_by].isin(keep_genes)]
        if var_df.shape[0] == 0:
            raise ValueError("No genes with >=2 isoforms remained after filtering.")

    return var_df, iso_counts, is_sparse_input


def _build_feature_groups(
    adata: AnnData,
    iso_counts: Any,
    var_df: pd.DataFrame,
    group_iso_by: str,
    gene_names: Optional[str],
    is_sparse_input: bool,
) -> _FeatureGroups:
    """Build per-gene torch count tensors without repeated full-matrix slicing."""
    iso_groups = list(var_df.groupby(group_iso_by, observed=True, sort=False))
    if len(iso_groups) == 0:
        raise ValueError("No genes remained after extracting isoform counts.")

    all_iso_names_flat: list[str] = []
    gene_slice_offsets: list[tuple[int, int]] = []
    resolved_gene_names: list[str] = []
    pos = 0
    for gene_id, group in iso_groups:
        iso_names = group.index.astype(str).tolist()
        all_iso_names_flat.extend(iso_names)
        gene_slice_offsets.append((pos, pos + len(iso_names)))
        pos += len(iso_names)
        resolved_gene_names.append(
            str(group[gene_names].iloc[0]) if gene_names is not None else str(gene_id)
        )

    var_index = pd.Index(adata.var_names)
    all_iso_indices = var_index.get_indexer(all_iso_names_flat)
    if (all_iso_indices < 0).any():
        missing = [n for n, idx in zip(all_iso_names_flat, all_iso_indices) if idx < 0]
        raise ValueError(
            f"Could not locate {len(missing)} isoform name(s) in adata.var_names "
            f"(first few: {missing[:5]}). This indicates an internal inconsistency."
        )

    if is_sparse_input:
        sub_csc = iso_counts[:, all_iso_indices].tocsc().astype(np.float32, copy=False)
    else:
        sub_arr = np.ascontiguousarray(iso_counts[:, all_iso_indices], dtype=np.float32)
        sub_tensor = torch.from_numpy(sub_arr)

    counts_list: list[torch.Tensor] = []
    for start, end in gene_slice_offsets:
        if is_sparse_input:
            coo = sub_csc[:, start:end].tocoo()
            indices = torch.from_numpy(
                np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
            )
            values = torch.from_numpy(coo.data)
            with torch.sparse.check_sparse_tensor_invariants(False):
                counts = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))
        else:
            counts = sub_tensor[:, start:end]
        counts_list.append(counts)

    return _FeatureGroups(counts_list, resolved_gene_names, all_iso_names_flat)


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
    Optional[torch.Tensor],
    list[str],
    Optional[Any],
    Optional[list[str]],
    "scipy.sparse.spmatrix | None",
    "AnnData | None",
]:
    """Extract and filter isoform count tensors from an AnnData object.

    Shared helper used by both :class:`splisosm.hyptest.np.SplisosmNP` and
    :class:`splisosm.hyptest.glmm.SplisosmGLMM` to prepare legacy-compatible
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
        Key in ``adata.obsm`` for spatial coordinates.  Optional when
        ``adj_key`` is provided: if the key is missing from ``adata.obsm`` the
        returned ``coordinates`` is ``None`` and downstream kernel construction
        proceeds from the adjacency matrix alone.  Raw coordinates are still
        required by SPARK-X and the GP-conditional DU test; those callers
        raise a dedicated error at call time when coordinates are absent.
    adj_key
        Key in ``adata.obsp`` for a pre-built adjacency matrix.
        When provided, it overrides the k-NN graph construction
        from coordinates and is used directly to build the spatial kernel.
        The adjacency matrix is symmetrized internally before being returned.
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
        tensor, array, sparse matrix, DataFrame of shape
        ``(n_spots, n_factors)``, a single obs-column name, or a list of
        obs-column names.
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
    coordinates : torch.Tensor or None
        Spatial coordinates with shape ``(n_spots, n_spatial_dims)`` and dtype
        float32. ``None`` when ``spatial_key`` is missing from ``adata.obsm``
        and ``adj_key`` supplies the neighborhood graph instead.
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
    _validate_prepare_inputs(
        adata=adata,
        layer=layer,
        group_iso_by=group_iso_by,
        spatial_key=spatial_key,
        adj_key=adj_key,
        min_counts=min_counts,
        min_bin_pct=min_bin_pct,
        gene_names=gene_names,
        design_mtx=design_mtx,
        min_component_size=min_component_size,
    )

    coordinates = _resolve_coordinates(adata, spatial_key, adj_key)
    adj_matrix = _load_or_build_adjacency(
        adata=adata,
        coordinates=coordinates,
        adj_key=adj_key,
        min_component_size=min_component_size,
        k_neighbors=k_neighbors,
    )
    spatial_inputs = _filter_small_components(
        adata=adata,
        coordinates=coordinates,
        adj_matrix=adj_matrix,
        design_mtx=design_mtx,
        spatial_key=spatial_key,
        min_component_size=min_component_size,
    )
    adata = spatial_inputs.adata
    coordinates = spatial_inputs.coordinates
    adj_matrix = spatial_inputs.adj_matrix
    design_mtx = spatial_inputs.design_mtx

    var_df, iso_counts, is_sparse_input = _filtered_feature_table(
        adata=adata,
        layer=layer,
        group_iso_by=group_iso_by,
        min_counts=min_counts,
        min_bin_pct=min_bin_pct,
        filter_single_iso_genes=filter_single_iso_genes,
        gene_names=gene_names,
    )
    features = _build_feature_groups(
        adata=adata,
        iso_counts=iso_counts,
        var_df=var_df,
        group_iso_by=group_iso_by,
        gene_names=gene_names,
        is_sparse_input=is_sparse_input,
    )

    resolved_design, resolved_covariates = _process_design_mtx(
        adata, design_mtx, covariate_names
    )
    filtered_adata = (
        adata[:, features.iso_names].copy() if return_filtered_anndata else None
    )

    return (
        features.counts_list,
        coordinates,
        features.gene_names,
        resolved_design,
        resolved_covariates,
        adj_matrix,
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


def compute_feature_summaries(
    adata: AnnData,
    gene_names: list[str],
    layer: str = "counts",
    group_iso_by: str = "gene_symbol",
    print_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute gene-level and isoform-level summary statistics.

    This is the shared implementation behind
    :meth:`~splisosm.SplisosmNP.extract_feature_summary`,
    :meth:`~splisosm.SplisosmFFT.extract_feature_summary`, and
    :meth:`~splisosm.SplisosmGLMM.extract_feature_summary`.

    Parameters
    ----------
    adata
        Annotated data matrix (typically the filtered AnnData from ``setup_data``).
    gene_names
        Ordered list of gene display names (length ``n_genes``).
    layer
        Layer in ``adata.layers`` containing raw isoform counts.
    group_iso_by
        Column in ``adata.var`` that groups isoforms by gene.
    print_progress
        Show a tqdm progress bar.

    Returns
    -------
    gene_summary : pandas.DataFrame
        Indexed by gene name with columns: ``n_isos``, ``perplexity``,
        ``pct_bin_on``, ``count_avg``, ``count_std``, ``major_ratio_avg``.
    isoform_summary : pandas.DataFrame
        Indexed by isoform name with original ``adata.var`` columns plus:
        ``pct_bin_on``, ``count_total``, ``count_avg``, ``count_std``,
        ``ratio_total``, ``ratio_avg``, ``ratio_std``.
    """
    iso_counts = adata.layers[layer]
    n_bins = iso_counts.shape[0]
    is_sparse = scipy.sparse.issparse(iso_counts)

    if is_sparse:
        if not scipy.sparse.isspmatrix_csc(iso_counts):
            iso_counts = iso_counts.tocsc()
    else:
        iso_counts = np.asarray(iso_counts, dtype=float)

    iso_groups = list(adata.var.groupby(group_iso_by, observed=True, sort=False))

    # gene_names may be display names (e.g. 'Gnai3') while groupby keys are
    # raw identifiers (e.g. 'ENSMUSG00000000001').  We only require that the
    # *count* matches — the order is guaranteed by construction (both come
    # from the same filtered, sorted adata.var).
    if len(iso_groups) != len(gene_names):
        raise ValueError(
            f"gene_names length ({len(gene_names)}) does not match the number "
            f"of gene groups ({len(iso_groups)}) in adata.var['{group_iso_by}']."
        )

    gene_rows: list[dict] = []
    iso_rows: list[dict] = []
    all_iso_names: list[str] = []

    iterator = tqdm(
        zip(gene_names, iso_groups),
        desc="Genes",
        total=len(gene_names),
        disable=not print_progress,
    )

    for gene_name, (_, iso_group_df) in iterator:
        iso_names = iso_group_df.index.tolist()
        iso_idx = adata.var_names.get_indexer(iso_names)

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
        iso_count_var = np.maximum((iso_sumsq / n_bins) - np.square(iso_count_avg), 0.0)
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
                ratio_sum = np.asarray(ratio_counts.sum(axis=0), dtype=float).ravel()
                ratio_sumsq = np.asarray(
                    ratio_counts.power(2).sum(axis=0), dtype=float
                ).ravel()
            else:
                ratio_counts = gene_counts[valid_rows] / row_sums[valid_rows, None]
                ratio_sum = ratio_counts.sum(axis=0)
                ratio_sumsq = np.square(ratio_counts).sum(axis=0)

            ratio_avg = ratio_sum / n_valid
            ratio_var = np.maximum((ratio_sumsq / n_valid) - np.square(ratio_avg), 0.0)
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
                "major_ratio_avg": (
                    float(ratio_total.max()) if ratio_total.size > 0 else 0.0
                ),
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

    gene_summary = pd.DataFrame(gene_rows).set_index("gene")
    var_df = adata.var.loc[all_iso_names].copy()
    stats_df = pd.DataFrame(iso_rows).set_index("isoform")
    isoform_summary = pd.concat([var_df, stats_df], axis=1)

    return gene_summary, isoform_summary


def extract_gene_level_statistics(
    adata: AnnData, layer: str = "counts", group_iso_by: str = "gene_symbol"
) -> pd.DataFrame:
    """Extract gene-level metadata from isoform-level counts anndata.

    .. deprecated::
        Use :func:`compute_feature_summaries` instead, which returns both
        gene-level and isoform-level statistics.

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
        Gene-level metadata.
    """
    warnings.warn(
        "extract_gene_level_statistics is deprecated; use compute_feature_summaries() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    gene_names = list(
        adata.var.groupby(group_iso_by, observed=True, sort=False).groups.keys()
    )
    gene_df, iso_df = compute_feature_summaries(
        adata, gene_names, layer=layer, group_iso_by=group_iso_by, print_progress=True
    )
    # Remap column names for backward compatibility
    gene_df = gene_df.rename(columns={"n_isos": "n_iso", "pct_bin_on": "pct_spot_on"})
    return gene_df
