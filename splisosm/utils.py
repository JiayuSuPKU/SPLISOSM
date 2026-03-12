from __future__ import annotations

import warnings
import json
from typing import Any, Optional, Literal
import re

from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse
from tqdm import tqdm
import pandas as pd
import torch
from matplotlib.image import imread
from anndata import AnnData
from smoother import SpatialWeightMatrix, SpatialLoss
from splisosm.likelihood import liu_sf

__all__ = [
    "get_cov_sp",
    "counts_to_ratios",
    "false_discovery_control",
    "load_visium_sp_meta",
    "load_visiumhd_spatialdata",
    "extract_counts_n_ratios",
    "extract_gene_level_statistics",
    "run_sparkx",
    "run_hsic_gc",
]


def get_cov_sp(
    coords: np.ndarray | torch.Tensor, k: int = 4, rho: float = 0.99
) -> torch.Tensor:
    """Wrapper function to get the spatial covariance matrix from spatial coordinates.

    It will first construct a mutual-k-nearest neighbor graph from the euclidean spatial coordinates,
    then convert the adjacency matrix to a standardized spatial covariance matrix using the
    intrinsic conditional autoregressive (ICAR) model with spatial autocorrelation coefficient rho.
    See :cite:`su2023smoother` for details.

    Parameters
    ----------
    coords
        Shape (n_spots, n_dims). Euclidean spatial coordinates of spots.
    k
        Number of nearest neighbors.
    rho
        Spatial autocorrelation coefficient.

    Returns
    -------
    cov_sp : torch.Tensor
        Shape (n_spots, n_spots). Spatial covariance matrix with standardized variance (== 1).
    """
    # first calculate the spatial weights matrix (swm)
    # here swm is the binary adjacency matrix of the knn graph
    weights = SpatialWeightMatrix()
    weights.calc_weights_knn(coords, k=k, verbose=False)

    # # convert the swm to spatial covariance matrix with standardized variance (== 1)
    # spatial_loss = SpatialLoss("icar", weights, rho=rho, standardize_cov=True)
    # cov_sp = torch.cholesky_inverse(torch.linalg.cholesky(spatial_loss.inv_cov[0].to_dense())) # n_spots x n_spots
    spatial_loss = SpatialLoss("icar", weights, rho=rho, standardize_cov=False)
    cov_sp = torch.cholesky_inverse(
        torch.linalg.cholesky(spatial_loss.inv_cov[0].to_dense())
    )  # n_spots x n_spots
    inv_sds = torch.diagflat(torch.diagonal(cov_sp) ** (-0.5))
    cov_sp = inv_sds @ cov_sp @ inv_sds

    return cov_sp


def counts_to_ratios(
    counts: np.ndarray | torch.Tensor,
    transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
    nan_filling: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Convert isoform counts to proportions.

    By default, isoform ratios at zero-coverage spots are filled with the mean ratio per isoform across all spots.
    After conversion, the isoform ratios can be further transformed using log-ratio-based transformations
    (clr, ilr, alr) or radial transformation :cite:`park2022kernel`.

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
        ``'mean'``: fill all-zero rows with the mean of the mean per column **before transformation**.
        ``'none'``: do not fill rows and return NaNs at all-zero rows.

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
                f"Please install scikit-bio to use ratio transformation='{transformation}'. Switching to 'none'."
            )
            transformation = "none"

    assert nan_filling in ["mean", "none"]

    if isinstance(counts, np.ndarray):
        counts = torch.from_numpy(counts).float()

    # identify zero rows to fill
    is_nan = counts.sum(1) == 0  # (n_spots,)

    # calculate isoform ratios
    if transformation in ["clr", "ilr", "alr"]:
        # add pseudocounts equal to 1% of the global mean per isoform to avoid zeros in the ratio
        y = (1 - 1e-2) * counts + 1e-2 * counts.mean(0, keepdim=True)
        y = y / y.sum(1, keepdim=True)  # isoform ratio without nans and zeros
    else:
        y = counts / counts.sum(1, keepdim=True)  # isoform ratio with nans
        # fill nan values with the mean ratio per column (isoform)
        if nan_filling == "mean":
            y[is_nan] = y[~is_nan].mean(0, keepdim=True)

    # apply transformation
    if transformation == "clr":
        y = torch.from_numpy(clr(y)).float()  # (n_spots, n_isos)
    elif transformation == "ilr":
        y = torch.from_numpy(ilr(y)).float()  # (n_spots, n_isos - 1)
    elif transformation == "alr":
        y = torch.from_numpy(alr(y)).float()  # (n_spots, n_isos - 1)
    elif transformation == "radial":
        y = y / y.norm(dim=1, keepdim=True)  # radial transformation with nans

    # fill back nan to rows with zero counts if needed
    if nan_filling == "none":
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
        warnings.warn("NaNs encountered in p-values. These will be ignored.")
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


# Similar to scanpy.read_visium
# https://github.com/scverse/scanpy/blob/main/scanpy/readwrite.py#L356-L512
def load_visium_sp_meta(
    adata: AnnData, path_to_spatial: str | Path, library_id: Optional[str] = None
) -> AnnData:
    """Helper function to load Visium spatial metadata.

    Parameters
    ----------
    adata
        Annotated data matrix to store the spatial metadata.
    path_to_spatial
        Path to the `spatial` folder generated by Space Ranger.
    library_id
        Library ID of the spatial data.

    Returns
    -------
    anndata : anndata.AnnData
        AnnData with spatial metadata.
    """
    if library_id is None:  # default library_id
        library_id = "library_id"

    adata.uns["spatial"] = dict()
    adata.uns["spatial"][library_id] = dict()

    path = Path(path_to_spatial)
    tissue_positions_file = (
        path / "tissue_positions.csv"
        if (path / "tissue_positions.csv").exists()
        else path / "tissue_positions_list.csv"
    )

    files = dict(
        tissue_positions_file=tissue_positions_file,
        scalefactors_json_file=path / "scalefactors_json.json",
        hires_image=path / "tissue_hires_image.png",
        lowres_image=path / "tissue_lowres_image.png",
    )

    # load images
    adata.uns["spatial"][library_id]["images"] = dict()
    for res in ["hires", "lowres"]:
        try:
            adata.uns["spatial"][library_id]["images"][res] = imread(
                str(files[f"{res}_image"])
            )
        except Exception:
            warnings.warn(
                f"Missing '{res}' image in {path_to_spatial}. Will be ignored."
            )
            adata.uns["spatial"][library_id]["images"][res] = None

    # read json scalefactors
    with open(files["scalefactors_json_file"]) as f:
        adata.uns["spatial"][library_id]["scalefactors"] = json.load(f)

    # read coordinates
    positions = pd.read_csv(
        files["tissue_positions_file"],
        header=0,
        index_col=0,
    )
    positions.columns = [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]

    # add coordinates to spot metadata
    adata.obs = adata.obs.join(positions, how="left")
    adata.obsm["spatial"] = adata.obs[
        ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    ].to_numpy()
    adata.obs.drop(
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
        inplace=True,
    )

    return adata


def _normalize_visiumhd_bin_name(bin_size: int | str) -> str:
    """Normalize Visium HD bin identifier to ``square_XXXum`` format."""
    if isinstance(bin_size, int):
        return f"square_{bin_size:03}um"

    bin_size_str = str(bin_size)
    if bin_size_str.startswith("square_") and bin_size_str.endswith("um"):
        return bin_size_str

    if re.fullmatch(r"\d+", bin_size_str):
        return f"square_{int(bin_size_str):03}um"

    raise ValueError(
        f"Unsupported bin size format: {bin_size!r}. Use int (e.g. 8) or 'square_008um'."
    )


def _aggregate_visiumhd_probe_counts(
    adata_2um: AnnData,
    barcode_mappings: pd.DataFrame,
    source_col: str,
    target_col: str,
    target_obs_names: pd.Index,
) -> scipy.sparse.csr_matrix:
    """Aggregate 2um probe counts into target bins according to mapping."""
    mapping_pairs = (
        barcode_mappings[[source_col, target_col]].dropna().drop_duplicates()
    )
    src_barcode = mapping_pairs[source_col].astype(str).to_numpy()
    tgt_barcode = mapping_pairs[target_col].astype(str).to_numpy()

    src_idx = adata_2um.obs_names.get_indexer(src_barcode)
    tgt_index = pd.Index(target_obs_names.astype(str))
    tgt_idx = tgt_index.get_indexer(tgt_barcode)

    valid = (src_idx >= 0) & (tgt_idx >= 0)
    if valid.sum() == 0:
        return scipy.sparse.csr_matrix(
            (len(tgt_index), adata_2um.n_vars), dtype=np.float32
        )

    aggregation = scipy.sparse.csr_matrix(
        (
            np.ones(valid.sum(), dtype=np.float32),
            (src_idx[valid], tgt_idx[valid]),
        ),
        shape=(adata_2um.n_obs, len(tgt_index)),
        dtype=np.float32,
    )

    x_2um = adata_2um.X
    if scipy.sparse.issparse(x_2um):
        x_2um = x_2um.tocsr()
    else:
        x_2um = scipy.sparse.csr_matrix(np.asarray(x_2um, dtype=np.float32))

    # (n_obs_2um, n_vars) -> (n_obs_target, n_vars)
    x_target = (x_2um.T @ aggregation).T
    return x_target.tocsr()


def load_visiumhd_spatialdata(
    path: str | Path,
    dataset_id: Optional[str] = None,
    bin_sizes: Optional[list[int | str]] = None,
    bins_as_squares: bool = True,
    fullres_image_file: Optional[str | Path] = None,
    load_all_images: bool = False,
    var_names_make_unique: bool = True,
    filtered_counts_file: bool = True,
    counts_layer_name: str = "counts",
) -> Any:
    """Load Visium HD outputs as SpatialData with probe-level binned tables.

    This wrapper uses ``binned_outputs/square_002um/raw_probe_bc_matrix.h5`` as the
    source matrix and aggregates counts to coarser bins (for example ``square_008um``
    and ``square_016um``) using ``barcode_mappings.parquet``.

    Parameters
    ----------
    path
        Path to Space Ranger ``outs`` directory for Visium HD.
    dataset_id
        Optional dataset ID passed to the SpatialData reader.
    bin_sizes
        Bin resolutions to include. Each entry can be ``int`` (for example ``8``)
        or Visium HD bin string (for example ``"square_008um"``). If ``None``,
        all available ``square_*um`` bins under ``binned_outputs`` are used.
    bins_as_squares
        Whether bins are represented as squares when loading shapes.
    fullres_image_file
        Path to the full-resolution image.
    load_all_images
        Whether to load all optional images via ``spatialdata_io`` reader.
    var_names_make_unique
        Whether to call ``var_names_make_unique()`` on probe table variables.
    filtered_counts_file
        Whether to keep only in-tissue 2um barcodes prior to aggregation.
        If ``True``, barcodes are taken from the source bin table loaded by
        ``visium_hd`` (``square_002um``). If unavailable, the function falls
        back to ``binned_outputs/square_002um/filtered_feature_bc_matrix.h5``.
    counts_layer_name
        Layer name used to store aggregated probe counts in each output table.

    Returns
    -------
    SpatialData
        A SpatialData object with probe-level tables for requested bins.

    Raises
    ------
    ImportError
        If required optional dependencies are not installed.
    ValueError
        If required files or requested bins are missing.
    """
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for load_visiumhd_spatialdata(). Install it via `pip install scanpy`."
        ) from e

    try:
        from spatialdata_io.readers.visium_hd import visium_hd
    except ImportError as e:
        raise ImportError(
            "spatialdata-io is required for load_visiumhd_spatialdata(). Install it via `pip install spatialdata-io`."
        ) from e

    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    binned_outputs = path / "binned_outputs"
    if not binned_outputs.exists():
        raise ValueError(f"Cannot find binned_outputs under: {path}")

    available_bins = sorted(
        [
            p.name
            for p in binned_outputs.iterdir()
            if p.is_dir() and p.name.startswith("square_")
        ]
    )
    if len(available_bins) == 0:
        raise ValueError(f"No square_*um bins found under: {binned_outputs}")

    if bin_sizes is None:
        requested_bins = available_bins
    else:
        requested_bins = [_normalize_visiumhd_bin_name(b) for b in bin_sizes]
        missing_bins = [b for b in requested_bins if b not in available_bins]
        if missing_bins:
            raise ValueError(
                f"Requested bins not found: {missing_bins}. Available bins: {available_bins}"
            )

    source_bin = "square_002um"
    if source_bin not in available_bins:
        raise ValueError("square_002um is required but not found in binned_outputs.")

    raw_probe_path = binned_outputs / source_bin / "raw_probe_bc_matrix.h5"
    if not raw_probe_path.exists():
        raise ValueError(f"Cannot find raw probe matrix at: {raw_probe_path}")

    mapping_path = path / "barcode_mappings.parquet"
    if not mapping_path.exists():
        # in case of file with a prefix, search recursively for the first match
        mapping_candidates = list(path.glob("*barcode_mappings.parquet"))
        if len(mapping_candidates) == 0:
            raise ValueError(f"Cannot find barcode_mappings.parquet under: {path}")
        mapping_path = mapping_candidates[0]

    # Load SpatialData geometry/images/tables template from Space Ranger outputs.
    requested_bins_um = [int(re.search(r"(\d{3})", b).group(1)) for b in requested_bins]
    sdata = visium_hd(
        path=path,
        dataset_id=dataset_id,
        filtered_counts_file=filtered_counts_file,
        load_segmentations_only=False,
        load_nucleus_segmentations=False,
        bin_size=requested_bins_um,
        bins_as_squares=bins_as_squares,
        annotate_table_by_labels=False,
        fullres_image_file=fullres_image_file,
        load_all_images=load_all_images,
        var_names_make_unique=False,
    )

    # Load 2um raw probe matrix and mapping.
    adata_2um = sc.read_10x_h5(raw_probe_path, gex_only=False)
    adata_2um.obs_names = adata_2um.obs_names.astype(str)
    if var_names_make_unique:
        adata_2um.var_names_make_unique()

    if filtered_counts_file:
        tissue_barcodes: pd.Index | None = None

        if source_bin in sdata.tables:
            tissue_barcodes = pd.Index(sdata.tables[source_bin].obs_names.astype(str))
        else:
            filtered_counts_path = (
                binned_outputs / source_bin / "filtered_feature_bc_matrix.h5"
            )
            if not filtered_counts_path.exists():
                raise ValueError(
                    "filtered_counts_file=True but no in-tissue barcode source was found. "
                    f"Expected table '{source_bin}' in SpatialData or file: {filtered_counts_path}"
                )
            adata_filtered = sc.read_10x_h5(filtered_counts_path, gex_only=False)
            tissue_barcodes = pd.Index(adata_filtered.obs_names.astype(str))

        in_tissue_mask = adata_2um.obs_names.isin(tissue_barcodes)
        adata_2um = adata_2um[in_tissue_mask].copy()
        if adata_2um.n_obs == 0:
            raise ValueError(
                "No in-tissue 2um barcodes remained after filtering raw_probe_bc_matrix.h5."
            )

    mapping_cols = [source_bin, *requested_bins]
    barcode_mappings = pd.read_parquet(mapping_path)
    for col in mapping_cols:
        if col not in barcode_mappings.columns:
            raise ValueError(
                f"Column '{col}' not found in barcode mappings: {mapping_path}"
            )
        if barcode_mappings[col].dtype == object:
            barcode_mappings[col] = barcode_mappings[col].map(
                lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )

    # Replace per-bin tables with probe-level aggregated counts.
    for bin_name in requested_bins:
        if bin_name not in sdata.tables:
            warnings.warn(
                f"Bin table '{bin_name}' not found in SpatialData object; skipping.",
                UserWarning,
            )
            continue

        template_table = sdata.tables[bin_name]

        if bin_name == source_bin:
            # The 2um table is the source itself; reindex to match template obs order.
            reindex = adata_2um.obs_names.get_indexer(
                template_table.obs_names.astype(str)
            )
            missing = reindex == -1
            if missing.any():
                x_source = (
                    adata_2um.X.tocsr()
                    if scipy.sparse.issparse(adata_2um.X)
                    else scipy.sparse.csr_matrix(
                        np.asarray(adata_2um.X, dtype=np.float32)
                    )
                )
                rows = np.where(~missing, reindex, 0)
                x_target = x_source[rows]
                if missing.any():
                    x_target = x_target.tolil()
                    x_target[missing] = 0
                    x_target = x_target.tocsr()
            else:
                x_source = adata_2um.X
                if scipy.sparse.issparse(x_source):
                    x_target = x_source.tocsr()[reindex]
                else:
                    x_target = scipy.sparse.csr_matrix(
                        np.asarray(x_source, dtype=np.float32)[reindex]
                    )
        else:
            x_target = _aggregate_visiumhd_probe_counts(
                adata_2um=adata_2um,
                barcode_mappings=barcode_mappings,
                source_col=source_bin,
                target_col=bin_name,
                target_obs_names=template_table.obs_names,
            )

        probe_table = AnnData(
            X=x_target,
            obs=template_table.obs.copy(),
            var=adata_2um.var.copy(),
        )

        for key, value in template_table.obsm.items():
            probe_table.obsm[key] = value.copy() if hasattr(value, "copy") else value
        for key, value in template_table.uns.items():
            probe_table.uns[key] = value.copy() if hasattr(value, "copy") else value

        probe_table.layers[counts_layer_name] = (
            probe_table.X.copy() if hasattr(probe_table.X, "copy") else probe_table.X
        )
        sdata.tables[bin_name] = probe_table

    return sdata


def extract_counts_n_ratios(
    adata: AnnData,
    layer: str = "counts",
    group_iso_by: str = "gene_symbol",
    return_sparse: bool = False,
    filter_single_iso_genes: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], Optional[np.ndarray]]:
    """Extract per-gene lists of isoform counts and ratios from anndata.

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

    for _gene, _group in tqdm(
        adata.var.reset_index().groupby(group_iso_by, observed=True)
    ):
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
    for _gene, _group in tqdm(
        adata.var.reset_index().groupby(group_iso_by, observed=True)
    ):
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
    counts_gene: np.ndarray | torch.Tensor,
    coordinates: np.ndarray | torch.Tensor,
    approx_rank: Optional[int] = None,
    **spatial_kernel_kwargs: Any,
) -> dict[str, Any]:
    """Function to compute HSIC-GC statistic for gene-level counts.

    This function is designed to be a plugin replacement for SPARK-X.

    Parameters
    ----------
    counts_gene
        Shape (n_spots, n_genes). Gene counts.
    coordinates
        Shape (n_spots, 2). Spatial coordinates of spots.
    approx_rank
        Approximate rank of the spatial kernel matrix.
    **spatial_kernel_kwargs
        Additional arguments for SpatialCovKernel.

    Returns
    -------
    dict
        Results of the HSIC-GC spatial variability test with keys:

        - ``'statistic'``: np.ndarray of shape (n_genes,). HSIC-GC statistics.
        - ``'pvalue'``: np.ndarray of shape (n_genes,). P-values.
        - ``'pvalue_adj'``: np.ndarray of shape (n_genes,). Adjusted p-values.
        - ``'method'``: str. Method name "hsic-gc".
    """

    from splisosm.kernel import SpatialCovKernel

    n_spots = counts_gene.shape[0]
    n_genes = counts_gene.shape[1]

    # determine the maximum rank for spatial kernel computation
    if n_spots > 5000:
        # 10x Visium has 4992 spots per slide. For larger datasets (i.e. Slideseq-V2),
        # it is recommended to use low-rank approximation
        max_rank = np.ceil(np.sqrt(n_spots) * 4).astype(int)
        approx_rank = (
            min(approx_rank, max_rank) if approx_rank is not None else max_rank
        )
    else:
        if approx_rank is not None:
            approx_rank = approx_rank if approx_rank < n_spots else None

    # set default spatial kernel kwargs
    default_spatial_kernel_kwargs = {
        "k_neighbors": 4,
        "model": "icar",
        "rho": 0.99,
        "standardize_cov": True,
        "centering": True,
        "approx_rank": approx_rank,
    }
    if spatial_kernel_kwargs is not None:
        if "centering" in spatial_kernel_kwargs:
            warnings.warn(
                "The 'centering' argument in spatial_kernel_kwargs will be ignored. It is always set to True for HSIC-GC."
            )
            spatial_kernel_kwargs.pop("centering")
        default_spatial_kernel_kwargs.update(spatial_kernel_kwargs)

    # compute the spatial kernel
    # Ensure coordinates is numpy for SpatialCovKernel/smoother
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()

    K_sp = SpatialCovKernel(coordinates, **default_spatial_kernel_kwargs)

    # get the eigenvalues
    lambda_sp = K_sp.eigenvalues()  # (rank,)
    lambda_sp = lambda_sp[lambda_sp > 1e-5]  # filter small eigenvalues

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
    for i in tqdm(range(n_genes)):
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

        lambda_y = counts.T @ counts  # (1, 1)
        lambda_spy = lambda_sp * lambda_y  # (rank, 1)
        hsic_scaled = torch.trace(K_sp.xtKx(counts))  # scalar
        pval = liu_sf((hsic_scaled * n_spots).numpy(), lambda_spy.numpy())

        hsic_list.append(hsic_scaled / (n_spots - 1) ** 2)  # HSIC statistic
        pvals_list.append(pval)

    sv_test_results = {
        "statistic": torch.tensor(hsic_list).numpy(),
        "pvalue": torch.tensor(pvals_list).numpy(),
        "method": "hsic-gc",
    }

    # calculate adjusted p-values
    sv_test_results["pvalue_adj"] = false_discovery_control(sv_test_results["pvalue"])

    return sv_test_results
