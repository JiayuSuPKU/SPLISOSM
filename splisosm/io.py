"""I/O loaders for SpatialData-based workflows.

This module provides high-level wrappers for constructing SpatialData objects
from platform-specific outputs.
"""

from __future__ import annotations
import re
import warnings
from pathlib import Path
from typing import Any, Optional
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from tqdm import tqdm

__all__ = [
    "load_visiumhd_probe",
    "load_xenium_codeword",
]


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
    # Table mapping 2um barcodes to 8um and 16um bins is available as
    # `barcode_mappings.parquet` in the Space Ranger output
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

    # Define a sparse aggregation matrix that maps 2um barcodes to target bins
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


def load_visiumhd_probe(
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
    spatialdata.SpatialData
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
            "scanpy is required for load_visiumhd_probe(). Install it via `pip install scanpy`."
        ) from e

    try:
        from spatialdata_io.readers.visium_hd import visium_hd
    except ImportError as e:
        raise ImportError(
            "spatialdata-io is required for load_visiumhd_probe(). Install it via `pip install spatialdata-io`."
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
        mapping_candidates = list(path.glob("*barcode_mappings.parquet"))
        if len(mapping_candidates) == 0:
            raise ValueError(f"Cannot find barcode_mappings.parquet under: {path}")
        mapping_path = mapping_candidates[0]

    # Call visium_hd() to load the SpatialData object with requested bins
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

    # Load probe-level 2um counts and filter to in-tissue barcodes if requested
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

    # Aggregate 2um probe counts into requested resolution bins
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

    for bin_name in requested_bins:
        if bin_name not in sdata.tables:
            warnings.warn(
                f"Bin table '{bin_name}' not found in SpatialData object; skipping.",
                UserWarning,
            )
            continue

        template_table = sdata.tables[bin_name]

        if bin_name == source_bin:
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

        # Save aggregated counts into a new table (AnnData) in the SpatialData object
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


def _xenium_locate_outs_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")

    outs_path = p / "outs"
    if (outs_path / "transcripts.zarr.zip").exists():
        return outs_path
    if (p / "transcripts.zarr.zip").exists():
        return p

    raise ValueError(
        "Could not locate transcripts.zarr.zip. Expected one of: "
        f"{outs_path / 'transcripts.zarr.zip'} or {p / 'transcripts.zarr.zip'}."
    )


def _compute_xenium_bins_from_meta(
    grid_origin: dict[str, float],
    native_grid_size: Sequence[float],
    native_nrows: int,
    native_ncols: int,
    spatial_resolution: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Bin X/Y coordinates into square bins at requested resolution."""
    x_min = float(grid_origin["x"])
    y_min = float(grid_origin["y"])
    x_max = x_min + int(native_ncols) * float(native_grid_size[0])
    y_max = y_min + int(native_nrows) * float(native_grid_size[1])

    ncols = int(np.ceil((x_max - x_min) / spatial_resolution))
    nrows = int(np.ceil((y_max - y_min) / spatial_resolution))
    x_bins = np.linspace(
        x_min, x_min + ncols * spatial_resolution, ncols + 1, dtype=np.float64
    )
    y_bins = np.linspace(
        y_min, y_min + nrows * spatial_resolution, nrows + 1, dtype=np.float64
    )
    return x_bins, y_bins, nrows, ncols


def _chunk_to_codeword_triplets(
    chunk: Any,
    codeword_to_target_col: np.ndarray,
    quality_threshold: float,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    nrows: int,
    ncols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For chunk data structure, see Xenium Onboard Analysis (XOA) output Zarr format documentation:
    # https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr#transcripts
    # `codeword_to_target_col` maps a raw Xenium codeword ID to the output matrix
    # column index. Non-target codewords are encoded as -1. This avoids repeated
    # `np.isin()` checks and dictionary lookups for every transcript in every chunk
    q_scores = np.asarray(chunk["quality_score"][:]).ravel()
    codeword_ids = np.asarray(chunk["codeword_identity"][:])
    if codeword_ids.ndim > 1:
        codeword_ids = codeword_ids[:, 0]

    # Transcript locations are stored as (x, y, z) coordinates in microns. We only need x and y
    locs = np.asarray(chunk["location"][:])
    if locs.ndim != 2 or locs.shape[1] < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    locs_xy = locs[:, :2]

    quality_mask = q_scores >= quality_threshold
    if not np.any(quality_mask):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    codeword_ids = codeword_ids[quality_mask]
    locs_xy = locs_xy[quality_mask]

    in_range = (codeword_ids >= 0) & (codeword_ids < codeword_to_target_col.shape[0])
    if not np.any(in_range):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    codeword_ids = codeword_ids[in_range]
    locs_xy = locs_xy[in_range]

    # Map transcript id to target column index. Non-target codewords are encoded as -1
    target_cols = codeword_to_target_col[codeword_ids.astype(np.int64)]
    in_target = target_cols >= 0
    if not np.any(in_target):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    target_cols = target_cols[in_target]
    locs_xy = locs_xy[in_target]

    x_idx = np.searchsorted(x_bins, locs_xy[:, 0], side="right") - 1
    y_idx = np.searchsorted(y_bins, locs_xy[:, 1], side="right") - 1
    valid = (x_idx >= 0) & (x_idx < ncols) & (y_idx >= 0) & (y_idx < nrows)
    if not np.any(valid):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    target_cols = target_cols[valid].astype(np.int64)

    # Each spatial bin is identified by a unique integer index based on its (row, col) position in the grid
    spot_idx = (y_idx * ncols + x_idx).astype(np.int64)
    n_target = int((codeword_to_target_col >= 0).sum())

    # Each (bin, codeword) pair is identified by a unique integer index based on its (spot_idx, target_col) pair
    # such that we can use np.unique to count occurrences of each unique pair
    pair_key = spot_idx * n_target + target_cols
    unique_key, counts = np.unique(pair_key, return_counts=True)

    # Parse the unique_key back into spatial and transcript indices
    rows = unique_key // n_target  # spot_idx, spatial bin indices
    cols = unique_key % n_target  # target_cols, transcript indices
    vals = counts.astype(np.float32)
    return rows, cols, vals


def _format_resolution_bin_name(spatial_resolution: float) -> str:
    if np.isclose(spatial_resolution, round(spatial_resolution)):
        return f"square_{int(round(spatial_resolution)):03}um"
    return f"square_{str(spatial_resolution).replace('.', 'p')}um"


def _format_resolution_obs_token(spatial_resolution: float) -> str:
    if np.isclose(spatial_resolution, round(spatial_resolution)):
        return f"{int(round(spatial_resolution)):03}um"
    return f"{str(spatial_resolution).replace('.', 'p')}um"


def _format_resolution_shape_name(spatial_resolution: float) -> str:
    return f"{_format_resolution_bin_name(spatial_resolution)}_bins"


def _assemble_xenium_codeword_table(
    counts_matrix: scipy.sparse.csr_matrix,
    spatial_resolution: float,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    nrows: int,
    ncols: int,
    codeword_id_to_gene: list[str],
    cw_ids: Sequence[int],
) -> AnnData:
    res_token = _format_resolution_obs_token(spatial_resolution)
    obs_names = [
        f"s_{res_token}_{int(r):05}_{int(c):05}-1"
        for r in range(nrows)
        for c in range(ncols)
    ]
    grid_cols, grid_rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    # Compute the center coordinates of each bin in microns
    x_centers = (x_bins[:-1] + x_bins[1:]) * 0.5
    y_centers = (y_bins[:-1] + y_bins[1:]) * 0.5

    obs_df = pd.DataFrame(
        {
            "array_row": grid_rows.ravel(),
            "array_col": grid_cols.ravel(),
            "x_center_um": x_centers[grid_cols].ravel(),
            "y_center_um": y_centers[grid_rows].ravel(),
        },
        index=obs_names,
    )

    # Feature metadata for each codeword, including gene symbol and feature ID
    var_metadata = [
        {
            "var_name": f"{codeword_id_to_gene[cw]}|{cw}",
            "codeword_id": str(cw),
            "gene_symbol": str(codeword_id_to_gene[cw]),
            "feature_id": f"{codeword_id_to_gene[cw]}|{cw}",
        }
        for cw in cw_ids
    ]
    var_df = pd.DataFrame(var_metadata).set_index("var_name")

    adata = AnnData(X=counts_matrix, obs=obs_df, var=var_df)
    adata.obsm["spatial"] = obs_df[["x_center_um", "y_center_um"]].to_numpy(
        dtype=np.float64
    )
    return adata


def _build_xenium_codeword_table_for_resolution(
    root: Any,
    level0_keys: Sequence[str],
    codeword_id_to_gene: list[str],
    target_codewords: list[int],
    spatial_resolution: float,
    quality_threshold: float,
    n_jobs: int,
    chunk_batch_size: int,
    show_progress: bool,
) -> tuple[AnnData, np.ndarray, np.ndarray, int, int]:
    density_gene = root["density"]["gene"]
    native_grid_size = density_gene.attrs["grid_size"]
    grid_origin = density_gene.attrs["origin"]
    native_nrows = int(density_gene.attrs["rows"])
    native_ncols = int(density_gene.attrs["cols"])

    x_bins, y_bins, nrows, ncols = _compute_xenium_bins_from_meta(
        grid_origin,
        native_grid_size,
        native_nrows,
        native_ncols,
        spatial_resolution,
    )
    total_spots = nrows * ncols
    n_target = len(target_codewords)

    # Map each codeword ID to its corresponding column index in the output matrix.
    # Non-target codewords are encoded as -1
    codeword_to_target_col = np.full(len(codeword_id_to_gene), -1, dtype=np.int64)
    codeword_to_target_col[np.asarray(target_codewords, dtype=np.int64)] = np.arange(
        n_target, dtype=np.int64
    )

    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None

    # Process a single chunk to extract (row, col, value) triplets for the sparse counts matrix
    # See https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr#transcripts
    # for the Zarr chunk structure.
    def process_key(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk_path = f"grids/0/{key}"
        if chunk_path not in root:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        return _chunk_to_codeword_triplets(
            root[chunk_path],
            codeword_to_target_col,
            quality_threshold,
            x_bins,
            y_bins,
            nrows,
            ncols,
        )

    counts_matrix = scipy.sparse.csr_matrix((total_spots, n_target), dtype=np.float32)
    batch_iter = range(0, len(level0_keys), chunk_batch_size)
    if show_progress:
        batch_iter = tqdm(
            batch_iter,
            desc=(
                f"Binning codewords @ {spatial_resolution:g}um "
                f"({len(level0_keys)} chunks)"
            ),
            leave=False,
        )

    for i in batch_iter:
        batch = level0_keys[i : i + chunk_batch_size]
        if Parallel is None or n_jobs == 1:
            batch_results = [process_key(k) for k in batch]
        else:
            batch_results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(process_key)(k) for k in batch
            )
        rows_parts = [r for r, _, _ in batch_results if r.size > 0]
        if len(rows_parts) == 0:
            continue
        cols_parts = [c for _, c, _ in batch_results if c.size > 0]
        vals_parts = [v for _, _, v in batch_results if v.size > 0]

        batch_rows = np.concatenate(rows_parts)
        batch_cols = np.concatenate(cols_parts)
        batch_vals = np.concatenate(vals_parts)

        batch_counts = scipy.sparse.coo_matrix(
            (batch_vals, (batch_rows, batch_cols)),
            shape=(total_spots, n_target),
            dtype=np.float32,
        ).tocsr()
        counts_matrix = counts_matrix + batch_counts

    adata = _assemble_xenium_codeword_table(
        counts_matrix=counts_matrix,
        spatial_resolution=spatial_resolution,
        x_bins=x_bins,
        y_bins=y_bins,
        nrows=nrows,
        ncols=ncols,
        codeword_id_to_gene=codeword_id_to_gene,
        cw_ids=target_codewords,
    )
    return adata, x_bins, y_bins, nrows, ncols


def load_xenium_codeword(
    path: str | Path,
    spatial_resolutions: Sequence[float] = (8.0, 16.0),
    quality_threshold: float = 20.0,
    n_jobs: int = -1,
    chunk_batch_size: int = 64,
    counts_layer_name: str = "counts",
    create_square_shapes: bool = True,
    cells_boundaries: bool = True,
    nucleus_boundaries: bool = True,
    cells_as_circles: bool = False,
    cells_labels: bool = True,
    nucleus_labels: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    aligned_images: bool = True,
    cells_table: bool = True,
    gex_only: bool = True,
    show_progress: bool = True,
) -> Any:
    """Load Xenium outputs and append multi-resolution codeword bin tables.

    This wrapper reads Xenium Ranger ``outs`` with ``spatialdata-io`` and then
    quantifies codewords into square spatial bins at one or more user-defined
    resolutions using transcript-level chunk data (``grids/0/*``). Counting is
    implemented with vectorized sparse aggregation over ``(spot, codeword)``
    pairs to reduce Python overhead while avoiding dependence on optional
    precomputed density matrices. For each resolution, a table named
    ``square_XXXum`` is added to ``sdata.tables``; optional square geometries
    with a ``_bins`` suffix are added to ``sdata.shapes`` so the tables can be
    used directly with :func:`spatialdata.rasterize_bins`.

    Parameters
    ----------
    path
        Path to Xenium Ranger output directory, or its parent containing
        ``outs/``.
    spatial_resolutions
        Spatial bin sizes in microns.
    quality_threshold
        Minimum transcript quality score to retain.
    n_jobs
        Parallel worker count for chunk processing. Use ``-1`` for all cores.
    chunk_batch_size
        Number of transcript chunks submitted per processing batch.
    counts_layer_name
        Layer name used to store codeword counts in each output table.
    create_square_shapes
        Whether to create square bin shapes for each table key.
    cells_boundaries
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    nucleus_boundaries
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    cells_as_circles
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    cells_labels
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    nucleus_labels
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    transcripts
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    morphology_mip
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    morphology_focus
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    aligned_images
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    cells_table
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    gex_only
        Passed to ``spatialdata_io.readers.xenium.xenium``.
    show_progress
        Whether to display progress bars while binning codewords.

    Returns
    -------
    spatialdata.SpatialData
        SpatialData object augmented with codeword-count tables at each
        requested resolution.

    Raises
    ------
    ImportError
        If required optional dependencies are not installed.
    ValueError
        If path/layout/arguments are invalid.
    """
    try:
        from spatialdata_io.readers.xenium import xenium
    except ImportError as e:
        raise ImportError(
            "spatialdata-io is required for load_xenium_codeword(). "
            "Install it via `pip install spatialdata-io`."
        ) from e

    try:
        import zarr
        from zarr.storage import ZipStore
    except ImportError as e:
        raise ImportError(
            "zarr is required for load_xenium_codeword(). "
            "Install it via `pip install zarr`."
        ) from e

    try:
        from spatialdata.models import TableModel
    except ImportError as e:
        raise ImportError(
            "spatialdata is required for load_xenium_codeword(). "
            "Install it via `pip install spatialdata`."
        ) from e

    if create_square_shapes:
        try:
            import geopandas as gpd
            from shapely.geometry import box
            from spatialdata.models import ShapesModel
            from spatialdata.transformations.transformations import Identity
        except ImportError as e:
            raise ImportError(
                "create_square_shapes=True requires geopandas, shapely, and spatialdata. "
                "Install via `pip install geopandas shapely spatialdata`."
            ) from e

    if len(spatial_resolutions) == 0:
        raise ValueError("`spatial_resolutions` cannot be empty.")
    if chunk_batch_size <= 0:
        raise ValueError("`chunk_batch_size` must be > 0.")
    if quality_threshold < 0:
        raise ValueError("`quality_threshold` must be >= 0.")

    resolutions = [float(x) for x in spatial_resolutions]
    if any(r <= 0 for r in resolutions):
        raise ValueError("All `spatial_resolutions` values must be > 0.")

    # Call spatialdata-io reader to load Xenium Ranger outputs into a SpatialData object
    outs_path = _xenium_locate_outs_path(path)
    sdata = xenium(
        path=outs_path,
        cells_boundaries=cells_boundaries,
        nucleus_boundaries=nucleus_boundaries,
        cells_as_circles=cells_as_circles,
        cells_labels=cells_labels,
        nucleus_labels=nucleus_labels,
        transcripts=transcripts,
        morphology_mip=morphology_mip,
        morphology_focus=morphology_focus,
        aligned_images=aligned_images,
        cells_table=cells_table,
        gex_only=gex_only,
    )

    # Binning transcripts into square bins at requested resolutions using transcript-level chunk data
    zarr_path = outs_path / "transcripts.zarr.zip"
    store = ZipStore(zarr_path, mode="r")
    root = zarr.open(store, mode="r")

    try:
        try:
            codeword_names_raw = root["density"]["codeword"].attrs.get(
                "codeword_names", []
            )
        except KeyError:
            raise ValueError(
                "Could not find `density/codeword` group in transcripts.zarr.zip. "
                "Consider re-running the Xenium Ranger (>=v3.1.0) relabel pipeline. "
                "https://www.10xgenomics.com/support/software/xenium-ranger/latest/analysis/running-pipelines/XR-relabel"
            )
        codeword_id_to_gene = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in codeword_names_raw
        ]
        if len(codeword_id_to_gene) == 0:
            raise ValueError("No codeword names found in transcripts.zarr.zip.")

        target_codewords = [
            i
            for i, name in enumerate(codeword_id_to_gene)
            if not str(name).startswith("UnassignedCodeword")
        ]
        if len(target_codewords) == 0:
            raise ValueError("No assigned codewords found in transcripts.zarr.zip.")

        level0_keys: list[str] = []
        try:
            grid_keys_attr = root["grids"].attrs.get("grid_keys", None)
        except Exception:
            grid_keys_attr = None

        if grid_keys_attr is not None and len(grid_keys_attr) > 0:
            level0_keys = [
                k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
                for k in list(grid_keys_attr[0])
            ]
        if len(level0_keys) == 0:
            raise ValueError(
                "Could not find transcript chunk keys under `grids/0` in transcripts.zarr.zip. "
                "This function requires transcript chunks for quantification."
            )

        for spatial_resolution in resolutions:
            adata, x_bins, y_bins, nrows, ncols = (
                _build_xenium_codeword_table_for_resolution(
                    root=root,
                    level0_keys=level0_keys,
                    codeword_id_to_gene=codeword_id_to_gene,
                    target_codewords=target_codewords,
                    spatial_resolution=spatial_resolution,
                    quality_threshold=quality_threshold,
                    n_jobs=n_jobs,
                    chunk_batch_size=chunk_batch_size,
                    show_progress=show_progress,
                )
            )

            table_key = _format_resolution_bin_name(spatial_resolution)
            shape_key = _format_resolution_shape_name(spatial_resolution)

            adata.layers[counts_layer_name] = (
                adata.X.copy() if hasattr(adata.X, "copy") else adata.X
            )
            adata.uns["spatial_bins"] = {
                "resolution_um": float(spatial_resolution),
                "shape_key": shape_key,
                "grid_origin_um": {
                    "x": float(x_bins[0]),
                    "y": float(y_bins[0]),
                },
                "grid_dims": {
                    "rows": int(nrows),
                    "cols": int(ncols),
                },
            }

            adata.obs["region"] = pd.Categorical([shape_key] * adata.n_obs)
            adata.obs["instance_id"] = adata.obs_names.astype(str)
            table = TableModel.parse(
                adata,
                region=shape_key,
                region_key="region",
                instance_key="instance_id",
            )
            sdata.tables[table_key] = table

            if create_square_shapes:
                rows = adata.obs["array_row"].to_numpy(dtype=np.int64)
                cols = adata.obs["array_col"].to_numpy(dtype=np.int64)
                polygons = [
                    box(x_bins[c], y_bins[r], x_bins[c + 1], y_bins[r + 1])
                    for r, c in zip(rows, cols, strict=False)
                ]
                geo_df = gpd.GeoDataFrame(
                    {"geometry": polygons},
                    index=adata.obs_names.astype(str),
                )
                sdata.shapes[shape_key] = ShapesModel.parse(
                    geo_df,
                    transformations={"global": Identity()},
                )
    finally:
        store.close()

    return sdata
