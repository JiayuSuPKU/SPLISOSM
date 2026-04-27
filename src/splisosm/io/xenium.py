"""10x Xenium loader utilities."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from tqdm import tqdm

__all__ = ["load_xenium_codeword"]


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


def _xenium_specs_and_transcripts_filenames() -> tuple[str, str]:
    """Resolve Xenium specs and transcripts parquet filenames."""
    specs_file = "experiment.xenium"
    transcripts_file = "transcripts.parquet"
    try:
        from spatialdata_io._constants._constants import XeniumKeys

        specs_file = str(XeniumKeys.XENIUM_SPECS)
        transcripts_file = str(XeniumKeys.TRANSCRIPTS_FILE)
    except Exception:
        # Keep defaults when spatialdata-io internals are not importable.
        pass
    return specs_file, transcripts_file


def _add_xenium_cell_codeword_table_from_parquet(
    sdata: Any,
    outs_path: Path,
    codeword_id_to_gene: Sequence[str],
    target_codewords: Sequence[int],
    quality_threshold: float,
    counts_layer_name: str,
) -> None:
    """Build and attach a cell-by-codeword table directly from transcripts parquet."""
    # See https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs#transcript-file
    # for the structure of the transcripts parquet file.
    _, transcripts_file = _xenium_specs_and_transcripts_filenames()
    transcripts_path = outs_path / transcripts_file
    if not transcripts_path.exists():
        warnings.warn(
            f"Could not find Xenium transcripts parquet at {transcripts_path}; "
            "skipping `table_codeword` generation.",
            UserWarning,
            stacklevel=2,
        )
        return

    try:
        import dask.dataframe as dd
    except ImportError as e:
        raise ImportError(
            "dask is required to build Xenium cell-by-codeword table from transcripts parquet. "
            "Install it via `pip install dask[dataframe]`."
        ) from e

    try:
        from spatialdata.models import TableModel
    except ImportError as e:
        raise ImportError(
            "spatialdata is required to parse Xenium cell-by-codeword table. "
            "Install it via `pip install spatialdata`."
        ) from e

    target_cw_index = pd.Index(np.asarray(target_codewords, dtype=np.int64))
    read_cols = ["cell_id", "codeword_index"]
    if quality_threshold > 0:
        read_cols.append("qv")

    tx = dd.read_parquet(transcripts_path, columns=read_cols)
    if quality_threshold > 0 and "qv" in tx.columns:
        tx = tx[tx["qv"] >= float(quality_threshold)]

    tx = tx[
        (tx["codeword_index"] >= 0)
        & (tx["codeword_index"].isin(target_cw_index.to_list()))
    ]
    grouped = tx.groupby(["cell_id", "codeword_index"]).size().compute()

    target_cell_ids = pd.Index(sdata.shapes["cell_boundaries"].index.astype(str))

    if grouped.shape[0] == 0:
        x_cells = scipy.sparse.csr_matrix(
            (len(target_cell_ids), len(target_cw_index)), dtype=np.float32
        )
    else:
        cell_vals = grouped.index.get_level_values(0)
        cell_series = pd.Series(cell_vals, dtype="object")
        if cell_series.size > 0 and isinstance(cell_series.iloc[0], (bytes, bytearray)):
            cell_ids = cell_series.str.decode("utf-8").to_numpy()
        else:
            cell_ids = cell_series.astype(str).to_numpy()

        cw_vals = grouped.index.get_level_values(1).to_numpy(dtype=np.int64, copy=False)
        rows = target_cell_ids.get_indexer(cell_ids)
        cols = target_cw_index.get_indexer(cw_vals)
        vals = grouped.to_numpy(dtype=np.float32, copy=False)

        valid = (rows >= 0) & (cols >= 0)
        x_cells = scipy.sparse.coo_matrix(
            (vals[valid], (rows[valid], cols[valid])),
            shape=(len(target_cell_ids), len(target_cw_index)),
            dtype=np.float32,
        ).tocsr()

    var_df = pd.DataFrame(
        {
            "var_name": [f"{codeword_id_to_gene[cw]}|{cw}" for cw in target_cw_index],
            "codeword_id": [str(cw) for cw in target_cw_index],
            "gene_symbol": [str(codeword_id_to_gene[cw]) for cw in target_cw_index],
            "feature_id": [f"{codeword_id_to_gene[cw]}|{cw}" for cw in target_cw_index],
        }
    ).set_index("var_name")

    adata_cells = AnnData(
        X=x_cells,
        obs=pd.DataFrame(index=target_cell_ids),
        var=var_df,
    )
    adata_cells.layers[counts_layer_name] = (
        adata_cells.X.copy() if hasattr(adata_cells.X, "copy") else adata_cells.X
    )
    adata_cells.obs["region"] = pd.Categorical(["cell_boundaries"] * adata_cells.n_obs)
    adata_cells.obs["instance_id"] = adata_cells.obs_names.astype(str)
    sdata.tables["table_codeword"] = TableModel.parse(
        adata_cells,
        region="cell_boundaries",
        region_key="region",
        instance_key="instance_id",
    )

    # Add cell meta in sdata.tables['table'].obs to sdata.tables['table_codeword'].obs
    if "table" in sdata.tables:
        cell_meta = sdata.tables["table"].obs.set_index("cell_id")
        sdata.tables["table_codeword"].obs = sdata.tables["table_codeword"].obs.join(
            cell_meta, how="left", rsuffix="_from_table"
        )
        # Copy spatial embeddings (e.g. cell centroids) from sdata.tables['table'].obsm
        for key, val in sdata.tables["table"].obsm.items():
            if key not in sdata.tables["table_codeword"].obsm:
                sdata.tables["table_codeword"].obsm[key] = (
                    val.copy() if hasattr(val, "copy") else val
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
    spatial_resolutions: Sequence[float] | None = (8.0, 16.0),
    quality_threshold: float = 20.0,
    n_jobs: int = -1,
    chunk_batch_size: int = 64,
    counts_layer_name: str = "counts",
    build_cell_codeword_table: bool = True,
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

    ``transcripts.zarr.zip`` is expected to contain the ``density/codeword`` group
    for codeword indexing (Xenium Ranger v3.1+ required).
    If ``build_cell_codeword_table=True`` and the ``transcripts.parquet`` file is available,
    a cell-by-codeword anndata named ``table_codeword`` will also be built and added to ``sdata.tables``.

    Parameters
    ----------
    path
        Path to Xenium Ranger output directory, or its parent containing
        ``outs/``.
    spatial_resolutions
        Spatial bin sizes in microns.  Pass ``None`` or an empty sequence to
        skip bin table creation entirely (cell-segmentation-only mode).
    quality_threshold
        Minimum transcript quality score to retain.
    n_jobs
        Parallel worker count for chunk processing. Use ``-1`` for all cores.
    chunk_batch_size
        Number of transcript chunks submitted per processing batch.
    counts_layer_name
        Layer name used to store codeword counts in each output table.
    build_cell_codeword_table
        Whether to build a cell-by-codeword table from the transcripts parquet file.
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
        SpatialData object augmented with bin-by-codeword count tables at each
        requested resolution and, when requested, a cell-by-codeword table
        named ``table_codeword``.

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

    resolutions: list[float] = (
        [] if spatial_resolutions is None else [float(x) for x in spatial_resolutions]
    )
    if any(r <= 0 for r in resolutions):
        raise ValueError("All `spatial_resolutions` values must be > 0.")
    if chunk_batch_size <= 0:
        raise ValueError("`chunk_batch_size` must be > 0.")
    if quality_threshold < 0:
        raise ValueError("`quality_threshold` must be >= 0.")

    if resolutions and create_square_shapes:
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

    # Open transcripts.zarr.zip only when codeword data is actually needed.
    if resolutions or build_cell_codeword_table:
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

            # If transcripts.parquet is available,
            # build a cell-by-codeword table directly from the parquet file.
            if build_cell_codeword_table:
                _add_xenium_cell_codeword_table_from_parquet(
                    sdata=sdata,
                    outs_path=outs_path,
                    codeword_id_to_gene=codeword_id_to_gene,
                    target_codewords=target_codewords,
                    quality_threshold=quality_threshold,
                    counts_layer_name=counts_layer_name,
                )

            if resolutions:
                level0_keys: list[str] = []
                try:
                    grid_keys_attr = root["grids"].attrs.get("grid_keys", None)
                except Exception:
                    grid_keys_attr = None

                if grid_keys_attr is not None and len(grid_keys_attr) > 0:
                    level0_keys = [
                        (
                            k.decode("utf-8")
                            if isinstance(k, (bytes, bytearray))
                            else str(k)
                        )
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
