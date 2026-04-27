"""10x Visium HD loader utilities."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

__all__ = ["load_visiumhd_probe"]


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
    path_to_feature_2um_h5: Optional[str | Path] = None,
) -> Any:
    """Load Visium HD outputs as SpatialData with probe-level binned tables.

    This wrapper uses ``binned_outputs/square_002um/raw_probe_bc_matrix.h5``
    (or a custom ``path_to_feature_2um_h5``) as the source feature count matrix.
    It aggregates probe/peak/isoform counts to coarser bins or cells (``square_008um``,
    ``square_016um`` and, when available, ``cell_id``) according to the spatial mapping
    ``barcode_mappings.parquet`` (Space Ranger v4.0+ required).

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
    path_to_feature_2um_h5
        Optional path to the raw 2um probe/peak/isoform counts matrix H5 or H5AD.
        If not provided, will look for ``binned_outputs/square_002um/raw_feature_bc_matrix.h5``.

    Returns
    -------
    spatialdata.SpatialData
        A SpatialData object with probe-level tables for requested bins and,
        if available, cell-level segmentation.

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

    if path_to_feature_2um_h5 is None:
        raw_probe_path = binned_outputs / source_bin / "raw_probe_bc_matrix.h5"
    else:
        raw_probe_path = Path(path_to_feature_2um_h5)

    if not raw_probe_path.exists():
        raise ValueError(
            f"Cannot find raw probe/peak/isoform matrix at: {raw_probe_path}"
        )

    mapping_path = path / "barcode_mappings.parquet"
    if not mapping_path.exists():
        mapping_candidates = list(path.glob("*barcode_mappings.parquet"))
        if len(mapping_candidates) == 0:
            raise ValueError(
                f"Cannot find barcode_mappings.parquet under: {path}. "
                "Consider re-running Space Ranger with v4.0+ to generate this file. "
                "https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/segmented-outputs"
            )
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
    # If already h5ad, load directly
    if raw_probe_path.suffix == ".h5ad":
        adata_2um = sc.read_h5ad(raw_probe_path)
    elif raw_probe_path.suffix in [".h5", ".hdf5"]:
        # Otherwise, assume it's in 10x H5 format and load with scanpy
        adata_2um = sc.read_10x_h5(raw_probe_path, gex_only=False)
    else:
        raise ValueError(
            f"Unsupported file format for raw probe matrix: {raw_probe_path.suffix}"
        )

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
                "No in-tissue 2um barcodes remained after filtering raw_probe_bc_matrix.h5. "
                "Please make sure barcodes are formatted correctly, e.g., 's_002um_xxxxx_xxxxx-1'."
            )

    # Aggregate 2um probe counts into requested resolution bins
    mapping_cols = [source_bin, *requested_bins]
    barcode_mappings = pd.read_parquet(mapping_path)

    # Check if cell_id column exists for cell-level aggregation
    has_cell_mapping = "cell_id" in barcode_mappings.columns
    if has_cell_mapping:
        mapping_cols.append("cell_id")

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
                stacklevel=2,
            )
            continue

        template_table = sdata.tables[bin_name]

        if bin_name == source_bin:
            reindex = adata_2um.obs_names.get_indexer(
                template_table.obs_names.astype(str)
            )
            missing = reindex == -1
            if missing.any():
                # Some barcodes in the template table are missing from the source 2um data; we will fill them with zeros
                x_source = (
                    adata_2um.X.tocsr()
                    if scipy.sparse.issparse(adata_2um.X)
                    else scipy.sparse.csr_matrix(
                        np.asarray(adata_2um.X, dtype=np.float32)
                    )
                )
                # Extract only valid rows (barcodes that exist in adata_2um)
                valid_mask = ~missing
                valid_indices_in_template = np.where(valid_mask)[0]
                valid_indices_in_source = reindex[valid_mask]

                # Extract valid rows from source
                x_valid = x_source.tocsr()[valid_indices_in_source]

                # Build output matrix using COO format to avoid memory bloat from LIL boolean indexing
                coo_valid = scipy.sparse.coo_matrix(x_valid)
                x_target = scipy.sparse.coo_matrix(
                    (
                        coo_valid.data,
                        (valid_indices_in_template[coo_valid.row], coo_valid.col),
                    ),
                    shape=(len(reindex), x_source.shape[1]),
                    dtype=np.float32,
                ).tocsr()
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

    # Aggregate 2um probe counts to cell segmentation level if available
    if has_cell_mapping and "cell_segmentations" in sdata.tables:
        cell_seg_table = sdata.tables["cell_segmentations"]

        # Get unique cell IDs in the cell_segmentations table
        target_cell_ids = pd.Index(cell_seg_table.obs_names.astype(str))

        # Aggregate probe counts from 2um barcodes to cells
        x_cells = _aggregate_visiumhd_probe_counts(
            adata_2um=adata_2um,
            barcode_mappings=barcode_mappings,
            source_col=source_bin,
            target_col="cell_id",
            target_obs_names=target_cell_ids,
        )

        # Create probe-level table for cells
        probe_table_cells = AnnData(
            X=x_cells,
            obs=cell_seg_table.obs.copy(),
            var=adata_2um.var.copy(),
        )

        for key, value in cell_seg_table.obsm.items():
            probe_table_cells.obsm[key] = (
                value.copy() if hasattr(value, "copy") else value
            )
        for key, value in cell_seg_table.uns.items():
            probe_table_cells.uns[key] = (
                value.copy() if hasattr(value, "copy") else value
            )

        probe_table_cells.layers[counts_layer_name] = (
            probe_table_cells.X.copy()
            if hasattr(probe_table_cells.X, "copy")
            else probe_table_cells.X
        )
        sdata.tables["cell_segmentations"] = probe_table_cells

    return sdata
