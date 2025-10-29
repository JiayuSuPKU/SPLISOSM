"""Convert 10x Xenium Prime 5K codeword quantification Zarr to AnnData with spatial binning using parallel processing."""
import os
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import zarr
from zarr.storage import ZipStore
import anndata
from tqdm import tqdm
from scipy.sparse import csc_matrix, csr_matrix, hstack
from joblib import Parallel, delayed


def _compute_bins_from_meta(grid_origin, native_grid_size, native_nrows, native_ncols, spatial_resolution):
    x_min, y_min = grid_origin['x'], grid_origin['y']
    x_max = x_min + native_ncols * native_grid_size[0]
    y_max = y_min + native_nrows * native_grid_size[1]
    ncols = int(np.ceil((x_max - x_min) / spatial_resolution))
    nrows = int(np.ceil((y_max - y_min) / spatial_resolution))
    x_bins = np.linspace(x_min, x_min + ncols * spatial_resolution, ncols + 1, dtype=np.float64)
    y_bins = np.linspace(y_min, y_min + nrows * spatial_resolution, nrows + 1, dtype=np.float64)
    return x_bins, y_bins, nrows, ncols


def _hist2d_sparse_column(locs_xy: np.ndarray, x_bins: np.ndarray, y_bins: np.ndarray, total_spots: int) -> csc_matrix:
    # np.histogram2d returns (nx, ny); we want (rows, cols) = (ny, nx), then flatten column-major
    hist, _, _ = np.histogram2d(locs_xy[:, 0], locs_xy[:, 1], bins=[x_bins, y_bins])
    hist_t = hist.T  # (nrows, ncols)
    return csc_matrix(hist_t.reshape(total_spots, 1))


def _chunk_to_partial_counts(chunk, target_codewords_set: set, quality_threshold: float,
                             x_bins: np.ndarray, y_bins: np.ndarray, nrows: int, ncols: int
                             ) -> Dict[int, csc_matrix]:
    # Read arrays
    q_scores = chunk['quality_score'][:].ravel()
    codeword_ids = chunk['codeword_identity'][:, 0]
    locs = chunk['location'][:]

    # Quality filter
    quality_mask = q_scores >= quality_threshold
    if not np.any(quality_mask):
        return {}

    codeword_ids = codeword_ids[quality_mask]
    locs = locs[quality_mask]

    # Keep only target codewords
    in_target = np.isin(codeword_ids, list(target_codewords_set))
    if not np.any(in_target):
        return {}

    codeword_ids = codeword_ids[in_target]
    locs = locs[in_target]

    # Bin once to 2D indices to avoid repeated histogram2d calls
    # We map each point to (row, col) bin and then aggregate per codeword via bincount
    # Compute bin indices (0..ncols-1, 0..nrows-1)
    x_idx = np.searchsorted(x_bins, locs[:, 0], side='right') - 1
    y_idx = np.searchsorted(y_bins, locs[:, 1], side='right') - 1
    valid = (x_idx >= 0) & (x_idx < ncols) & (y_idx >= 0) & (y_idx < nrows)
    if not np.any(valid):
        return {}

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    codeword_ids = codeword_ids[valid]

    # Flatten (row, col) to linear index with row-major order matching our (nrows, ncols) layout
    lin = y_idx * ncols + x_idx  # 0..total_spots-1
    total_spots = nrows * ncols

    # Aggregate by codeword via grouped bincount
    # Sort by codeword to do one pass bincount per group
    order = np.argsort(codeword_ids, kind='mergesort')
    codeword_ids_sorted = codeword_ids[order]
    lin_sorted = lin[order]

    unique_cw, starts = np.unique(codeword_ids_sorted, return_index=True)
    partial = {}
    for i, cw in enumerate(unique_cw):
        if cw not in target_codewords_set:
            continue
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else len(codeword_ids_sorted)
        lin_cw = lin_sorted[start:end]
        # bincount to dense vector then make sparse column
        counts = np.bincount(lin_cw, minlength=total_spots)
        if counts.sum() == 0:
            continue
        col = csc_matrix(counts.reshape(total_spots, 1))
        partial[int(cw)] = col
    return partial


def convert_xenium_to_anndata_parallel(
    data_dir: str,
    res_dir: str,
    spatial_resolution: float,
    quality_threshold: float = 20.0,
    n_jobs: int = -1,
    chunk_batch_size: int = 64,
):
    zarr_path = os.path.join(data_dir, 'outs', 'transcripts.zarr.zip')
    if not os.path.exists(zarr_path):
        zarr_path = os.path.join(data_dir, 'transcripts.zarr.zip')
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Missing transcripts.zarr.zip in {data_dir}/outs or {data_dir}")
    os.makedirs(res_dir, exist_ok=True)

    root = zarr.open(ZipStore(zarr_path), mode='r')

    codeword_id_to_gene = root.density.codeword.attrs.get('codeword_names', [])
    target_codewords = [i for i, g in enumerate(codeword_id_to_gene) if not str(g).startswith('UnassignedCodeword')]
    if not target_codewords:
        raise ValueError("No assigned codewords found.")
    target_codewords_set = set(target_codewords)

    # Create spatial grid for binning
    native_grid_size = root.density.gene.attrs['grid_size']
    grid_origin = root.density.gene.attrs['origin']
    native_nrows = root.density.gene.attrs['rows']
    native_ncols = root.density.gene.attrs['cols']

    x_bins, y_bins, nrows, ncols = _compute_bins_from_meta(
        grid_origin, native_grid_size, native_nrows, native_ncols, spatial_resolution
    )
    total_spots = nrows * ncols

    level0_keys = list(root['grids'].attrs['grid_keys'][0])

    # Process chunks in parallel producing partial per-codeword columns, then reduce by summation
    def process_key(key: str) -> Dict[int, csc_matrix]:
        chunk_path = f'grids/0/{key}'
        if chunk_path not in root:
            return {}
        chunk = root[chunk_path]
        return _chunk_to_partial_counts(
            chunk, target_codewords_set, quality_threshold, x_bins, y_bins, nrows, ncols
        )

    # Run in parallel with batching to reduce scheduler overhead
    partials: List[Dict[int, csc_matrix]] = []
    with tqdm(total=len(level0_keys), desc="Reading + binning chunks (parallel)") as pbar:
        # manual batching
        for i in range(0, len(level0_keys), chunk_batch_size):
            batch = level0_keys[i:i + chunk_batch_size]
            batch_results = Parallel(n_jobs=n_jobs, prefer="threads", backend="loky")(
                delayed(process_key)(k) for k in batch
            )
            partials.extend(batch_results)
            pbar.update(len(batch))

    # Reduce: sum columns for each codeword
    # Use dictionary of lists -> sum
    acc: Dict[int, List[csc_matrix]] = {cw: [] for cw in target_codewords}
    for d in partials:
        for cw, col in d.items():
            acc[cw].append(col)

    # Assemble counts matrix in parallel by splitting codewords into chunks
    def sum_columns(cols: List[csc_matrix]) -> csc_matrix:
        if not cols:
            return csc_matrix((total_spots, 1), dtype=np.float64)
        if len(cols) == 1:
            return cols[0]
        # Sum via CSR to speed up
        s = cols[0].tocsr()
        for c in cols[1:]:
            s = s + c.tocsr()
        return s.tocsc()

    # Parallel reduce per codeword
    cw_ids = target_codewords
    results = Parallel(n_jobs=n_jobs, prefer="threads", backend="loky")(
        delayed(sum_columns)(acc[cw]) for cw in tqdm(cw_ids, desc="Reducing codewords")
    )

    counts_matrix: csr_matrix = hstack(results, format='csr')

    # Build AnnData
    obs_names = [f'spot_{i}' for i in range(total_spots)]
    grid_cols, grid_rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    obs_df = pd.DataFrame({
        'array_row': grid_rows.ravel(),
        'array_col': grid_cols.ravel()
    }, index=obs_names)

    var_metadata = [{
        'var_name': str(cw),
        'gene_symbol': codeword_id_to_gene[cw],
        'feature_id': f"{codeword_id_to_gene[cw]}-{cw}"
    } for cw in cw_ids]
    var_df = pd.DataFrame(var_metadata).set_index('var_name')

    adata = anndata.AnnData(X=counts_matrix, obs=obs_df, var=var_df)
    adata.layers['counts'] = adata.X.copy()
    adata.uns['spatial'] = {
        'resolution_um': spatial_resolution,
        'grid_origin_um': grid_origin,
        'grid_dims': {'rows': nrows, 'cols': ncols}
    }

    out_path = os.path.join(res_dir, f"codeword_quant_res_{int(spatial_resolution)}um.h5ad")
    adata.write_h5ad(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Parallel Xenium codeword quantification to AnnData")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--res_dir', type=str, required=True)
    parser.add_argument('--spatial_resolution', type=float, required=True)
    parser.add_argument('--quality_threshold', type=float, default=20.0)
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of workers for parallelism (-1 uses all cores)")
    parser.add_argument('--chunk_batch_size', type=int, default=64, help="Number of Zarr chunks per joblib submit batch")
    args = parser.parse_args()

    path = convert_xenium_to_anndata_parallel(
        data_dir=args.data_dir,
        res_dir=args.res_dir,
        spatial_resolution=args.spatial_resolution,
        quality_threshold=args.quality_threshold,
        n_jobs=args.n_jobs,
        chunk_batch_size=args.chunk_batch_size,
    )
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()