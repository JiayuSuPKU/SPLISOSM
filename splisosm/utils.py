import warnings
import json

from pathlib import Path
import numpy as np
import scipy.sparse
import pandas as pd
import torch
from matplotlib.image import imread
from anndata import AnnData
from smoother import SpatialWeightMatrix, SpatialLoss
from splisosm.likelihood import liu_sf

def get_cov_sp(coords, k=4, rho=0.99):
    """Wrapper function to get the spatial covariance matrix from spatial coordinates.

    Args:
        coords: n_spots x 2. Spatial coordinates of spots.
        rho: float. Spatial autocorrelation coefficient.

    Returns:
        cov_sp: n_spots x n_spots. Spatial covariance matrix with standardized variance (== 1).
    """
    # first calculate the spatial weights matrix (swm)
    # here swm is the binary adjacency matrix of the knn graph
    weights = SpatialWeightMatrix()
    weights.calc_weights_knn(coords, k=k, verbose=False)

    # # convert the swm to spatial covariance matrix with standardized variance (== 1)
    # spatial_loss = SpatialLoss("icar", weights, rho=rho, standardize_cov=True)
    # cov_sp = torch.cholesky_inverse(torch.linalg.cholesky(spatial_loss.inv_cov[0].to_dense())) # n_spots x n_spots
    spatial_loss = SpatialLoss("icar", weights, rho=rho, standardize_cov=False)
    cov_sp = torch.cholesky_inverse(torch.linalg.cholesky(spatial_loss.inv_cov[0].to_dense())) # n_spots x n_spots
    inv_sds = torch.diagflat(torch.diagonal(cov_sp) ** (-0.5))
    cov_sp = inv_sds @ cov_sp @ inv_sds

    return cov_sp

def counts_to_ratios(counts, transformation = "none", nan_filling = "mean"):
    """Convert counts to proportions.

    Note: the kernel in Aitchison geometry (euclidean distance after clr transformation) is sensitive to zeroes.
        For data with many near-zero proportions, it is recommended to using the radial transformation instead.
        See:
            Park, Junyoung, et al. "Kernel methods for radial transformed compositional data with many zeros."
            International Conference on Machine Learning. PMLR, 2022.
            https://proceedings.mlr.press/v162/park22d/park22d.pdf

    Args:
        counts: dense torch.Tensor of shape (n_spots, n_isos). Isoform counts.
        transformation: str. Transformation applied to the proportions. One of "none", "clr", "ilr", "alr", "radial".
        nan_filling: str. Method to fill all zero rows. One of "mean", "none".
            - "mean": fill missing values with the mean of the mean per column **befor transformation**.
            - "none": do not fill missing values and return NaNs.

    Returns:
        ratios: dense torch.Tensor of shape (n_spots, n_isos) or (n_spots, n_isos - 1) if ilr or alr transformation is used.
    """
    assert transformation in ["none", "clr", "ilr", "alr", "radial"]
    if transformation in ["clr", "ilr", "alr"]:
        try:
            from skbio.stats.composition import clr, ilr, alr # for ratio transformation
        except ImportError:
            warnings.warn(f"Please install scikit-bio to use ratio transformation='{transformation}'. Switching to 'none'.")
            transformation = "none"

    assert nan_filling in ["mean", "none"]

    if isinstance(counts, np.ndarray):
        counts = torch.from_numpy(counts).float()

    # identify zero rows to fill
    is_nan = counts.sum(1) == 0 # (n_spots,)

    # calculate isoform ratios
    if transformation in ["clr", "ilr", "alr"]:
        # add pseudocounts equal to 1% of the global mean per isoform to avoid zeros in the ratio
        y = (1 - 1e-2) * counts + 1e-2 * counts.mean(0, keepdim=True)
        y = y / y.sum(1, keepdim=True) # isoform ratio without nans and zeros
    else:
        y = counts / counts.sum(1, keepdim=True) # isoform ratio with nans
        # fill nan values with the mean ratio per column (isoform)
        if nan_filling == "mean":
            y[is_nan] = y[~is_nan].mean(0, keepdim=True)

    # apply transformation
    if transformation == "clr":
        y = torch.from_numpy(clr(y)).float() # (n_spots, n_isos)
    elif transformation == "ilr":
        y = torch.from_numpy(ilr(y)).float() # (n_spots, n_isos - 1)
    elif transformation == "alr":
        y = torch.from_numpy(alr(y)).float() # (n_spots, n_isos - 1)
    elif transformation == "radial":
        y = y / y.norm(dim=1, keepdim=True) # radial transformation with nans

    # fill back nan to rows with zero counts if needed
    if nan_filling == "none":
        y[is_nan] = torch.nan

    return y


# From scipy v1.13.1
# https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_morestats.py#L4737
def false_discovery_control(ps, *, axis=0, method='bh'):
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps : 1D array_like
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis : int
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method : {'bh', 'by'}
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg [1]_ (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
        [2]_ (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adjusted : array_like
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    combine_pvalues
    statsmodels.stats.multitest.multipletests

    Notes
    -----
    In multiple hypothesis testing, false discovery control procedures tend to
    offer higher power than familywise error rate control procedures (e.g.
    Bonferroni correction [1]_).

    If the p-values correspond with independent tests (or tests with
    "positive regression dependencies" [2]_), rejecting null hypotheses
    corresponding with Benjamini-Hochberg-adjusted p-values below :math:`q`
    controls the false discovery rate at a level less than or equal to
    :math:`q m_0 / m`, where :math:`m_0` is the number of true null hypotheses
    and :math:`m` is the total number of null hypotheses tested. The same is
    true even for dependent tests when the p-values are adjusted accorded to
    the more conservative Benjaminini-Yekutieli procedure.

    The adjusted p-values produced by this function are comparable to those
    produced by the R function ``p.adjust`` and the statsmodels function
    `statsmodels.stats.multitest.multipletests`. Please consider the latter
    for more advanced methods of multiple comparison correction.

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    .. [2] Benjamini, Yoav, and Daniel Yekutieli. "The control of the false
           discovery rate in multiple testing under dependency." Annals of
           statistics (2001): 1165-1188.

    .. [3] TileStats. FDR - Benjamini-Hochberg explained - Youtube.
           https://www.youtube.com/watch?v=rZKa4tW2NKs.

    .. [4] Neuhaus, Karl-Ludwig, et al. "Improved thrombolysis in acute
           myocardial infarction with front-loaded administration of alteplase:
           results of the rt-PA-APSAC patency study (TAPS)." Journal of the
           American College of Cardiology 19.5 (1992): 885-891.
    """
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    # Handle NaNs
    if np.isnan(ps).any():
        warnings.warn("NaNs encountered in p-values. These will be ignored.")
        # ignore NaNs in the p-values
        ps_in_range = (np.issubdtype(ps.dtype, np.number)
                    and np.all(ps[~np.isnan(ps)] == np.clip(ps[~np.isnan(ps)], 0, 1)))
    else:
        ps_in_range = (np.issubdtype(ps.dtype, np.number)
                    and np.all(ps == np.clip(ps, 0, 1)))

    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
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
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
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
def load_visium_sp_meta(adata: AnnData, path_to_spatial, library_id = None):
    """Helper function to load Visium spatial metadata.

    Args:
        adata: AnnData to store the spatial meta. Annotated data matrix.
        path_to_spatial: str. Path to the spatial folder generated by spaceranger.
        library_id: str. Library ID of the spatial data.

    Returns:
        adata: AnnData with spatial metadata.
    """
    if library_id is None: # default library_id
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
    for res in ['hires', 'lowres']:
        try:
            adata.uns["spatial"][library_id]['images'][res] = imread(
                str(files[f'{res}_image'])
            )
        except Exception:
            warnings.warn(f"Missing '{res}' image in {path_to_spatial}. Will be ignored.")
            adata.uns["spatial"][library_id]['images'][res] = None

    # read json scalefactors
    with open(files['scalefactors_json_file']) as f:
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


def extract_counts_n_ratios(adata: AnnData, layer = 'counts', group_iso_by = 'gene_symbol', return_sparse = False):
    """ Extract per-gene lists of isoform counts and ratios from anndata.

    Args:
        adata: AnnData. Annotated data matrix.
        layer: str. Layer to extract isoform counts (adata.layers[layer]).
        group_iso_by: str. Gene index in adata.var to group isoforms by.

    Returns:
        counts_list: List[torch.Tensor]. Isoform counts per gene, each of (n_spots, n_isos).
        ratios_list: List[torch.Tensor]. Isoform ratios per gene, each of (n_spots, n_isos).
        gene_name_list: List[str]. Gene names.
        ratio_obs_merged: np.ndarray. Observed isoform ratios, each of (n_spots, n_isos_total).
        return_sparse: bool. Whether to return sparse torch tensors for counts_list.
            if True, ratios_list will be empty and ratio_obs_merged will be None.
    """
    # extract isoform counts
    iso_counts = adata.layers[layer] # (n_spots, n_isos_total)

    # Check if input is sparse
    is_sparse_input = scipy.sparse.issparse(iso_counts)
    if is_sparse_input and not scipy.sparse.isspmatrix_csc(iso_counts) and not scipy.sparse.isspmatrix_csr(iso_counts):
        # convert to csr for efficient slicing if it is other sparse format
        iso_counts = iso_counts.tocsr()

    counts_list = [] # isoform counts per gene, each of (n_spots, n_isos)
    ratios_list = [] # isoform ratios per gene, each of (n_spots, n_isos)
    gene_name_list = [] # of length n_genes
    iso_ind_list = [] # of length n_genes

    for _gene, _group in adata.var.reset_index().groupby(group_iso_by, observed = True):
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
            _counts = torch.from_numpy(_counts_np).float() # (n_spots, n_isos)
            _ratios = counts_to_ratios(_counts, transformation='none') # (n_spots, n_isos)

            counts_list.append(_counts)
            ratios_list.append(_ratios)

    if is_sparse_input and return_sparse:
        ratio_obs_merged = None
    else:
        # reshape and store the observed ratio in anndata
        ratio_obs_merged = torch.concat(ratios_list, axis = 1).numpy() # (n_spots, n_isos_total)
        ratio_obs_merged = ratio_obs_merged[:, np.argsort(np.concatenate(iso_ind_list))] # (n_spots, n_isos_total)

    return counts_list, ratios_list, gene_name_list, ratio_obs_merged


def extract_gene_level_statistics(adata: AnnData, layer = 'counts', group_iso_by = 'gene_symbol'):
    """ Extract gene-level metadata from isoform-level counts anndata.

    Args:
        adata: AnnData. Annotated data matrix.
        layer: str. Layer to extract isoform counts (adata.layers[layer]).
        group_iso_by: str. Gene index in adata.var to group isoforms by.

    Returns:
        df_gene_meta: pd.DataFrame. Gene-level metadata with columns:
            - n_iso: int. Number of isoforms per gene.
            - pct_spot_on: float. Percentage of spots with non-zero counts.
            - count_avg: float. Average counts per gene.
            - count_std: float. Standard deviation of counts per gene.
            - perplexity: float. Expression-based effective number of isoforms.
            - major_ratio_avg: float. Average ratio of the major isoform.

    """
    # extract isoform counts
    iso_counts = adata.layers[layer] # (n_spots, n_isos_total)

    # Check if input is sparse
    is_sparse_input = scipy.sparse.issparse(iso_counts)
    if is_sparse_input and not scipy.sparse.isspmatrix_csc(iso_counts) and not scipy.sparse.isspmatrix_csr(iso_counts):
        iso_counts = iso_counts.tocsr()

    df_list = []
    # loop through genes
    for _gene, _group in adata.var.reset_index().groupby(group_iso_by, observed = True):
        # extract isoform counts and relative ratio
        iso_indices = _group.index.tolist()
        _counts = iso_counts[:, iso_indices]

        if is_sparse_input:
             # Calculate statistics on sparse matrix without densifying
             _sum_per_iso = np.asarray(_counts.sum(0)).flatten() # (n_isos,)
             _row_sums = np.asarray(_counts.sum(1)).flatten() # (n_spots,)
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
            _ratios_avg = (_sum_per_iso / _total_sum) # (n_isos,)

        # calculate and store gene-level statistics
        # handle zeros in log
        with np.errstate(divide='ignore', invalid='ignore'):
             entropy = - (np.log(_ratios_avg) * _ratios_avg)
             entropy = np.nan_to_num(entropy).sum()

        df_list.append({
            'gene': _gene,
            'n_iso': _group.shape[0],
            'pct_spot_on': pct_spot_on,
            'count_avg': count_avg,
            'count_std': count_std,
            'perplexity': np.exp(entropy),
            'major_ratio_avg': _ratios_avg.max() if _ratios_avg.size > 0 else 0.0,
        })

    df_gene_meta = pd.DataFrame(df_list).set_index('gene')

    return df_gene_meta

def run_sparkx(counts_gene, coordinates):
    """Wrapper for running the SPARK-X test for spatial gene expression variability.

    Args:
        counts_gene: array(n_spots, n_genes), the observed gene counts.
        coordinates: array(n_spots, 2), the spatial coordinates.

    Returns:
        sv_sparkx: dict. Results of the SPARK-X spatial variability test with keys:
            - 'statistic': np.ndarray of shape (n_genes,). Mean SPARK-X statistics.
            - 'pvalue': np.ndarray of shape (n_genes,). Combined p-values.
            - 'pvalue_adj': np.ndarray of shape (n_genes,). Adjusted combined p-values.
            - 'method': str. Method name "spark-x".
    """
    if scipy.sparse.issparse(counts_gene):
        raise ValueError("run_sparkx does not support sparse input for counts_gene.")
    if isinstance(counts_gene, torch.Tensor) and counts_gene.is_sparse:
        raise ValueError("run_sparkx does not support sparse input for counts_gene.")

    # load packages neccessary for running SPARK-X
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    spark = importr('SPARK')

    # prepare robject inputs
    if isinstance(counts_gene, torch.Tensor):
        counts_gene = counts_gene.numpy()
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.numpy()

    with (ro.default_converter + numpy2ri.converter).context():
        coords_r = ro.conversion.get_conversion().py2rpy(coordinates) # (n_spots, 2)
        counts_r = ro.conversion.get_conversion().py2rpy(counts_gene.T) # (n_genes, n_spots)

    counts_r.colnames = ro.vectors.StrVector(r['rownames'](coords_r))

    # run SPARK-X and extract outputs
    sparkx_res = spark.sparkx(counts_r, coords_r)
    with (ro.default_converter + numpy2ri.converter).context():
        sv_sparkx = {
            'statistic': ro.conversion.get_conversion().rpy2py(sparkx_res.rx2('stats')).mean(1),
            'pvalue': ro.conversion.get_conversion().rpy2py(sparkx_res.rx2('res_mtest'))['combinedPval'],
            'pvalue_adj': ro.conversion.get_conversion().rpy2py(sparkx_res.rx2('res_mtest'))['adjustedPval'],
            'method': 'spark-x',
        }

    return sv_sparkx

def run_hsic_gc(counts_gene, coordinates, approx_rank = None, **spatial_kernel_kwargs):
    """Function to compute HSIC-GC statistic for gene-level counts.

    This function is designed to be a plugin replacement for SPARK-X.

    Args:
        counts_gene: dense or sparse array-like of shape (n_spots, n_genes). Gene counts.
        coordinates: array of shape (n_spots, 2). Spatial coordinates of spots.
        approx_rank: int. Approximate rank of the spatial kernel matrix.
        spatial_kernel_kwargs: dict. Additional arguments for SpatialCovKernel.
    Returns:
        sv_test_results: dict. Results of the HSIC-GC spatial variability test with keys:
            - 'statistic': np.ndarray of shape (n_genes,). HSIC-GC statistics.
            - 'pvalue': np.ndarray of shape (n_genes,). P-values.
            - 'pvalue_adj': np.ndarray of shape (n_genes,). Adjusted p-values.
            - 'method': str. Method name "hsic-gc".
    """

    from splisosm.kernel import SpatialCovKernel
    n_spots = counts_gene.shape[0]
    n_genes = counts_gene.shape[1]

    # determine the maximum rank for spatial kernel computation
    if n_spots > 5000:
        # 10x Visium has 4992 spots per slide. For larger datasets (i.e. Slideseq-V2),
        # it is recommended to use low-rank approximation
        max_rank = np.ceil(np.sqrt(n_spots) * 4).astype(int)
        approx_rank = min(approx_rank, max_rank) if approx_rank is not None else max_rank
    else:
        if approx_rank is not None:
            approx_rank = approx_rank if approx_rank < n_spots else None

    # set default spatial kernel kwargs
    default_spatial_kernel_kwargs = {
        'k_neighbors': 4,
        'model': 'icar',
        'rho': 0.99,
        'standardize_cov': True,
        'centering': True,
        'approx_rank': approx_rank
    }
    if spatial_kernel_kwargs is not None:
        if 'centering' in spatial_kernel_kwargs:
            warnings.warn("The 'centering' argument in spatial_kernel_kwargs will be ignored. It is always set to True for HSIC-GC.")
            spatial_kernel_kwargs.pop('centering')
        default_spatial_kernel_kwargs.update(spatial_kernel_kwargs)

    # compute the spatial kernel
    # Ensure coordinates is numpy for SpatialCovKernel/smoother
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()

    K_sp = SpatialCovKernel(
        coordinates, **default_spatial_kernel_kwargs
    )

    # get the eigenvalues
    lambda_sp = K_sp.eigenvalues() # (rank,)
    lambda_sp = lambda_sp[lambda_sp > 1e-5] # filter small eigenvalues

    # compute the HSIC-GC statistic per-gene
    is_scipy_sparse = scipy.sparse.issparse(counts_gene)
    is_torch_sparse = isinstance(counts_gene, torch.Tensor) and counts_gene.is_sparse

    if is_scipy_sparse:
        n_spots, n_genes = counts_gene.shape
        # Ensure efficient column slicing
        if not scipy.sparse.isspmatrix_csc(counts_gene) and not scipy.sparse.isspmatrix_csr(counts_gene):
             counts_gene = counts_gene.tocsc()
    elif is_torch_sparse:
        n_spots, n_genes = counts_gene.shape
        if counts_gene.dtype != torch.float32 and counts_gene.dtype != torch.float64:
             counts_gene = counts_gene.float()
        if not counts_gene.is_coalesced():
            counts_gene = counts_gene.coalesce()
        # Pre-allocate selection vector for column extraction
        selection_vec = torch.zeros(n_genes, 1, device=counts_gene.device, dtype=counts_gene.dtype)
    else:
        if not isinstance(counts_gene, torch.Tensor):
            counts_gene = torch.from_numpy(counts_gene).float() # (n_spots, n_genes)
        n_spots, n_genes = counts_gene.shape
        y_dense = counts_gene - counts_gene.mean(0, keepdim=True) # center the counts, (n_spots, n_genes)

    hsic_list, pvals_list = [], []
    for i in range(n_genes):
        if is_scipy_sparse:
             col = counts_gene[:, i].toarray() # dense (n_spots, 1)
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
             counts = y_dense[:, i:i+1] # (n_spots, 1)

        lambda_y = counts.T @ counts # (1, 1)
        lambda_spy = lambda_sp * lambda_y # (rank, 1)
        hsic_scaled = torch.trace(K_sp.xtKx(counts)) # scalar
        pval = liu_sf((hsic_scaled * n_spots).numpy(), lambda_spy.numpy())

        hsic_list.append(hsic_scaled / (n_spots - 1) ** 2) # HSIC statistic
        pvals_list.append(pval)

    sv_test_results = {
        'statistic': torch.tensor(hsic_list).numpy(),
        'pvalue': torch.tensor(pvals_list).numpy(),
        'method': 'hsic-gc',
    }

    # calculate adjusted p-values
    sv_test_results['pvalue_adj'] = false_discovery_control(sv_test_results['pvalue'])

    return sv_test_results