# Non-parametric hypothesis testing for spatial splicing patterns
import warnings
import re
from scipy.stats import chi2, ttest_ind, combine_pvalues
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from smoother.weights import coordinate_to_weights_knn_sparse
from splisosm.utils import get_cov_sp, counts_to_ratios, false_discovery_control
from splisosm.kernel import SpatialCovKernel
from splisosm.likelihood import liu_sf

def _run_sparkx(coordinates, counts_list):
    """Wrapper for running the SPARK-X test for spatial gene expression variability.

    Args:
        coordinates: tensor(n_spots, 2), the spatial coordinates.
        counts_list: list of tensor(n_spots, n_isos), the observed isoform counts for each gene.

    Returns:
        sv_sparkx: dict, the results from the SPARK-X test.
    """
    # load packages neccessary for running SPARK-X
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    pandas2ri.activate()
    spark = importr('SPARK')

    # prepare robject inputs
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.numpy()
    coords_r = ro.conversion.py2rpy(coordinates) # (n_spots, 2)

    # merge isoform counts into gene counts
    # isog_counts = adata_ont[:, mapping_matrix.index].layers['counts'] @ mapping_matrix
    counts_g = torch.concat([_counts.sum(1, keepdim=True) for _counts in counts_list], axis = 1)
    counts_r = ro.conversion.py2rpy(counts_g.T.numpy()) # (n_genes, n_spots)
    counts_r.colnames = ro.vectors.StrVector(r['rownames'](coords_r))

    # run SPARK-X and extract outputs
    sparkx_res = spark.sparkx(counts_r, coords_r)
    sv_sparkx = ro.conversion.rpy2py(sparkx_res.rx['res_mtest'][0])
    sv_sparkx = {
        'statistic': ro.conversion.rpy2py(sparkx_res.rx['stats'][0]).mean(1),
        'pvalue': ro.conversion.rpy2py(sparkx_res.rx['res_mtest'][0])['combinedPval'].values,
        'pvalue_adj': ro.conversion.rpy2py(sparkx_res.rx['res_mtest'][0])['adjustedPval'].values,
        'method': 'spark-x',
    }

    return sv_sparkx

def _calc_ttest_differential_usage(data, groups, combine_pval = True, combine_method = 'tippett'):
    """Calculate the two-sample t-test statistic for differential usage.

    The t-test is applied to each isoform independently and combined if combine_pval is True.

    Args:
        data: tensor(n_spots, n_isos), the observed isoforms counts for a given gene.
        groups: tensor(n_spots), the binary group labels for each spot.
        combine_pval: bool, whether to combine p-values across isoforms.
        combine_method: str, the method to combine p-values.
              See scipy.stats.combine_pvalues() for more details.

    Returns:
        stats: tensor(n_isos) or 1, the t-test statistic.
        pval: tensor(n_isos) or 1, the p-value.
    """
    # check if groups contains more than two unique values
    _g = torch.unique(groups) # group labels
    if len(_g) > 2:
        raise ValueError("More than two groups detected. Only two are allowed for the two-sample t-test.")

    # run t-test per isoform
    t1 = data[groups == _g[0], :] # (k, n_isos)
    t2 = data[groups == _g[1], :] # (n_spots - k, n_isos)
    stats, pval = ttest_ind(t1, t2, axis=0, nan_policy='omit') # each of len n_isos

    # combine p-values across isoforms
    if combine_pval:
        stats, pval = combine_pvalues(pval, method = combine_method) # each of len 1

    return stats, pval


def linear_hsic_test(X, Y, centering = True):
    """The linear HSIC test.

    Equivalent to a multivariate extension of pearson correlation.

    Args:
        X: tensor(n_samples, n_features_x)
        Y: tensor(n_samples, n_features_y)
        centering: bool, whether to center the data.
    Returns:
        hsic: float, the HSIC statistic.
        pvalue: float, the p-value.
    """
    # if a sample contains NaN values in either X or Y, remove it
    is_nan = torch.isnan(X).any(1) | torch.isnan(Y).any(1)
    X = X[~is_nan]
    Y = Y[~is_nan]

    if centering:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

    n_samples = X.shape[0]
    eigv_th = 1e-5

    # calculate the HSIC statistic
    hsic_scaled = torch.norm(Y.T @ X, p = 'fro').pow(2)

    # find the eigenvalues of the kernel matrices
    lambda_x = torch.linalg.eigvalsh(X.T @ X) # length of n_features_x
    lambda_x = lambda_x[lambda_x > eigv_th] # remove small eigenvalues
    lambda_y = torch.linalg.eigvalsh(Y.T @ Y) # length of n_features_y
    lambda_y = lambda_y[lambda_y > eigv_th] # remove small eigenvalues

    # asymptotic null distribution
    lambda_xy = (lambda_x.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1) # length of n_features_x * n_features_y
    pval = liu_sf((hsic_scaled * n_samples).numpy(), lambda_xy.numpy())

    return (hsic_scaled / (n_samples - 1) ** 2), pval


def get_kernel_regression_residual_op(Kx, epsilon):
    """Calculate the residuals of kernel regression.

    Args:
        Kx: tensor(n_samples, n_samples), the kernel matrix of X.
        epsilon: float, regularization parameter.

    Returns:
        Rx: tensor(n_samples, n_samples), residual operator. Residuals(Y) := Y - Y_pred(X) = Rx @ Y.
    """
    Kx = 0.5 * (Kx + Kx.T) # symmetrize
    Rx = epsilon * torch.linalg.inv(Kx + epsilon * torch.eye(Kx.shape[0]))

    return Rx


def get_knn_regression_residual_op(X, k = 6):
    """Calculate the residuals of KNN regression.

    Args:
        X: tensor(n_samples, d). Input data.
        k: int, number of neighbors.

    Returns:
        Rx: tensor(n_samples, n_samples), residual operator. Residuals(Y) := Y - Y_pred(X) = Rx @ Y.
    """
    n_samples = X.shape[0]

    # build the KNN graph and convert to a row-normalized weights matrix
    # w.shape == (n_samples, n_samples)
    # w.sum(1) == [1] * n_samples
    w = coordinate_to_weights_knn_sparse(X, k=k, symmetric=True, row_scale=True) # sparse matrix

    # remove diagonals in the weights matrix if any to free unconnected samples
    w_i = w.indices()
    w_v = w.values()
    ndiag_mask = (w_i[0] != w_i[1]) # non-diagonal indices

    # calculate the residual operator (I - W)
    w_i_new = torch.concatenate(
        [w_i[:, ndiag_mask], torch.arange(n_samples).repeat(2, 1)], axis = 1
    )
    w_v_new = torch.concatenate(
        [w_v[ndiag_mask] * (-1), torch.ones(n_samples)]
    )
    Rx = torch.sparse_coo_tensor(w_i_new, w_v_new, w.shape, dtype=torch.float32).coalesce()

    return Rx


def fit_kernel_gpr(X, Y, normalize_x = True, normalize_y = True, return_residuals = True,
                   constant_value = 1.0, constant_value_bounds = (1e-3, 1e3),
                   length_scale = 1.0, length_scale_bounds = (1e-2, 1e+2)):
    """Fit a Gaussian process regression to learn parameters for kernel regression.

    Args:
        X: tensor(n_samples, d). Input data of d features.
        Y: tensor(n_samples, m). Output data of m targets.
        normalize_x: bool. Normalize the input data.
        return_residuals: bool. Return the residuals.
        Possible kernel configurations:
            constant_value: float. Constant kernel value.
            constant_value_bounds: tuple. Bounds for the constant kernel.
            length_scale: float. Length scale for the RBF kernel.
            length_scale_bounds: tuple. Bounds for the length scale.

    Returns:
        Kxy: tensor(n_samples, n_samples). Kernel matrix.
        epsilon: float. Regularization parameter.
        Y_residuals: tensor(n_samples, m). Residuals.
    """
    # remove samples that contains NaN values in Y
    n_samples_original = Y.shape[0]
    is_nan = torch.isnan(Y).any(1)
    X = X[~is_nan]
    Y = Y[~is_nan]
    n_samples = Y.shape[0]

    # normalize the input and target data if needed
    if normalize_x:
        X = (X - X.mean(0)) / X.std(0)
        X[torch.isinf(X)] = 0 # for constant columns

    if normalize_y:
        Y = (Y - Y.mean(0)) / Y.std(0)
        Y[torch.isinf(Y)] = 0 # for constant columns

    # specify the kernel choice
    # KernelX = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
    KernelX = C(constant_value, constant_value_bounds) * RBF(length_scale, length_scale_bounds) + \
        WhiteKernel(0.1, (1e-5, 1e+1))
    gpx = GaussianProcessRegressor(kernel=KernelX)

    # fit Gaussian process, including hyperparameter optimization
    gpx.fit(X, Y)

    # get the kernel matrix and regularization parameter
    Kxy = torch.from_numpy(gpx.kernel_.k1(X, X)).float()
    epsilon = np.exp(gpx.kernel_.theta[-1])

    if not return_residuals:
        return Kxy, epsilon

    # calculate the residuals
    Rx = get_kernel_regression_residual_op(Kxy, epsilon)
    Y_residuals = Rx @ Y

    if n_samples_original == n_samples:
        return Y_residuals

    # insert NaN values back to the residuals in the original order
    Y_residuals_full = torch.full((n_samples_original, Y_residuals.shape[1]), float('nan'))
    Y_residuals_full[~is_nan] = Y_residuals

    return Y_residuals_full


class SplisosmNP():
    """Non-parametric spatial isoform statistical model.

    Usages:
    - Spatial variability test:
        model = SplisosmNP()
        model.setup_data(data, coordinates)
        model.test_spatial_variability(method = 'hsic-ir', ...)

    - Differential usage test:
        model = SplisosmNP()
        model.setup_data(data, coordinates, design_mtx)
        model.test_diffential_usage(method = 'hsic', ...)

    - Retreive results:
        sv_results = model.get_formatted_test_results('sv')
        du_results = model.get_formatted_test_results('du')
    """
    def __init__(self):
        # to be set after running setup_data()
        self.n_genes = None # number of genes
        self.n_spots = None # number of spots
        self.n_isos = None # list of number of isoforms for each gene
        self.n_factors = None # number of covariates to test for differential usage

        # to store the spatial variability test results after running test_spatial_variability()
        self.sv_test_results = {}

        # to store the differential usage test results after running test_differential_usage()
        self.du_test_results = {}

    def __str__(self):
        _sv_status = f"Completed ({self.sv_test_results['method']})" if len(self.sv_test_results) > 0 else "NA"
        _du_status = f"Completed ({self.du_test_results['method']})" if len(self.du_test_results) > 0 else "NA"
        return f"=== Non-parametric SPLISOSM model for spatial isoform testings\n" + \
                f"- Number of genes: {self.n_genes}\n" + \
                f"- Number of spots: {self.n_spots}\n" + \
                f"- Number of covariates: {self.n_factors}\n" + \
                f"- Average number of isoforms per gene: {np.mean(self.n_isos) if self.n_isos is not None else None}\n" + \
                 "=== Test results\n" + \
                f"- Spatial variability test: {_sv_status}\n" + \
                f"- Differential usage test: {_du_status}"

    def setup_data(self, data, coordinates, approx_rank = None,
                   design_mtx = None, gene_names = None, covariate_names = None):
        """Setup the data for the model.

        Args:
            data: list of tensor(n_spots, n_isos), the observed isoform counts for each gene.
            coordinates: tensor(n_spots, 2), the spatial coordinates.
            approx_rank: int, the rank of the low-rank approximation for the spatial covariance matrix.
                If None, use the full-rank dense covariance matrix.
                For larger datasets (n_spots > 5,000), the maximum rank is set to 4 * sqrt(n_spots).
            design_mtx: tensor(n_spots, n_factors), the design matrix for the fixed effects.
            gene_names: list of str, the gene names.
        """
        self.n_genes = len(data) # number of genes
        self.n_spots = len(data[0]) # number of spots
        self.n_isos = [data_g.shape[1] for data_g in data] # number of isoforms for each gene
        self.gene_names = gene_names if gene_names is not None else [f"gene_{i + 1}" for i in range(self.n_genes)]
        assert len(self.gene_names) == self.n_genes, "Gene names must match the number of genes."
        assert min(self.n_isos) > 1, "At least two isoforms are required for each gene."

        # convert numpy.array to torch.tensor float if not already
        _data = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in data]
        self.data = [data_g.float() for data_g in _data] # [tensor(n_spots, n_isos)] * n_genes

        # create spatial covariance matrix from the coordinates
        assert coordinates.shape[0] == self.n_spots
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.from_numpy(coordinates).float()
        elif isinstance(coordinates, pd.DataFrame):
            coordinates = torch.from_numpy(coordinates.values).float()

        self.coordinates = coordinates

        # determine the maximum rank for spatial kernel computation
        if self.n_spots > 5000:
            # 10x Visium has 4992 spots per slide. For larger datasets (i.e. Slideseq-V2),
            # it is recommended to use low-rank approximation
            max_rank = np.ceil(np.sqrt(self.n_spots) * 4).astype(int)
            approx_rank = min(approx_rank, max_rank) if approx_rank is not None else max_rank
        else:
            if approx_rank is not None:
                approx_rank = approx_rank if approx_rank < self.n_spots else None

        # compute the spatial kernel
        K_sp = SpatialCovKernel(
            coordinates, k_neighbors=4, rho=0.99, centering=True, standardize_cov=True,
            approx_rank=approx_rank
        )

        # self.corr_sp = get_cov_sp(coordinates, k = 4, rho=0.99)
        self.corr_sp = K_sp

        # check the design matrix
        if design_mtx is not None:
            assert design_mtx.shape[0] == self.n_spots
            if isinstance(design_mtx, np.ndarray):
                design_mtx = torch.from_numpy(design_mtx)

            if design_mtx.dim() == 1: # in case of a single covariate
                design_mtx = design_mtx.unsqueeze(1)

            # convert to float tensor
            design_mtx = design_mtx.float()

            if covariate_names is not None:	# set default names
                assert len(covariate_names) == design_mtx.shape[1], "Covariate names must match the number of factors."
            else:
                covariate_names = [f"factor_{i + 1}" for i in range(design_mtx.shape[1])]

            # check for constant covariates
            _ind = torch.where(design_mtx.std(0) < 1e-5)[0]
            for _i in _ind:
                warnings.warn(f"{covariate_names[_i]} has zero variance. Please remove it.")

        self.design_mtx = design_mtx
        self.n_factors = design_mtx.shape[1] if design_mtx is not None else 0
        self.covariate_names = covariate_names

        # store the eigendecomposition of the spatial covariance matrix
        # try:
        #     corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.corr_sp)
        # except RuntimeError:
        #     # fall back to eig if eigh fails
        #     # related to a pytorch bug on M1 macs, see https://github.com/pytorch/pytorch/issues/83818
        #     corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eig(self.corr_sp)
        #     corr_sp_eigvals = torch.real(corr_sp_eigvals)
        #     corr_sp_eigvecs = torch.real(corr_sp_eigvecs)

        # self._corr_sp_eigvals = corr_sp_eigvals
        # self._corr_sp_eigvecs = corr_sp_eigvecs
        self._corr_sp_eigvals = self.corr_sp.eigenvalues()

    def get_formatted_test_results(self, test_type):
        """Get the formatted test results as data frame."""
        assert test_type in ['sv', 'du'], "Invalid test type. Must be one of 'sv' or 'du'."
        if test_type == 'sv':
            # check if the spatial variability test has been run
            assert len(self.sv_test_results) > 0, "No spatial variability test results found. Please run test_spatial_variability() first."
            # format the results
            res = pd.DataFrame({
                'gene': self.gene_names,
                'statistic': self.sv_test_results['statistic'],
                'pvalue': self.sv_test_results['pvalue'],
                'pvalue_adj': self.sv_test_results['pvalue_adj'],
            })
            return res
        else:
            # check if the differential usage test has been run
            assert len(self.du_test_results) > 0, "No differential usage test results found. Please run test_differential_usage() first."
            # format the results
            res = pd.DataFrame({
                'gene': np.repeat(self.gene_names, self.n_factors),
                'covariate': np.tile(self.covariate_names, self.n_genes),
                'statistic': self.du_test_results['statistic'].reshape(-1),
                'pvalue': self.du_test_results['pvalue'].reshape(-1),
                'pvalue_adj': self.du_test_results['pvalue_adj'].reshape(-1),
            })
            return res

    def test_spatial_variability(self, method = "hsic-ir",
                                 ratio_transformation = 'none', nan_filling = 'mean',
                                 use_perm_null = False, n_perms_per_gene = None,
                                 return_results = False, print_progress = True):
        """Test for spatial variability.

        Args:
            method: str, the test method. Must be one of "hsic-ir", "hsic-ic", "hsic-gc", "spark-x".
                Isoform-level tests (still one test per gene):
                    - "hsic-ir": HSIC test between multivariate isoform ratios (IR) and spatial locations.
                    - "hsic-ic": HSIC test between multivariate isoform counts (IC) and spatial locations.
                Gene-level tests:
                    - "hsic-gc": HSIC test between univariate gene-level counts (GC) and spatial locations.
                    - "spark-x": the SPARK-X test for variable gene expression. See [1].
            ratio_transformation: str, if using the isoform ratio ("hsic-ic"), what compositional transformation to use.
                Can be one of 'none', 'clr', 'ilr', 'alr', 'radial'[2].
            nan_filling: str, how to fill the NaN values in the isoform ratios. Can be 'mean' or 'none'.
            use_perm_null: bool, whether to generate the null distribution from permutation.
                If False, use the asymptotic distributions of chi-square mixtures with df = 1. See [3].
            return_results: bool, whether to return the test statistics and p-values.
            print_progress: bool, whether to show the progress bar. Default to True.

        References:
            [1] Zhu, Jiaqiang, Shiquan Sun, and Xiang Zhou. "SPARK-X: non-parametric modeling enables scalable
                   and robust detection of spatial expression patterns for large spatial transcriptomic studies."
                Genome biology 22.1 (2021): 184.
            [2] Park, Junyoung, et al. "Kernel methods for radial transformed compositional data with many zeros."
                International Conference on Machine Learning. PMLR, 2022.
            [3] Zhang, Kun, et al. "Kernel-based conditional independence test and application in causal discovery."
                arXiv preprint arXiv:1202.3775 (2012).
        """

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["mean", "none"]
        assert method in valid_methods, f"Invalid method. Must be one of {valid_methods}."
        assert ratio_transformation in valid_transformations, f"Invalid ratio transformation. Must be one of {valid_transformations}."
        assert nan_filling in valid_nan_filling, f"Invalid NaN filling method. Must be one of {valid_nan_filling}."

        if method == 'spark-x': # run the gene-level SPARK-X test
            self.sv_test_results = _run_sparkx(self.coordinates, self.data)
        else:
            # use a global spatial kernel unless nan_filling is 'none'
            n_spots = self.n_spots
            # H = torch.eye(n_spots) - 1/n_spots
            # K_sp = H @ self.corr_sp @ H # centered spatial kernel
            K_sp = self.corr_sp # the Kernel class object was already centered

            # calculate the eigenvalues of the spatial kernel
            if not use_perm_null:
                # lambda_sp = torch.linalg.eigvalsh(K_sp) # eigenvalues of length n_spots
                lambda_sp = self._corr_sp_eigvals # use precomputed eigenvalues
                lambda_sp = lambda_sp[lambda_sp > 1e-5] # remove small eigenvalues

            # prepare inputs for generating the null distribution
            if use_perm_null:
                n_nulls = n_perms_per_gene if n_perms_per_gene is not None else 1000

            # iterate over genes and calculate the HSIC statistic
            hsic_list, pvals_list = [], []
            for counts in tqdm(self.data, disable=(not print_progress)):
                if method == 'hsic-ir' and nan_filling == 'none':
                    # spetial treatment for the isoform ratio test when nan_filling is 'none'
                    # need to adjust the effective spot number (non NaN spots) and spatial kernel
                    y = counts_to_ratios(
                        counts, transformation = ratio_transformation, nan_filling = nan_filling
                    )
                    # remove spots with NaN values
                    is_nan = torch.isnan(y).any(1) # spots with NaN values
                    y = y[~is_nan] # (n_non_nan, n_isos)

                    # adjust the effective number of spots and update the spatial kernel
                    n_spots = y.shape[0]
                    # H = torch.eye(n_spots) - 1/n_spots
                    # K_sp = H @ self.corr_sp[~is_nan, :][:, ~is_nan] @ H # centered spatial kernel
                    K_sp = self.corr_sp.realization()[~is_nan, :][:, ~is_nan]
                    K_sp = K_sp - K_sp.mean(dim=0, keepdim=True)
                    K_sp = K_sp - K_sp.mean(dim=1, keepdim=True)

                    # calculate the eigenvalues of the new per-gene spatial kernel
                    if not use_perm_null:
                        lambda_sp = torch.linalg.eigvalsh(K_sp) # eigenvalues of length n_spots
                        lambda_sp = lambda_sp[lambda_sp > 1e-5] # remove small eigenvalues

                    # compute the hsic statistic
                    hsic_scaled = torch.trace(y.T @ K_sp @ y)

                else: # one global spatial kernel for all genes
                    if method == 'hsic-ic': # use isoform-level count data
                        y = counts - counts.mean(0, keepdim=True) # centering per isoform
                    elif method == 'hsic-gc': # use gene-level count data
                        y = counts.sum(1, keepdim=True)
                        y = y - y.mean() # centering per isoform
                    else: # use isoform ratio
                        # calculate the isoform ratio from counts
                        y = counts_to_ratios(
                            counts, transformation = ratio_transformation, nan_filling = nan_filling
                        )
                        y = y - y.mean(0, keepdim=True) # centering per isoform

                    # calculate the HSIC statistic
                    hsic_scaled = torch.trace(K_sp.xtKx(y)) # equivalent to y.T @ K_sp @ y

                hsic_list.append(hsic_scaled / (n_spots - 1) ** 2)

                if use_perm_null: # permutation-based null distribution
                    # randomly shuffle the spatial locations
                    perm_idx = torch.stack([torch.randperm(n_spots) for _ in range(n_nulls)])
                    yy = y[perm_idx,:] # (n_nulls, n_spots, n_isos)

                    # calculate the null HSIC statistics
                    null_m = torch.einsum('bii->b', (yy.transpose(1,2) @ K_sp.unsqueeze(0) @ yy))

                    # calculate the p-value
                    pval = (null_m > hsic_scaled).sum() / n_nulls

                else: # asymptotic null distribution
                    lambda_y = torch.linalg.eigvalsh(y.T @ y) # length of n_isos
                    lambda_spy = (lambda_sp.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1) # n_spots * (n_isos or 1)
                    pval = liu_sf((hsic_scaled * n_spots).numpy(), lambda_spy.numpy())

                pvals_list.append(pval)

                # store the results
                self.sv_test_results = {
                    'statistic': torch.tensor(hsic_list).numpy(),
                    'pvalue': torch.tensor(pvals_list).numpy(),
                    'method': method,
                    'use_perm_null': use_perm_null,
                }

            # calculate adjusted p-values
            self.sv_test_results['pvalue_adj'] = false_discovery_control(self.sv_test_results['pvalue'])

        # return results
        if return_results:
            return self.sv_test_results

    def test_differential_usage(self, method = "hsic-gp",
                                ratio_transformation = 'none', nan_filling = 'mean',
                                hsic_eps = 1e-3, gp_configs = None,
                                print_progress = True, return_results = False):
        """Test for spatial isoform differential usage.

        Args:
            method: str, the test method. Must be one of "hsic", "hsic-knn", "hsic-gp", "t-fisher", "t-tippett".
                HSIC tests:
                    - "hsic": HSIC test for isoform differential usage along each factor in the design matrix.
                        - hsic_eps: float, the regularization parameter for HSIC.
                            A kernel regression is used to remove the spatial effect from y and z. See [1].
                              If None, test the unconditional H_0: y /independent z.
                            Otherwise, test the conditional H_0: y /independent z | x where x is the spatial coordinates.
                        - For continuous factors, it is equivalent to the (partial) pearson correlation test.
                        - For binary factors, it is equivalent to the two-sample t-test.
                    - "hsic-knn": conditional HSIC test using KNN regression to remove spatial effect.
                    - "hsic-gp": conditional HSIC test using kernels learned from Gaussian process regression.
                        - gp_configs: dict, the kernel configurations for the GP regression to pass to fit_kernel_gpr().
                            Using fixed length scale by default for efficiency.
                T-tests (binary factors only): "t-fisher", "t-tippett".
                    - Two-sample t-test for isoform differential usage along each factor in the design matrix.
                    - The test is applied to the ratio of each isoform independently and combined using one of 'fisher' or 'tippett'.
            ratio_transformation: str, what compositional transformation to use for isoform ratio.
                Can be one of 'none', 'clr', 'ilr', 'alr', 'radial' [2].
            nan_filling: str, how to fill the NaN values in the isoform ratios. Can be 'mean' or 'none'.
            print_progress: bool, whether to show the progress bar. Default to True.
            return_results: bool, whether to return the test statistics and p-values.

        References:
            [1] Park, Junyoung, et al. "Kernel methods for radial transformed compositional data with many zeros."
                International Conference on Machine Learning. PMLR, 2022.
            [2] Zhang, Kun, et al. "Kernel-based conditional independence test and application in causal discovery."
                arXiv preprint arXiv:1202.3775 (2012).
        """
        if self.design_mtx is None:
            raise ValueError("Cannot find the design matrix. Perhaps you forgot to set it up using setup_data().")

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["hsic", "hsic-knn", "hsic-gp", "t-fisher", "t-tippett"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["none", "mean"]
        assert method in valid_methods, f"Invalid method. Must be one of {valid_methods}."
        assert ratio_transformation in valid_transformations, f"Invalid transformation. Must be one of {valid_transformations}."
        assert nan_filling in valid_nan_filling, f"Invalid nan_filling. Must be one of {valid_nan_filling}."

        # TODO
        if method in ['hsic', 'hsic-knn']: # HSIC-based test with pre-specified kernel
            # x: spatial coordinates, z: factor of interest, y: isoform usage
            # need to first regress out the spatial effect x from x and y

            if nan_filling == 'mean': # no NaN values in the ratio
                # use the same spatial kernel matrix for all genes
                if method == 'hsic-knn': # use KNN regression
                    Rx = get_knn_regression_residual_op(self.coordinates, k = 4)
                else: # use kernel regression
                    # calculate the residual operator Rx
                    # if hsic_eps is None, testing the unconditional H_0: y \independent z
                    # otherwise, testing the conditional H_0: y \independent z | x
                    if hsic_eps is None: # unconditional HSIC test
                        Rx = torch.eye(n_spots)
                    else: # conditional HSIC test
                        assert hsic_eps > 0, "The regularization parameter hsic_eps must be positive."
                        # prepare the spatial kernel matrix
                        # H = torch.eye(n_spots) - 1/n_spots
                        # Kx = H @ self.corr_sp @ H # centered kernel for spatial coordinates
                        Kx = self.corr_sp.realization() # the Kernel class object was already centered
                        # regularized kernel regression
                        Rx = get_kernel_regression_residual_op(Kx, hsic_eps)

                hsic_list, pvals_list = [], []
                # iterate over factors
                for _ind in tqdm(range(n_factors), disable=(not print_progress), dynamic_ncols=True):
                    # center the factor of interest
                    z = self.design_mtx[:, _ind].clone() # len of n_spots
                    assert z.std() > 0, f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                    z = Rx @ z # regression residual
                    z = (z - z.mean()) / z.std() # normalize the factor of interest for stability
                    z = z.unsqueeze(1) # (n_spots, 1)

                    _hsic_ind, _pvals_ind = [], []
                    # iterate over genes and calculate the HSIC statistic
                    for counts in self.data:
                        # calculate isoform usage ratio (n_spots, n_isos)
                        y = counts_to_ratios(counts, transformation = ratio_transformation, nan_filling='mean')
                        y = Rx @ y # regression residual, (n_spots, n_isos)
                        y = y - y.mean(0, keepdim=True) # centering per isoform

                        # calculate the HSIC statistic
                        _hsic, _pval = linear_hsic_test(z, y, centering = True)

                        _hsic_ind.append(_hsic)
                        _pvals_ind.append(_pval)

                    # stack the results
                    hsic_list.append(torch.tensor(_hsic_ind))
                    pvals_list.append(torch.tensor(_pvals_ind))

                # combine results
                hsic_all = torch.stack(hsic_list, dim=1)
                pvals_all = torch.stack(pvals_list, dim=1)

            else: # nan_filling == 'none', NaN values in the ratio
                hsic_list, pvals_list = [], []
                # iterate over genes and use different spatial kernel matrix for different genes
                for counts in tqdm(self.data, disable=(not print_progress), dynamic_ncols=True):
                    # calculate isoform usage ratio (n_spots, n_isos)
                    y = counts_to_ratios(counts, transformation = ratio_transformation, nan_filling='none')

                    # remove NaN spots
                    is_nan = torch.isnan(y).any(1) # spots with NaN values, (n_spots,)

                    # calculate the residual operator Rx
                    if method == 'hsic-knn': # use KNN regression
                        Rx = get_knn_regression_residual_op(self.coordinates[~is_nan], k = 4)
                    else: # use kernel regression
                        # calculate the residual operator Rx
                        # if hsic_eps is None, testing the unconditional H_0: y \independent z
                        # otherwise, testing the conditional H_0: y \independent z | x
                        if hsic_eps is None: # unconditional HSIC test
                            Rx = torch.eye(n_spots - is_nan.sum())
                        else: # conditional HSIC test
                            assert hsic_eps > 0, "The regularization parameter hsic_eps must be positive."
                            # create the spatial kernel matrix as the principal submatrix
                            # Kx = self.corr_sp[~is_nan,:][:,~is_nan] # (n_non_nan, n_non_nan)
                            # H = torch.eye(Kx.shape[0]) - 1/Kx.shape[0]
                            # Kx = H @ Kx @ H # centered spatial kernel, (n_non_nan, n_non_nan)
                            K_x = self.corr_sp.realization()[~is_nan, :][:, ~is_nan]
                            K_x = K_x - K_x.mean(dim=0, keepdim=True)
                            K_x = K_sp - K_x.mean(dim=1, keepdim=True) # (n_non_nan, n_non_nan)

                            # regularized kernel regression
                            Rx = get_kernel_regression_residual_op(Kx, hsic_eps)

                    # calculate the residuals
                    y = Rx @ y[~is_nan] # regression residual, (n_non_nan, n_isos)

                    # center the factor of interest
                    y = y - y.mean(0, keepdim=True) # centering per isoform

                    _hsic_ind, _pvals_ind = [], []
                    # iterate over factors
                    for _ind in range(n_factors):
                        # center the factor of interest
                        z = self.design_mtx[~is_nan, _ind].clone() # (n_non_nan,)
                        assert z.std() > 0, f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                        z = Rx @ z # regression residual, (n_non_nan,)
                        z = (z - z.mean()) / z.std() # normalize the factor of interest for stability
                        z = z.unsqueeze(1) # (n_non_nan, 1)

                        # calculate the HSIC statistic
                        _hsic, _pval = linear_hsic_test(z, y, centering = True)

                        _hsic_ind.append(_hsic)
                        _pvals_ind.append(_pval)

                    # stack the results
                    hsic_list.append(torch.tensor(_hsic_ind))
                    pvals_list.append(torch.tensor(_pvals_ind))

                # combine results
                hsic_all = torch.stack(hsic_list, dim=0) # (n_genes, n_factors)
                pvals_all = torch.stack(pvals_list, dim=0) # (n_genes, n_factors)

            # store the results
            self.du_test_results = {
                'statistic': hsic_all.numpy(), # (n_genes, n_factors)
                'pvalue': pvals_all.numpy(), # (n_genes, n_factors)
                'method': method,
            }

        elif method == 'hsic-gp': # HSIC-based test with kernel learned by Gaussian process regression
            # specify the GP kernel configurations
            _default_gp_configs = {
                'constant_value_covariate': 1.0,
                'length_scale_covariate': 1.0,
                'constant_value_bounds_covariate': (1e-3, 1e3),
                'length_scale_bounds_covariate': 'fixed',
                'constant_value_isoform': 1e-3,
                'length_scale_isoform': 1.0,
                'constant_value_bounds_isoform': 'fixed',
                'length_scale_bounds_isoform': 'fixed',
            }
            if gp_configs is None:
                gp_configs = _default_gp_configs
            else: # update the config
                gp_configs = {**_default_gp_configs, **gp_configs}

            # normalize the spatial coordinates
            x = self.coordinates.clone() # (n_spots, 2)
            x = (x - x.mean(0)) / x.std(0) # normalize the spatial coordinates
            x[torch.isinf(x)] = 0 # for constant columns

            # run GP regression for every factor
            z_res_list = []
            for _ind in tqdm(range(n_factors), disable=(not print_progress), dynamic_ncols=True):
                # center the factor of interest
                z = self.design_mtx[:, _ind].clone() # (n_spots,)
                assert z.std() > 0, f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z = (z - z.mean()) / z.std()
                z = z.unsqueeze(1) # (n_spots, 1)

                # fit the kernel regression and store results
                z_res = fit_kernel_gpr(
                    x, z, normalize_x = False, normalize_y = False, return_residuals = True,
                    constant_value = gp_configs['constant_value_covariate'],
                    constant_value_bounds = gp_configs['constant_value_bounds_covariate'],
                    length_scale = gp_configs['length_scale_covariate'],
                    length_scale_bounds = gp_configs['length_scale_bounds_covariate']
                )
                z_res_list.append(z_res)

            # run GP regression for every gene
            y_res_list = []
            for counts in tqdm(self.data, disable=(not print_progress), dynamic_ncols=True):
                # calculate isoform usage ratio (n_spots, n_isos)
                y = counts_to_ratios(counts, transformation = ratio_transformation, nan_filling = nan_filling)

                # center the isoform usage for non-NaN spots
                if nan_filling == 'none':
                    is_nan = torch.isnan(y).any(1) # spots with NaN values
                    y[~is_nan] = y[~is_nan] - y[~is_nan].mean(0, keepdim=True) # y still of (n_spots, n_isos)
                else:
                    y = y - y.mean(0, keepdim=True) # (n_spots, n_isos)

                # fit the kernel regression and store results
                y_res = fit_kernel_gpr(
                    x, y, normalize_x = False, normalize_y = False, return_residuals = True,
                    constant_value = gp_configs['constant_value_isoform'],
                    constant_value_bounds = gp_configs['constant_value_bounds_isoform'],
                    length_scale = gp_configs['length_scale_isoform'],
                    length_scale_bounds = gp_configs['length_scale_bounds_isoform']
                )
                y_res_list.append(y_res)

            # calculate the HSIC statistic
            hsic_list, pvals_list = [], []
            # iterate over factors
            for _z in z_res_list:
                _hsic_ind, _pvals_ind = [], []
                # iterate over genes and calculate the HSIC statistic
                for _y in y_res_list:
                    # calculate the HSIC statistic
                    _hsic, _pval = linear_hsic_test(_z, _y, centering = True)
                    _hsic_ind.append(_hsic)
                    _pvals_ind.append(_pval)

                # stack the results
                hsic_list.append(torch.tensor(_hsic_ind))
                pvals_list.append(torch.tensor(_pvals_ind))

            # combine results
            hsic_all = torch.stack(hsic_list, dim=1)
            pvals_all = torch.stack(pvals_list, dim=1)

            # store the results
            self.du_test_results = {
                'statistic': hsic_all.numpy(), # (n_genes, n_factors)
                'pvalue': pvals_all.numpy(), # (n_genes, n_factors)
                'method': method,
            }

        else: # two-sample t-test
            # method to combine p-values across isoforms, either 'fisher' or 'tippett'
            combine_method = re.findall(r'^t-(.+)', method)[0]

            _du_ttest_stats_all, _du_ttest_pvals_all = [], []

            # iterate over factors
            for _ind in range(n_factors):
                # iterate over genes and calculate the t-test statistic
                _du_ttest_stats, _du_ttest_pvals = [], []
                for counts in self.data:
                    # calculate isoform usage ratio (n_spots, n_isos)
                    ratios = counts_to_ratios(
                        counts, transformation = ratio_transformation, nan_filling = nan_filling
                    )

                    # run t-test and combine p-values
                    _stats, _pvals = _calc_ttest_differential_usage(
                        ratios, self.design_mtx[:, _ind],
                        combine_pval=True, combine_method=combine_method
                    )
                    _du_ttest_stats.append(_stats)
                    _du_ttest_pvals.append(_pvals)

                _du_ttest_stats = np.stack(_du_ttest_stats, axis=0)
                _du_ttest_pvals = np.stack(_du_ttest_pvals, axis=0)

                _du_ttest_stats_all.append(_du_ttest_stats)
                _du_ttest_pvals_all.append(_du_ttest_pvals)

            # combine results
            _du_ttest_stats_all = np.stack(_du_ttest_stats_all, axis=1)
            _du_ttest_pvals_all = np.stack(_du_ttest_pvals_all, axis=1)

            # store the results
            self.du_test_results = {
                'statistic': _du_ttest_stats_all, # (n_genes, n_factors)
                'pvalue': _du_ttest_pvals_all, # (n_genes, n_factors)
                'method': method,
            }

        # calculate adjusted p-values (independently for each factor)
        self.du_test_results['pvalue_adj'] = false_discovery_control(
            self.du_test_results['pvalue'], axis=0
        )

        # return the results if needed
        if return_results:
            return self.du_test_results

