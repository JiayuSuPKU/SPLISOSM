"""Non-parametric hypothesis tests for spatial isoform usage."""

from __future__ import annotations

import warnings
import re
from typing import Any, Optional, Union, Literal
from scipy.stats import ttest_ind, combine_pvalues
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from anndata import AnnData

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from smoother.weights import coordinate_to_weights_knn_sparse
from splisosm.utils import (
    counts_to_ratios,
    false_discovery_control,
    prepare_inputs_from_anndata,
    run_sparkx,
)
from splisosm.kernel import SpatialCovKernel
from splisosm.likelihood import liu_sf

__all__ = [
    "linear_hsic_test",
    "fit_kernel_gpr",
    "SplisosmNP",
]


def _calc_ttest_differential_usage(
    data, groups, combine_pval=True, combine_method="tippett"
):
    """Calculate the two-sample t-test statistic for differential usage.

    The t-test is applied to each isoform independently and combined if combine_pval is True.

    Parameters
    ----------
    data : torch.Tensor
        Shape (n_spots, n_isos), the observed isoforms counts for a given gene.
    groups : torch.Tensor
        Shape (n_spots), the binary group labels for each spot.
    combine_pval : bool, optional
        Whether to combine p-values across isoforms.
    combine_method : str, optional
        The method to combine p-values. See scipy.stats.combine_pvalues() for more details.

    Returns
    -------
    stats : torch.Tensor
        Shape (n_isos) or (1,), the t-test statistic.
    pval : torch.Tensor
        Shape (n_isos) or (1,), the p-value.
    """
    # check if groups contains more than two unique values
    _g = torch.unique(groups)  # group labels
    if len(_g) > 2:
        raise ValueError(
            "More than two groups detected. Only two are allowed for the two-sample t-test."
        )

    # run t-test per isoform
    t1 = data[groups == _g[0], :]  # (k, n_isos)
    t2 = data[groups == _g[1], :]  # (n_spots - k, n_isos)
    stats, pval = ttest_ind(t1, t2, axis=0, nan_policy="omit")  # each of len n_isos

    # combine p-values across isoforms
    if combine_pval:
        stats, pval = combine_pvalues(pval, method=combine_method)  # each of len 1

    return stats, pval


def linear_hsic_test(
    X: torch.Tensor, Y: torch.Tensor, centering: bool = True
) -> tuple[float, float]:
    """The linear HSIC test.

    Equivalent to a multivariate extension of Pearson correlation.

    Parameters
    ----------
    X
        Shape (n_samples, n_features_x).
    Y
        Shape (n_samples, n_features_y).
    centering
        Whether to center the data. If False, assume the data is already centered.

    Returns
    -------
    hsic : float
        The HSIC statistic.
    pvalue : float
        The p-value.
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
    hsic_scaled = torch.norm(Y.T @ X, p="fro").pow(2)

    # find the eigenvalues of the kernel matrices
    lambda_x = torch.linalg.eigvalsh(X.T @ X)  # length of n_features_x
    lambda_x = lambda_x[lambda_x > eigv_th]  # remove small eigenvalues
    lambda_y = torch.linalg.eigvalsh(Y.T @ Y)  # length of n_features_y
    lambda_y = lambda_y[lambda_y > eigv_th]  # remove small eigenvalues

    # asymptotic null distribution
    lambda_xy = (lambda_x.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(
        -1
    )  # length of n_features_x * n_features_y
    pval = liu_sf((hsic_scaled * n_samples).numpy(), lambda_xy.numpy())

    return (hsic_scaled / (n_samples - 1) ** 2), pval


def get_kernel_regression_residual_op(Kx: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Calculate the residuals of kernel regression.

    Parameters
    ----------
    Kx : torch.Tensor
        Shape (n_samples, n_samples), the kernel matrix of X.
    epsilon : float
        Regularization parameter.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, n_samples), residual operator. Residuals(Y) := Y - Y_pred(X) = Rx @ Y.
    """
    Kx = 0.5 * (Kx + Kx.T)  # symmetrize
    Rx = epsilon * torch.linalg.inv(Kx + epsilon * torch.eye(Kx.shape[0]))

    return Rx


def get_knn_regression_residual_op(X: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Calculate the residuals of KNN regression.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, d). Input data.
    k : int, optional
        Number of neighbors.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, n_samples), residual operator. Residuals(Y) := Y - Y_pred(X) = Rx @ Y.
    """
    n_samples = X.shape[0]

    # build the KNN graph and convert to a row-normalized weights matrix
    # w.shape == (n_samples, n_samples)
    # w.sum(1) == [1] * n_samples
    w = coordinate_to_weights_knn_sparse(
        X, k=k, symmetric=True, row_scale=True
    )  # sparse matrix

    # remove diagonals in the weights matrix if any to free unconnected samples
    w_i = w.indices()
    w_v = w.values()
    ndiag_mask = w_i[0] != w_i[1]  # non-diagonal indices

    # calculate the residual operator (I - W)
    w_i_new = torch.concatenate(
        [w_i[:, ndiag_mask], torch.arange(n_samples).repeat(2, 1)], axis=1
    )
    w_v_new = torch.concatenate([w_v[ndiag_mask] * (-1), torch.ones(n_samples)])
    Rx = torch.sparse_coo_tensor(
        w_i_new, w_v_new, w.shape, dtype=torch.float32
    ).coalesce()

    return Rx


def fit_kernel_gpr(
    X: torch.Tensor,
    Y: torch.Tensor,
    normalize_x: bool = True,
    normalize_y: bool = True,
    return_residuals: bool = True,
    constant_value: float = 1.0,
    constant_value_bounds: tuple[float, float] = (1e-3, 1e3),
    length_scale: float = 1.0,
    length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
) -> Union[tuple[torch.Tensor, float], torch.Tensor]:
    """Fit a Gaussian process regression to learn parameters for kernel regression.

    Y ~ GaussianProcessRegressor(X, kernel = ConstantKernel * RBF + WhiteNoise)

    Parameters
    ----------
    X
        Shape (n_samples, d). Input data of d features.
    Y
        Shape (n_samples, m). Output data of m targets.
    normalize_x
        Whether to normalize the input data.
    normalize_y
        Whether to normalize the output data.
    return_residuals
        Whether to return the regression residuals ``Y_residuals`` :math:`Y - \\hat{Y}`.
    constant_value
        Constant kernel value.
    constant_value_bounds
        Bounds for the constant kernel value to search.
    length_scale
        Length scale for the RBF kernel.
    length_scale_bounds
        Bounds for the RBF length scale to search.

    Returns
    -------
    tuple[torch.Tensor, float] or torch.Tensor
        If `return_residuals` is False, returns the estimated kernel and regularization strength (``Kxy``, ``epsilon``).
        If `return_residuals` is True, returns the regression residual ``Y_residuals`` of shape (n_samples, m).

    Notes
    -----
    This function is a wrapper of `sklearn.gaussian_process.GaussianProcessRegressor`.
    It is possible to speed up model fitting via more efficient implementations of Gaussian process regression, e.g., `GPyTorch`.
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
        X[torch.isinf(X)] = 0  # for constant columns

    if normalize_y:
        Y = (Y - Y.mean(0)) / Y.std(0)
        Y[torch.isinf(Y)] = 0  # for constant columns

    # specify the kernel choice
    # KernelX = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
    KernelX = C(constant_value, constant_value_bounds) * RBF(
        length_scale, length_scale_bounds
    ) + WhiteKernel(0.1, (1e-5, 1e1))
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
    Y_residuals_full = torch.full(
        (n_samples_original, Y_residuals.shape[1]), float("nan")
    )
    Y_residuals_full[~is_nan] = Y_residuals

    return Y_residuals_full


class SplisosmNP:
    """Non-parametric spatial isoform statistical model.

    Examples
    --------
    Setup data:

    >>> from splisosm import SplisosmNP
    >>> import torch
    >>> # Simulate data for 10 genes with different number of isoforms
    >>> data_3_iso = [torch.randint(low=0, high=5, size=(100, 3)) for _ in range(5)]  # 5 genes with 3 isoforms
    >>> data_4_iso = [torch.randint(low=0, high=5, size=(100, 4)) for _ in range(5)]  # 5 genes with 4 isoforms
    >>> data = data_3_iso + data_4_iso
    >>> coordinates = torch.rand(100, 2)  # 100 spots with 2D coordinates
    >>> design_mtx = torch.rand(100, 2)  # 100 spots with 2 covariates

    Spatial variability test:

    >>> model = SplisosmNP()
    >>> model.setup_data(data, coordinates)
    >>> model.test_spatial_variability(method = 'hsic-ir')
    >>> sv_results = model.get_formatted_test_results('sv')
    >>> print(sv_results.head())

    Differential usage test:

    >>> model = SplisosmNP()
    >>> model.setup_data(data, coordinates, design_mtx=design_mtx)
    >>> model.test_differential_usage(method = 'hsic')
    >>> du_results = model.get_formatted_test_results('du')
    >>> print(du_results.head())
    """

    n_genes: int
    """Number of genes."""

    n_spots: int
    """Number of spots."""

    n_isos: list[int]
    """List of numbers of isoforms per gene."""

    n_factors: int
    """Number of covariates to test for differential usage."""

    sv_test_results: dict
    """Dictionary to store the spatial variability test results after running test_spatial_variability().
    It contains the following keys:
    
    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes,), the test statistic for each gene.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes,), the p-value for each gene.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes,), the BH adjusted p-value for each gene.
    """

    du_test_results: dict
    """Dictionary to store the differential usage test results after running test_differential_usage(). 
    It contains the following keys:
    
    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes, n_factors), the test statistic for each gene and covariate.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes, n_factors), the p-value for each gene and covariate.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes, n_factors), the BH adjusted p-value for each gene and covariate. Each column/covariate is adjusted separately.
    """

    def __init__(self) -> None:
        # to be set after running setup_data()
        self.n_genes = None  # number of genes
        self.n_spots = None  # number of spots
        self.n_isos = None  # list of number of isoforms for each gene
        self.n_factors = None  # number of covariates to test for differential usage
        self.adata = None  # optional anndata source for the new setup path
        self.setup_input_mode = None  # "legacy" or "anndata"

        # to store the spatial variability test results after running test_spatial_variability()
        self.sv_test_results = {}

        # to store the differential usage test results after running test_differential_usage()
        self.du_test_results = {}

    def __str__(self) -> str:
        """Return string representation of the model."""
        _sv_status = (
            f"Completed ({self.sv_test_results['method']})"
            if len(self.sv_test_results) > 0
            else "NA"
        )
        _du_status = (
            f"Completed ({self.du_test_results['method']})"
            if len(self.du_test_results) > 0
            else "NA"
        )
        return (
            "=== Non-parametric SPLISOSM model for spatial isoform testings\n"
            + f"- Number of genes: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Average number of isoforms per gene: {np.mean(self.n_isos) if self.n_isos is not None else None}\n"
            + "=== Test results\n"
            + f"- Spatial variability test: {_sv_status}\n"
            + f"- Differential usage test: {_du_status}"
        )

    def setup_data(
        self,
        data: Optional[list[Union[torch.Tensor, np.ndarray]]] = None,
        coordinates: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        approx_rank: Optional[int] = None,
        design_mtx: Optional[
            Union[torch.Tensor, np.ndarray, pd.DataFrame, str, list[str]]
        ] = None,
        gene_names: Optional[Union[list[str], str]] = None,
        covariate_names: Optional[list[str]] = None,
        *,
        adata: Optional[AnnData] = None,
        spatial_key: str = "spatial",
        layer: str = "counts",
        group_iso_by: str = "gene_symbol",
        min_counts: int = 10,
        min_bin_pct: float = 0.0,
        filter_single_iso_genes: bool = True,
    ) -> None:
        """Setup isoform-level spatial data for hypothesis testing.

        This method supports two input modes for backward compatibility.

        - Legacy mode: pass ``data`` and ``coordinates`` directly.
        - AnnData mode: pass ``adata``, where counts are extracted from
          ``adata.layers[layer]`` grouped by ``group_iso_by``, and coordinates
          are read from ``adata.obsm[spatial_key]``.
          See :func:`splisosm.utils.prepare_inputs_from_anndata` for details.

        Parameters
        ----------
        data
            Legacy mode only. List of tensors/arrays with shape
            ``(n_spots, n_isos)`` containing isoform counts for each gene.
        coordinates
            Legacy mode only. Shape ``(n_spots, 2)``, spatial coordinates.
        approx_rank
            The rank of the low-rank approximation for the spatial covariance matrix.
            If None, use the full-rank dense covariance matrix.
            For larger datasets (n_spots > 5,000), the maximum rank is set to ``4 * sqrt(n_spots)``.
        design_mtx
            Design matrix for differential usage tests.

            - Legacy mode: tensor/array/dataframe of shape ``(n_spots, n_factors)``.
            - AnnData mode: tensor/array/dataframe, or one obs-column name
              (str), or a list of obs-column names.
        gene_names
            Gene names.

            - Legacy mode: list of gene names.
            - AnnData mode: optional column name in ``adata.var`` used as
              display names for grouped genes; if None, use grouped gene IDs.
        covariate_names
            List of covariate names.
        adata
            AnnData object used in the new input mode.
        spatial_key
            Key in ``adata.obsm`` for spatial coordinates.
        layer
            Counts layer in ``adata.layers``.
        group_iso_by
            Column in ``adata.var`` used to group isoforms by gene.
        min_counts
            Minimum total isoform count across spots required to retain an isoform
            in AnnData mode.
        min_bin_pct
            Minimum percentage/fraction of spots where an isoform is expressed in
            AnnData mode. Values in ``[0, 1]`` are treated as fractions; values in
            ``(1, 100]`` are treated as percentages.
        filter_single_iso_genes
            AnnData mode only. Whether to remove genes with fewer than two retained
            isoforms.

        Raises
        ------
        ValueError
            If input arguments are invalid or required fields are missing.
        """
        if adata is not None:
            if data is not None or coordinates is not None:
                raise ValueError(
                    "When `adata` is provided, `data` and `coordinates` should not be provided."
                )

            (
                extracted_data,
                extracted_coordinates,
                extracted_gene_names,
                extracted_design_mtx,
                extracted_covariate_names,
            ) = prepare_inputs_from_anndata(
                adata=adata,
                design_mtx=design_mtx,
                gene_names=gene_names,
                covariate_names=covariate_names,
                spatial_key=spatial_key,
                layer=layer,
                group_iso_by=group_iso_by,
                min_counts=min_counts,
                min_bin_pct=min_bin_pct,
                filter_single_iso_genes=filter_single_iso_genes,
            )

            data = extracted_data
            coordinates = extracted_coordinates
            design_mtx = extracted_design_mtx
            gene_names = extracted_gene_names
            covariate_names = extracted_covariate_names

            self.adata = adata
            self.setup_input_mode = "anndata"
        else:
            if data is None or coordinates is None:
                raise ValueError(
                    "Provide either (`data`, `coordinates`) for legacy mode, or `adata` for AnnData mode."
                )

            if isinstance(gene_names, str):
                raise ValueError(
                    "In legacy mode, `gene_names` must be a list of names or None."
                )

            if isinstance(design_mtx, str) or (
                isinstance(design_mtx, list)
                and len(design_mtx) > 0
                and isinstance(design_mtx[0], str)
            ):
                raise ValueError(
                    "In legacy mode, `design_mtx` must be a matrix-like object, not column names."
                )

            self.adata = None
            self.setup_input_mode = "legacy"

        self.n_genes = len(data)  # number of genes
        self.n_spots = len(data[0])  # number of spots
        self.n_isos = [
            data_g.shape[1] for data_g in data
        ]  # number of isoforms for each gene
        self.gene_names = (
            gene_names
            if gene_names is not None
            else [f"gene_{i + 1}" for i in range(self.n_genes)]
        )
        if len(self.gene_names) != self.n_genes:
            raise ValueError("Gene names must match the number of genes.")
        if min(self.n_isos) <= 1:
            raise ValueError("At least two isoforms are required for each gene.")

        # convert numpy.array to torch.tensor float if not already
        _data = [
            torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
            for arr in data
        ]
        self.data = [
            data_g.float() for data_g in _data
        ]  # [tensor(n_spots, n_isos)] * n_genes

        # create spatial covariance matrix from the coordinates
        if coordinates.shape[0] != self.n_spots:
            raise ValueError(
                "The number of coordinate rows must match the number of spots."
            )
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
            approx_rank = (
                min(approx_rank, max_rank) if approx_rank is not None else max_rank
            )
        else:
            if approx_rank is not None:
                approx_rank = approx_rank if approx_rank < self.n_spots else None

        # compute the spatial kernel
        K_sp = SpatialCovKernel(
            coordinates,
            k_neighbors=4,
            rho=0.99,
            centering=True,
            standardize_cov=True,
            approx_rank=approx_rank,
        )

        # self.corr_sp = get_cov_sp(coordinates, k = 4, rho=0.99)
        self.corr_sp = K_sp

        # check the design matrix
        if design_mtx is not None:
            if isinstance(design_mtx, pd.DataFrame):
                design_mtx = torch.from_numpy(design_mtx.values)
            elif isinstance(design_mtx, np.ndarray):
                design_mtx = torch.from_numpy(design_mtx)

            if design_mtx.shape[0] != self.n_spots:
                raise ValueError(
                    "The design matrix must have the same number of rows as spots."
                )

            if design_mtx.dim() == 1:  # in case of a single covariate
                design_mtx = design_mtx.unsqueeze(1)

            # convert to float tensor
            design_mtx = design_mtx.float()

            if covariate_names is not None:  # set default names
                if len(covariate_names) != design_mtx.shape[1]:
                    raise ValueError(
                        "Covariate names must match the number of factors."
                    )
            else:
                covariate_names = [
                    f"factor_{i + 1}" for i in range(design_mtx.shape[1])
                ]

            # check for constant covariates
            _ind = torch.where(design_mtx.std(0) < 1e-5)[0]
            for _i in _ind:
                warnings.warn(
                    f"{covariate_names[_i]} has zero variance. Please remove it."
                )

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

    def get_formatted_test_results(
        self, test_type: Literal["sv", "du"]
    ) -> pd.DataFrame:
        """Get the formatted test results as data frame.

        Parameters
        ----------
        test_type
            Type of test results to retrieve. Can be one of ``'sv'`` (spatial variability) or ``'du'`` (differential usage).

        Returns
        -------
        pandas.DataFrame
            Formatted test results.
        """
        assert test_type in [
            "sv",
            "du",
        ], "Invalid test type. Must be one of 'sv' or 'du'."
        if test_type == "sv":
            # check if the spatial variability test has been run
            assert (
                len(self.sv_test_results) > 0
            ), "No spatial variability test results found. Please run test_spatial_variability() first."
            # format the results
            res = pd.DataFrame(
                {
                    "gene": self.gene_names,
                    "statistic": self.sv_test_results["statistic"],
                    "pvalue": self.sv_test_results["pvalue"],
                    "pvalue_adj": self.sv_test_results["pvalue_adj"],
                }
            )
            return res
        else:
            # check if the differential usage test has been run
            assert (
                len(self.du_test_results) > 0
            ), "No differential usage test results found. Please run test_differential_usage() first."
            # format the results
            res = pd.DataFrame(
                {
                    "gene": np.repeat(self.gene_names, self.n_factors),
                    "covariate": np.tile(self.covariate_names, self.n_genes),
                    "statistic": self.du_test_results["statistic"].reshape(-1),
                    "pvalue": self.du_test_results["pvalue"].reshape(-1),
                    "pvalue_adj": self.du_test_results["pvalue_adj"].reshape(-1),
                }
            )
            return res

    def test_spatial_variability(
        self,
        method: Literal["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"] = "hsic-ir",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        use_perm_null: bool = False,
        n_perms_per_gene: Optional[int] = None,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial variability.

        Kernel-based multivariate hypothesis testing for spatial variability in

        - gene-level total counts (``"hsic-gc"`` or ``"spark-x"`` :cite:`zhu2021spark`)
        - isoform usage ratios (``"hsic-ir"``)
        - isoform counts (``"hsic-ic"``)

        Test statistics and p-values are computed per gene for each gene separately.

        Parameters
        ----------
        method
            Must be one of ``"hsic-ir"``, ``"hsic-ic"``, ``"hsic-gc"``, or ``"spark-x"``.
        ratio_transformation
            If using isoform ratios, the compositional transformation to apply.
            Can be one of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, or ``'radial'`` :cite:`park2022kernel`.
            See :func:`splisosm.utils.counts_to_ratios` for more details.
        nan_filling
            How to fill the NaN values in the isoform ratios. Can be one of ``'mean'`` or ``'none'``.
            See :func:`splisosm.utils.counts_to_ratios` for more details.
        use_perm_null
            Whether to generate the null distribution from permutation.
            If False, use the asymptotic distribution of chi-square mixtures with Liu's method :cite:`liu2009new`.
        n_perms_per_gene
            Number of permutations per gene for permutation test.
        return_results
            Whether to return the test statistics and p-values.
            Default to False, in which case the results are stored in ``self.sv_test_results``.
        print_progress
            Whether to show the progress bar. Default to True.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise, returns None and stores results in self.sv_test_results.

        Notes
        -----
        To run the SPARK-X test, the R-package `SPARK` must be installed and accessible from Python via `rpy2`.
        """

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["mean", "none"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid ratio transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid NaN filling method. Must be one of {valid_nan_filling}."

        if method == "spark-x":  # run the gene-level SPARK-X test
            # prepare the data in gene-level counts
            counts_g = torch.concat(
                [_counts.sum(1, keepdim=True) for _counts in self.data], axis=1
            )  # tensor(n_spots, n_genes)
            self.sv_test_results = run_sparkx(
                counts_g.numpy(), self.coordinates.numpy()
            )
        else:
            # use a global spatial kernel unless nan_filling is 'none'
            n_spots = self.n_spots
            # H = torch.eye(n_spots) - 1/n_spots
            # K_sp = H @ self.corr_sp @ H # centered spatial kernel
            K_sp = self.corr_sp  # the Kernel class object was already centered

            # calculate the eigenvalues of the spatial kernel
            if not use_perm_null:
                # lambda_sp = torch.linalg.eigvalsh(K_sp) # eigenvalues of length n_spots
                lambda_sp = self._corr_sp_eigvals  # use precomputed eigenvalues
                lambda_sp = lambda_sp[lambda_sp > 1e-5]  # remove small eigenvalues

            # prepare inputs for generating the null distribution
            if use_perm_null:
                n_nulls = n_perms_per_gene if n_perms_per_gene is not None else 1000

            # iterate over genes and calculate the HSIC statistic
            hsic_list, pvals_list = [], []
            for counts in tqdm(self.data, disable=(not print_progress)):
                if counts.is_sparse:
                    counts = counts.to_dense()

                if method == "hsic-ir" and nan_filling == "none":
                    # spetial treatment for the isoform ratio test when nan_filling is 'none'
                    # need to adjust the effective spot number (non NaN spots) and spatial kernel
                    y = counts_to_ratios(
                        counts,
                        transformation=ratio_transformation,
                        nan_filling=nan_filling,
                    )
                    # remove spots with NaN values
                    is_nan = torch.isnan(y).any(1)  # spots with NaN values
                    y = y[~is_nan]  # (n_non_nan, n_isos)

                    # adjust the effective number of spots and update the spatial kernel
                    n_spots = y.shape[0]
                    # H = torch.eye(n_spots) - 1/n_spots
                    # K_sp = H @ self.corr_sp[~is_nan, :][:, ~is_nan] @ H # centered spatial kernel
                    K_sp = self.corr_sp.realization()[~is_nan, :][:, ~is_nan]
                    K_sp = K_sp - K_sp.mean(dim=0, keepdim=True)
                    K_sp = K_sp - K_sp.mean(dim=1, keepdim=True)

                    # calculate the eigenvalues of the new per-gene spatial kernel
                    if not use_perm_null:
                        lambda_sp = torch.linalg.eigvalsh(
                            K_sp
                        )  # eigenvalues of length n_spots
                        lambda_sp = lambda_sp[
                            lambda_sp > 1e-5
                        ]  # remove small eigenvalues

                    # compute the hsic statistic
                    hsic_scaled = torch.trace(y.T @ K_sp @ y)

                else:  # one global spatial kernel for all genes
                    if method == "hsic-ic":  # use isoform-level count data
                        y = counts - counts.mean(
                            0, keepdim=True
                        )  # centering per isoform
                    elif method == "hsic-gc":  # use gene-level count data
                        y = counts.sum(1, keepdim=True)
                        y = y - y.mean()  # centering per isoform
                    else:  # use isoform ratio
                        # calculate the isoform ratio from counts
                        y = counts_to_ratios(
                            counts,
                            transformation=ratio_transformation,
                            nan_filling=nan_filling,
                        )
                        y = y - y.mean(0, keepdim=True)  # centering per isoform

                    # calculate the HSIC statistic
                    hsic_scaled = torch.trace(
                        K_sp.xtKx(y)
                    )  # equivalent to y.T @ K_sp @ y

                hsic_list.append(hsic_scaled / (n_spots - 1) ** 2)

                if use_perm_null:  # permutation-based null distribution
                    # randomly shuffle the spatial locations
                    perm_idx = torch.stack(
                        [torch.randperm(n_spots) for _ in range(n_nulls)]
                    )
                    yy = y[perm_idx, :]  # (n_nulls, n_spots, n_isos)

                    # calculate the null HSIC statistics
                    null_m = torch.einsum(
                        "bii->b", (yy.transpose(1, 2) @ K_sp.unsqueeze(0) @ yy)
                    )

                    # calculate the p-value
                    pval = (null_m > hsic_scaled).sum() / n_nulls

                else:  # asymptotic null distribution
                    lambda_y = torch.linalg.eigvalsh(y.T @ y)  # length of n_isos
                    lambda_spy = (
                        lambda_sp.unsqueeze(0) * lambda_y.unsqueeze(1)
                    ).reshape(
                        -1
                    )  # n_spots * (n_isos or 1)
                    pval = liu_sf((hsic_scaled * n_spots).numpy(), lambda_spy.numpy())

                pvals_list.append(pval)

                # store the results
                self.sv_test_results = {
                    "statistic": torch.tensor(hsic_list).numpy(),
                    "pvalue": torch.tensor(pvals_list).numpy(),
                    "method": method,
                    "use_perm_null": use_perm_null,
                }

            # calculate adjusted p-values
            self.sv_test_results["pvalue_adj"] = false_discovery_control(
                self.sv_test_results["pvalue"]
            )

        # return results
        if return_results:
            return self.sv_test_results

    def test_differential_usage(
        self,
        method: Literal[
            "hsic", "hsic-knn", "hsic-gp", "t-fisher", "t-tippett"
        ] = "hsic-gp",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        hsic_eps: Optional[float] = 1e-3,
        gp_configs: Optional[dict[str, Any]] = None,
        print_progress: bool = True,
        return_results: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial isoform differential usage.

        Before running this function, the design matrix must be set up using :func:`setup_data`.
        Each column of the design matrix corresponds to a covariate to test for differential association
        with the isoform usage ratios of each gene.
        Test statistics and p-values are computed per (gene, covariate) pair separately.

        Two types of association tests are supported:

        - Unconditional (``"hsic"`` with ``hsic_eps=None``, ``"t-fisher"``, ``"t-tippett"``): test the unconditional association between isoform usage ratios and the covariate of interest.
        - Conditional (``"hsic"`` with ``hsic_eps>0``, ``"hsic-knn"``, ``"hsic-gp"``): test the association conditioned on the spatial coordinates. See :cite:`zhang2012kernel` for more details.

        Parameters
        ----------
        method
            Method for association testing can be one of the HSIC tests

            * ``"hsic"``: HSIC test for isoform differential usage along each factor in the design matrix.
              For continuous factors, it is equivalent to the (partial) pearson correlation test.
              For binary factors, it is equivalent to the two-sample t-test.
            * ``"hsic-knn"``: conditional HSIC test using KNN regression to remove spatial effect.
            * ``"hsic-gp"``: conditional HSIC test using kernels learned from Gaussian process regression.

            Or, one of T-tests (binary factors only), ``"t-fisher"``, ``"t-tippett"``, where each isoform
            is tested independently for association with the factor of interest,
            and the p-values are combined to gene-level using either Fisher's or Tippett's approach.
        ratio_transformation
            What compositional transformation to use for isoform ratio.
            Can be one of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, or ``'radial'`` :cite:`park2022kernel`.
            See :func:`splisosm.utils.counts_to_ratios` for more details.
        nan_filling
            How to fill the NaN values in the isoform ratios. Can be one of ``'mean'`` or ``'none'``.
            See :func:`splisosm.utils.counts_to_ratios` for more details.
        hsic_eps
            The regularization parameter for conditional HSIC when ``method='hsic'``.
            If None, the test is unconditional. This parameter does not apply to ``method='hsic-knn'`` or ``method='hsic-gp'``.
        gp_configs
            The kernel configurations for the Gaussian process regression.
            See :func:`splisosm.hyptest_np.fit_kernel_gpr` for more details.
            If `None`, defaults to the configuration below. For efficiency,
            we fix some parameters to be constant::

                {
                    "constant_value_covariate": 1.0,
                    "length_scale_covariate": 1.0,
                    "constant_value_bounds_covariate": (1e-3, 1e3),
                    "length_scale_bounds_covariate": "fixed",
                    "constant_value_isoform": 1e-3,
                    "length_scale_isoform": 1.0,
                    "constant_value_bounds_isoform": "fixed",
                    "length_scale_bounds_isoform": "fixed",
                }

        print_progress
            Whether to show the progress bar. Default to True.
        return_results
            Whether to return the test statistics and p-values.
            If False, the results are stored in ``self.du_test_results``.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise, returns None and stores results in self.du_test_results.
        """
        if self.design_mtx is None:
            raise ValueError(
                "Cannot find the design matrix. Perhaps you forgot to set it up using setup_data()."
            )

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["hsic", "hsic-knn", "hsic-gp", "t-fisher", "t-tippett"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["none", "mean"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid nan_filling. Must be one of {valid_nan_filling}."

        # TODO
        if method in ["hsic", "hsic-knn"]:  # HSIC-based test with pre-specified kernel
            # x: spatial coordinates, z: factor of interest, y: isoform usage
            # need to first regress out the spatial effect x from x and y

            if nan_filling == "mean":  # no NaN values in the ratio
                # use the same spatial kernel matrix for all genes
                if method == "hsic-knn":  # use KNN regression
                    Rx = get_knn_regression_residual_op(self.coordinates, k=4)
                else:  # use kernel regression
                    # calculate the residual operator Rx
                    # if hsic_eps is None, testing the unconditional H_0: y \independent z
                    # otherwise, testing the conditional H_0: y \independent z | x
                    if hsic_eps is None:  # unconditional HSIC test
                        Rx = torch.eye(n_spots)
                    else:  # conditional HSIC test
                        assert (
                            hsic_eps > 0
                        ), "The regularization parameter hsic_eps must be positive."
                        # prepare the spatial kernel matrix
                        # H = torch.eye(n_spots) - 1/n_spots
                        # Kx = H @ self.corr_sp @ H # centered kernel for spatial coordinates
                        Kx = (
                            self.corr_sp.realization()
                        )  # the Kernel class object was already centered
                        # regularized kernel regression
                        Rx = get_kernel_regression_residual_op(Kx, hsic_eps)

                hsic_list, pvals_list = [], []
                # iterate over factors
                for _ind in tqdm(
                    range(n_factors), disable=(not print_progress), dynamic_ncols=True
                ):
                    # center the factor of interest
                    z = self.design_mtx[:, _ind].clone()  # len of n_spots
                    assert (
                        z.std() > 0
                    ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                    z = Rx @ z  # regression residual
                    z = (
                        z - z.mean()
                    ) / z.std()  # normalize the factor of interest for stability
                    z = z.unsqueeze(1)  # (n_spots, 1)

                    _hsic_ind, _pvals_ind = [], []
                    # iterate over genes and calculate the HSIC statistic
                    for counts in self.data:
                        if counts.is_sparse:
                            counts = counts.to_dense()

                        # calculate isoform usage ratio (n_spots, n_isos)
                        y = counts_to_ratios(
                            counts,
                            transformation=ratio_transformation,
                            nan_filling="mean",
                        )
                        y = Rx @ y  # regression residual, (n_spots, n_isos)
                        y = y - y.mean(0, keepdim=True)  # centering per isoform

                        # calculate the HSIC statistic
                        _hsic, _pval = linear_hsic_test(z, y, centering=True)

                        _hsic_ind.append(_hsic)
                        _pvals_ind.append(_pval)

                    # stack the results
                    hsic_list.append(torch.tensor(_hsic_ind))
                    pvals_list.append(torch.tensor(_pvals_ind))

                # combine results
                hsic_all = torch.stack(hsic_list, dim=1)
                pvals_all = torch.stack(pvals_list, dim=1)

            else:  # nan_filling == 'none', NaN values in the ratio
                if counts.is_sparse:
                    counts = counts.to_dense()

                hsic_list, pvals_list = [], []
                # iterate over genes and use different spatial kernel matrix for different genes
                for counts in tqdm(
                    self.data, disable=(not print_progress), dynamic_ncols=True
                ):
                    # calculate isoform usage ratio (n_spots, n_isos)
                    y = counts_to_ratios(
                        counts, transformation=ratio_transformation, nan_filling="none"
                    )

                    # remove NaN spots
                    is_nan = torch.isnan(y).any(1)  # spots with NaN values, (n_spots,)

                    # calculate the residual operator Rx
                    if method == "hsic-knn":  # use KNN regression
                        Rx = get_knn_regression_residual_op(
                            self.coordinates[~is_nan], k=4
                        )
                    else:  # use kernel regression
                        # calculate the residual operator Rx
                        # if hsic_eps is None, testing the unconditional H_0: y \independent z
                        # otherwise, testing the conditional H_0: y \independent z | x
                        if hsic_eps is None:  # unconditional HSIC test
                            Rx = torch.eye(n_spots - is_nan.sum())
                        else:  # conditional HSIC test
                            assert (
                                hsic_eps > 0
                            ), "The regularization parameter hsic_eps must be positive."
                            # create the spatial kernel matrix as the principal submatrix
                            # Kx = self.corr_sp[~is_nan,:][:,~is_nan] # (n_non_nan, n_non_nan)
                            # H = torch.eye(Kx.shape[0]) - 1/Kx.shape[0]
                            # Kx = H @ Kx @ H # centered spatial kernel, (n_non_nan, n_non_nan)
                            K_x = self.corr_sp.realization()[~is_nan, :][:, ~is_nan]
                            K_x = K_x - K_x.mean(dim=0, keepdim=True)
                            K_x = K_x - K_x.mean(
                                dim=1, keepdim=True
                            )  # (n_non_nan, n_non_nan)

                            # regularized kernel regression
                            Rx = get_kernel_regression_residual_op(Kx, hsic_eps)

                    # calculate the residuals
                    y = Rx @ y[~is_nan]  # regression residual, (n_non_nan, n_isos)

                    # center the factor of interest
                    y = y - y.mean(0, keepdim=True)  # centering per isoform

                    _hsic_ind, _pvals_ind = [], []
                    # iterate over factors
                    for _ind in range(n_factors):
                        # center the factor of interest
                        z = self.design_mtx[~is_nan, _ind].clone()  # (n_non_nan,)
                        assert (
                            z.std() > 0
                        ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                        z = Rx @ z  # regression residual, (n_non_nan,)
                        z = (
                            z - z.mean()
                        ) / z.std()  # normalize the factor of interest for stability
                        z = z.unsqueeze(1)  # (n_non_nan, 1)

                        # calculate the HSIC statistic
                        _hsic, _pval = linear_hsic_test(z, y, centering=True)

                        _hsic_ind.append(_hsic)
                        _pvals_ind.append(_pval)

                    # stack the results
                    hsic_list.append(torch.tensor(_hsic_ind))
                    pvals_list.append(torch.tensor(_pvals_ind))

                # combine results
                hsic_all = torch.stack(hsic_list, dim=0)  # (n_genes, n_factors)
                pvals_all = torch.stack(pvals_list, dim=0)  # (n_genes, n_factors)

            # store the results
            self.du_test_results = {
                "statistic": hsic_all.numpy(),  # (n_genes, n_factors)
                "pvalue": pvals_all.numpy(),  # (n_genes, n_factors)
                "method": method,
            }

        elif (
            method == "hsic-gp"
        ):  # HSIC-based test with kernel learned by Gaussian process regression
            # specify the GP kernel configurations
            _default_gp_configs = {
                "constant_value_covariate": 1.0,
                "length_scale_covariate": 1.0,
                "constant_value_bounds_covariate": (1e-3, 1e3),
                "length_scale_bounds_covariate": "fixed",
                "constant_value_isoform": 1e-3,
                "length_scale_isoform": 1.0,
                "constant_value_bounds_isoform": "fixed",
                "length_scale_bounds_isoform": "fixed",
            }
            if gp_configs is None:
                gp_configs = _default_gp_configs
            else:  # update the config
                gp_configs = {**_default_gp_configs, **gp_configs}

            # normalize the spatial coordinates
            x = self.coordinates.clone()  # (n_spots, 2)
            x = (x - x.mean(0)) / x.std(0)  # normalize the spatial coordinates
            x[torch.isinf(x)] = 0  # for constant columns

            # run GP regression for every factor
            z_res_list = []
            for _ind in tqdm(
                range(n_factors), disable=(not print_progress), dynamic_ncols=True
            ):
                # center the factor of interest
                z = self.design_mtx[:, _ind].clone()  # (n_spots,)
                assert (
                    z.std() > 0
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z = (z - z.mean()) / z.std()
                z = z.unsqueeze(1)  # (n_spots, 1)

                # fit the kernel regression and store results
                z_res = fit_kernel_gpr(
                    x,
                    z,
                    normalize_x=False,
                    normalize_y=False,
                    return_residuals=True,
                    constant_value=gp_configs["constant_value_covariate"],
                    constant_value_bounds=gp_configs["constant_value_bounds_covariate"],
                    length_scale=gp_configs["length_scale_covariate"],
                    length_scale_bounds=gp_configs["length_scale_bounds_covariate"],
                )
                z_res_list.append(z_res)

            # run GP regression for every gene
            y_res_list = []
            for counts in tqdm(
                self.data, disable=(not print_progress), dynamic_ncols=True
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()

                # calculate isoform usage ratio (n_spots, n_isos)
                y = counts_to_ratios(
                    counts, transformation=ratio_transformation, nan_filling=nan_filling
                )

                # center the isoform usage for non-NaN spots
                if nan_filling == "none":
                    is_nan = torch.isnan(y).any(1)  # spots with NaN values
                    y[~is_nan] = y[~is_nan] - y[~is_nan].mean(
                        0, keepdim=True
                    )  # y still of (n_spots, n_isos)
                else:
                    y = y - y.mean(0, keepdim=True)  # (n_spots, n_isos)

                # fit the kernel regression and store results
                y_res = fit_kernel_gpr(
                    x,
                    y,
                    normalize_x=False,
                    normalize_y=False,
                    return_residuals=True,
                    constant_value=gp_configs["constant_value_isoform"],
                    constant_value_bounds=gp_configs["constant_value_bounds_isoform"],
                    length_scale=gp_configs["length_scale_isoform"],
                    length_scale_bounds=gp_configs["length_scale_bounds_isoform"],
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
                    _hsic, _pval = linear_hsic_test(_z, _y, centering=True)
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
                "statistic": hsic_all.numpy(),  # (n_genes, n_factors)
                "pvalue": pvals_all.numpy(),  # (n_genes, n_factors)
                "method": method,
            }

        else:  # two-sample t-test
            # method to combine p-values across isoforms, either 'fisher' or 'tippett'
            combine_method = re.findall(r"^t-(.+)", method)[0]

            _du_ttest_stats_all, _du_ttest_pvals_all = [], []

            # iterate over factors
            for _ind in range(n_factors):
                # iterate over genes and calculate the t-test statistic
                _du_ttest_stats, _du_ttest_pvals = [], []
                for counts in self.data:
                    if counts.is_sparse:
                        counts = counts.to_dense()

                    # calculate isoform usage ratio (n_spots, n_isos)
                    ratios = counts_to_ratios(
                        counts,
                        transformation=ratio_transformation,
                        nan_filling=nan_filling,
                    )

                    # run t-test and combine p-values
                    _stats, _pvals = _calc_ttest_differential_usage(
                        ratios,
                        self.design_mtx[:, _ind],
                        combine_pval=True,
                        combine_method=combine_method,
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
                "statistic": _du_ttest_stats_all,  # (n_genes, n_factors)
                "pvalue": _du_ttest_pvals_all,  # (n_genes, n_factors)
                "method": method,
            }

        # calculate adjusted p-values (independently for each factor)
        self.du_test_results["pvalue_adj"] = false_discovery_control(
            self.du_test_results["pvalue"], axis=0
        )

        # return the results if needed
        if return_results:
            return self.du_test_results
