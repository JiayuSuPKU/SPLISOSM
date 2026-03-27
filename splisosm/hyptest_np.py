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

from splisosm.utils import (
    counts_to_ratios,
    false_discovery_control,
    prepare_inputs_from_anndata,
    run_sparkx,
)
from splisosm.kernel import SpatialCovKernel
from splisosm.likelihood import liu_sf
from splisosm.kernel_gpr import (
    linear_hsic_test,
    fit_kernel_gpr,
    make_kernel_gpr,
    _DEFAULT_GPR_CONFIGS,
)

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
    >>> model.test_spatial_variability(method='hsic-ir')
    >>> sv_results = model.get_formatted_test_results('sv')
    >>> print(sv_results.head())

    Differential usage test:

    >>> model = SplisosmNP()
    >>> model.setup_data(data, coordinates, design_mtx=design_mtx)
    >>> model.test_differential_usage(method='hsic-gp', residualize='cov_only')
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

    gene_names: list[str]
    """List of gene names corresponding to the genes in the model."""

    covariate_names: list[str]
    """List of covariate names corresponding to columns of the design matrix."""

    adata: Optional[AnnData]
    """Source :class:`anndata.AnnData` object when using AnnData input mode;
    ``None`` in legacy (list-of-tensors) mode."""

    data: list[torch.Tensor]
    """List of isoform count tensors, one per gene, each of shape
    ``(n_spots, n_isos)``.  Set by :meth:`setup_data`."""

    coordinates: torch.Tensor
    """Spatial coordinates of shape ``(n_spots, 2)``.
    Set by :meth:`setup_data`."""

    corr_sp: Any
    """Spatial covariance kernel (:class:`~splisosm.kernel.SpatialCovKernel`)
    constructed from :attr:`coordinates`.  Set by :meth:`setup_data`."""

    design_mtx: Optional[torch.Tensor]
    """Design matrix of shape ``(n_spots, n_factors)``; ``None`` if no
    covariates were provided to :meth:`setup_data`."""

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
        self._setup_input_mode = None  # "legacy" or "anndata"

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
        data : list of torch.Tensor or numpy.ndarray, optional
            Legacy mode only. List of isoform count arrays, one per gene, each of
            shape ``(n_spots, n_isos)``.
        coordinates : torch.Tensor, numpy.ndarray, or pandas.DataFrame, optional
            Legacy mode only. Spatial coordinates of shape ``(n_spots, 2)``.
        approx_rank : int or None, optional
            Rank of the low-rank approximation for the spatial covariance matrix.
            If ``None``, the full-rank dense matrix is used.
            For datasets with ``n_spots > 5000`` the maximum rank is capped at
            ``4 * sqrt(n_spots)``.
        design_mtx : torch.Tensor, numpy.ndarray, pandas.DataFrame, str, or list of str, optional
            Design matrix for differential usage tests.

            - Legacy mode: array/tensor/DataFrame of shape ``(n_spots, n_factors)``.
            - AnnData mode: array/tensor/DataFrame, one obs-column name (str),
              or a list of obs-column names.

            Categorical obs columns are one-hot encoded automatically. Covariate names
            are inferred when not explicitly provided (see ``covariate_names``).
        gene_names : list of str, str, or None, optional
            Gene names.

            - Legacy mode: list of gene name strings.
            - AnnData mode: column name in ``adata.var`` used as display names for
              grouped genes; if ``None``, the values of ``group_iso_by`` are used.
        covariate_names : list of str or None, optional
            Covariate names for columns of the design matrix. If not provided,
            names are inferred as follows:

            - **AnnData mode with column name(s)**: column names are used, with
              categorical columns expanded to one-hot names (e.g., ``col_cat0``,
              ``col_cat1`` for ``col``).
            - **Legacy mode with DataFrame**: DataFrame column names are used.
            - **Otherwise**: auto-generated as ``factor_1``, ``factor_2``, etc.

            When explicitly provided, must match the number of factors in the design
            matrix after any categorical encoding.
        adata : anndata.AnnData or None, optional
            AnnData object for AnnData input mode.
        spatial_key : str, optional
            Key in ``adata.obsm`` for spatial coordinates.
        layer : str, optional
            Layer in ``adata.layers`` that stores isoform counts.
        group_iso_by : str, optional
            Column in ``adata.var`` used to group isoforms by gene.
        min_counts : int, optional
            Minimum total isoform count across spots required to retain an isoform
            in AnnData mode.
        min_bin_pct : float, optional
            Minimum fraction/percentage of spots where an isoform must be expressed.
            Values in ``[0, 1]`` are fractions; values in ``(1, 100]`` are percentages.
        filter_single_iso_genes : bool, optional
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
            self._setup_input_mode = "anndata"
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
            self._setup_input_mode = "legacy"

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
        if filter_single_iso_genes and min(self.n_isos) <= 1:
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

        # check and process the design matrix
        if design_mtx is not None:
            # Infer covariate names from DataFrame columns if available
            inferred_cov_names = None
            if isinstance(design_mtx, pd.DataFrame):
                inferred_cov_names = list(design_mtx.columns)
                design_mtx = design_mtx.values

            # Convert sparse matrices to dense
            if hasattr(design_mtx, "toarray"):  # scipy sparse matrix
                design_mtx = design_mtx.toarray()

            # Convert numpy to torch
            if isinstance(design_mtx, np.ndarray):
                design_mtx = torch.from_numpy(design_mtx.astype(np.float32))
            elif isinstance(design_mtx, torch.Tensor):
                design_mtx = design_mtx.float()
            else:
                raise TypeError(
                    f"Unsupported design_mtx type: {type(design_mtx)}. "
                    "Expected numpy array, torch tensor, pandas DataFrame, or sparse matrix."
                )

            # Validate shape
            if design_mtx.shape[0] != self.n_spots:
                raise ValueError(
                    f"Design matrix row count ({design_mtx.shape[0]}) must match "
                    f"number of spots ({self.n_spots})."
                )

            # Handle 1D design matrix (single covariate)
            if design_mtx.dim() == 1:
                design_mtx = design_mtx.unsqueeze(1)

            # Ensure float dtype
            design_mtx = design_mtx.float()

            # Determine covariate names with priority: explicit > inferred > generated
            n_factors = design_mtx.shape[1]
            if covariate_names is not None:
                # Explicit covariate_names provided by user
                if len(covariate_names) != n_factors:
                    raise ValueError(
                        f"Number of covariate_names ({len(covariate_names)}) must match "
                        f"design matrix columns ({n_factors})."
                    )
            elif inferred_cov_names is not None:
                # Inferred from DataFrame columns
                if len(inferred_cov_names) != n_factors:
                    raise ValueError(
                        f"DataFrame column count ({len(inferred_cov_names)}) does not match "
                        f"design matrix columns ({n_factors})."
                    )
                covariate_names = inferred_cov_names
            else:
                # Generate default covariate names
                covariate_names = [f"factor_{i+1}" for i in range(n_factors)]

            # Check for constant/zero-variance covariates
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                cov_stds = design_mtx.std(dim=0)
                zero_var_indices = torch.where(cov_stds < 1e-5)[0]
                for idx in zero_var_indices:
                    warnings.warn(
                        f"Covariate '{covariate_names[idx]}' has near-zero variance "
                        "(std < 1e-5). Consider removing it."
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
        """Get formatted test results as a pandas DataFrame.

        Parameters
        ----------
        test_type : {"sv", "du"}
            Which results to retrieve: ``"sv"`` for spatial variability or
            ``"du"`` for differential usage.

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
        method : {"hsic-ir", "hsic-ic", "hsic-gc", "spark-x"}, optional
            Test target: ``"hsic-ir"`` (isoform usage ratios), ``"hsic-ic"``
            (isoform counts), ``"hsic-gc"`` (gene-level counts), or
            ``"spark-x"`` (SPARK-X :cite:`zhu2021spark`).
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Compositional transformation applied to isoform ratios when
            ``method="hsic-ir"``.  See :func:`splisosm.utils.counts_to_ratios`
            and :cite:`park2022kernel` for details.
        nan_filling : {"mean", "none"}, optional
            Strategy for NaN values in isoform ratios.
            See :func:`splisosm.utils.counts_to_ratios` for details.
        use_perm_null : bool, optional
            If ``True``, estimate the null distribution by permutation.
            If ``False`` (default), use the asymptotic chi-square mixture with
            Liu's method :cite:`liu2009new`.
        n_perms_per_gene : int or None, optional
            Number of permutations per gene when ``use_perm_null=True``.
            Defaults to 1000 when ``None``.
        return_results : bool, optional
            If ``True``, return the result dict.  Otherwise store results in
            :attr:`sv_test_results` and return ``None``.
        print_progress : bool, optional
            Whether to show a progress bar.

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
        method: Literal["hsic", "hsic-gp", "t-fisher", "t-tippett"] = "hsic-gp",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        gpr_backend: Literal["sklearn", "gpytorch"] = "sklearn",
        gpr_configs: Optional[dict[str, Any]] = None,
        residualize: Literal["cov_only", "both"] = "cov_only",
        print_progress: bool = True,
        return_results: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial isoform differential usage.

        Before running this function, the design matrix must be set up using :func:`setup_data`.
        Each column of the design matrix corresponds to a covariate to test for differential
        association with the isoform usage ratios of each gene.
        Test statistics and p-values are computed per (gene, covariate) pair separately.

        Two types of association tests are supported:

        - Unconditional (``"hsic"``, ``"t-fisher"``, ``"t-tippett"``): test the
          unconditional association between isoform usage ratios and the covariate.
        - Conditional (``"hsic-gp"``): test the association conditioned on spatial
          coordinates via Gaussian process regression.  See :cite:`zhang2012kernel`
          for more details.

        Parameters
        ----------
        method : str, optional
            Method for association testing:

            * ``"hsic"``: Unconditional HSIC test (multivariate RV coefficient).
              For continuous factors, equivalent to the multivariate Pearson correlation
              test.  For binary factors, equivalent to the two-sample Hotelling T**2 test.
            * ``"hsic-gp"``: Conditional HSIC test.  Spatial effects are removed via
              Gaussian process regression before computing the HSIC statistic.

            Or one of the T-tests (binary factors only):

            * ``"t-fisher"``, ``"t-tippett"``: each isoform is tested independently
              and p-values are combined gene-wise via Fisher's or Tippett's method.
        ratio_transformation : str, optional
            Compositional transformation for isoform ratios.
            One of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, ``'radial'``
            :cite:`park2022kernel`.  See :func:`splisosm.utils.counts_to_ratios`.
        nan_filling : str, optional
            How to fill NaN values in isoform ratios.  One of ``'mean'`` or ``'none'``.
            See :func:`splisosm.utils.counts_to_ratios`.
        gpr_backend : str, optional
            GPR backend to use for ``method='hsic-gp'``.
            One of ``'sklearn'`` (default) or ``'gpytorch'``.
            For FFT-accelerated spatial GP on regular grids use
            :class:`~splisosm.hyptest_fft.SplisosmFFT` instead.
        gpr_configs : dict, optional
            Nested configuration dict for the GPR objects, with optional keys
            ``'covariate'`` and/or ``'isoform'``.  Each sub-dict is forwarded to
            :func:`splisosm.kernel_gpr.make_kernel_gpr`.  Unspecified keys use the
            defaults from :data:`splisosm.kernel_gpr._DEFAULT_GPR_CONFIGS`::

                {
                    "covariate": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                        "n_inducing": 5000,
                    },
                    "isoform": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                        "n_inducing": 5000,
                    },
                }

            ``"n_inducing"`` *(int or None)* controls the scale of spatial GP
            fitting for each backend:

            * **sklearn** — maximum number of observations used for
              hyperparameter fitting.  Full exact GP when ``n_obs ≤ n_inducing``
              (or ``None``); a randomly sub-sampled **subset-of-data** of
              ``n_inducing`` points otherwise (**not** the same inducing-point
              approximation as gpytorch).  Default: ``5000``.  Set to ``None``
              to use all observations (warns when ``n_obs > 10_000``).
            * **gpytorch** — FITC sparse-GP inducing-point approximation with
              ``n_inducing`` points; set to ``None`` for exact GP.
              Default: ``5000``.

        residualize : {"cov_only", "both"}, optional
            Controls which signals are spatially residualized when
            ``method="hsic-gp"``:

            * ``"cov_only"`` (default): residualize covariates only; test
              HSIC(Z_res, Y_raw).  Fastest; calibration matches ``"both"``
              when covariate GPR captures most spatial confounding.
            * ``"both"``: residualize both covariates and isoform ratios.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        return_results : bool, optional
            Whether to return the test statistics and p-values.
            If False, the results are stored in ``self.du_test_results``.

        Returns
        -------
        results : dict or None
            If ``return_results`` is True, returns dict with test statistics and
            p-values. Otherwise, returns None and stores results in
            ``self.du_test_results``.
        """
        if self.design_mtx is None:
            raise ValueError(
                "Cannot find the design matrix. Perhaps you forgot to set it up using setup_data()."
            )

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["hsic", "hsic-gp", "t-fisher", "t-tippett"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["none", "mean"]
        valid_residualize = ["cov_only", "both"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid nan_filling. Must be one of {valid_nan_filling}."
        assert (
            residualize in valid_residualize
        ), f"Invalid residualize. Must be one of {valid_residualize}."

        n_genes = self.n_genes

        if method == "hsic":  # unconditional HSIC test (multivariate RV coefficient)
            # Pre-compute normalized covariates (n_factors small tensors, never sparse)
            z_list = []
            for _ind in range(n_factors):
                z = self.design_mtx[:, _ind].clone().unsqueeze(1)  # (n_spots, 1)
                assert (
                    z.std() > 0
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z_list.append(z)

            hsic_all = torch.empty(n_genes, n_factors)
            pvals_all = torch.empty(n_genes, n_factors)

            # Outer loop: one gene at a time to avoid materialising all dense counts
            for _g, counts in enumerate(
                tqdm(self.data, disable=(not print_progress), dynamic_ncols=True)
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()
                y = counts_to_ratios(
                    counts, transformation=ratio_transformation, nan_filling=nan_filling
                )
                for _f, z in enumerate(z_list):
                    hsic_all[_g, _f], pvals_all[_g, _f] = linear_hsic_test(
                        z, y, centering=True
                    )

            self.du_test_results = {
                "statistic": hsic_all.numpy(),
                "pvalue": pvals_all.numpy(),
                "method": method,
            }

        elif method == "hsic-gp":  # conditional HSIC via GP regression residuals
            # Build GPR configs (merge user overrides over defaults)
            cov_config = {**_DEFAULT_GPR_CONFIGS["covariate"]}
            iso_config = {**_DEFAULT_GPR_CONFIGS["isoform"]}
            if gpr_configs is not None:
                if "covariate" in gpr_configs:
                    cov_config.update(gpr_configs["covariate"])
                if "isoform" in gpr_configs:
                    iso_config.update(gpr_configs["isoform"])

            # Normalize spatial coordinates once
            x = self.coordinates.clone()  # (n_spots, 2)
            x = (x - x.mean(0)) / x.std(0)
            x[torch.isinf(x)] = 0  # guard against constant coordinate axes

            # Fit GPR for covariates and get residuals (n_factors small tensors, never sparse)
            gpr_cov = make_kernel_gpr(gpr_backend, **cov_config)
            z_res_list = []
            for _ind in tqdm(
                range(n_factors), disable=(not print_progress), dynamic_ncols=True
            ):
                z = self.design_mtx[:, _ind].clone()
                assert (
                    z.std() > 0
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z = (z - z.mean()) / z.std()
                z_res_list.append(gpr_cov.fit_residuals(x, z.unsqueeze(1)))

            # Optionally fit GPR for isoform ratios to residualize spatial effects in the response as well.
            if residualize == "both":
                gpr_iso = make_kernel_gpr(gpr_backend, **iso_config)
                # Warm up the shared eigendecomposition for backends that support it
                # (e.g. sklearn with fixed signal bounds) so the first gene does not
                # pay the cost of a redundant full GP fit.
                if (
                    hasattr(gpr_iso, "precompute_shared_kernel")
                    and gpr_iso.signal_bounds_fixed
                ):
                    gpr_iso.precompute_shared_kernel(x)

            # --- Main loop: densify and process one gene at a time ---
            hsic_all = torch.empty(n_genes, n_factors)
            pvals_all = torch.empty(n_genes, n_factors)

            for _g, counts in enumerate(
                tqdm(self.data, disable=(not print_progress), dynamic_ncols=True)
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()
                y = counts_to_ratios(
                    counts, transformation=ratio_transformation, nan_filling=nan_filling
                )

                if residualize == "both":
                    y = gpr_iso.fit_residuals(x, y)

                for _f, z_res in enumerate(z_res_list):
                    hsic_all[_g, _f], pvals_all[_g, _f] = linear_hsic_test(
                        z_res, y, centering=True
                    )

            self.du_test_results = {
                "statistic": hsic_all.numpy(),
                "pvalue": pvals_all.numpy(),
                "method": method,
            }

        else:  # two-sample t-test
            # method to combine p-values across isoforms, either 'fisher' or 'tippett'
            combine_method = re.findall(r"^t-(.+)", method)[0]

            stats_all = np.empty((n_genes, n_factors))
            pvals_all = np.empty((n_genes, n_factors))

            # Outer loop: one gene at a time so each sparse tensor is densified once
            for _g, counts in enumerate(
                tqdm(self.data, disable=(not print_progress), dynamic_ncols=True)
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()
                ratios = counts_to_ratios(
                    counts,
                    transformation=ratio_transformation,
                    nan_filling=nan_filling,
                )
                for _ind in range(n_factors):
                    _stats, _pvals = _calc_ttest_differential_usage(
                        ratios,
                        self.design_mtx[:, _ind],
                        combine_pval=True,
                        combine_method=combine_method,
                    )
                    stats_all[_g, _ind] = _stats
                    pvals_all[_g, _ind] = _pvals

            self.du_test_results = {
                "statistic": stats_all,  # (n_genes, n_factors)
                "pvalue": pvals_all,  # (n_genes, n_factors)
                "method": method,
            }

        # calculate adjusted p-values (independently for each factor)
        self.du_test_results["pvalue_adj"] = false_discovery_control(
            self.du_test_results["pvalue"], axis=0
        )

        # return the results if needed
        if return_results:
            return self.du_test_results
