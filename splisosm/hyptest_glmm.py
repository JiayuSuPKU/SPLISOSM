"""GLMM-based hypothesis tests for spatial isoform usage."""

from __future__ import annotations

import warnings
from timeit import default_timer as timer
from typing import Any, Optional, Union, Literal

import pandas as pd
import numpy as np
import scipy.sparse
from scipy.stats import chi2
import torch
from anndata import AnnData
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from splisosm.utils import (
    false_discovery_control,
    prepare_inputs_from_anndata,
)
from splisosm.dataset import IsoDataset
from splisosm.kernel import SpatialCovKernel
from splisosm.model import MultinomGLM
from splisosm._glmm_workers import (
    IsoFullModel,
    IsoNullNoSpVar,
    _fit_model_one_gene,
    _fit_null_full_sv_one_gene,
    _fit_perm_one_gene,
    _calc_llr_spatial_variability,
    _calc_score_differential_usage,
    _calc_wald_differential_usage,
)

__all__ = ["SplisosmGLMM"]

# Sentinel object that signals "auto-select approx_rank at setup_data time".
# Distinct from ``None``, which means "force full rank regardless of n_spots".
_APPROX_RANK_AUTO = object()


class SplisosmGLMM:
    """Parametric spatial isoform statistical modeling using GLMM.

    This is a convenience class that wraps around the :class:`splisosm.model.MultinomGLMM`
    for batched model fitting and spatial variability and differential usage testing.

    Examples
    --------
    Setup data:

    >>> from splisosm import SplisosmGLMM
    >>> import torch
    >>> # Simulate data for 10 genes with different number of isoforms
    >>> data_3_iso = [torch.randint(low=0, high=5, size=(100, 3)) for _ in range(5)]  # 5 genes with 3 isoforms
    >>> data_4_iso = [torch.randint(low=0, high=5, size=(100, 4)) for _ in range(5)]  # 5 genes with 4 isoforms
    >>> data = data_3_iso + data_4_iso
    >>> coordinates = torch.rand(100, 2)  # 100 spots with 2D coordinates
    >>> design_mtx = torch.rand(100, 2)  # 100 spots with 2 covariates

    Model fitting:

    >>> model = SplisosmGLMM(model_type='glmm-full')
    >>> model.setup_data(data, coordinates, design_mtx=design_mtx, group_gene_by_n_iso=True)
    >>> model.fit(n_jobs=1, batch_size=5, with_design_mtx=False)
    >>> fitted_models = model.get_fitted_models()
    >>> print(fitted_models[0])  # print the fitted model for the first gene

    Differential usage test:

    >>> model.test_differential_usage(method='score')
    >>> du_results = model.get_formatted_test_results('du')
    >>> print(du_results.head())
    """

    n_genes: int
    """Number of genes."""

    n_spots: int
    """Number of spots."""

    n_isos_per_gene: list[int]
    """List of numbers of isoforms per gene (equivalent to ``n_isos`` in :class:`SplisosmNP` and :class:`SplisosmFFT`)."""

    n_factors: int
    """Number of covariates to test for differential usage."""

    gene_names: list[str]
    """List of gene names corresponding to the genes in the model."""

    covariate_names: list[str]
    """List of covariate names corresponding to columns of the design matrix."""

    adata: Optional[AnnData]
    """Source :class:`anndata.AnnData` object when using AnnData input mode;
    ``None`` in legacy (list-of-tensors) mode."""

    coordinates: torch.Tensor
    """Spatial coordinates of shape ``(n_spots, 2)``.
    Set by :meth:`setup_data`."""

    corr_sp: torch.Tensor
    """Spatial covariance matrix of shape ``(n_spots, n_spots)``.
    Constructed from :attr:`coordinates` by :meth:`setup_data`."""

    design_mtx: Optional[torch.Tensor]
    """Design matrix of shape ``(n_spots, n_factors)``; ``None`` if no
    covariates were provided to :meth:`setup_data`."""

    sv_test_results: dict
    """Dictionary to store the spatial variability test results after running test_spatial_variability().
    It contains the following keys:

    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes,), the log likelihood ratio test statistic for each gene.
    - ``'df'``: int, the degrees of freedom for the test statistic.
    - ``'use_perm_null'``: bool, whether the null distribution is estimated using permutation.
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

    model_type: Literal["glmm-full", "glmm-null", "glm"]
    """The type of GLM or GLMM model."""

    model_configs: dict
    """Dictionary of model configurations, with keys
    ``share_variance``, ``var_parameterization_sigma_theta``,
    ``var_fix_sigma``, ``var_prior_model``, ``var_prior_model_params``,
    ``init_ratio``, ``fitting_method``, and ``fitting_configs``."""

    fitting_results: dict
    """Dictionary to store lists of fitted models, with keys
    ``'glmm-full'``, ``'glmm-null'``, ``'glm'``."""

    def __init__(
        self,
        model_type: Literal["glmm-full", "glmm-null", "glm"] = "glmm-full",
        share_variance: bool = True,
        var_parameterization_sigma_theta: bool = True,
        var_fix_sigma: bool = False,
        var_prior_model: str = "none",
        var_prior_model_params: dict = {},
        init_ratio: str = "observed",
        fitting_method: str = "joint_gd",
        fitting_configs: dict = {"max_epochs": -1},
        k_neighbors: int = 4,
        rho: float = 0.99,
        approx_rank=_APPROX_RANK_AUTO,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        """
        Parameters
        ----------
        model_type
            Which model to fit. Can be one of ``'glmm-full'`` (Multinomial GLMM with spatial random effects),
            ``'glmm-null'`` (Multinomial GLMM with white noise), ``'glm'`` (Multinomial GLM).
        share_variance
            Whether to share the variance component across isoforms.
        var_parameterization_sigma_theta
            Whether to parameterize the variance components as (``sigma``, ``theta_logit``) or (``sigma_sp``, ``sigma_nsp``).
            If True, the variance components will be (``sigma``, ``theta_logit``), where ``sigma`` is the total variance and
            ``theta_logit`` is the logit of the spatial variance proportion.
            If False, the variance components will be (``sigma_sp``, ``sigma_nsp``), where ``sigma_sp`` is the spatial
            variance and ``sigma_nsp`` is the non-spatial variance.
        var_fix_sigma
            Whether to fix the total variance (``sigma``) or not. If True, the total variance will be fixed to the initial value,
            which is the average per-spot variance of isoform counts normalized by its mean expression.
            See `MultinomGLMM._initialize_params` for details.
        var_prior_model
            The prior model on the total variance ``sigma``. Default is ``'none'`` with no prior.
            Other options are ``'gamma'`` (Gamma prior) and ``'inv_gamma'`` (Inverse Gamma prior).
        var_prior_model_params
            The parameters for the prior model on the total variance ``sigma``.
            For ``'gamma'``, the default parameters are ``{'alpha': 2.0, 'beta': 0.3}``.
            For ``'inv_gamma'``, the default parameters are ``{'alpha': 3, 'beta': 0.5}``.
        init_ratio
            The initialization method for the logit isoform usage ratio. Options are ``'observed'`` (initialize using observed counts)
            and ``'uniform'`` (equal isoform usage across space).
        fitting_method
            The fitting method to use when ``model_type='glmm-full'`` or ``'glmm-null'``.
            Options are ``'joint_gd'`` (joint likelihood with gradient descent),
            ``'joint_newton'`` (joint likelihood with Newton's method),
            ``'marginal_gd'`` (marginal likelihood with gradient descent),
            and ``'marginal_newton'`` (marginal likelihood with Newton's method).
        fitting_configs
            A dictionary of fitting configurations with the following keys:

            - ``'lr'``: float, learning rate
            - ``'optim'``: str, optimization method (one of ``'adam'``, ``'sgd'``, or ``'lbfgs'``)
            - ``'tol'``: float, tolerance for convergence
            - ``'max_epochs'``: int, maximum number of epochs
            - ``'patience'``: int, number of epochs to wait for improvement before stopping
            - ``'update_nu_every_k'``: int, number of iterations to update ``nu`` when using ``fitting_method='marginal_newton'``
        k_neighbors : int, optional
            Number of nearest neighbours used to build the spatial k-NN adjacency graph
            (default 4).  Passed to :class:`splisosm.kernel.SpatialCovKernel`.
            Ignored when ``adj_key`` is provided to :meth:`setup_data`.
        rho : float, optional
            Spectral smoothing parameter for the ICAR-like spatial kernel (default 0.99).
            Values close to 1 produce smoother spatial covariance.
        approx_rank : int or None, optional
            Number of leading eigenvectors of the spatial kernel to retain.

            * **Not specified (default)**: automatically selects the rank at
              :meth:`setup_data` time — full rank when ``n_spots ≤ 5000``,
              otherwise ``ceil(sqrt(n_spots) * 4)``.
            * ``None``: always use full rank regardless of ``n_spots`` (a warning
              is emitted when ``n_spots > 5000`` because the eigendecomposition of
              a large dense matrix is expensive).
            * Positive integer: use exactly that many eigenvectors.
        device : {"cpu", "cuda", "mps"}, optional
            Device for all model computation (default ``"cpu"``).
            Use ``"cuda"`` for NVIDIA GPU or ``"mps"`` for Apple Silicon.
            Parallel fitting (``n_jobs > 1``) is not supported for non-CPU devices
            and will automatically fall back to single-core with a warning.

        See also
        --------
        :class:`splisosm.model.MultinomGLMM` for more details on the model configurations.
        """
        # specify the model type to fit
        assert model_type in ["glmm-full", "glmm-null", "glm"]
        self.model_type = model_type

        self.model_configs = {
            "share_variance": share_variance,
            "var_parameterization_sigma_theta": var_parameterization_sigma_theta,
            "var_fix_sigma": var_fix_sigma,
            "var_prior_model": var_prior_model,
            "var_prior_model_params": var_prior_model_params,
            "init_ratio": init_ratio,
            "fitting_method": fitting_method,
            "fitting_configs": fitting_configs,
        }

        # kernel construction configs (used in setup_data)
        self._kernel_k_neighbors = k_neighbors
        self._kernel_rho = rho
        self._kernel_standardize_cov = True  # always standardize; not user-configurable
        self._approx_rank = approx_rank  # sentinel, None, or int
        self._device = device

        # to be set after running setup_data()
        self.n_genes = None  # number of genes
        self.n_spots = None  # number of spots
        self.n_isos_per_gene = None  # list of number of isoforms for each gene
        self.n_factors = None  # number of covariates to test for differential usage
        self.adata = None  # optional anndata source for the new setup path
        self._setup_input_mode = None  # "legacy" or "anndata"

        # feature summary cache (populated by _compute_feature_summaries)
        self._filtered_adata = None
        self._counts_layer = None
        self._group_iso_by = None
        self._gene_summary = None
        self._isoform_summary = None

        # to store the fitted models after running fit()
        self._is_trained = False
        self.fitting_results = {
            "models_glmm-full": [],
            "models_glmm-null": [],
            "models_glm": [],
        }

        # to store the spatial variability test results after running test_spatial_variability()
        self.sv_test_results = {}

        # to store the differential usage test results after running test_differential_usage()
        self.du_test_results = {}

    def __str__(self):
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
            "=== Parametric SPLISOSM model for spatial isoform testings\n"
            + f"- Number of genes: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Average number of isoforms per gene: {np.mean(self.n_isos_per_gene) if self.n_isos_per_gene is not None else None}\n"
            + "=== Model configurations\n"
            + f"- Model type: {self.model_type}\n"
            + f"- Parameterized using sigma and theta: {self.model_configs['var_parameterization_sigma_theta']}\n"
            + f"- Learnable variance: {not self.model_configs['var_fix_sigma']}\n"
            + f"- Same variance across classes: {self.model_configs['share_variance']}\n"
            + f"- Prior on total variance: {self.model_configs['var_prior_model']}\n"
            + f"- Initialization method: {self.model_configs['init_ratio']}\n"
            + "=== Fitting configurations \n"
            + f"- Trained: {self._is_trained}\n"
            + f"- Fitting methods: {self.model_configs['fitting_method']}\n"
            + f"- Parameters: {self.model_configs['fitting_configs']}\n"
            + "=== Test results\n"
            + f"- Spatial variability test: {_sv_status}\n"
            + f"- Differential usage test: {_du_status}"
        )

    @property
    def n_isos(self) -> list[int]:
        """List of numbers of isoforms per gene.

        Alias for :attr:`n_isos_per_gene` for API consistency with
        :class:`SplisosmNP` and :class:`SplisosmFFT`.
        """
        return self.n_isos_per_gene

    def setup_data(
        self,
        adata: AnnData,
        *,
        spatial_key: str = "spatial",
        adj_key: Optional[str] = None,
        layer: str = "counts",
        group_iso_by: str = "gene_symbol",
        gene_names: Optional[str] = None,
        group_gene_by_n_iso: bool = False,
        design_mtx: Optional[
            Union[np.ndarray, torch.Tensor, pd.DataFrame, str, list[str]]
        ] = None,
        covariate_names: Optional[list[str]] = None,
        min_counts: int = 10,
        min_bin_pct: float = 0.0,
        filter_single_iso_genes: bool = True,
        min_component_size: int = 1,
    ) -> None:
        """Setup the data for the GLMM model.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix.  Counts are read from
            ``adata.layers[layer]`` grouped by ``group_iso_by``, and
            spatial coordinates from ``adata.obsm[spatial_key]``.
        spatial_key : str, optional
            Key in ``adata.obsm`` for spatial coordinates (default
            ``"spatial"``).
        adj_key : str or None, optional
            Key in ``adata.obsp`` for a pre-built adjacency matrix.
            When provided, it overrides the k-NN graph construction
            from coordinates and be used directly to build the spatial kernel.
            The adjacency matrix is symmetrized internally.
        layer : str, optional
            Layer in ``adata.layers`` storing isoform counts (default
            ``"counts"``).
        group_iso_by : str, optional
            Column in ``adata.var`` used to group isoforms by gene
            (default ``"gene_symbol"``).
        gene_names : str or None, optional
            Column name in ``adata.var`` used as display names for genes.
            If ``None``, the values of ``group_iso_by`` are used.
        group_gene_by_n_iso : bool, optional
            If ``True``, genes are sorted and grouped by their isoform
            count before batching.  Required for ``batch_size > 1`` in
            :meth:`fit` because the GLMM model expects a fixed isoform
            count per batch (default ``False``).
        design_mtx : tensor, array, DataFrame, str, or list of str, optional
            Design matrix for fixed effects.  Accepts an
            array/tensor/DataFrame of shape ``(n_spots, n_factors)``, a
            single obs-column name (str), or a list of obs-column names.
        covariate_names : list of str or None, optional
            Explicit covariate names.  When not provided, names are
            inferred from column names or auto-generated.
        min_counts : int, optional
            Minimum total isoform count across spots to retain an isoform
            (default 10).
        min_bin_pct : float, optional
            Minimum fraction/percentage of spots where an isoform must be
            expressed (default 0.0).
        filter_single_iso_genes : bool, optional
            Whether to remove genes with fewer than two retained isoforms
            (default ``True``).
        min_component_size : int, optional
            Minimum number of spots a connected component must contain to
            be retained.  Spots in smaller components are removed from all
            data structures before the spatial kernel is built.  Default 1
            disables filtering.  A ``UserWarning`` is issued when spots are
            removed.

        Raises
        ------
        ValueError
            If input arguments are invalid or required fields are missing.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("`adata` must be an AnnData object.")

        (
            data,
            coordinates,
            resolved_gene_names,
            resolved_design,
            resolved_cov_names,
            adj_matrix,
            filtered_adata,
        ) = prepare_inputs_from_anndata(
            adata=adata,
            layer=layer,
            group_iso_by=group_iso_by,
            spatial_key=spatial_key,
            min_counts=min_counts,
            min_bin_pct=min_bin_pct,
            filter_single_iso_genes=filter_single_iso_genes,
            gene_names=gene_names,
            design_mtx=design_mtx,
            covariate_names=covariate_names,
            min_component_size=min_component_size,
            adj_key=adj_key,
            k_neighbors=self._kernel_k_neighbors,
            return_filtered_anndata=True,
        )

        self.adata = adata
        self._setup_input_mode = "anndata"
        self._filtered_adata = filtered_adata
        self._counts_layer = layer
        self._group_iso_by = group_iso_by

        # Build dataset (handles grouping by n_isos for batching)
        _dataset = IsoDataset(data, resolved_gene_names, group_gene_by_n_iso)
        self.n_genes, self.n_spots, self.n_isos_per_gene = (
            _dataset.n_genes,
            _dataset.n_spots,
            _dataset.n_isos_per_gene,
        )
        self.gene_names = _dataset.gene_names
        self._dataset = _dataset
        self._group_gene_by_n_iso = group_gene_by_n_iso
        self.coordinates = coordinates

        # Build spatial kernel only when the model actually uses spatial random effects.
        # For 'glm' there are no random effects at all.
        # For 'glmm-null' the spatial variance is fixed at 0 (theta = -inf), so the
        # kernel cancels out.  We use a rank-1 dummy covariance (constant vector scaled
        # to unit norm) which, combined with theta = 0, gives the exact identity
        # covariance C = σ²I via the low-rank Woodbury formula.
        n_spots = self.n_spots

        if self.model_type == "glm":
            # No spatial random effects — eigvals/eigvecs are not used.
            self.corr_sp = None
            self._corr_sp_eigvals = None
            self._corr_sp_eigvecs = None

        elif self.model_type == "glmm-null":
            # Spatial variance is always zero in the null model.
            # Rank-1 dummy: V = (1/√n) 1_n, λ = 1.  With theta=0:
            #   _cov_eigvals  = σ²(0·1 + 1) = σ²
            #   residual_eigval = σ²(1-0) = σ²
            #   correction = 1/σ² - 1/σ² = 0  →  C⁻¹ = (1/σ²)I  ✓
            self.corr_sp = None
            self._corr_sp_eigvals = torch.ones(1)  # (1,)
            self._corr_sp_eigvecs = torch.full(
                (n_spots, 1), 1.0 / np.sqrt(n_spots)
            )  # (n_spots, 1)

        else:
            # model_type == "glmm-full" — build the real spatial kernel.
            _kernel_kwargs = dict(
                rho=self._kernel_rho,
                standardize_cov=self._kernel_standardize_cov,
                centering=False,
            )
            if adj_matrix is not None:
                _kernel = SpatialCovKernel(
                    coords=None,
                    adj_matrix=adj_matrix,
                    **_kernel_kwargs,
                )
            else:
                _kernel = SpatialCovKernel(
                    coords=coordinates,
                    adj_matrix=None,
                    k_neighbors=self._kernel_k_neighbors,
                    **_kernel_kwargs,
                )

            # Determine the effective number of eigenvectors to retain.
            if self._approx_rank is _APPROX_RANK_AUTO:
                # Auto-select: full rank for small grids, truncated for large ones.
                k = (
                    None
                    if n_spots <= SpatialCovKernel.DENSE_THRESHOLD
                    else int(np.ceil(np.sqrt(n_spots) * 4))
                )
            elif self._approx_rank is None:
                # User explicitly requested full rank.
                if n_spots > SpatialCovKernel.DENSE_THRESHOLD:
                    warnings.warn(
                        f"approx_rank=None forces a full eigendecomposition of a "
                        f"{n_spots}×{n_spots} matrix which may be very slow and "
                        f"memory-intensive.  Consider omitting approx_rank (auto) "
                        f"or passing a positive integer.",
                        stacklevel=2,
                    )
                k = None
            else:
                k = min(int(self._approx_rank), n_spots)

            # Trigger eigendecomposition via SpatialCovKernel (numpy eigh / eigsh).
            # For dense mode (n ≤ DENSE_THRESHOLD), K_eigvecs always has n_spots columns
            # regardless of k, so we slice explicitly when k is given.
            _kernel.eigenvalues(
                k=k
            )  # populates K_eigvals, K_eigvecs (and K_sp for dense)
            self.corr_sp = _kernel.realization() if k is None else None
            if k is None:
                # Full decomp: use all cached eigenpairs
                self._corr_sp_eigvals = _kernel.K_eigvals  # (n_spots,)
                self._corr_sp_eigvecs = _kernel.K_eigvecs  # (n_spots, n_spots)
            else:
                # Truncated: dense mode may have cached all n_spots; implicit stores exactly k.
                self._corr_sp_eigvals = _kernel.K_eigvals[:k]  # (k,)
                self._corr_sp_eigvecs = _kernel.K_eigvecs[:, :k]  # (n_spots, k)

        # Process design matrix.
        # GLMM fitting requires dense torch tensors, so sparse design matrices are
        # densified here (unlike SplisosmNP which keeps them sparse).
        if resolved_design is not None:
            import scipy.sparse as _sp

            if _sp.issparse(resolved_design):
                resolved_design = resolved_design.toarray()
            design_mtx_t = torch.from_numpy(
                np.asarray(resolved_design, dtype=np.float32)
            )
            if design_mtx_t.dim() == 1:
                design_mtx_t = design_mtx_t.unsqueeze(1)
            n_factors = design_mtx_t.shape[1]

            if design_mtx_t.shape[0] != self.n_spots:
                raise ValueError(
                    f"Design matrix row count ({design_mtx_t.shape[0]}) must "
                    f"match number of spots ({self.n_spots}) after filtering."
                )

            # Check for constant/zero-variance covariates
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                cov_stds = design_mtx_t.std(dim=0)
                zero_var_indices = torch.where(cov_stds < 1e-5)[0]
                for idx in zero_var_indices:
                    _cname = (
                        resolved_cov_names[idx]
                        if resolved_cov_names is not None
                        else str(idx.item())
                    )
                    warnings.warn(
                        f"Covariate '{_cname}' has near-zero variance "
                        "(std < 1e-5). Consider removing it."
                    )

            self.n_factors = n_factors
            self.design_mtx = design_mtx_t
            self.covariate_names = resolved_cov_names
        else:
            self.n_factors = 0
            self.design_mtx = None
            self.covariate_names = None

        # Move spatial eigenpairs and design matrix to the target device
        _dev = torch.device(self._device)
        if self._corr_sp_eigvals is not None:
            self._corr_sp_eigvals = self._corr_sp_eigvals.to(_dev)
            self._corr_sp_eigvecs = self._corr_sp_eigvecs.to(_dev)
        if self.design_mtx is not None:
            self.design_mtx = self.design_mtx.to(_dev)

    def _setup_from_prebuilt(
        self,
        data: list,
        coordinates: torch.Tensor,
        corr_sp: torch.Tensor,
        corr_sp_eigvals: torch.Tensor,
        corr_sp_eigvecs: torch.Tensor,
        design_mtx: Optional[torch.Tensor],
        gene_names: list,
        group_gene_by_n_iso: bool,
        covariate_names: Optional[list],
    ) -> None:
        """Internal fast-path setup from pre-built tensors (used by permutation loop).

        Skips AnnData parsing, k-NN construction, and eigendecomposition —
        all of which are inherited from the parent model.
        """
        _dataset = IsoDataset(data, gene_names, group_gene_by_n_iso)
        self.n_genes = _dataset.n_genes
        self.n_spots = _dataset.n_spots
        self.n_isos_per_gene = _dataset.n_isos_per_gene
        self.gene_names = _dataset.gene_names
        self._dataset = _dataset
        self._group_gene_by_n_iso = group_gene_by_n_iso
        self.coordinates = coordinates
        self.corr_sp = corr_sp
        self._corr_sp_eigvals = corr_sp_eigvals
        self._corr_sp_eigvecs = corr_sp_eigvecs
        self.design_mtx = design_mtx
        self.n_factors = design_mtx.shape[1] if design_mtx is not None else 0
        self.covariate_names = covariate_names
        self.adata = None
        self._setup_input_mode = "prebuilt"

    # ------------------------------------------------------------------
    # Feature summaries
    # ------------------------------------------------------------------

    def _compute_feature_summaries(self, print_progress: bool = True) -> None:
        """Compute and cache both gene-level and isoform-level summaries."""
        if self._filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if self._gene_summary is not None and self._isoform_summary is not None:
            return

        adata = self._filtered_adata
        n_bins = int(adata.n_obs)
        iso_counts = adata.layers[self._counts_layer]
        is_sparse = scipy.sparse.issparse(iso_counts)

        if is_sparse:
            if not scipy.sparse.isspmatrix_csc(iso_counts):
                iso_counts = iso_counts.tocsc()
        else:
            iso_counts = np.asarray(iso_counts, dtype=float)

        # Derive per-gene isoform lists from the filtered adata var.
        iso_groups = list(
            adata.var.groupby(self._group_iso_by, observed=True, sort=False)
        )

        gene_rows: list[dict] = []
        iso_rows: list[dict] = []

        iterator = zip(self.gene_names, iso_groups)
        if print_progress:
            iterator = tqdm(iterator, desc="Genes", total=len(self.gene_names))

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
            iso_count_var = np.maximum(
                (iso_sumsq / n_bins) - np.square(iso_count_avg), 0.0
            )
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
                    ratio_sum = np.asarray(
                        ratio_counts.sum(axis=0), dtype=float
                    ).ravel()
                    ratio_sumsq = np.asarray(
                        ratio_counts.power(2).sum(axis=0), dtype=float
                    ).ravel()
                else:
                    ratio_counts = gene_counts[valid_rows] / row_sums[valid_rows, None]
                    ratio_sum = ratio_counts.sum(axis=0)
                    ratio_sumsq = np.square(ratio_counts).sum(axis=0)

                ratio_avg = ratio_sum / n_valid
                ratio_var = np.maximum(
                    (ratio_sumsq / n_valid) - np.square(ratio_avg), 0.0
                )
                ratio_std = np.sqrt(ratio_var)
            else:
                ratio_avg = np.zeros(len(iso_names), dtype=float)
                ratio_std = np.zeros(len(iso_names), dtype=float)

            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -(np.log(ratio_total) * ratio_total)
                entropy = float(np.nan_to_num(entropy).sum())

            gene_count_avg = float(gene_total / n_bins)
            gene_count_sumsq = float(np.square(row_sums).sum())
            gene_count_var = max(gene_count_sumsq / n_bins - gene_count_avg**2, 0.0)
            gene_pct_bin_on = float(np.count_nonzero(row_sums) / n_bins)

            gene_rows.append(
                {
                    "gene": gene_name,
                    "n_isos": len(iso_names),
                    "perplexity": float(np.exp(entropy)),
                    "pct_bin_on": gene_pct_bin_on,
                    "count_avg": gene_count_avg,
                    "count_std": float(np.sqrt(gene_count_var)),
                }
            )

            for i, iso_name in enumerate(iso_names):
                iso_rows.append(
                    {
                        **iso_group_df.loc[iso_name].to_dict(),
                        "isoform": iso_name,
                        "gene": gene_name,
                        "pct_bin_on": float(iso_pct_bin_on[i]),
                        "count_total": float(iso_total[i]),
                        "count_avg": float(iso_count_avg[i]),
                        "count_std": float(iso_count_std[i]),
                        "ratio_total": float(ratio_total[i]),
                        "ratio_avg": float(ratio_avg[i]),
                        "ratio_std": float(ratio_std[i]),
                    }
                )

        gene_df = pd.DataFrame(gene_rows).set_index("gene")
        iso_df = pd.DataFrame(iso_rows).set_index("isoform")
        self._gene_summary = gene_df
        self._isoform_summary = iso_df

    def extract_feature_summary(
        self,
        level: Literal["gene", "isoform"] = "gene",
        print_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute filtered feature-level summary statistics.

        Gene-level statistics are aggregated across all isoforms that passed
        the filters applied in :meth:`setup_data`.  Isoform-level statistics
        are computed per isoform and augmented onto the corresponding rows of
        ``adata.var``.

        Results are cached: repeated calls with the same ``level`` return the
        cached :class:`pandas.DataFrame` without recomputation.

        Parameters
        ----------
        level
            Summary granularity.
            ``'gene'``: one row per gene.
            ``'isoform'``: one row per isoform that passed filtering.
        print_progress
            Whether to show a progress bar.

        Returns
        -------
        pandas.DataFrame
            For ``level='gene'``, the index is the gene display name and the
            columns are:

            - ``'n_isos'``: int. Number of isoforms retained after filtering.
            - ``'perplexity'``: float. Effective number of isoforms based on
              the marginal isoform usage entropy.
            - ``'pct_bin_on'``: float. Fraction of spots with non-zero total
              gene counts.
            - ``'count_avg'``: float. Mean per-spot total count for the gene.
            - ``'count_std'``: float. Std of per-spot total count for the gene.

            For ``level='isoform'``, the index is the isoform name (matching
            ``adata.var_names``) and the columns are the original ``adata.var``
            columns plus:

            - ``'pct_bin_on'``: float. Fraction of spots with count > 0.
            - ``'count_total'``: float. Total counts across all spots.
            - ``'count_avg'``: float. Mean count per spot.
            - ``'count_std'``: float. Std of count per spot.
            - ``'ratio_total'``: float. Fraction of total gene counts
              attributable to this isoform.
            - ``'ratio_avg'``: float. Mean per-spot isoform usage ratio
              (computed over spots with non-zero gene coverage).
            - ``'ratio_std'``: float. Std of per-spot isoform usage ratio
              (computed over spots with non-zero gene coverage).

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        ValueError
            If ``level`` is not ``'gene'`` or ``'isoform'``.
        """
        if self._filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if level not in {"gene", "isoform"}:
            raise ValueError("`level` must be one of 'gene' or 'isoform'.")

        self._compute_feature_summaries(print_progress=print_progress)

        if level == "gene":
            return self._gene_summary
        return self._isoform_summary

    def get_formatted_test_results(
        self, test_type: Literal["sv", "du"]
    ) -> pd.DataFrame:
        """Get the formatted test results as data frame.

        Parameters
        ----------
        test_type : {"sv", "du"}
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

    def fit(
        self,
        n_jobs: int = 1,
        batch_size: int = 1,
        quiet: bool = True,
        print_progress: bool = True,
        with_design_mtx: bool = False,
        from_null: bool = False,
        refit_null: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Model fitting.

        Parameters
        ----------
        n_jobs : int, optional
            The number of cores to use for parallel fitting. Default to 1.
        batch_size : int, optional
            The maximum number of genes per job to fit in parallel. Default to 1.
        quiet : bool, optional
            Whether to suppress the fitting logs. Default to True.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        with_design_mtx : bool, optional
            Whether to include the design matrix for the fixed effects. Default to False.
        from_null : bool, optional
            Whether to initialize the full model from a null model
            with zero spatial variability (white-noise random effect).
        refit_null : bool, optional
            Whether to refit the null model after fitting the full model.
        random_seed : int or None, optional
            The random seed for reproducibility. Default to None.

        See also
        --------
        :func:`splisosm.model.MultinomGLMM.fit` for fitting a single model.
        """

        if batch_size > 1 and not self._group_gene_by_n_iso:
            warnings.warn(
                "Ignoring batch size argument since the dataset is not grouped. "
                + "For batch fitting please set 'group_gene_by_n_iso = True' when setup_data()"
            )
            batch_size = 1

        if from_null:
            # fit the null and full model sequentially
            self._fit_null_full_sv(
                n_jobs=n_jobs,
                batch_size=batch_size,
                quiet=quiet,
                print_progress=print_progress,
                refit_null=refit_null,
                with_design_mtx=with_design_mtx,
                random_seed=random_seed,
            )
        else:
            # fit the full model only de novo
            self._fit_denovo(
                n_jobs=n_jobs,
                batch_size=batch_size,
                quiet=quiet,
                print_progress=print_progress,
                with_design_mtx=with_design_mtx,
                random_seed=random_seed,
            )

        # store the fitting configurations
        self.model_configs["fitting_configs"].update(
            {
                "with_design_mtx": with_design_mtx,
                "from_null": from_null,
                "refit_null": refit_null,
                "batch_size": batch_size,
            }
        )

        self._is_trained = True

    def save(self, path: str) -> None:
        """Save the fitted models to a file.

        This function is designed to overcome a limitation of ``torch.save()``,
        which naively saves `n_genes` copies of the spatial kernel matrix ``self.corr_sp``.
        The kernel matrix is reconstructed per fitted model using ``self._corr_sp_eigvals``
        and ``self._corr_sp_eigvecs``.

        Parameters
        ----------
        path
            The path to save the fitted models.
        """
        for key in ["models_glmm-full", "models_glmm-null"]:
            if key in self.fitting_results and len(self.fitting_results[key]) > 0:
                # update model.corr_sp as a reference to self.corr_sp
                fitted_models = self.fitting_results[key]
                for model in fitted_models:
                    model.corr_sp = self.corr_sp

        torch.save(self, path)

    def get_fitted_models(self) -> list[Any]:
        """Get the fitted models after running fit().

        Returns
        -------
        models: list of fitted models
            - ``model_type='glmm-full'``: list[splisosm.hyptest_glmm.IsoFullModel]
            - ``model_type='glmm-null'``: list[splisosm.hyptest_glmm.IsoNullNoSpVar]
            - ``model_type='glm'``: list[splisosm.model.MultinomGLM]
        """
        if self.model_type == "glmm-full":
            return self.fitting_results["models_glmm-full"]
        elif self.model_type == "glmm-null":
            return self.fitting_results["models_glmm-null"]
        elif self.model_type == "glm":
            return self.fitting_results["models_glm"]
        else:
            raise ValueError(f"Invalid model type {self.model_type}.")

    def _ungroup_fitted_models(
        self, fitted_models: list[Any], batch_size: int, with_design_mtx: bool
    ) -> list[Any]:
        """Ungroup the fitted models to match the original gene names.

        Parameters
        ----------
        fitted_models : list
            List of length n_batches.
        batch_size : int
            The maximum number of genes per job to fit in parallel.
        with_design_mtx : bool
            Whether to include the design matrix for the fixed effects.

        Returns
        -------
        list
            Fitted models ungrouped, list of length n_genes in the original order.
        """
        data = self._dataset.get_dataloader(batch_size=batch_size)

        fitted_models_ungrouped = []
        gene_names_ungroupped = []

        # loop over each batch
        for grouped_m, batch in zip(fitted_models, data):
            # unwrap the batch
            b_n_isos, b_counts, b_gene_names = (
                batch["n_isos"],
                batch["x"],
                batch["gene_name"],
            )
            assert b_n_isos[0] == grouped_m.n_isos

            # add the gene names to the list
            gene_names_ungroupped.extend(b_gene_names)

            # extract batched parameters
            return_par_names = [
                "nu",
                "beta",
                "bias_eta",
                "sigma",
                "theta_logit",
                "sigma_sp",
                "sigma_nsp",
            ]
            pars = {
                k: v.detach()
                for k, v in grouped_m.state_dict().items()
                if k in return_par_names
            }

            # loop over each gene in the batch
            for _g in range(b_counts.shape[0]):
                # extract the counts and fitted parameters for the gene
                if b_counts.is_sparse:
                    b_counts = b_counts.to_dense()
                _g_counts = b_counts[_g : (_g + 1), ...]  # (1, n_spots, b_n_isos)
                _g_pars = {k: v[_g : (_g + 1), ...] for k, v in pars.items()}

                # initialize and setup the model
                if self.model_type == "glm":
                    model = MultinomGLM()
                    model.setup_data(
                        _g_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        device=self._device,
                    )
                else:
                    if self.model_type == "glmm-full":
                        model = IsoFullModel(**self.model_configs)
                    elif self.model_type == "glmm-null":
                        model = IsoNullNoSpVar(**self.model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self.model_type}.")

                    model.setup_data(
                        _g_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals,
                        corr_sp_eigvecs=self._corr_sp_eigvecs,
                        device=self._device,
                    )

                # update model parameters
                model.update_params_from_dict(_g_pars)
                fitted_models_ungrouped.append(model)

        # reorder the fitted models by gene names
        _order = [gene_names_ungroupped.index(_g) for _g in self.gene_names]
        fitted_models_ungrouped = [fitted_models_ungrouped[i] for i in _order]

        return fitted_models_ungrouped

    def _fit_denovo(
        self,
        n_jobs: int = 1,
        batch_size: int = 1,
        quiet: bool = True,
        print_progress: bool = True,
        with_design_mtx: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """Fit the selected model to the data de novo.

        Parameters
        ----------
        n_jobs : int, optional
            The number of cores to use for parallel fitting. Default to 1.
        batch_size : int, optional
            The maximum number of genes per job to fit in parallel. Default to 1.
        quiet : bool, optional
            Whether to suppress the fitting logs. Default to True.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        with_design_mtx : bool, optional
            Whether to include the design matrix for the fixed effects. Default to False.
        random_seed : int, optional
            The random seed for reproducibility. Default to None.
        """
        # empty existing models before the new run
        fitted_models = []

        # decide whether to use multiprocessing
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_jobs > 1 and self._device != "cpu":
            warnings.warn(
                f"Parallel fitting (n_jobs={n_jobs}) is not supported for "
                f"device={self._device!r}. Falling back to n_jobs=1.",
                UserWarning,
                stacklevel=2,
            )
            n_jobs = 1

        # start timer
        t_start = timer()

        # extract the dataloader
        n_batches = sum(1 for _ in self._dataset.get_dataloader(batch_size=batch_size))
        data = self._dataset.get_dataloader(batch_size=batch_size)

        if n_jobs == 1:  # use single core
            if print_progress:
                print(
                    f"Fitting with single core for {self.n_genes} genes (batch_size={batch_size})."
                )

            # iterate over genes and fit the selected model
            for batch in tqdm(data, disable=not print_progress, total=n_batches):
                _, b_counts, _ = (batch["n_isos"], batch["x"], batch["gene_name"])

                if b_counts.is_sparse:
                    b_counts = b_counts.to_dense()

                # initialize and setup the model
                if self.model_type == "glm":
                    model = MultinomGLM()
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        device=self._device,
                    )
                else:
                    if self.model_type == "glmm-full":
                        model = IsoFullModel(**self.model_configs)
                    elif self.model_type == "glmm-null":
                        model = IsoNullNoSpVar(**self.model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self.model_type}.")
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals,
                        corr_sp_eigvecs=self._corr_sp_eigvecs,
                        device=self._device,
                    )

                # fit the model
                model.fit(
                    quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed
                )
                fitted_models.append(model)
        else:
            if print_progress:
                print(
                    f"Fitting with {n_jobs} cores for {self.n_genes} genes (batch_size={batch_size})."
                )
                print(
                    "Note: the progress bar is updated before each fitting, rather than when it finishes."
                )

            # Prepare tasks with delayed to ensure they're ready for parallel execution
            tasks_gen = (
                delayed(_fit_model_one_gene)(
                    self.model_configs,
                    self.model_type,
                    batch["x"],
                    self._corr_sp_eigvals,
                    self._corr_sp_eigvecs,
                    self.design_mtx if with_design_mtx else None,
                    quiet,
                    random_seed,
                    self._device,
                )
                for batch in data
            )

            fitted_pars = Parallel(n_jobs=n_jobs)(
                tqdm(tasks_gen, total=n_batches, disable=not print_progress)
            )

            # convert the fitted parameters to models
            for batch, pars in zip(
                self._dataset.get_dataloader(batch_size=batch_size), fitted_pars
            ):
                # unwrap the batch
                b_counts = batch["x"]

                # initialize and setup the model
                if self.model_type == "glm":
                    model = MultinomGLM()
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        device=self._device,
                    )
                else:
                    if self.model_type == "glmm-full":
                        model = IsoFullModel(**self.model_configs)
                    elif self.model_type == "glmm-null":
                        model = IsoNullNoSpVar(**self.model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self.model_type}.")
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals,
                        corr_sp_eigvecs=self._corr_sp_eigvecs,
                        device=self._device,
                    )

                # update model parameters
                model.update_params_from_dict(pars)
                fitted_models.append(model)

        # ungroup the fitted models to match the original gene names
        if batch_size > 1:
            fitted_models = self._ungroup_fitted_models(
                fitted_models, batch_size, with_design_mtx
            )

        # store the fitted models
        if self.model_type == "glmm-full":
            self.fitting_results["models_glmm-full"] = fitted_models
        elif self.model_type == "glmm-null":
            self.fitting_results["models_glmm-null"] = fitted_models
        elif self.model_type == "glm":
            self.fitting_results["models_glm"] = fitted_models
        else:
            raise ValueError(f"Invalid model type {self.model_type}.")

        # stop timer
        t_end = timer()

        if print_progress:
            print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

    def _fit_null_full_sv(
        self,
        refit_null=True,
        n_jobs=1,
        batch_size=1,
        quiet=True,
        print_progress=True,
        with_design_mtx=True,
        random_seed=None,
    ):
        """Fit the null (no spatial random effect) and the full model to the data sequentially.

        Parameters
        ----------
        refit_null : bool, optional
            Whether to refit the null model after fitting the full model.
            Default to True.
        n_jobs : int, optional
            The number of cores to use for parallel fitting. Default to 1.
        batch_size : int, optional
            The maximum number of genes per job to fit in parallel. Default to 1.
        quiet : bool, optional
            Whether to suppress the fitting logs. Default to True.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
            Only applicable when n_jobs = 1.
        with_design_mtx : bool, optional
            Whether to include the design matrix for the fixed effects. Default to True.
        random_seed : int, optional
            The random seed for reproducibility. Default to None.
        """
        # empty existing models before the new run
        fitted_null_models_sv = []
        fitted_full_models = []

        # decide whether to use multiprocessing
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_jobs > 1 and self._device != "cpu":
            warnings.warn(
                f"Parallel fitting (n_jobs={n_jobs}) is not supported for "
                f"device={self._device!r}. Falling back to n_jobs=1.",
                UserWarning,
                stacklevel=2,
            )
            n_jobs = 1

        # start timer
        t_start = timer()

        # extract the dataloader
        n_batches = sum(1 for _ in self._dataset.get_dataloader(batch_size=batch_size))
        data = self._dataset.get_dataloader(batch_size=batch_size)

        if n_jobs == 1:  # use single core
            if print_progress:
                print(
                    f"Fitting with single core for {self.n_genes} genes (batch_size={batch_size})."
                )

            # iterate over genes and fit the selected model
            for batch in tqdm(data, disable=not print_progress, total=n_batches):
                _, b_counts, _ = (batch["n_isos"], batch["x"], batch["gene_name"])

                # fit the null model
                null = IsoNullNoSpVar(**self.model_configs)
                null.setup_data(
                    b_counts,
                    design_mtx=self.design_mtx if with_design_mtx else None,
                    corr_sp_eigvals=self._corr_sp_eigvals,
                    corr_sp_eigvecs=self._corr_sp_eigvecs,
                    device=self._device,
                )
                null.fit(
                    quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed
                )

                # fit the full model from the null
                full = IsoFullModel.from_trained_null_no_sp_var_model(null)
                full.fit(
                    quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed
                )

                # refit the null model if needed
                if refit_null:
                    null_refit = IsoNullNoSpVar.from_trained_full_model(full)
                    null_refit.fit(
                        quiet=quiet,
                        verbose=False,
                        diagnose=False,
                        random_seed=random_seed,
                    )

                    # update the null if larger log-likelihood
                    if (
                        null_refit().mean() > null().mean()
                    ):  # null() returns shape of (n_genes,)
                        null = null_refit

                    # refit the full model from the null if likelihood decreases
                    if null().mean() > full().mean():
                        full_refit = IsoFullModel.from_trained_null_no_sp_var_model(
                            null
                        )
                        full_refit.fit(
                            quiet=quiet,
                            verbose=False,
                            diagnose=False,
                            random_seed=random_seed,
                        )
                        if full_refit().mean() > full().mean():
                            full = full_refit

                fitted_null_models_sv.append(null)
                fitted_full_models.append(full)

        else:  # use multiprocessing
            if print_progress:
                print(
                    f"Fitting with {n_jobs} cores for {self.n_genes} genes (batch_size={batch_size})."
                )
                print(
                    "Note: the progress bar is updated before each fitting, rather than when it finishes."
                )

            # Prepare tasks with delayed to ensure they're ready for parallel execution
            tasks_gen = (
                delayed(_fit_null_full_sv_one_gene)(
                    self.model_configs,
                    batch["x"],
                    self._corr_sp_eigvals,
                    self._corr_sp_eigvecs,
                    self.design_mtx if with_design_mtx else None,
                    refit_null,
                    quiet,
                    random_seed,
                    self._device,
                )
                for batch in data
            )

            fitted_pars = Parallel(n_jobs=n_jobs)(
                tqdm(tasks_gen, total=n_batches, disable=not print_progress)
            )

            # convert the fitted parameters to models
            for batch, (n_par, f_par) in zip(
                self._dataset.get_dataloader(batch_size=batch_size), fitted_pars
            ):
                # unwrap the batch
                b_counts = batch["x"]

                # null models
                null = IsoNullNoSpVar(**self.model_configs)
                null.setup_data(
                    b_counts,
                    design_mtx=self.design_mtx if with_design_mtx else None,
                    corr_sp_eigvals=self._corr_sp_eigvals,
                    corr_sp_eigvecs=self._corr_sp_eigvecs,
                    device=self._device,
                )
                # update model parameters
                null.update_params_from_dict(n_par)

                # full models
                full = IsoFullModel.from_trained_null_no_sp_var_model(null)
                full.update_params_from_dict(f_par)

                fitted_null_models_sv.append(null)
                fitted_full_models.append(full)

        # ungroup the fitted models to match the original gene names
        if batch_size > 1:
            fitted_null_models_sv = self._ungroup_fitted_models(
                fitted_null_models_sv, batch_size, with_design_mtx
            )
            fitted_full_models = self._ungroup_fitted_models(
                fitted_full_models, batch_size, with_design_mtx
            )

        # store the fitted models
        self.fitting_results["models_glmm-null"] = fitted_null_models_sv
        self.fitting_results["models_glmm-full"] = fitted_full_models

        # stop timer
        t_end = timer()

        if print_progress:
            print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

    def _fit_sv_llr_perm(
        self,
        n_perms: int = 20,
        n_jobs: int = 1,
        print_progress: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Calculate the null distribution of likelihood ratio using permutation.

        Parameters
        ----------
        n_perms : int, optional
            The number of permutations to run per gene. Default to 20.
        n_jobs : int, optional
            The number of cores to use for parallel fitting. Default to 1.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        random_seed : int, optional
            The random seed for reproducibility. Default to None.
        """
        # fit permutated data using the same null model
        fitting_configs = self.model_configs["fitting_configs"]

        try:
            with_design_mtx = fitting_configs["with_design_mtx"]
            refit_null = fitting_configs["refit_null"]
            batch_size = fitting_configs["batch_size"]
        except KeyError:
            raise ValueError(
                "Null models not found. Please run fit() with from_null = True first."
            )

        if random_seed is not None:  # set random seed for reproducibility
            torch.manual_seed(random_seed)

        # extract the likelihood ratio statistics from each permutation
        _sv_llr_perm_stats = []

        # decide whether to use multiprocessing
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_jobs > 1 and self._device != "cpu":
            warnings.warn(
                f"Parallel fitting (n_jobs={n_jobs}) is not supported for "
                f"device={self._device!r}. Falling back to n_jobs=1.",
                UserWarning,
                stacklevel=2,
            )
            n_jobs = 1

        # start timer
        t_start = timer()

        if n_jobs == 1:  # use single core
            if print_progress:
                print(
                    f"Running permutation with single core for {self.n_genes} genes "
                    f"(n_perms={n_perms}, batch_size={batch_size})."
                )

            # run n_perms permutations for each gene
            for _ in tqdm(range(n_perms), disable=not print_progress):
                # randomly shuffle the spatial locations
                perm_idx = torch.randperm(self.n_spots)

                # fit a new SplisosmGLMM model using pre-built tensors (fast path)
                new_model = SplisosmGLMM(**self.model_configs, device=self._device)
                new_design_mtx = (
                    self.design_mtx[perm_idx, :]
                    if (self.design_mtx is not None and with_design_mtx)
                    else None
                )
                new_data = [_d[perm_idx, :] for _d in self._dataset.data]
                new_model._setup_from_prebuilt(
                    data=new_data,
                    coordinates=self.coordinates,
                    corr_sp=self.corr_sp,
                    corr_sp_eigvals=self._corr_sp_eigvals,
                    corr_sp_eigvecs=self._corr_sp_eigvecs,
                    design_mtx=new_design_mtx,
                    gene_names=self.gene_names,
                    group_gene_by_n_iso=self._group_gene_by_n_iso,
                    covariate_names=self.covariate_names,
                )
                new_model._fit_null_full_sv(
                    refit_null=refit_null,
                    n_jobs=1,
                    batch_size=batch_size,
                    quiet=True,
                    print_progress=False,
                    with_design_mtx=with_design_mtx,
                    random_seed=random_seed,
                )

                # calculate the likelihood ratio statistic
                _sv_llr_stats = []
                for full_m, null_m in zip(
                    new_model.fitting_results["models_glmm-full"],
                    new_model.fitting_results["models_glmm-null"],
                ):
                    # use marginal likelihood for stability
                    llr, _ = _calc_llr_spatial_variability(null_m, full_m)
                    _sv_llr_stats.append(llr)

                _sv_llr_stats = torch.tensor(_sv_llr_stats)
                _sv_llr_perm_stats.append(_sv_llr_stats)

            # save the llr statistics from permutated data
            self.fitting_results["sv_llr_perm_stats"] = torch.concat(
                _sv_llr_perm_stats, dim=0
            )

        else:  # use multiprocessing
            if print_progress:
                print(
                    f"Running permutation with {n_jobs} cores for {self.n_genes} genes "
                    f"(n_perms={n_perms}, batch_size={batch_size})."
                )
                print(
                    "Note: the progress bar is updated before each fitting, rather than when it finishes."
                )

            # extract the dataloader
            n_batches = sum(
                1 for _ in self._dataset.get_dataloader(batch_size=batch_size)
            )
            data = self._dataset.get_dataloader(batch_size=batch_size)

            # Prepare tasks with delayed to ensure they're ready for parallel execution
            tasks_gen = (
                delayed(_fit_perm_one_gene)(
                    torch.randperm(self.n_spots),
                    self.model_configs,
                    batch["x"],
                    self._corr_sp_eigvals,
                    self._corr_sp_eigvecs,
                    self.design_mtx if with_design_mtx else None,
                    refit_null,
                    random_seed,
                    self._device,
                )
                for batch in data
                for _ in range(n_perms)
            )

            _sv_llr_perm_stats = Parallel(n_jobs=n_jobs)(
                tqdm(tasks_gen, total=(n_batches * n_perms), disable=not print_progress)
            )

            self.fitting_results["sv_llr_perm_stats"] = torch.concat(
                _sv_llr_perm_stats, dim=0
            )

        # stop timer
        t_end = timer()

        if print_progress:
            print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

    def test_spatial_variability(
        self,
        method: str = "llr",
        use_perm_null: bool = False,
        return_results: bool = False,
        print_progress: bool = True,
        n_perms_per_gene: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Parametric test for spatial variability.

        .. caution::
            The likelihood ratio statistic is not well-calibrated for sparse data.
            We recommend the non-parametric HSIC-based tests in :class:`splisosm.hyptest_np.SplisosmNP`
            for spatial variability testing.

        Note that the parametric and non-parametric tests are assymptotically equivalent under the null.
        See :cite:`su2026consistent` for detailed theoretical analysis.

        Parameters
        ----------
        method : {"llr"}, optional
            The test method.
            Currently only support ``"llr"``, the likelihood ratio test (H_0: ``sigma_sp`` = 0).
        use_perm_null : bool, optional
            Whether to generate the null distribution from permutation.
            If False, use the chi-square with df = n_var_components as the null.
        return_results : bool, optional
            Whether to return the test statistics and p-values.
            If False, the results will be stored in ``self.sv_test_results``.
        print_progress : bool, optional
            Whether to show the progress bar for permutation. Default to True.
        n_perms_per_gene : int or None, optional
            Number of permutations per gene when ``use_perm_null=True``. Default to 20.
        **kwargs : Any
            Additional arguments passed to `self._fit_sv_llr_perm()` if ``use_perm_null = True``.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise returns None.

        See also
        --------
        :func:`splisosm.hyptest_np.SplisosmNP.test_spatial_variability` for non-parametric tests.
        """

        valid_methods = ["llr"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."

        # Parametric likelihood ratio test for spatial variability. Need to fit the null and full models.
        if len(self.fitting_results["models_glmm-null"]) == 0:
            raise ValueError(
                "Null models not found. Please run fit() with from_null = True first."
            )

        _sv_llr_stats, _sv_llr_dfs = [], []
        # iterate over genes and calculate the likelihood ratio statistic
        for full_m, null_m in zip(
            self.fitting_results["models_glmm-full"],
            self.fitting_results["models_glmm-null"],
        ):
            # use marginal likelihood for stability
            llr, df = _calc_llr_spatial_variability(null_m, full_m)
            _sv_llr_stats.append(llr)
            _sv_llr_dfs.append(df)

        _sv_llr_stats = torch.tensor(_sv_llr_stats)
        _sv_llr_dfs = torch.tensor(_sv_llr_dfs)

        if use_perm_null:
            # use permutation to calculate the p-value.
            if "sv_llr_perm_stats" not in self.fitting_results.keys():
                self._fit_sv_llr_perm(
                    n_perms=n_perms_per_gene if n_perms_per_gene is not None else 20,
                    print_progress=print_progress,
                    **kwargs,
                )
            else:  # use the cached results if available
                print("Using cached permutation results...")

            _sv_llr_perm = self.fitting_results["sv_llr_perm_stats"]
            _sv_llr_pvals = 1 - (_sv_llr_stats[:, None] > _sv_llr_perm[None, :]).sum(
                1
            ) / len(_sv_llr_perm)
        else:
            # calculate the p-value using chi-square distribution
            # move to CPU before scipy (scipy does not support non-CPU tensors)
            _sv_llr_pvals = 1 - chi2.cdf(_sv_llr_stats.cpu(), df=_sv_llr_dfs.cpu())
            _sv_llr_pvals = torch.tensor(_sv_llr_pvals)

        # store the results (always on CPU for downstream pandas/numpy usage)
        self.sv_test_results = {
            "statistic": _sv_llr_stats.cpu().numpy(),
            "pvalue": _sv_llr_pvals.cpu().numpy(),
            "df": _sv_llr_dfs.cpu().numpy(),
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
        method: Literal["score", "wald"] = "score",
        print_progress: bool = True,
        return_results: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Parametric test for spatial isoform differential usage.

        Before running this function, the design matrix must be set up using :func:`setup_data`.
        Each column of the design matrix corresponds to a covariate to test for differential association
        with the isoform usage ratios of each gene.
        Test statistics and p-values are computed per (gene, covariate) pair separately.

        Similar to :func:`splisosm.hyptest_np.SplisosmNP.test_differential_usage`, here we also support two types of association tests but **implicitly**:

        - Unconditional (when ``model_type='glm'``): test the unconditional association between isoform usage ratios and the covariate of interest (H_0: ``beta`` = 0).
        - Conditional (when ``model_type='glmm-full'``): test for association (H_0: ``beta`` = 0) conditioned on the spatial random effect.

        Parameters
        ----------
        method : {"score", "wald"}, optional
            Depending on whether the design matrix is used for model fitting,
            different methods must be used for hypothesis testing:

            - ``"wald"`` when models were fit with ``fit(..., with_design_mtx=True)``.
            - ``"score"`` when models were fit with ``fit(..., with_design_mtx=False)``.

            .. caution::
                The Wald statistic with GLM/GLMM is empirically anti-conserved (i.e., lots of false positives).
                Always use ``method="score"`` when possible.

        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        return_results : bool, optional
            Whether to return the test statistics and p-values.
            If False, the results are stored in ``self.du_test_results``.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise, returns None and stores results in self.du_test_results.
        """
        if self.design_mtx is None:
            raise ValueError("No design matrix is provided. Run setup_data() first.")

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["wald", "score"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."

        if method == "score":  # Score test
            # extract the fitted full models
            fitted_models = self.get_fitted_models()
            if len(fitted_models) == 0:
                raise ValueError(
                    "Fitted full models not found. Run fit(..., with_design_mtx = False) first."
                )
            if self.model_configs["fitting_configs"]["with_design_mtx"]:
                raise ValueError(
                    "Design matrix is included in the fitted models. "
                    "Perhaps you want to use the wald test. Otherwise please run fit() with with_design_mtx = False."
                )

            _du_score_stats, _du_score_dfs = [], []
            # iterate over genes and calculate the score statistic
            for m in tqdm(fitted_models, disable=(not print_progress)):
                score_stat, score_df = _calc_score_differential_usage(
                    m, self.design_mtx
                )
                _du_score_stats.append(score_stat)
                _du_score_dfs.append(score_df)

            _du_score_stats = torch.stack(_du_score_stats, dim=0).reshape(
                -1, n_factors
            )  # (n_genes, n_factors)
            _du_score_dfs = (
                torch.tensor(_du_score_dfs).unsqueeze(-1).expand(-1, n_factors)
            )  # (n_genes, n_factors)

            # calculate the p-value using chi-square distribution
            # move to CPU before scipy (scipy does not support non-CPU tensors)
            _du_score_pvals = 1 - chi2.cdf(
                _du_score_stats.cpu(), df=_du_score_dfs.cpu()
            )
            _du_score_pvals = torch.tensor(_du_score_pvals)

            # store the results (always on CPU for downstream pandas/numpy usage)
            self.du_test_results = {
                "statistic": _du_score_stats.cpu(),  # (n_genes, n_factors)
                "pvalue": _du_score_pvals.cpu(),  # (n_genes, n_factors)
                "method": method,
            }

        else:  # method == 'wald', Wald test (anti-conservative)
            # extract the fitted full models
            fitted_models = self.get_fitted_models()
            if len(fitted_models) == 0:
                raise ValueError("Fitted full models not found. Run fit() first.")
            if not self.model_configs["fitting_configs"]["with_design_mtx"]:
                raise ValueError(
                    "Design matrix is not included in the fitted models. "
                    "Perhaps you want to use the score test. Otherwise please run fit() with with_design_mtx = True."
                )

            _du_wald_stats, _du_wald_dfs = [], []
            # iterate over genes and calculate the Wald statistic
            for m in tqdm(fitted_models, disable=(not print_progress)):
                wald_stat, wald_df = _calc_wald_differential_usage(m)
                _du_wald_stats.append(wald_stat)
                _du_wald_dfs.append(wald_df)

            _du_wald_stats = torch.stack(_du_wald_stats, dim=0).reshape(
                -1, n_factors
            )  # (n_genes, n_factors)
            _du_wald_dfs = (
                torch.tensor(_du_wald_dfs).unsqueeze(-1).expand(-1, n_factors)
            )  # (n_genes, n_factors)

            # calculate the p-value using chi-square distribution
            # move to CPU before scipy (scipy does not support non-CPU tensors)
            _du_wald_pvals = 1 - chi2.cdf(
                _du_wald_stats.cpu(), df=_du_wald_dfs.cpu()
            )
            _du_wald_pvals = torch.tensor(_du_wald_pvals)

            # store the results (always on CPU for downstream pandas/numpy usage)
            self.du_test_results = {
                "statistic": _du_wald_stats.cpu(),  # (n_genes, n_factors)
                "pvalue": _du_wald_pvals.cpu(),  # (n_genes, n_factors)
                "method": method,
            }

        # calculate adjusted p-values (independently for each factor)
        self.du_test_results["pvalue_adj"] = false_discovery_control(
            self.du_test_results["pvalue"], axis=0
        )

        # return the results if needed
        if return_results:
            return self.du_test_results
