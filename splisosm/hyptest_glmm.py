"""GLMM-based hypothesis tests for spatial isoform usage."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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
from splisosm.kernel import IdentityKernel, SpatialCovKernel
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


@dataclass
class _FittedGeneState:
    """Minimal per-gene state extracted after fitting."""

    state_dict: dict[str, torch.Tensor]
    convergence: bool
    best_loss: float
    best_epoch: int
    fitting_time: float
    n_isos: int


class SplisosmGLMM:
    """Parametric spatial isoform statistical modeling using GLMM.

    This is a convenience class that wraps around the :class:`splisosm.model.MultinomGLMM`
    for batched model fitting and spatial variability and differential usage testing.

    Examples
    --------
    Prepare an AnnData object and fit a GLMM:

    >>> from splisosm import SplisosmGLMM
    >>> # adata : AnnData of shape (n_spots, n_isoforms)
    >>> #   adata.layers["counts"]    — raw isoform counts
    >>> #   adata.var["gene_symbol"]  — column grouping isoforms by gene
    >>> #   adata.obsm["spatial"]     — (n_spots, 2) spatial coordinates
    >>> #   adata.obs["covariate"]    — optional covariate column for DU testing
    >>> model = SplisosmGLMM(
    ...     model_type="glmm-full",    # 'glmm-full' | 'glmm-null' | 'glm'
    ...     fitting_method="joint_gd", # 'joint_gd' | 'joint_newton' | 'marginal_gd' | 'marginal_newton'
    ...     device="cpu",              # 'cpu' | 'cuda' (NVIDIA) | 'mps' (Apple Silicon)
    ...     approx_rank=None,          # None (exact) or int (low-rank spatial kernel)
    ... )
    >>> model.setup_data(
    ...     adata,
    ...     layer="counts",
    ...     group_iso_by="gene_symbol",
    ...     group_gene_by_n_iso=True,  # required for batch_size > 1 in fit()
    ...     design_mtx="covariate",    # obs column name, or (n_spots, n_factors) array
    ... )
    >>> model.fit(
    ...     n_jobs=1, batch_size=20,
    ...     with_design_mtx=False,     # False → score test (recommended for DU)
    ... )
    >>> fitted_models = model.get_fitted_models()

    Differential usage test:

    >>> model.test_differential_usage(method="score")  # or "wald" if with_design_mtx=True
    >>> du_results = model.get_formatted_test_results("du")
    >>> print(du_results.head())
    """

    # -- Public attributes (populated by :meth:`setup_data`) ------------------

    n_genes: int
    """Number of genes after filtering."""

    n_spots: int
    """Number of spatial spots."""

    n_isos_per_gene: list[int]
    """Number of isoforms per gene (list of length :attr:`n_genes`)."""

    n_factors: int
    """Number of covariates for differential usage testing."""

    gene_names: list[str]
    """Gene display names (length :attr:`n_genes`)."""

    covariate_names: list[str]
    """Covariate display names (length :attr:`n_factors`)."""

    adata: Optional[AnnData]
    """Source :class:`~anndata.AnnData` object; ``None`` before :meth:`setup_data`."""

    sp_kernel: Any
    """Spatial kernel (:class:`~splisosm.kernel.SpatialCovKernel` for ``'glmm-full'``,
    :class:`~splisosm.kernel.IdentityKernel` for ``'glmm-null'``, ``None`` for ``'glm'``).
    Set by :meth:`setup_data`."""

    design_mtx: Optional[torch.Tensor]
    """Design matrix ``(n_spots, n_factors)``; ``None`` if no covariates."""

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
        self._model_type = model_type

        self._model_configs = {
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
        self._fitted_states: dict[str, list[_FittedGeneState]] = {
            "glmm-full": [],
            "glmm-null": [],
            "glm": [],
        }

        # to store permutation LLR statistics (populated by _fit_sv_llr_perm)
        self._sv_llr_perm_stats: Optional[torch.Tensor] = None

        # to store the spatial variability test results after running test_spatial_variability()
        self._sv_test_results = {}

        # to store the differential usage test results after running test_differential_usage()
        self._du_test_results = {}

    def __str__(self):
        _sv_status = (
            f"Completed ({self._sv_test_results['method']})"
            if len(self._sv_test_results) > 0
            else "NA"
        )
        _du_status = (
            f"Completed ({self._du_test_results['method']})"
            if len(self._du_test_results) > 0
            else "NA"
        )
        if self._is_trained:
            try:
                key = self._model_key_for_type()
                n_conv = sum(s.convergence for s in self._fitted_states[key])
                _fit_line = (
                    f"- Trained: True (converged: {n_conv} / {self.n_genes} genes)\n"
                )
            except Exception:
                _fit_line = "- Trained: True\n"
        else:
            _fit_line = "- Trained: False\n"

        _config_desc = ""
        if self._model_type in ["glmm-full", "glmm-null"]:
            if self._model_type == "glmm-full":
                _config_desc = (
                    f"- Mutual neighbors K: {self._kernel_k_neighbors}\n"
                    f"- Spatial autocorrelation rho: {self._kernel_rho}\n"
                    f"- Variance parameterized with theta_logit: {self._model_configs['var_parameterization_sigma_theta']}\n"
                )
            _config_desc += (
                f"- Learnable variance: {not self._model_configs['var_fix_sigma']}\n"
                f"- Same variance across classes: {self._model_configs['share_variance']}\n"
                f"- Prior on total variance: {self._model_configs['var_prior_model']}\n"
                f"- Initialization method: {self._model_configs['init_ratio']}\n"
            )
        return (
            "=== SplisosmGLMM\n"
            + f"- Number of genes: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Average isoforms per gene: {np.mean(self.n_isos_per_gene) if self.n_isos_per_gene is not None else None}\n"
            + "=== Model configurations\n"
            + f"- Model type: {self._model_type}\n"
            + _config_desc
            + "=== Fitting configurations\n"
            + _fit_line
            + f"- Fitting method: {self._model_configs['fitting_method']}\n"
            + f"- Parameters: {self._model_configs['fitting_configs']}\n"
            + "=== Test results\n"
            + f"- Spatial variability: {_sv_status}\n"
            + f"- Differential usage: {_du_status}"
        )

    __repr__ = __str__

    @property
    def filtered_adata(self) -> AnnData:
        """The filtered AnnData of shape (:attr:`n_spots`, sum(:attr:`n_isos_per_gene`)).

        This is the data used internally after :meth:`setup_data`.
        It is a copy of the input :attr:`adata`, subsetted to the retained spots and isoforms after filtering.

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        """
        if self._filtered_adata is None:
            raise RuntimeError("Data not initialised. Call setup_data() first.")
        return self._filtered_adata

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
        self._coordinates = coordinates

        # Build spatial kernel only when the model actually uses spatial random effects.
        # For 'glm' there are no random effects at all.
        # For 'glmm-null' the spatial variance is fixed at 0 (theta = -inf), so the
        # kernel cancels out.  We use a rank-1 dummy covariance (constant vector scaled
        # to unit norm) which, combined with theta = 0, gives the exact identity
        # covariance C = σ²I via the low-rank Woodbury formula.
        n_spots = self.n_spots

        if self._model_type == "glm":
            # No spatial random effects — eigvals/eigvecs are not used.
            self.sp_kernel = None
            self._corr_sp_eigvals = None
            self._corr_sp_eigvecs = None

        elif self._model_type == "glmm-null":
            # Spatial variance is always zero in the null model (theta → 0).
            # Use IdentityKernel; extract rank-1 dummy eigenpairs for the
            # Woodbury formula: C = σ²I regardless of eigenvectors.
            self.sp_kernel = IdentityKernel(n_spots)
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
                        UserWarning,
                        stacklevel=2,
                    )
                k = None
            else:
                k = min(int(self._approx_rank), n_spots)

            # Trigger eigendecomposition via SpatialCovKernel (numpy eigh / eigsh).
            _kernel.eigenvalues(k=k)
            # Store the kernel object (for inspection / save-load)
            self.sp_kernel = _kernel
            if k is None:
                self._corr_sp_eigvals = _kernel.K_eigvals  # (n_spots,)
                self._corr_sp_eigvecs = _kernel.K_eigvecs  # (n_spots, n_spots)
            else:
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
                        "(std < 1e-5). Consider removing it.",
                        UserWarning,
                        stacklevel=2,
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
        sp_kernel,
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
        self._coordinates = coordinates
        self.sp_kernel = sp_kernel
        self._corr_sp_eigvals = corr_sp_eigvals
        self._corr_sp_eigvecs = corr_sp_eigvecs
        self.design_mtx = design_mtx
        self.n_factors = design_mtx.shape[1] if design_mtx is not None else 0
        self.covariate_names = covariate_names
        self.adata = None
        self._setup_input_mode = "prebuilt"

    # ------------------------------------------------------------------
    # Lean per-gene state extraction / reconstruction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_gene_state(model) -> _FittedGeneState:
        """Extract a lightweight :class:`_FittedGeneState` from a fitted model."""
        sd = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if k in model._parameters
        }
        lg = getattr(model, "logger", None)
        return _FittedGeneState(
            state_dict=sd,
            convergence=bool(lg.convergence[0].item()) if lg is not None else False,
            best_loss=float(lg.best_loss[0].item()) if lg is not None else float("nan"),
            best_epoch=int(lg.best_epoch[0].item()) if lg is not None else 0,
            fitting_time=float(getattr(model, "fitting_time", 0.0)),
            n_isos=model.n_isos,
        )

    def _get_gene_counts(self, gene_idx: int) -> torch.Tensor:
        """Return counts for gene *gene_idx* as ``(1, n_spots, n_isos)``."""
        return self._dataset.data[gene_idx].unsqueeze(0)

    def _reconstruct_gene_model(self, gene_idx: int, model_key: str = "glmm-full"):
        """Reconstruct a functional per-gene model from stored state."""
        state = self._fitted_states[model_key][gene_idx]
        counts = self._get_gene_counts(gene_idx)

        with_design_mtx = self._model_configs.get("fitting_configs", {}).get(
            "with_design_mtx", False
        )
        design = self.design_mtx if with_design_mtx else None

        if model_key == "glm":
            model = MultinomGLM()
            model.setup_data(counts, design_mtx=design, device=self._device)
        else:
            if model_key == "glmm-full":
                model = IsoFullModel(**self._model_configs)
            elif model_key == "glmm-null":
                model = IsoNullNoSpVar(**self._model_configs)
            else:
                raise ValueError(f"Invalid model key {model_key!r}.")
            model.setup_data(
                counts,
                design_mtx=design,
                corr_sp_eigvals=self._corr_sp_eigvals,
                corr_sp_eigvecs=self._corr_sp_eigvecs,
                device=self._device,
            )

        # Load fitted parameters (strict=False because buffers are not in state_dict)
        model.load_state_dict(state.state_dict, strict=False)
        return model

    def _model_key_for_type(self, model_type: Optional[str] = None) -> str:
        """Map a model_type string to its ``_fitted_states`` key."""
        mt = model_type or self._model_type
        if mt == "glmm-full":
            return "glmm-full"
        elif mt == "glmm-null":
            return "glmm-null"
        elif mt == "glm":
            return "glm"
        raise ValueError(f"Invalid model type {mt!r}.")

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

        iterator = tqdm(
            zip(self.gene_names, iso_groups),
            desc="Genes",
            total=len(self.gene_names),
            disable=not print_progress,
        )

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
        self,
        test_type: Literal["sv", "du"],
        with_gene_summary: bool = False,
    ) -> pd.DataFrame:
        """Get the formatted test results as data frame.

        Parameters
        ----------
        test_type : {"sv", "du"}
            Type of test results to retrieve.
        with_gene_summary : bool, optional
            If ``True``, append gene-level summary statistics from
            :meth:`extract_feature_summary`.

        Returns
        -------
        pandas.DataFrame
            Formatted test results.
        """
        if test_type not in {"sv", "du"}:
            raise ValueError("test_type must be 'sv' or 'du'.")
        if test_type == "sv":
            if len(self._sv_test_results) == 0:
                raise ValueError(
                    "No spatial variability results. Run test_spatial_variability() first."
                )
            df = pd.DataFrame(
                {
                    "gene": self.gene_names,
                    "statistic": self._sv_test_results["statistic"],
                    "pvalue": self._sv_test_results["pvalue"],
                    "pvalue_adj": self._sv_test_results["pvalue_adj"],
                }
            )
        else:
            if len(self._du_test_results) == 0:
                raise ValueError(
                    "No differential usage results. Run test_differential_usage() first."
                )
            covariate_names = self.covariate_names or [
                f"factor_{i}" for i in range(self.n_factors)
            ]
            df = pd.DataFrame(
                {
                    "gene": np.repeat(self.gene_names, self.n_factors),
                    "covariate": np.tile(covariate_names, self.n_genes),
                    "statistic": self._du_test_results["statistic"].reshape(-1),
                    "pvalue": self._du_test_results["pvalue"].reshape(-1),
                    "pvalue_adj": self._du_test_results["pvalue_adj"].reshape(-1),
                }
            )

        if with_gene_summary:
            gene_df = self.extract_feature_summary(level="gene", print_progress=False)
            df = df.merge(gene_df, left_on="gene", right_index=True, how="left")

        return df

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
            Only applicable when ``from_null=True``.
        random_seed : int or None, optional
            The random seed for reproducibility. Default to None.

        See also
        --------
        :func:`splisosm.model.MultinomGLMM.fit` for fitting a single model.
        """

        if batch_size > 1 and not self._group_gene_by_n_iso:
            warnings.warn(
                "Ignoring batch size argument since the dataset is not grouped. "
                "For batch fitting please set 'group_gene_by_n_iso = True' when setup_data().",
                UserWarning,
                stacklevel=2,
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
        self._model_configs["fitting_configs"].update(
            {
                "with_design_mtx": with_design_mtx,
                "from_null": from_null,
                "refit_null": refit_null,
                "batch_size": batch_size,
            }
        )

        self._is_trained = True

    def save(self, path: str) -> None:
        """Save the fitted model to a file.

        Only lightweight per-gene :class:`_FittedGeneState` objects (fitted
        parameters + convergence metadata) are stored -- no full ``nn.Module``
        objects.  The raw ``.adata`` is **not** saved; only
        ``_filtered_adata`` is persisted.  After loading, ``.adata`` will be
        ``None``.

        Parameters
        ----------
        path
            The path to save the fitted models.
        """
        # Don't persist raw adata (only _filtered_adata is kept)
        adata_backup = self.adata
        self.adata = None

        # Don't persist the kernel object (eigenpairs suffice for reconstruction)
        kernel_backup = self.sp_kernel
        if self._corr_sp_eigvals is not None:
            self.sp_kernel = None

        try:
            torch.save(self, path)
        finally:
            self.sp_kernel = kernel_backup
            self.adata = adata_backup

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "SplisosmGLMM":
        """Load a :class:`SplisosmGLMM` model previously saved with :meth:`save`.

        Parameters
        ----------
        path
            Path to the file written by :meth:`save`.
        map_location
            Optional device string (e.g. ``'cpu'``, ``'cuda:0'``) to remap
            tensor storage when loading a model that was saved on a different
            device.  Defaults to ``None`` (preserves original device mapping).

        Returns
        -------
        SplisosmGLMM
            The fully restored model, including fitted parameters, test results,
            and all metadata.

        Examples
        --------
        >>> model.save("model.pt")
        >>> loaded = SplisosmGLMM.load("model.pt")
        >>> loaded.get_training_summary()
        """
        obj = torch.load(path, map_location=map_location, weights_only=False)
        return obj

    def get_fitted_models(self) -> list[Any]:
        """Get the fitted models after running fit().

        Each call reconstructs the full model objects from the stored
        lightweight state.  The returned list is a fresh reconstruction;
        mutations to the returned models are *not* persisted.

        Returns
        -------
        models: list of fitted models
            - ``model_type='glmm-full'``: list[splisosm.hyptest_glmm.IsoFullModel]
            - ``model_type='glmm-null'``: list[splisosm.hyptest_glmm.IsoNullNoSpVar]
            - ``model_type='glm'``: list[splisosm.model.MultinomGLM]
        """
        key = self._model_key_for_type()
        return [
            self._reconstruct_gene_model(i, key)
            for i in range(len(self._fitted_states[key]))
        ]

    def get_gene_model(self, gene_name: str) -> Any:
        """Return the fitted model for a single gene by display name.

        Each call reconstructs a fresh model from stored state.

        Parameters
        ----------
        gene_name
            Display name of the gene (must appear in :attr:`gene_names`).

        Returns
        -------
        Fitted ``MultinomGLM``, ``IsoFullModel``, or ``IsoNullNoSpVar`` instance
        corresponding to ``gene_name``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        KeyError
            If ``gene_name`` is not found in :attr:`gene_names`.
        """
        if not self._is_trained:
            raise RuntimeError("Call fit() first.")
        try:
            idx = self.gene_names.index(gene_name)
        except ValueError:
            preview = self.gene_names[:5]
            raise KeyError(
                f"Gene {gene_name!r} not found in gene_names. "
                f"First 5 available: {preview}"
            ) from None
        key = self._model_key_for_type()
        return self._reconstruct_gene_model(idx, key)

    def __getitem__(self, gene_name: str) -> Any:
        """Shorthand for :meth:`get_gene_model`.

        Example
        -------
        >>> model['GENE_NAME']  # returns the fitted model for GENE_NAME
        """
        return self.get_gene_model(gene_name)

    def get_training_summary(self) -> pd.DataFrame:
        """Return a per-gene training summary as a :class:`pandas.DataFrame`.

        The DataFrame is indexed by gene name and contains the following columns:

        - ``'model_type'``: str. Model type (``'glmm-full'``, ``'glmm-null'``, or ``'glm'``).
        - ``'converged'``: bool. Whether the gene converged during training.
        - ``'best_loss'``: float. Best (lowest) negative log-likelihood achieved.
        - ``'best_epoch'``: int. Epoch at which the best loss was recorded.
        - ``'fitting_time_s'``: float. Wall-clock seconds spent fitting this gene.

        Only available after calling :meth:`fit`.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_trained:
            raise RuntimeError("Call fit() first.")

        key = self._model_key_for_type()
        rows = []
        for gene, state in zip(self.gene_names, self._fitted_states[key]):
            rows.append(
                {
                    "gene": gene,
                    "model_type": self._model_type,
                    "converged": state.convergence,
                    "best_loss": state.best_loss,
                    "best_epoch": state.best_epoch,
                    "fitting_time_s": state.fitting_time,
                }
            )
        return pd.DataFrame(rows).set_index("gene")

    def get_fitted_ratios_anndata(self, layer_name: str = "fitted_ratios") -> AnnData:
        """Return a copy of the filtered AnnData with fitted isoform ratios as a new layer.

        For each gene the per-spot softmax isoform ratios are taken from the
        fitted model (``model.get_isoform_ratio()``) and placed into the
        corresponding isoform columns of a ``(n_spots, n_filtered_isoforms)``
        dense matrix, which is stored under ``layer_name``.

        **Isoform ordering guarantee**: the columns of the new layer follow
        the exact row order of ``self._filtered_adata.var``.  This holds
        because :meth:`setup_data` builds per-gene count tensors by slicing
        ``filtered_adata`` isoforms in their ``var`` row order (via
        ``groupby(sort=False)``), and :meth:`~splisosm.model.MultinomGLM.get_isoform_ratio`
        returns ratios in the same column order as the input counts.

        Parameters
        ----------
        layer_name
            Name of the new layer.  Defaults to ``'fitted_ratios'``.

        Returns
        -------
        AnnData
            A *copy* of ``self._filtered_adata`` (shape
            ``(n_spots, n_filtered_isoforms)``) with ``layer_name`` added.
            The ``var`` DataFrame is identical to ``self._filtered_adata.var``,
            so isoform metadata (e.g. the gene-grouping column) is intact.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called, or if :meth:`setup_data` was
            not called with an :class:`~anndata.AnnData` object (i.e. raw
            tensors were provided instead).

        Notes
        -----
        Requires that the per-gene model parameters (``beta``, ``bias_eta``,
        ``nu``) are still accessible.  If :meth:`free_memory` was previously
        called with ``strip_model_data=True``, models fitted with covariates
        (``n_factors > 0``) will raise an error inside
        ``get_isoform_ratio()`` because ``X_spot`` has been freed.
        """
        if not self._is_trained:
            raise RuntimeError("Call fit() first.")
        if self._filtered_adata is None:
            raise RuntimeError(
                "No filtered AnnData is available. Call setup_data() with an "
                "AnnData object (not raw tensors) before calling "
                "get_fitted_ratios_anndata()."
            )

        ad_out = self._filtered_adata.copy()
        n_fvars = ad_out.n_vars
        ratio_mat = np.full((self.n_spots, n_fvars), np.nan, dtype=np.float32)

        for gene, model in zip(self.gene_names, self.get_fitted_models()):
            # column indices for this gene in _filtered_adata.var
            gene_mask = (ad_out.var[self._group_iso_by] == gene).values
            col_idx = np.where(gene_mask)[0]

            # (1, n_spots, n_isos) → (n_spots, n_isos)
            ratio = model.get_isoform_ratio().detach().cpu().squeeze(0).numpy()
            ratio_mat[:, col_idx] = ratio

        ad_out.layers[layer_name] = ratio_mat

        return ad_out

    def free_memory(
        self, strip_model_data: bool = True, free_kernel: bool = True
    ) -> None:
        """Release large tensors that are no longer needed after fitting.

        Parameters
        ----------
        strip_model_data
            Accepted for backward compatibility but has no effect.
            Per-gene models are no longer stored as full ``nn.Module``
            objects; only lightweight :class:`_FittedGeneState` dataclasses
            are kept after fitting.
        free_kernel
            If ``True``, set ``self.sp_kernel = None``.  The dense
            ``(n_spots × n_spots)`` spatial kernel is only retained when a
            full-rank eigendecomposition was used; eigenpairs
            (``_corr_sp_eigvals``, ``_corr_sp_eigvecs``) are always preserved
            and are sufficient for all downstream operations.
        """
        if free_kernel:
            self.sp_kernel = None

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
                if self._model_type == "glm":
                    model = MultinomGLM()
                    model.setup_data(
                        _g_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        device=self._device,
                    )
                else:
                    if self._model_type == "glmm-full":
                        model = IsoFullModel(**self._model_configs)
                    elif self._model_type == "glmm-null":
                        model = IsoNullNoSpVar(**self._model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self._model_type}.")

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

        # reorder the fitted models by gene names — O(N) dict lookup instead of O(N²) list.index
        _name_to_idx = {name: i for i, name in enumerate(gene_names_ungroupped)}
        _order = [_name_to_idx[g] for g in self.gene_names]
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
            for batch in tqdm(
                data, desc="Fitting", total=n_batches, disable=not print_progress
            ):
                _, b_counts, _ = (batch["n_isos"], batch["x"], batch["gene_name"])

                # initialize and setup the model
                if self._model_type == "glm":
                    model = MultinomGLM()
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        device=self._device,
                    )
                else:
                    if self._model_type == "glmm-full":
                        model = IsoFullModel(**self._model_configs)
                    elif self._model_type == "glmm-null":
                        model = IsoNullNoSpVar(**self._model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self._model_type}.")
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals,
                        corr_sp_eigvecs=self._corr_sp_eigvecs,
                        device=self._device,
                    )

                # fit the model
                model.fit(quiet=quiet, verbose=False, random_seed=random_seed)
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
                    self._model_configs,
                    self._model_type,
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
                tqdm(
                    tasks_gen,
                    desc="Fitting",
                    total=n_batches,
                    disable=not print_progress,
                )
            )

            # convert the fitted parameters to models
            for batch, pars in zip(
                self._dataset.get_dataloader(batch_size=batch_size), fitted_pars
            ):
                # unwrap the batch
                b_counts = batch["x"]

                # initialize and setup the model
                if self._model_type == "glm":
                    model = MultinomGLM()
                    model.setup_data(
                        b_counts,
                        design_mtx=self.design_mtx if with_design_mtx else None,
                        device=self._device,
                    )
                else:
                    if self._model_type == "glmm-full":
                        model = IsoFullModel(**self._model_configs)
                    elif self._model_type == "glmm-null":
                        model = IsoNullNoSpVar(**self._model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self._model_type}.")
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

        # extract lightweight state and discard full model objects
        key = self._model_key_for_type()
        self._fitted_states[key] = [self._extract_gene_state(m) for m in fitted_models]
        del fitted_models

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
            for batch in tqdm(
                data, desc="Fitting", total=n_batches, disable=not print_progress
            ):
                _, b_counts, _ = (batch["n_isos"], batch["x"], batch["gene_name"])

                # fit the null model
                null = IsoNullNoSpVar(**self._model_configs)
                null.setup_data(
                    b_counts,
                    design_mtx=self.design_mtx if with_design_mtx else None,
                    corr_sp_eigvals=self._corr_sp_eigvals,
                    corr_sp_eigvecs=self._corr_sp_eigvecs,
                    device=self._device,
                )
                null.fit(quiet=quiet, verbose=False, random_seed=random_seed)

                # fit the full model from the null
                full = IsoFullModel.from_trained_null_no_sp_var_model(null)
                full.fit(quiet=quiet, verbose=False, random_seed=random_seed)

                # refit the null model if needed
                if refit_null:
                    null_refit = IsoNullNoSpVar.from_trained_full_model(full)
                    null_refit.fit(
                        quiet=quiet,
                        verbose=False,
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
                    self._model_configs,
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
                tqdm(
                    tasks_gen,
                    desc="Fitting",
                    total=n_batches,
                    disable=not print_progress,
                )
            )

            # convert the fitted parameters to models
            for batch, (n_par, f_par) in zip(
                self._dataset.get_dataloader(batch_size=batch_size), fitted_pars
            ):
                # unwrap the batch
                b_counts = batch["x"]

                # null models
                null = IsoNullNoSpVar(**self._model_configs)
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

        # extract lightweight state and discard full model objects
        self._fitted_states["glmm-null"] = [
            self._extract_gene_state(m) for m in fitted_null_models_sv
        ]
        self._fitted_states["glmm-full"] = [
            self._extract_gene_state(m) for m in fitted_full_models
        ]
        del fitted_null_models_sv, fitted_full_models

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
        fitting_configs = self._model_configs["fitting_configs"]

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
            for _ in tqdm(
                range(n_perms),
                desc="Permutations",
                total=n_perms,
                disable=not print_progress,
            ):
                # randomly shuffle the spatial locations
                perm_idx = torch.randperm(self.n_spots)

                # fit a new SplisosmGLMM model using pre-built tensors (fast path)
                new_model = SplisosmGLMM(**self._model_configs, device=self._device)
                new_design_mtx = (
                    self.design_mtx[perm_idx, :]
                    if (self.design_mtx is not None and with_design_mtx)
                    else None
                )
                new_data = [_d[perm_idx, :] for _d in self._dataset.data]
                new_model._setup_from_prebuilt(
                    data=new_data,
                    coordinates=self._coordinates,
                    sp_kernel=self.sp_kernel,
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
                for g_idx in range(new_model.n_genes):
                    full_m = new_model._reconstruct_gene_model(g_idx, "glmm-full")
                    null_m = new_model._reconstruct_gene_model(g_idx, "glmm-null")
                    # use marginal likelihood for stability
                    llr, _ = _calc_llr_spatial_variability(null_m, full_m)
                    _sv_llr_stats.append(llr)

                _sv_llr_stats = torch.tensor(_sv_llr_stats)
                _sv_llr_perm_stats.append(_sv_llr_stats)

            # save the llr statistics from permutated data
            self._sv_llr_perm_stats = torch.concat(_sv_llr_perm_stats, dim=0)

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
                    self._model_configs,
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
                tqdm(
                    tasks_gen,
                    desc="Permutations",
                    total=(n_batches * n_perms),
                    disable=not print_progress,
                )
            )

            self._sv_llr_perm_stats = torch.concat(_sv_llr_perm_stats, dim=0)

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
            If False, the results will be stored in ``self._sv_test_results``.
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
        if len(self._fitted_states["glmm-null"]) == 0:
            raise ValueError(
                "Null models not found. Please run fit() with from_null = True first."
            )

        _sv_llr_stats, _sv_llr_dfs = [], []
        # iterate over genes and calculate the likelihood ratio statistic
        for g_idx in range(self.n_genes):
            full_m = self._reconstruct_gene_model(g_idx, "glmm-full")
            null_m = self._reconstruct_gene_model(g_idx, "glmm-null")
            # use marginal likelihood for stability
            llr, df = _calc_llr_spatial_variability(null_m, full_m)
            _sv_llr_stats.append(llr)
            _sv_llr_dfs.append(df)

        _sv_llr_stats = torch.tensor(_sv_llr_stats)
        _sv_llr_dfs = torch.tensor(_sv_llr_dfs)

        if use_perm_null:
            # use permutation to calculate the p-value.
            if self._sv_llr_perm_stats is None:
                self._fit_sv_llr_perm(
                    n_perms=n_perms_per_gene if n_perms_per_gene is not None else 20,
                    print_progress=print_progress,
                    **kwargs,
                )
            else:  # use the cached results if available
                print("Using cached permutation results...")

            _sv_llr_perm = self._sv_llr_perm_stats
            _sv_llr_pvals = 1 - (_sv_llr_stats[:, None] > _sv_llr_perm[None, :]).sum(
                1
            ) / len(_sv_llr_perm)
        else:
            # calculate the p-value using chi-square distribution
            # move to CPU before scipy (scipy does not support non-CPU tensors)
            _sv_llr_pvals = 1 - chi2.cdf(_sv_llr_stats.cpu(), df=_sv_llr_dfs.cpu())
            _sv_llr_pvals = torch.tensor(_sv_llr_pvals)

        # store the results (always on CPU for downstream pandas/numpy usage)
        self._sv_test_results = {
            "statistic": _sv_llr_stats.cpu().numpy(),
            "pvalue": _sv_llr_pvals.cpu().numpy(),
            "df": _sv_llr_dfs.cpu().numpy(),
            "method": method,
            "use_perm_null": use_perm_null,
        }

        # calculate adjusted p-values
        self._sv_test_results["pvalue_adj"] = false_discovery_control(
            self._sv_test_results["pvalue"]
        )

        # return results
        if return_results:
            return self._sv_test_results

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
            If False, the results are stored in ``self._du_test_results``.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise, returns None and stores results in self._du_test_results.
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
            if self._model_configs["fitting_configs"]["with_design_mtx"]:
                raise ValueError(
                    "Design matrix is included in the fitted models. "
                    "Perhaps you want to use the wald test. Otherwise please run fit() with with_design_mtx = False."
                )

            _du_score_stats, _du_score_dfs = [], []
            # iterate over genes and calculate the score statistic
            for m in tqdm(
                fitted_models,
                desc=f"DU [{method}]",
                total=len(fitted_models),
                disable=not print_progress,
            ):
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
            self._du_test_results = {
                "statistic": _du_score_stats.cpu(),  # (n_genes, n_factors)
                "pvalue": _du_score_pvals.cpu(),  # (n_genes, n_factors)
                "method": method,
            }

        else:  # method == 'wald', Wald test (anti-conservative)
            # extract the fitted full models
            fitted_models = self.get_fitted_models()
            if len(fitted_models) == 0:
                raise ValueError("Fitted full models not found. Run fit() first.")
            if not self._model_configs["fitting_configs"]["with_design_mtx"]:
                raise ValueError(
                    "Design matrix is not included in the fitted models. "
                    "Perhaps you want to use the score test. Otherwise please run fit() with with_design_mtx = True."
                )

            _du_wald_stats, _du_wald_dfs = [], []
            # iterate over genes and calculate the Wald statistic
            for m in tqdm(
                fitted_models,
                desc=f"DU [{method}]",
                total=len(fitted_models),
                disable=not print_progress,
            ):
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
            _du_wald_pvals = 1 - chi2.cdf(_du_wald_stats.cpu(), df=_du_wald_dfs.cpu())
            _du_wald_pvals = torch.tensor(_du_wald_pvals)

            # store the results (always on CPU for downstream pandas/numpy usage)
            self._du_test_results = {
                "statistic": _du_wald_stats.cpu(),  # (n_genes, n_factors)
                "pvalue": _du_wald_pvals.cpu(),  # (n_genes, n_factors)
                "method": method,
            }

        # calculate adjusted p-values (independently for each factor)
        self._du_test_results["pvalue_adj"] = false_discovery_control(
            self._du_test_results["pvalue"], axis=0
        )

        # return the results if needed
        if return_results:
            return self._du_test_results
