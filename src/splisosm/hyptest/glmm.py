"""GLMM-based hypothesis tests for spatial isoform usage."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Optional, Union, Literal

import pandas as pd
import numpy as np
from scipy.stats import chi2
import torch
from anndata import AnnData
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm

from splisosm.hyptest._base import _FeatureSummaryMixin, _ResultsMixin
from splisosm.utils.preprocessing import (
    prepare_inputs_from_anndata,
)
from splisosm.utils.stats import false_discovery_control
from splisosm.glmm.dataset import IsoDataset
from splisosm.glmm.glm import MultinomGLM
from splisosm.kernel import IdentityKernel, SpatialCovKernel
from splisosm.glmm.workers import (
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


class SplisosmGLMM(_ResultsMixin, _FeatureSummaryMixin):
    """Parametric spatial isoform statistical modeling using GLMM.

    This is a convenience class that wraps around the :class:`splisosm.glmm.MultinomGLMM`
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
    ...     # approx_rank: omit for auto | None (force full rank) | int (fixed low-rank)
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
        var_fix_sigma: bool = True,
        var_prior_model: str = "none",
        var_prior_model_params: dict | None = None,
        init_ratio: str = "uniform",
        fitting_method: str = "joint_gd",
        fitting_configs: dict | None = None,
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
        var_fix_sigma
            Whether to fix the total variance (``sigma``) to the Fano-factor
            initial estimate.  When ``True`` (default), only ``theta_logit``
            is learned, producing conservative but well-calibrated hypothesis
            tests.  Set to ``False`` to learn sigma jointly with other
            parameters; this may yield higher power for the SV test but can
            inflate false positive rates for both SV and DU tests.
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
        :class:`splisosm.glmm.MultinomGLMM` for more details on the model configurations.
        """
        # specify the model type to fit
        valid_model_types = ["glmm-full", "glmm-null", "glm"]
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model type. Must be one of {valid_model_types}.")
        self._model_type = model_type

        self._model_configs = {
            "share_variance": share_variance,
            "var_fix_sigma": var_fix_sigma,
            "var_prior_model": var_prior_model,
            "var_prior_model_params": var_prior_model_params,
            "init_ratio": init_ratio,
            "fitting_method": fitting_method,
            "fitting_configs": (
                fitting_configs if fitting_configs is not None else {"max_epochs": 500}
            ),
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

    def _store_anndata_setup(
        self,
        adata: AnnData,
        layer: str,
        group_iso_by: str,
        filtered_adata: AnnData,
    ) -> None:
        """Store AnnData provenance and reset feature-summary caches."""
        self.adata = adata
        self._setup_input_mode = "anndata"
        self._filtered_adata = filtered_adata
        self._counts_layer = layer
        self._group_iso_by = group_iso_by
        self._gene_summary = None
        self._isoform_summary = None

    def _store_dataset(
        self,
        data: list[torch.Tensor],
        gene_names: list[str],
        group_gene_by_n_iso: bool,
        coordinates: Optional[torch.Tensor],
    ) -> None:
        """Build and store the per-gene dataset used by fitting."""
        dataset = IsoDataset(data, gene_names, group_gene_by_n_iso)
        self.n_genes = dataset.n_genes
        self.n_spots = dataset.n_spots
        self.n_isos_per_gene = dataset.n_isos_per_gene
        self.gene_names = dataset.gene_names
        self._dataset = dataset
        self._group_gene_by_n_iso = group_gene_by_n_iso
        self._coordinates = coordinates

    def _resolve_spatial_rank(self, n_spots: int) -> Optional[int]:
        """Resolve the number of spatial eigenvectors retained for GLMM fitting."""
        if self._approx_rank is _APPROX_RANK_AUTO:
            return (
                None
                if n_spots <= SpatialCovKernel.DENSE_THRESHOLD
                else int(np.ceil(np.sqrt(n_spots) * 4))
            )
        if self._approx_rank is None:
            if n_spots > SpatialCovKernel.DENSE_THRESHOLD:
                warnings.warn(
                    f"approx_rank=None forces a full eigendecomposition of a "
                    f"{n_spots}×{n_spots} matrix which may be very slow and "
                    f"memory-intensive.  Consider omitting approx_rank (auto) "
                    f"or passing a positive integer.",
                    UserWarning,
                    stacklevel=2,
                )
            return None
        return min(int(self._approx_rank), n_spots)

    def _setup_spatial_kernel(
        self,
        coordinates: Optional[torch.Tensor],
        adj_matrix: Optional[Any],
    ) -> None:
        """Build spatial kernel/eigenpairs according to the configured model type."""
        n_spots = self.n_spots
        if self._model_type == "glm":
            self.sp_kernel = None
            self._corr_sp_eigvals = None
            self._corr_sp_eigvecs = None
            return

        if self._model_type == "glmm-null":
            self.sp_kernel = IdentityKernel(n_spots)
            self._corr_sp_eigvals = torch.ones(1)
            self._corr_sp_eigvecs = torch.full((n_spots, 1), 1.0 / np.sqrt(n_spots))
            return

        kernel_kwargs = dict(
            rho=self._kernel_rho,
            standardize_cov=self._kernel_standardize_cov,
            centering=False,
        )
        kernel = SpatialCovKernel(
            coords=None if adj_matrix is not None else coordinates,
            adj_matrix=adj_matrix,
            k_neighbors=None if adj_matrix is not None else self._kernel_k_neighbors,
            **kernel_kwargs,
        )
        k = self._resolve_spatial_rank(n_spots)
        kernel.eigenvalues(k=k)
        self.sp_kernel = kernel
        self._corr_sp_eigvals = kernel.K_eigvals if k is None else kernel.K_eigvals[:k]
        self._corr_sp_eigvecs = (
            kernel.K_eigvecs if k is None else kernel.K_eigvecs[:, :k]
        )

    def _store_design_matrix(
        self,
        resolved_design: Optional[Any],
        resolved_cov_names: Optional[list[str]],
    ) -> None:
        """Store dense torch design matrix expected by GLMM fitting."""
        if resolved_design is None:
            self.n_factors = 0
            self.design_mtx = None
            self.covariate_names = None
            return

        import scipy.sparse as _sp

        if _sp.issparse(resolved_design):
            resolved_design = resolved_design.toarray()
        design_mtx_t = torch.from_numpy(np.asarray(resolved_design, dtype=np.float32))
        if design_mtx_t.dim() == 1:
            design_mtx_t = design_mtx_t.unsqueeze(1)
        if design_mtx_t.shape[0] != self.n_spots:
            raise ValueError(
                f"Design matrix row count ({design_mtx_t.shape[0]}) must "
                f"match number of spots ({self.n_spots}) after filtering."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            for idx in torch.where(design_mtx_t.std(dim=0) < 1e-5)[0]:
                cov_name = (
                    resolved_cov_names[int(idx)]
                    if resolved_cov_names is not None
                    else str(idx.item())
                )
                warnings.warn(
                    f"Covariate '{cov_name}' has near-zero variance "
                    "(std < 1e-5). Consider removing it.",
                    UserWarning,
                    stacklevel=2,
                )

        self.n_factors = design_mtx_t.shape[1]
        self.design_mtx = design_mtx_t
        self.covariate_names = resolved_cov_names

    def _move_setup_tensors_to_device(self) -> None:
        """Move spatial eigenpairs and the design matrix onto the configured device."""
        device = torch.device(self._device)
        if self._corr_sp_eigvals is not None:
            self._corr_sp_eigvals = self._corr_sp_eigvals.to(device)
            self._corr_sp_eigvecs = self._corr_sp_eigvecs.to(device)
        if self.design_mtx is not None:
            self.design_mtx = self.design_mtx.to(device)

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
            ``"spatial"``).  Optional when ``adj_key`` is provided: if the key
            is missing from ``adata.obsm`` the spatial kernel is built from
            the adjacency alone.  SplisosmGLMM does not require raw
            coordinates past kernel construction.
        adj_key : str or None, optional
            Key in ``adata.obsp`` for a pre-built adjacency matrix.
            When provided, it overrides the k-NN graph construction
            from coordinates and be used directly to build the spatial kernel.
            Also makes ``spatial_key`` optional (see above).
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
            filter_single_iso_genes=True,  # GLMM requires ≥2 isoforms per gene
            gene_names=gene_names,
            design_mtx=design_mtx,
            covariate_names=covariate_names,
            min_component_size=min_component_size,
            adj_key=adj_key,
            k_neighbors=self._kernel_k_neighbors,
            return_filtered_anndata=True,
        )

        self._store_anndata_setup(adata, layer, group_iso_by, filtered_adata)
        self._store_dataset(data, resolved_gene_names, group_gene_by_n_iso, coordinates)
        self._setup_spatial_kernel(coordinates, adj_matrix)
        self._store_design_matrix(resolved_design, resolved_cov_names)
        self._move_setup_tensors_to_device()

    def _setup_from_prebuilt(
        self,
        data: list,
        coordinates: Optional[torch.Tensor],
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
        if self._dataset is None:
            raise RuntimeError(
                "Count data is not available. Call setup_data() first, or re-attach "
                "the dataset after loading a saved model."
            )
        return self._dataset.data[gene_idx].unsqueeze(0)

    def _reconstruct_gene_model(self, gene_idx: int, model_key: str = "glmm-full"):
        """Reconstruct a functional per-gene model from stored state."""
        state = self._fitted_states[model_key][gene_idx]
        counts = self._get_gene_counts(gene_idx)

        with_design_mtx = self._model_configs.get("fitting_configs", {}).get(
            "with_design_mtx", False
        )
        model = self._new_model_for_counts(
            counts,
            model_type=model_key,
            design_mtx=self._design_for_fit(with_design_mtx),
        )
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

    def _resolve_fit_n_jobs(self, n_jobs: int) -> int:
        """Resolve CPU worker count and enforce non-CPU serial fitting."""
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_jobs > 1 and self._device != "cpu":
            warnings.warn(
                f"Parallel fitting (n_jobs={n_jobs}) is not supported for "
                f"device={self._device!r}. Falling back to n_jobs=1.",
                UserWarning,
                stacklevel=2,
            )
            return 1
        return n_jobs

    def _design_for_fit(self, with_design_mtx: bool) -> Optional[torch.Tensor]:
        """Return the design matrix for fitting when requested."""
        return self.design_mtx if with_design_mtx else None

    def _batch_loader(self, batch_size: int):
        """Return ``(n_batches, dataloader)`` for a fitting pass."""
        n_batches = sum(1 for _ in self._dataset.get_dataloader(batch_size=batch_size))
        return n_batches, self._dataset.get_dataloader(batch_size=batch_size)

    def _new_model_for_counts(
        self,
        counts: torch.Tensor,
        *,
        model_type: Optional[str] = None,
        design_mtx: Optional[torch.Tensor] = None,
    ):
        """Construct and setup a GLM/GLMM for one batch of gene counts."""
        model_type = model_type or self._model_type
        if model_type == "glm":
            model = MultinomGLM()
            model.setup_data(counts, design_mtx=design_mtx, device=self._device)
            return model
        if model_type == "glmm-full":
            model = IsoFullModel(**self._model_configs)
        elif model_type == "glmm-null":
            model = IsoNullNoSpVar(**self._model_configs)
        else:
            raise ValueError(f"Invalid model type {model_type}.")
        model.setup_data(
            counts,
            design_mtx=design_mtx,
            corr_sp_eigvals=self._corr_sp_eigvals,
            corr_sp_eigvecs=self._corr_sp_eigvecs,
            device=self._device,
        )
        return model

    def _print_fit_start(
        self,
        *,
        label: str,
        n_jobs: int,
        batch_size: int,
        print_progress: bool,
    ) -> None:
        """Print standard GLMM fitting progress preamble."""
        if not print_progress:
            return
        if n_jobs == 1:
            print(
                f"{label} with single core for {self.n_genes} genes "
                f"(batch_size={batch_size})."
            )
        else:
            print(
                f"{label} with {n_jobs} cores for {self.n_genes} genes "
                f"(batch_size={batch_size})."
            )
            print(
                "Note: the progress bar is updated before each fitting, rather than when it finishes."
            )

    def _fit_single_batch_model(
        self,
        counts: torch.Tensor,
        *,
        with_design_mtx: bool,
        quiet: bool,
        random_seed: Optional[int],
    ):
        """Fit the configured model for one batch in-process."""
        model = self._new_model_for_counts(
            counts, design_mtx=self._design_for_fit(with_design_mtx)
        )
        model.fit(quiet=quiet, verbose=False, random_seed=random_seed)
        return model

    def _rebuild_batch_model_from_params(
        self,
        counts: torch.Tensor,
        pars: dict[str, torch.Tensor],
        *,
        with_design_mtx: bool,
    ):
        """Reconstruct a fitted batch model from parameters returned by a worker."""
        model = self._new_model_for_counts(
            counts, design_mtx=self._design_for_fit(with_design_mtx)
        )
        model.update_params_from_dict(pars)
        return model

    def _fit_null_full_batch(
        self,
        counts: torch.Tensor,
        *,
        refit_null: bool,
        with_design_mtx: bool,
        quiet: bool,
        random_seed: Optional[int],
    ) -> tuple[Any, Any]:
        """Fit one null/full SV model pair in-process."""
        null = self._new_model_for_counts(
            counts,
            model_type="glmm-null",
            design_mtx=self._design_for_fit(with_design_mtx),
        )
        null.fit(quiet=quiet, verbose=False, random_seed=random_seed)
        full = IsoFullModel.from_trained_null_no_sp_var_model(null)
        full.fit(quiet=quiet, verbose=False, random_seed=random_seed)

        if refit_null:
            null_refit = IsoNullNoSpVar.from_trained_full_model(full)
            null_refit.fit(quiet=quiet, verbose=False, random_seed=random_seed)
            if null_refit().mean() > null().mean():
                null = null_refit
            if null().mean() > full().mean():
                full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
                full_refit.fit(quiet=quiet, verbose=False, random_seed=random_seed)
                if full_refit().mean() > full().mean():
                    full = full_refit
        return null, full

    def _rebuild_null_full_from_params(
        self,
        counts: torch.Tensor,
        null_pars: dict[str, torch.Tensor],
        full_pars: dict[str, torch.Tensor],
        *,
        with_design_mtx: bool,
    ) -> tuple[Any, Any]:
        """Reconstruct fitted null/full batch models from worker parameters."""
        null = self._new_model_for_counts(
            counts,
            model_type="glmm-null",
            design_mtx=self._design_for_fit(with_design_mtx),
        )
        null.update_params_from_dict(null_pars)
        full = IsoFullModel.from_trained_null_no_sp_var_model(null)
        full.update_params_from_dict(full_pars)
        return null, full

    def _store_fitted_states(
        self,
        model_key: str,
        fitted_models: list[Any],
        *,
        batch_size: int,
        with_design_mtx: bool,
    ) -> None:
        """Ungroup fitted batch models if needed and store lightweight states."""
        if batch_size > 1:
            fitted_models = self._ungroup_fitted_models(
                fitted_models, batch_size, with_design_mtx
            )
        self._fitted_states[model_key] = [
            self._extract_gene_state(model) for model in fitted_models
        ]

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
        :func:`splisosm.glmm.MultinomGLMM.fit` for fitting a single model.
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
            - ``model_type='glmm-full'``: list[splisosm.hyptest.glmm.IsoFullModel]
            - ``model_type='glmm-null'``: list[splisosm.hyptest.glmm.IsoNullNoSpVar]
            - ``model_type='glm'``: list[splisosm.glmm.MultinomGLM]
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
        ``groupby(sort=False)``), and :meth:`~splisosm.glmm.MultinomGLM.get_isoform_ratio`
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
            if b_n_isos[0] != grouped_m.n_isos:
                raise RuntimeError(
                    "Fitted model isoform count does not match the grouped batch."
                )

            # add the gene names to the list
            gene_names_ungroupped.extend(b_gene_names)

            # extract batched parameters
            return_par_names = [
                "nu",
                "beta",
                "bias_eta",
                "sigma",
                "theta_logit",
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
                model = self._new_model_for_counts(
                    _g_counts, design_mtx=self._design_for_fit(with_design_mtx)
                )
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
        fitted_models = []
        n_jobs = self._resolve_fit_n_jobs(n_jobs)
        t_start = timer()
        n_batches, data = self._batch_loader(batch_size)
        self._print_fit_start(
            label="Fitting",
            n_jobs=n_jobs,
            batch_size=batch_size,
            print_progress=print_progress,
        )

        if n_jobs == 1:  # use single core
            for batch in tqdm(
                data, desc="Fitting", total=n_batches, disable=not print_progress
            ):
                fitted_models.append(
                    self._fit_single_batch_model(
                        batch["x"],
                        with_design_mtx=with_design_mtx,
                        quiet=quiet,
                        random_seed=random_seed,
                    )
                )
        else:
            tasks = (
                delayed(_fit_model_one_gene)(
                    self._model_configs,
                    self._model_type,
                    batch["x"],
                    self._corr_sp_eigvals,
                    self._corr_sp_eigvecs,
                    self._design_for_fit(with_design_mtx),
                    quiet,
                    random_seed,
                    self._device,
                )
                for batch in data
            )
            fitted_pars = Parallel(n_jobs=n_jobs)(
                tqdm(
                    tasks,
                    desc="Fitting",
                    total=n_batches,
                    disable=not print_progress,
                )
            )
            for batch, pars in zip(
                self._dataset.get_dataloader(batch_size=batch_size), fitted_pars
            ):
                fitted_models.append(
                    self._rebuild_batch_model_from_params(
                        batch["x"], pars, with_design_mtx=with_design_mtx
                    )
                )

        self._store_fitted_states(
            self._model_key_for_type(),
            fitted_models,
            batch_size=batch_size,
            with_design_mtx=with_design_mtx,
        )
        del fitted_models

        if print_progress:
            print(f"Fitting finished. Time elapsed: {timer() - t_start:.2f} seconds.")

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
        fitted_null_models_sv = []
        fitted_full_models = []
        n_jobs = self._resolve_fit_n_jobs(n_jobs)
        t_start = timer()
        n_batches, data = self._batch_loader(batch_size)
        self._print_fit_start(
            label="Fitting",
            n_jobs=n_jobs,
            batch_size=batch_size,
            print_progress=print_progress,
        )

        if n_jobs == 1:
            for batch in tqdm(
                data, desc="Fitting", total=n_batches, disable=not print_progress
            ):
                null, full = self._fit_null_full_batch(
                    batch["x"],
                    refit_null=refit_null,
                    with_design_mtx=with_design_mtx,
                    quiet=quiet,
                    random_seed=random_seed,
                )
                fitted_null_models_sv.append(null)
                fitted_full_models.append(full)
        else:
            tasks = (
                delayed(_fit_null_full_sv_one_gene)(
                    self._model_configs,
                    batch["x"],
                    self._corr_sp_eigvals,
                    self._corr_sp_eigvecs,
                    self._design_for_fit(with_design_mtx),
                    refit_null,
                    quiet,
                    random_seed,
                    self._device,
                )
                for batch in data
            )
            fitted_pars = Parallel(n_jobs=n_jobs)(
                tqdm(
                    tasks,
                    desc="Fitting",
                    total=n_batches,
                    disable=not print_progress,
                )
            )
            for batch, (null_pars, full_pars) in zip(
                self._dataset.get_dataloader(batch_size=batch_size), fitted_pars
            ):
                null, full = self._rebuild_null_full_from_params(
                    batch["x"],
                    null_pars,
                    full_pars,
                    with_design_mtx=with_design_mtx,
                )
                fitted_null_models_sv.append(null)
                fitted_full_models.append(full)

        self._store_fitted_states(
            "glmm-null",
            fitted_null_models_sv,
            batch_size=batch_size,
            with_design_mtx=with_design_mtx,
        )
        self._store_fitted_states(
            "glmm-full",
            fitted_full_models,
            batch_size=batch_size,
            with_design_mtx=with_design_mtx,
        )
        del fitted_null_models_sv, fitted_full_models

        if print_progress:
            print(f"Fitting finished. Time elapsed: {timer() - t_start:.2f} seconds.")

    def _permutation_fit_configs(self) -> tuple[bool, bool, int]:
        """Return fit settings required by the SV permutation path."""
        fitting_configs = self._model_configs["fitting_configs"]
        try:
            return (
                fitting_configs["with_design_mtx"],
                fitting_configs["refit_null"],
                fitting_configs["batch_size"],
            )
        except KeyError:
            raise ValueError(
                "Null models not found. Please run fit() with from_null = True first."
            ) from None

    def _setup_permuted_model(
        self,
        perm_idx: torch.Tensor,
        *,
        with_design_mtx: bool,
    ) -> "SplisosmGLMM":
        """Create a fast-path GLMM wrapper with permuted spot order."""
        new_design_mtx = (
            self.design_mtx[perm_idx, :]
            if (self.design_mtx is not None and with_design_mtx)
            else None
        )
        new_model = SplisosmGLMM(**self._model_configs, device=self._device)
        new_model._setup_from_prebuilt(
            data=[data[perm_idx, :] for data in self._dataset.data],
            coordinates=self._coordinates,
            sp_kernel=self.sp_kernel,
            corr_sp_eigvals=self._corr_sp_eigvals,
            corr_sp_eigvecs=self._corr_sp_eigvecs,
            design_mtx=new_design_mtx,
            gene_names=self.gene_names,
            group_gene_by_n_iso=self._group_gene_by_n_iso,
            covariate_names=self.covariate_names,
        )
        return new_model

    @staticmethod
    def _llr_stats_from_fitted_model(model: "SplisosmGLMM") -> torch.Tensor:
        """Calculate per-gene marginal LLR statistics from fitted null/full states."""
        stats = []
        for gene_idx in range(model.n_genes):
            full_m = model._reconstruct_gene_model(gene_idx, "glmm-full")
            null_m = model._reconstruct_gene_model(gene_idx, "glmm-null")
            llr, _ = _calc_llr_spatial_variability(null_m, full_m)
            stats.append(llr)
        return torch.tensor(stats)

    def _run_one_sv_permutation(
        self,
        *,
        with_design_mtx: bool,
        refit_null: bool,
        batch_size: int,
        random_seed: Optional[int],
    ) -> torch.Tensor:
        """Fit one permuted model and return its per-gene LLR statistics."""
        new_model = self._setup_permuted_model(
            torch.randperm(self.n_spots), with_design_mtx=with_design_mtx
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
        return self._llr_stats_from_fitted_model(new_model)

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
        if n_perms < 1:
            raise ValueError("`n_perms` must be a positive integer.")

        with_design_mtx, refit_null, batch_size = self._permutation_fit_configs()

        if random_seed is not None:  # set random seed for reproducibility
            torch.manual_seed(random_seed)

        perm_stats = []
        n_jobs = self._resolve_fit_n_jobs(n_jobs)
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
                perm_stats.append(
                    self._run_one_sv_permutation(
                        with_design_mtx=with_design_mtx,
                        refit_null=refit_null,
                        batch_size=batch_size,
                        random_seed=random_seed,
                    )
                )

            self._sv_llr_perm_stats = torch.stack(perm_stats, dim=0)

        else:  # use multiprocessing
            if print_progress:
                print(
                    f"Running permutation with {n_jobs} cores for {self.n_genes} genes "
                    f"(n_perms={n_perms}, batch_size={batch_size})."
                )
                print(
                    "Note: the progress bar is updated before each fitting, rather than when it finishes."
                )

            n_batches, data = self._batch_loader(batch_size)
            name_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
            batch_infos = [
                (
                    [name_to_idx[name] for name in batch["gene_name"]],
                    batch["x"],
                )
                for batch in data
            ]
            task_meta = []
            tasks = []
            for perm_idx in range(n_perms):
                perm = torch.randperm(self.n_spots)
                for gene_indices, counts in batch_infos:
                    task_meta.append((perm_idx, gene_indices))
                    tasks.append(
                        delayed(_fit_perm_one_gene)(
                            perm,
                            self._model_configs,
                            counts,
                            self._corr_sp_eigvals,
                            self._corr_sp_eigvecs,
                            self._design_for_fit(with_design_mtx),
                            refit_null=refit_null,
                            random_seed=random_seed,
                            device=self._device,
                        )
                    )
            perm_stats = Parallel(n_jobs=n_jobs)(
                tqdm(
                    tasks,
                    desc="Permutations",
                    total=(n_batches * n_perms),
                    disable=not print_progress,
                )
            )
            perm_matrix = torch.empty((n_perms, self.n_genes), dtype=torch.float32)
            for (perm_idx, gene_indices), batch_stats in zip(task_meta, perm_stats):
                perm_matrix[perm_idx, gene_indices] = batch_stats.detach().cpu().float()
            self._sv_llr_perm_stats = perm_matrix

        if print_progress:
            print(f"Fitting finished. Time elapsed: {timer() - t_start:.2f} seconds.")

    def _validate_sv_test(self, method: str) -> None:
        """Validate spatial variability test state."""
        valid_methods = ["llr"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        if len(self._fitted_states["glmm-null"]) == 0:
            raise ValueError(
                "Null models not found. Please run fit() with from_null = True first."
            )

    def _calc_sv_llr_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate per-gene LLR statistics and degrees of freedom."""
        stats, dfs = [], []
        for gene_idx in range(self.n_genes):
            if self.n_isos_per_gene[gene_idx] <= 1:
                stats.append(torch.tensor(0.0))
                dfs.append(torch.tensor(1))
                continue
            full_m = self._reconstruct_gene_model(gene_idx, "glmm-full")
            null_m = self._reconstruct_gene_model(gene_idx, "glmm-null")
            llr, df = _calc_llr_spatial_variability(null_m, full_m)
            stats.append(llr)
            dfs.append(df)
        return torch.tensor(stats), torch.tensor(dfs)

    def _sv_llr_pvalues(
        self,
        stats: torch.Tensor,
        dfs: torch.Tensor,
        *,
        use_perm_null: bool,
        n_perms_per_gene: Optional[int],
        print_progress: bool,
        perm_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Return SV LLR p-values from chi-square or permutation null."""
        if use_perm_null:
            if self._sv_llr_perm_stats is None:
                self._fit_sv_llr_perm(
                    n_perms=n_perms_per_gene if n_perms_per_gene is not None else 20,
                    print_progress=print_progress,
                    **perm_kwargs,
                )
            else:
                print("Using cached permutation results...")
            perm = self._sv_llr_perm_stats
            if perm.ndim == 1:
                if perm.numel() % self.n_genes != 0:
                    raise RuntimeError(
                        "Cached permutation statistics must have one column per gene."
                    )
                perm = perm.reshape(-1, self.n_genes)
            if perm.shape[1] != self.n_genes:
                raise RuntimeError(
                    "Cached permutation statistics must have shape "
                    "(n_perms, n_genes)."
                )
            perm = perm.to(dtype=stats.dtype, device=stats.device)
            pvals = (1 + (perm >= stats[None, :]).sum(dim=0)) / (perm.shape[0] + 1)
        else:
            pvals = torch.tensor(1 - chi2.cdf(stats.cpu(), df=dfs.cpu()))

        single_iso_mask = torch.tensor(
            [n <= 1 for n in self.n_isos_per_gene], dtype=torch.bool
        )
        pvals[single_iso_mask] = 1.0
        return pvals

    def _store_sv_results(
        self,
        stats: torch.Tensor,
        pvals: torch.Tensor,
        dfs: torch.Tensor,
        *,
        method: str,
        use_perm_null: bool,
    ) -> None:
        """Store formatted SV results."""
        self._sv_test_results = {
            "statistic": stats.cpu().numpy(),
            "pvalue": pvals.cpu().numpy(),
            "df": dfs.cpu().numpy(),
            "method": method,
            "use_perm_null": use_perm_null,
        }
        self._sv_test_results["pvalue_adj"] = false_discovery_control(
            self._sv_test_results["pvalue"]
        )

    def _validate_du_test(self, method: str) -> int:
        """Validate differential-usage test state and return n_factors."""
        if self.design_mtx is None:
            raise ValueError("No design matrix is provided. Run setup_data() first.")
        valid_methods = ["wald", "score"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}.")
        fitted_with_design = self._model_configs["fitting_configs"].get(
            "with_design_mtx", False
        )
        if method == "score" and fitted_with_design:
            raise ValueError(
                "Design matrix is included in the fitted models. "
                "Perhaps you want to use the wald test. Otherwise please run fit() with with_design_mtx = False."
            )
        if method == "wald" and not fitted_with_design:
            raise ValueError(
                "Design matrix is not included in the fitted models. "
                "Perhaps you want to use the score test. Otherwise please run fit() with with_design_mtx = True."
            )
        return self.design_mtx.shape[1]

    def _du_test_fn(self, method: str):
        """Return the DU statistic function for the selected method."""
        return (
            (lambda model: _calc_score_differential_usage(model, self.design_mtx))
            if method == "score"
            else _calc_wald_differential_usage
        )

    def _calc_du_stats(
        self,
        *,
        method: str,
        n_factors: int,
        print_progress: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate DU statistics and degrees of freedom for all genes."""
        fitted_models = self.get_fitted_models()
        if len(fitted_models) == 0:
            suffix = (
                "Run fit() first."
                if method == "wald"
                else ("Run fit(..., with_design_mtx = False) first.")
            )
            raise ValueError(f"Fitted full models not found. {suffix}")

        stat_fn = self._du_test_fn(method)
        stats, dfs = [], []
        for gene_idx, model in enumerate(
            tqdm(
                fitted_models,
                desc=f"DU [{method}]",
                total=len(fitted_models),
                disable=not print_progress,
            )
        ):
            if self.n_isos_per_gene[gene_idx] <= 1:
                stats.append(torch.zeros(n_factors))
                dfs.append(0)
                continue
            stat, df = stat_fn(model)
            stats.append(stat)
            dfs.append(df)

        stats = torch.stack(stats, dim=0).reshape(-1, n_factors)
        dfs = torch.tensor(dfs).unsqueeze(-1).expand(-1, n_factors)
        return stats, dfs

    def _du_pvalues_from_chi2(
        self, stats: torch.Tensor, dfs: torch.Tensor
    ) -> torch.Tensor:
        """Return DU chi-square p-values with single-isoform genes set to one."""
        pvals = torch.tensor(1 - chi2.cdf(stats.cpu(), df=dfs.cpu()))
        pvals[dfs[:, 0] == 0] = 1.0
        return pvals

    def _store_du_results(
        self,
        stats: torch.Tensor,
        pvals: torch.Tensor,
        *,
        method: str,
    ) -> None:
        """Store formatted DU results."""
        self._du_test_results = {
            "statistic": stats.cpu(),
            "pvalue": pvals.cpu(),
            "method": method,
        }
        self._du_test_results["pvalue_adj"] = false_discovery_control(
            self._du_test_results["pvalue"], axis=0
        )

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
            We recommend the non-parametric HSIC-based tests in :class:`splisosm.hyptest.np.SplisosmNP`
            for spatial variability testing.

        Note that the parametric and non-parametric tests are assymptotically equivalent under the null.
        See :cite:`su2026consistent` for detailed theoretical analysis.

        Parameters
        ----------
        method : {"llr"}, optional
            The test method.
            Currently only support ``"llr"``, the likelihood ratio test (H_0: ``theta`` = 0, i.e. no spatial variance).
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
        :func:`splisosm.hyptest.np.SplisosmNP.test_spatial_variability` for non-parametric tests.
        """

        self._validate_sv_test(method)
        stats, dfs = self._calc_sv_llr_stats()
        pvals = self._sv_llr_pvalues(
            stats,
            dfs,
            use_perm_null=use_perm_null,
            n_perms_per_gene=n_perms_per_gene,
            print_progress=print_progress,
            perm_kwargs=kwargs,
        )
        self._store_sv_results(
            stats,
            pvals,
            dfs,
            method=method,
            use_perm_null=use_perm_null,
        )
        if return_results:
            return self._sv_test_results
        return None

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

        Similar to :func:`splisosm.hyptest.np.SplisosmNP.test_differential_usage`, here we also support two types of association tests but **implicitly**:

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
        n_factors = self._validate_du_test(method)
        stats, dfs = self._calc_du_stats(
            method=method,
            n_factors=n_factors,
            print_progress=print_progress,
        )
        self._store_du_results(
            stats,
            self._du_pvalues_from_chi2(stats, dfs),
            method=method,
        )
        if return_results:
            return self._du_test_results
        return None
