"""Spatial Gaussian-process residualization backends for SPLISOSM.

This module provides an abstract interface and concrete implementations for
GP residualization used by conditional differential-usage tests.

GPR-residualizer hierarchy
---------------------------
``KernelGPR`` (abstract):

    - ``SklearnKernelGPR``  - dense sklearn backend for small to moderate data.
    - ``GPyTorchKernelGPR`` - GPyTorch backend (optional dep); lazy tensors.
    - ``FFTKernelGPR``      - FFT-based kernel; suitable for regular grids.
    - ``NUFFTKernelGPR``    - FINUFFT-based kernel; suitable for irregular 2-D coordinates.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import scipy.fft
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from splisosm._gpr.operators import (
    DenseKernelOp,
    FFTKernelOp,
    NUFFTKernelOp,
    SpatialKernelOp,
    _as_numpy_coords_2d,
    _regular_grid_shape_spacing,
    _require_finufft,
)
from splisosm._gpr.statistics import (
    _build_rbf_cross_kernel,
    _build_rbf_kernel,
    _kernel_residuals_from_eigdecomp,
    linear_hsic_test,
)

__all__ = [
    # # Kernel linear operators
    # "SpatialKernelOp",
    # "DenseKernelOp",
    # "FFTKernelOp",
    # "NUFFTKernelOp",
    # GPR residualizers
    "KernelGPR",
    "SklearnKernelGPR",
    "GPyTorchKernelGPR",
    "FFTKernelGPR",
    "NUFFTKernelGPR",
    "make_kernel_gpr",
    # HSIC utilities
    "linear_hsic_test",
]


# ---------------------------------------------------------------------------
# Abstract KernelGPR base class
# ---------------------------------------------------------------------------


class KernelGPR(ABC):
    """Abstract GP residualizer.

    Subclasses determine how hyperparameters are optimized (MLE, variational,
    fixed) and whether the kernel matrix is materialized (dense) or kept
    implicit.

    All subclasses expose the same ``fit_residuals`` interface so that the
    DU-test pipeline in ``SplisosmNP`` and ``SplisosmFFT`` is backend-agnostic.
    """

    @abstractmethod
    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit a spatial GP and return residuals.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Spatial coordinates, pre-normalized by
            the caller.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        residuals : torch.Tensor
            Shape ``(n_spots, m)``. Residuals
            ``Y - GP_smooth(Y | coords)``. NaN rows in ``Y`` are preserved
            as NaN.
        """

    def fit_residuals_batch(
        self,
        coords: torch.Tensor,
        Y_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Fit residuals for many targets sharing the same coordinates.

        The default implementation calls :meth:`fit_residuals` sequentially.
        Subclasses may override this to amortize work across targets
        (e.g. ``SklearnKernelGPR`` precomputes a shared eigendecomposition
        when signal bounds are fixed).

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape ``(n_spots, m_i)``.

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape ``(n_spots, m_i)``.
        """
        return [self.fit_residuals(coords, Y) for Y in Y_list]

    def get_kernel_op(self, coords: torch.Tensor) -> SpatialKernelOp:
        """Build the kernel operator for given coordinates.

        Returns a :class:`SpatialKernelOp` whose hyperparameters have been set
        (but not yet optimized for a specific target Y).  Useful when the
        caller needs the operator for its own purposes (e.g. computing
        eigenvalues for the HSIC null distribution).

        Not all backends support this method.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Spatial coordinates.

        Returns
        -------
        SpatialKernelOp
            Kernel operator for the given coordinates.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose an explicit SpatialKernelOp."
        )


# ---------------------------------------------------------------------------
# sklearn backend
# ---------------------------------------------------------------------------

_DEFAULT_GP_PARAMS = {
    "constant_value": 1.0,
    "constant_value_bounds": (1e-3, 1e3),
    "length_scale": 1.0,
    "length_scale_bounds": "fixed",
}

_DEFAULT_GPR_SCALE_PARAMS = {
    # sklearn: subset-of-data size for hyperparameter fitting.
    # gpytorch: FITC inducing-point count.
    "n_inducing": 5_000,
}

_DEFAULT_NUFFT_PARAMS = {
    "epsilon_bounds": (1e-5, 1e1),
    "n_modes": None,
    "max_auto_modes": None,
    "nufft_eps": 1e-6,
    "nufft_opts": None,
    "lml_approx_rank": 256,
    "lml_exact_max_n": 512,
    "eigsh_tol": 1e-4,
    "period_margin": 0.5,
    "cg_rtol": 1e-5,
    "cg_maxiter": None,
    "workers": None,
}

_DEFAULT_GPR_CONFIGS = {
    # Covariate GPR: optimize signal amplitude via MLE; length scale fixed
    # at 1.0 because coordinates are z-score normalized before fitting.
    "covariate": {
        **_DEFAULT_GP_PARAMS,
        **_DEFAULT_GPR_SCALE_PARAMS,
        **_DEFAULT_NUFFT_PARAMS,
    },
    # Isoform GPR: used only when residualize='both'.
    "isoform": {
        **_DEFAULT_GP_PARAMS,
        **_DEFAULT_GPR_SCALE_PARAMS,
        **_DEFAULT_NUFFT_PARAMS,
    },
}

_COMMON_GPR_KWARGS = set(_DEFAULT_GP_PARAMS)
_SKLEARN_GPR_KWARGS = _COMMON_GPR_KWARGS | {"n_inducing"}
_GPYTORCH_GPR_KWARGS = _COMMON_GPR_KWARGS | {"n_inducing", "n_iter", "lr", "device"}
_FFT_GPR_KWARGS = _COMMON_GPR_KWARGS | {"epsilon_bounds", "workers"}
_NUFFT_GPR_KWARGS = _COMMON_GPR_KWARGS | set(_DEFAULT_NUFFT_PARAMS)
_KNOWN_GPR_KWARGS = (
    _SKLEARN_GPR_KWARGS | _GPYTORCH_GPR_KWARGS | _FFT_GPR_KWARGS | _NUFFT_GPR_KWARGS
)


class SklearnKernelGPR(KernelGPR):
    """Dense sklearn GP residualizer.

    Uses ``GaussianProcessRegressor`` with a ``ConstantKernel x RBF +
    WhiteKernel`` to learn spatial hyperparameters and residualize targets.

    When both signal bounds are ``"fixed"`` (:attr:`signal_bounds_fixed`),
    the base kernel matrix is the same for every target sharing the same
    coordinates.  Calling :meth:`precompute_shared_kernel` once before a
    loop over many targets reduces per-target cost from ``O(n_spots^3)`` GP
    fitting to ``O(n_spots^2)`` spectral operations.

    Parameters
    ----------
    constant_value : float
        Initial signal amplitude.
    constant_value_bounds : tuple or ``"fixed"``
        Search bounds for the signal amplitude.  ``"fixed"`` disables
        optimization of this parameter.
    length_scale : float
        Initial RBF length scale.
    length_scale_bounds : tuple or ``"fixed"``
        Search bounds for the length scale.
    n_inducing : int or None
        Maximum number of observations used for hyperparameter fitting.
        When ``n_spots <= n_inducing`` (or ``n_inducing`` is ``None``), all
        observations are used for an exact GP fit. When
        ``n_spots > n_inducing``,
        a randomly sampled **subset-of-data** of ``n_inducing`` points is used
        for hyperparameter search, then the fitted kernel is applied to predict
        on all observations.  This is **not** an inducing-point (FITC/VFE)
        approximation; the subset is only used to determine hyperparameters.
        Setting ``n_inducing=None`` disables the subset shortcut and always
        uses the full dataset; a warning is issued when ``n_spots > 10_000``.

    Notes
    -----
    This backend always materializes a dense ``n_spots x n_spots`` kernel
    matrix (up to
    ``n_inducing`` x ``n_inducing`` when the subset path is taken).  For
    large data, use ``NUFFTKernelGPR`` for irregular 2-D coordinates,
    ``FFTKernelGPR`` for regular raster grids, or ``GPyTorchKernelGPR`` with
    FITC inducing points.
    """

    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
        length_scale: float = 1.0,
        length_scale_bounds: Union[tuple, str] = "fixed",
        n_inducing: Optional[int] = 5_000,
    ) -> None:
        self._constant_value = constant_value
        self._constant_value_bounds = constant_value_bounds
        self._length_scale = length_scale
        self._length_scale_bounds = length_scale_bounds
        self.n_inducing = n_inducing

        # Cached eigendecomposition of the shared kernel (when bounds fixed)
        self._shared_eigvecs: Optional[torch.Tensor] = None
        self._shared_eigvals: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SklearnKernelGPR":
        """Construct from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dict with optional keys: ``constant_value``,
            ``constant_value_bounds``, ``length_scale``, ``length_scale_bounds``,
            ``n_inducing``.

        Returns
        -------
        SklearnKernelGPR
            New instance constructed from the configuration.
        """
        return cls(
            constant_value=config.get("constant_value", 1.0),
            constant_value_bounds=config.get("constant_value_bounds", (1e-3, 1e3)),
            length_scale=config.get("length_scale", 1.0),
            length_scale_bounds=config.get("length_scale_bounds", "fixed"),
            n_inducing=config.get("n_inducing", 5_000),
        )

    @property
    def signal_bounds_fixed(self) -> bool:
        """True when both constant_value and length_scale bounds are fixed."""
        return (
            self._constant_value_bounds == "fixed"
            and self._length_scale_bounds == "fixed"
        )

    def precompute_shared_kernel(self, coords: torch.Tensor) -> None:
        """Precompute and cache the eigendecomposition of the fixed-param kernel.

        Call this once before a loop over many targets that share the same
        coordinates and have ``signal_bounds_fixed == True``.  Subsequent calls
        to :meth:`fit_residuals` will use the cached decomposition and run in
        ``O(n_spots^2)`` rather than ``O(n_spots^3)``.

        Has no effect (and logs a warning) when ``signal_bounds_fixed`` is
        False.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Normalized spatial coordinates.

        Returns
        -------
        None
        """
        if self.n_inducing is not None and coords.shape[0] > self.n_inducing:
            warnings.warn(
                f"n={coords.shape[0]} > n_inducing={self.n_inducing}; "
                "large-n subset-of-data path will be used; precompute_shared_kernel skipped.",
                UserWarning,
                stacklevel=2,
            )
            return
        if not self.signal_bounds_fixed:
            warnings.warn(
                "precompute_shared_kernel() has no effect when signal bounds "
                "are not fixed; the full GP will still be fitted per target.",
                UserWarning,
                stacklevel=2,
            )
            return
        K = _build_rbf_kernel(coords, self._constant_value, self._length_scale)
        K = 0.5 * (K + K.T)
        eigvals, eigvecs = torch.linalg.eigh(K)  # ascending
        self._shared_eigvals = eigvals
        self._shared_eigvecs = eigvecs

    def get_kernel_op(self, coords: torch.Tensor) -> DenseKernelOp:
        """Return a ``DenseKernelOp`` built from the current (fixed) params.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Normalized spatial coordinates.

        Returns
        -------
        DenseKernelOp
            Kernel operator for the given coordinates.
        """
        K = _build_rbf_kernel(coords, self._constant_value, self._length_scale)
        return DenseKernelOp(K)

    # ------------------------------------------------------------------
    # Core residualization
    # ------------------------------------------------------------------

    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP residuals for a single target matrix.

        Chooses the fast spectral path when :attr:`signal_bounds_fixed` and
        the shared eigendecomposition has been precomputed; otherwise falls
        back to a full sklearn GP fit when ``n_spots <= n_inducing``, or a
        subset-of-data hyperparameter fit when ``n_spots > n_inducing``.

        NaN rows in ``Y`` are excluded from fitting and reinserted as NaN
        in the output.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Normalized spatial coordinates.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor
            Shape ``(n_spots, m)``. Spatial residuals. NaN rows from ``Y`` are
            preserved as NaN.
        """
        is_nan = torch.isnan(Y).any(1)
        if is_nan.any():
            Y_clean = Y[~is_nan]
            coords_clean = coords[~is_nan]
            res_clean = self._fit_no_nan(coords_clean, Y_clean)
            out = torch.full_like(Y, float("nan"))
            out[~is_nan] = res_clean
            return out

        return self._fit_no_nan(coords, Y)

    def _fit_large_n(self, coords: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Residualization for n > n_inducing via subset-of-data GP + chunked prediction.

        Fits an sklearn GP on a randomly sub-sampled ``n_inducing`` reference set
        to determine hyperparameters, then predicts on all ``n`` points using
        chunked matrix operations to limit peak memory.

        This is a **subset-of-data** approximation: hyperparameters are estimated
        from a subset but residuals are computed for all observations.  It is *not*
        an inducing-point (FITC/VFE) approximation.

        Returns
        -------
        torch.Tensor, shape (n, m)
            Residuals ``Y - K(X, X_m) @ (K_mm + eps·I)^{-1} @ Y_m``.
        """
        n = coords.shape[0]
        m = self.n_inducing
        rng = np.random.default_rng(0)
        idx = rng.choice(n, m, replace=False)
        idx_t = torch.from_numpy(idx)
        X_m = coords[idx_t]  # (m, d)
        Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)
        Y_m = Y2d[idx_t].float()  # (m, targets)

        # Fit sklearn GP on subset to get hyperparameters
        y_m_mean = Y_m.mean(dim=1).numpy()  # (m,) — column mean for multi-output
        kernel = C(self._constant_value, self._constant_value_bounds) * RBF(
            self._length_scale, self._length_scale_bounds
        ) + WhiteKernel(0.1, (1e-5, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gp.fit(X_m.numpy(), y_m_mean)

        # Extract fitted hyperparameters
        fitted_k = gp.kernel_
        sigma2 = float(fitted_k.k1.k1.constant_value)
        ls = float(fitted_k.k1.k2.length_scale)
        eps = float(np.exp(fitted_k.k2.theta[0]))

        # Build (m, m) kernel and solve alpha = (K_mm + eps·I)^{-1} @ Y_m
        K_mm = _build_rbf_kernel(X_m, sigma2, ls)  # (m, m)
        A = K_mm + eps * torch.eye(m)
        L = torch.linalg.cholesky(A)
        alpha = torch.cholesky_solve(Y_m, L)  # (m, targets)

        # Predict in chunks: res = Y - K(X, X_m) @ alpha
        _CHUNK = 10_000
        res = Y2d.float().clone()
        for start in range(0, n, _CHUNK):
            end = min(start + _CHUNK, n)
            K_chunk = _build_rbf_cross_kernel(coords[start:end], X_m, sigma2, ls)
            res[start:end] -= K_chunk @ alpha

        return res.squeeze(1) if Y.dim() == 1 else res

    def _fit_no_nan(self, coords: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Inner residualization without NaN handling."""
        if self.n_inducing is None:
            if coords.shape[0] > 10_000:
                warnings.warn(
                    f"n_inducing=None: using all {coords.shape[0]} observations for GP "
                    "hyperparameter fitting. This may be slow for large datasets. "
                    "Consider setting n_inducing to limit the subset size.",
                    UserWarning,
                    stacklevel=3,
                )
        elif coords.shape[0] > self.n_inducing:
            return self._fit_large_n(coords, Y)
        if (
            self.signal_bounds_fixed
            and self._shared_eigvals is not None
            and Y.shape[0] == self._shared_eigvecs.shape[0]
        ):
            # Fast path: reuse precomputed eigendecomposition.
            # Shape guard ensures we only use the cached decomp when coords haven't
            # been filtered (e.g., no NaN rows were removed for this gene).
            return _kernel_residuals_from_eigdecomp(
                self._shared_eigvecs, self._shared_eigvals, Y
            )
        # Full sklearn GP fit.
        return _sklearn_gpr_residuals(
            coords,
            Y,
            constant_value=self._constant_value,
            constant_value_bounds=self._constant_value_bounds,
            length_scale=self._length_scale,
            length_scale_bounds=self._length_scale_bounds,
        )

    def fit_residuals_batch(
        self,
        coords: torch.Tensor,
        Y_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Fit residuals for multiple targets, amortizing kernel setup.

        If :attr:`signal_bounds_fixed` and no eigendecomposition has been
        precomputed yet, this method precomputes it before the loop.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Normalized spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape ``(n_spots, m_i)``.

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape ``(n_spots, m_i)``.
        """
        if self.signal_bounds_fixed and self._shared_eigvals is None:
            self.precompute_shared_kernel(coords)
        return [self.fit_residuals(coords, Y) for Y in Y_list]


def _sklearn_gpr_residuals(
    X: torch.Tensor,
    Y: torch.Tensor,
    constant_value: float,
    constant_value_bounds: Union[tuple, str],
    length_scale: float,
    length_scale_bounds: Union[tuple, str],
) -> torch.Tensor:
    """Fit a single sklearn GPR and return residuals (no NaN handling)."""
    x_np = X.numpy()
    kernel = C(constant_value, constant_value_bounds) * RBF(
        length_scale, length_scale_bounds
    ) + WhiteKernel(0.1, (1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_np, Y.numpy())

    Kxy = torch.from_numpy(gp.kernel_.k1(x_np, x_np)).float()
    epsilon = float(np.exp(gp.kernel_.theta[-1]))

    Kxy = 0.5 * (Kxy + Kxy.T)
    Rx = epsilon * torch.linalg.inv(Kxy + epsilon * torch.eye(Kxy.shape[0]))
    return Rx @ Y


# ---------------------------------------------------------------------------
# GPyTorch backend (optional dependency)
# ---------------------------------------------------------------------------


class _GPyTorchExactGPModel:
    """Internal helper: wraps a gpytorch ExactGP with ScaleKernel(RBFKernel).

    Constructed lazily to avoid importing gpytorch at module load time.
    """

    @staticmethod
    def build(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Any,
        constant_value: float,
        constant_value_bounds: Union[tuple, str],
        length_scale: float,
        length_scale_bounds: Union[tuple, str],
    ) -> Any:
        """Build and return a gpytorch ExactGP model."""
        import gpytorch

        class _Model(gpytorch.models.ExactGP):
            def __init__(self) -> None:
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                # Set initial values
                covar.outputscale = constant_value
                covar.base_kernel.lengthscale = length_scale
                # Apply bounds / fix parameters
                if constant_value_bounds == "fixed":
                    covar.raw_outputscale.requires_grad_(False)
                else:
                    lo, hi = constant_value_bounds
                    covar.register_constraint(
                        "raw_outputscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    covar.outputscale = constant_value  # re-apply after constraint
                if length_scale_bounds == "fixed":
                    covar.base_kernel.raw_lengthscale.requires_grad_(False)
                else:
                    lo, hi = length_scale_bounds
                    covar.base_kernel.register_constraint(
                        "raw_lengthscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    covar.base_kernel.lengthscale = length_scale
                self.covar_module = covar

            def forward(self, x: torch.Tensor) -> Any:
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return _Model()

    @staticmethod
    def build_sparse(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Any,
        inducing_points: torch.Tensor,
        constant_value: float,
        constant_value_bounds: Union[tuple, str],
        length_scale: float,
        length_scale_bounds: Union[tuple, str],
    ) -> Any:
        """Build a GPyTorch SGPR model (FITC) using InducingPointKernel."""
        import gpytorch

        class _SparseModel(gpytorch.models.ExactGP):
            def __init__(self) -> None:
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                # Set initial values
                base.outputscale = constant_value
                base.base_kernel.lengthscale = length_scale
                # Apply bounds / fix parameters — mirror build() exactly
                if constant_value_bounds == "fixed":
                    base.raw_outputscale.requires_grad_(False)
                else:
                    lo, hi = constant_value_bounds
                    base.register_constraint(
                        "raw_outputscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    base.outputscale = constant_value  # re-apply after constraint
                if length_scale_bounds == "fixed":
                    base.base_kernel.raw_lengthscale.requires_grad_(False)
                else:
                    lo, hi = length_scale_bounds
                    base.base_kernel.register_constraint(
                        "raw_lengthscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    base.base_kernel.lengthscale = length_scale
                self.covar_module = gpytorch.kernels.InducingPointKernel(
                    base,
                    inducing_points=inducing_points,
                    likelihood=likelihood,
                )

            def forward(self, x: torch.Tensor) -> Any:
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return _SparseModel()


def _subsample_inducing(X: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """Uniformly sub-sample m rows from X as inducing points."""
    idx = np.random.default_rng(seed).choice(len(X), m, replace=False)
    return X[torch.from_numpy(idx)]


class GPyTorchKernelGPR(KernelGPR):
    """GPyTorch GP residualizer.

    Uses GPyTorch's lazy-tensor infrastructure so that all linear-system
    solves during training use Lanczos-based conjugate gradients rather
    than dense Cholesky.  Hyperparameters are optimized with Adam on the
    marginal log-likelihood.

    For moderate datasets, the exact-GP path is used. Passing
    ``n_inducing > 0`` enables the FITC sparse GP approximation via
    InducingPointKernel.

    Parameters
    ----------
    constant_value : float
        Initial signal amplitude (outputscale).
    constant_value_bounds : tuple or "fixed"
        Search bounds for the signal amplitude.  ``"fixed"`` disables
        optimization of this parameter.
    length_scale : float
        Initial RBF length scale.
    length_scale_bounds : tuple or "fixed"
        Search bounds for the length scale.
    n_inducing : int or None
        Number of inducing points for FITC sparse GP approximation. When set,
        memory scales as ``O(n_spots * n_inducing)`` rather than
        ``O(n_spots^2)``. Set to ``None`` to use exact GP.
    n_iter : int
        Adam optimizer iterations for hyperparameter learning.
    lr : float
        Adam learning rate.
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Notes
    -----
    Requires ``gpytorch`` (optional dependency)::

        pip install gpytorch

    Hyperparameter search uses the mean of all target columns as the
    training signal, then applies the fitted kernel to every column.
    This matches the sklearn backend behavior and amortizes GP fitting across
    output columns.

    The noise (epsilon) search is handled by GPyTorch's
    ``GaussianLikelihood``, which is always optimized regardless of the
    signal bounds setting.
    """

    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
        length_scale: float = 1.0,
        length_scale_bounds: Union[tuple, str] = "fixed",
        n_inducing: Optional[int] = None,
        n_iter: int = 100,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        try:
            import gpytorch as _gpytorch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GPyTorchKernelGPR requires gpytorch. "
                "Install it with:  pip install gpytorch"
            ) from exc
        self._constant_value = constant_value
        self._constant_value_bounds = constant_value_bounds
        self._length_scale = length_scale
        self._length_scale_bounds = length_scale_bounds
        self.n_inducing = n_inducing
        self.n_iter = n_iter
        self.lr = lr
        self.device = device

    @property
    def signal_bounds_fixed(self) -> bool:
        """True when both constant_value and length_scale bounds are ``"fixed"``."""
        return (
            self._constant_value_bounds == "fixed"
            and self._length_scale_bounds == "fixed"
        )

    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP residuals using GPyTorch's exact-GP backend.

        Hyperparameters are optimized on the mean of all target columns.
        The fitted kernel and noise level are then applied to every column
        of ``Y`` to compute residuals.

        NaN rows are excluded from fitting and reinserted as NaN in output.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Normalized spatial coordinates.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor
            Shape ``(n_spots, m)``. Spatial residuals
            ``epsilon * (K + epsilon * I)**(-1) @ Y``. NaN rows from ``Y``
            are preserved as NaN.
        """
        is_nan = torch.isnan(Y).any(1)
        if is_nan.any():
            coords_c = coords[~is_nan]
            Y_c = Y[~is_nan]
            res_clean = self._fit_no_nan(coords_c, Y_c)
            out = torch.full_like(Y, float("nan"))
            out[~is_nan] = res_clean
            return out
        return self._fit_no_nan(coords, Y)

    def _fit_no_nan(self, coords: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Inner residualization assuming no NaN rows."""
        import gpytorch

        Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)  # (n, m)
        n = Y2d.shape[0]
        dev = torch.device(self.device)

        train_x = coords.float().to(dev)
        # Optimize hyperparameters on the column mean
        train_y = Y2d.float().mean(dim=1).to(dev)  # (n,)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
        # Use FITC only when there are strictly more observations than the
        # requested inducing set (i.e. an actual sparse approximation).
        use_sparse = self.n_inducing is not None and n > self.n_inducing
        if use_sparse:
            inducing_pts = _subsample_inducing(train_x, self.n_inducing)
            model = _GPyTorchExactGPModel.build_sparse(
                train_x,
                train_y,
                likelihood,
                inducing_pts,
                self._constant_value,
                self._constant_value_bounds,
                self._length_scale,
                self._length_scale_bounds,
            ).to(dev)
        else:
            model = _GPyTorchExactGPModel.build(
                train_x,
                train_y,
                likelihood,
                self._constant_value,
                self._constant_value_bounds,
                self._length_scale,
                self._length_scale_bounds,
            ).to(dev)

        model.train()
        likelihood.train()
        # model.parameters() already includes likelihood params via ExactGP
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable and self.n_iter > 0:
            optimizer = torch.optim.Adam(trainable, lr=self.lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(self.n_iter):
                optimizer.zero_grad()
                loss = -mll(model(train_x), train_y)
                loss.backward()
                optimizer.step()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            eps = likelihood.noise.item()
            K_lazy = model.covar_module(train_x)  # lazy (n, n)
            Y_dev = Y2d.float().to(dev)  # (n, m)
            # Solve (K + eps*I) @ alpha = Y via CG (no explicit n x n matrix)
            K_noisy = K_lazy.add_diagonal(torch.full((n,), eps, device=dev))
            alpha = K_noisy.solve(Y_dev)  # (n, m)
            res = eps * alpha  # epsilon * (K + eps*I)^{-1} @ Y

        return res.cpu()


# ---------------------------------------------------------------------------
# FFT GPR backend (regular grid, RBF kernel)
# ---------------------------------------------------------------------------


class FFTKernelGPR(KernelGPR):
    """FFT GP residualizer for regular ``(ny, nx)`` grids.

    Accepts the same hyperparameter interface as :class:`SklearnKernelGPR`.
    Kernel eigenvalues are computed with :class:`FFTKernelOp`; matrix-vector
    operations cost ``O(n_grid log n_grid)`` and no dense
    ``n_grid x n_grid`` kernel is formed.

    Notes
    -----
    **Hyperparameter optimization strategy**

    The spectral log-marginal likelihood (LML) for ``Y`` with
    ``n_grid = ny * nx`` cells and ``m`` response channels is::

        -2 LML = m * sum_k log(lambda_k + eps)
                 + (1 / n_grid) * sum_k ||Y_hat_k||^2 / (lambda_k + eps)

    where ``lambda_k = sigma^2 * s_k(l)`` are the kernel eigenvalues with
    signal variance ``sigma^2`` and unit-spectrum eigenvalues ``s_k(l)`` at
    length scale ``l``; ``eps`` is the white-noise variance; and
    ``Y_hat = FFT2(Y_cube)`` (unnormalized 2-D DFT).

    Four optimization cases based on which bounds (constant_value_bounds, length_scale_bounds)
    are ``"fixed"``:

    1. **Both fixed** → 1-D bounded scalar search over ``log(eps)`` only.
       Fastest; unit spectrum cached across repeated calls.
    2. **cv free, ls fixed** → 2-D L-BFGS-B over ``(log_sigma2, log_eps)``.
       Unit spectrum cached.
    3. **cv fixed, ls free** → 2-D L-BFGS-B over ``(log_l, log_eps)``.
       A fresh ``FFTKernelOp`` is built at each optimizer step.
    4. **Both free** → 3-D L-BFGS-B over ``(log_sigma2, log_l, log_eps)``.
       A fresh ``FFTKernelOp`` is built at each optimizer step.

    Parameters
    ----------
    constant_value : float
        Initial signal variance ``sigma^2``.
    constant_value_bounds : tuple or "fixed"
        Bounds for ``sigma^2``. ``"fixed"`` pins ``sigma^2``.
    length_scale : float
        Initial RBF length scale.
    length_scale_bounds : tuple or "fixed"
        Bounds for the RBF length scale.  ``"fixed"`` pins it; a 2-tuple
        ``(lo, hi)`` of positive floats enables length-scale optimization.
    epsilon_bounds : tuple[float, float]
        Search bounds for the white-noise / regularization level.
    workers : int or None
        Number of ``scipy.fft`` workers.
    """

    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
        length_scale: float = 1.0,
        length_scale_bounds: Union[tuple, str] = "fixed",
        epsilon_bounds: tuple = (1e-5, 1e1),
        workers: Optional[int] = None,
    ) -> None:
        if length_scale_bounds != "fixed":
            if not (
                isinstance(length_scale_bounds, (tuple, list))
                and len(length_scale_bounds) == 2
                and float(length_scale_bounds[0]) > 0
                and float(length_scale_bounds[1]) > float(length_scale_bounds[0])
            ):
                raise ValueError(
                    "length_scale_bounds must be 'fixed' or a 2-tuple of positive "
                    f"floats (lo, hi) with lo < hi; got {length_scale_bounds!r}."
                )
        self._constant_value = float(constant_value)
        self._constant_value_bounds = constant_value_bounds
        self._length_scale = float(length_scale)
        self._length_scale_bounds = length_scale_bounds
        self._epsilon_bounds = tuple(epsilon_bounds)
        self._workers = workers

        # Grid state — populated on first use
        self._ny: Optional[int] = None
        self._nx: Optional[int] = None
        self._dy: float = 1.0
        self._dx: float = 1.0

        # Cached unit-spectrum op (sigma^2 = 1, reused for 1-D eps search)
        self._unit_op: Optional[FFTKernelOp] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def signal_bounds_fixed(self) -> bool:
        """True when BOTH ``constant_value_bounds`` and ``length_scale_bounds`` are ``"fixed"``."""
        return (
            self._constant_value_bounds == "fixed"
            and self._length_scale_bounds == "fixed"
        )

    @property
    def ny(self) -> Optional[int]:
        """Grid height (set after first residualization)."""
        return self._ny

    @property
    def nx(self) -> Optional[int]:
        """Grid width (set after first residualization)."""
        return self._nx

    @property
    def n(self) -> Optional[int]:
        """Total grid cells (ny * nx)."""
        if self._ny is None or self._nx is None:
            return None
        return self._ny * self._nx

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _detect_grid(
        self, coords: torch.Tensor
    ) -> tuple[int, int, float, float, np.ndarray, np.ndarray]:
        """Auto-detect regular grid from 2-D coordinates.

        Parameters
        ----------
        coords : torch.Tensor, shape (n_obs, 2)

        Returns
        -------
        ny, nx : int
        dy, dx : float
        rows, cols : np.ndarray[int]

        Raises
        ------
        ValueError
            If coordinates do not form a regular grid.
        """
        xy = coords.numpy()
        uy = np.unique(xy[:, 0])
        ux = np.unique(xy[:, 1])
        ny, nx = len(uy), len(ux)

        if ny * nx != len(xy):
            raise ValueError(
                f"FFTKernelGPR requires a regular grid but got {len(xy)} "
                f"points for a {ny}x{nx} grid."
            )

        dy = float(np.diff(uy).mean()) if ny > 1 else 1.0
        dx = float(np.diff(ux).mean()) if nx > 1 else 1.0

        if ny > 1 and not np.allclose(np.diff(uy), dy, rtol=1e-4):
            raise ValueError("y-coordinates are not evenly spaced.")
        if nx > 1 and not np.allclose(np.diff(ux), dx, rtol=1e-4):
            raise ValueError("x-coordinates are not evenly spaced.")

        y_to_row = {v: i for i, v in enumerate(uy)}
        x_to_col = {v: i for i, v in enumerate(ux)}
        rows = np.array([y_to_row[float(v)] for v in xy[:, 0]], dtype=int)
        cols = np.array([x_to_col[float(v)] for v in xy[:, 1]], dtype=int)
        return ny, nx, dy, dx, rows, cols

    def _get_unit_op(self, ny: int, nx: int, dy: float, dx: float) -> FFTKernelOp:
        """Build/cache a unit-spectrum FFTKernelOp (sigma^2 = 1)."""
        if (
            self._unit_op is None
            or self._ny != ny
            or self._nx != nx
            or self._dy != dy
            or self._dx != dx
        ):
            self._unit_op = FFTKernelOp(
                ny=ny,
                nx=nx,
                dy=dy,
                dx=dx,
                constant_value=1.0,
                length_scale=self._length_scale,
                workers=self._workers,
            )
        self._ny, self._nx, self._dy, self._dx = ny, nx, dy, dx
        return self._unit_op

    # ------------------------------------------------------------------
    # Spectral LML helpers
    # ------------------------------------------------------------------

    def _yhat_power(self, cube: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Compute FFT2-power summed over channels.

        Parameters
        ----------
        cube : np.ndarray, shape (ny, nx, m)
            Centred, NaN-imputed cube.

        Returns
        -------
        power : np.ndarray, shape (ny, nx)
            ``sum_c |FFT2(cube_c)[k]|^2`` for each frequency k.
        n : int
        m : int
        """
        y_hat = scipy.fft.fft2(cube, axes=(0, 1), workers=self._workers)
        power = (np.abs(y_hat) ** 2).sum(axis=2)
        return power, cube.shape[0] * cube.shape[1], cube.shape[2]

    def _optimize(
        self,
        power: np.ndarray,
        n: int,
        m: int,
        ny: int,
        nx: int,
        dy: float,
        dx: float,
    ) -> tuple[float, float, float]:
        """Optimize (sigma^2, length_scale, eps) via spectral LML.

        Four cases depending on which bounds are ``"fixed"``:

        1. Both fixed → 1-D scalar search over ``log(eps)``.
        2. cv free, ls fixed → 2-D L-BFGS-B over ``(log_sigma2, log_eps)``.
        3. cv fixed, ls free → 2-D L-BFGS-B over ``(log_l, log_eps)``.
        4. Both free → 3-D L-BFGS-B over ``(log_sigma2, log_l, log_eps)``.

        Returns
        -------
        sigma2 : float
        length_scale : float
        eps : float
        """
        power_flat = power.ravel()
        lo_e, hi_e = self._epsilon_bounds
        ls_fixed = self._length_scale_bounds == "fixed"
        cv_fixed = self._constant_value_bounds == "fixed"

        if ls_fixed:
            # Cases 1 & 2: unit spectrum can be cached
            unit_op = self._get_unit_op(ny, nx, dy, dx)
            lam0 = unit_op.eigenvalues_2d.ravel()

            if cv_fixed:
                # Case 1: 1-D search over log(eps)
                lam_fixed = self._constant_value * lam0

                def neg_lml_1d(log_eps: float) -> float:
                    eps = float(np.exp(log_eps))
                    le = lam_fixed + eps
                    return 0.5 * float(
                        m * np.sum(np.log(le)) + np.sum(power_flat / le) / n
                    )

                res = minimize_scalar(
                    neg_lml_1d, bounds=(np.log(lo_e), np.log(hi_e)), method="bounded"
                )
                return self._constant_value, self._length_scale, float(np.exp(res.x))

            else:
                # Case 2: 2-D over (log_sigma2, log_eps)
                lo_s, hi_s = self._constant_value_bounds

                def neg_lml_2d(params: np.ndarray) -> float:
                    sigma2 = float(np.exp(params[0]))
                    eps = float(np.exp(params[1]))
                    le = sigma2 * lam0 + eps
                    return 0.5 * float(
                        m * np.sum(np.log(le)) + np.sum(power_flat / le) / n
                    )

                x0 = np.array(
                    [np.log(self._constant_value), np.log(np.sqrt(lo_e * hi_e))]
                )
                bounds_lb = [
                    (np.log(float(lo_s)), np.log(float(hi_s))),
                    (np.log(lo_e), np.log(hi_e)),
                ]
                res = minimize(neg_lml_2d, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
                return (
                    float(np.exp(res.x[0])),
                    self._length_scale,
                    float(np.exp(res.x[1])),
                )

        else:
            # Cases 3 & 4: length scale is free; rebuild FFTKernelOp per step.
            # eigenvalues are sigma^2-scaled inside the op (constant_value argument).
            lo_l, hi_l = self._length_scale_bounds

            if cv_fixed:
                # Case 3: 2-D over (log_l, log_eps)
                sigma2_fixed = self._constant_value

                def neg_lml_3(params: np.ndarray) -> float:
                    length_scale = float(np.exp(params[0]))
                    eps = float(np.exp(params[1]))
                    tmp_op = FFTKernelOp(
                        ny, nx, dy, dx, sigma2_fixed, length_scale, self._workers
                    )
                    lam = tmp_op.eigenvalues_2d.ravel()
                    le = lam + eps
                    return 0.5 * float(
                        m * np.sum(np.log(le)) + np.sum(power_flat / le) / n
                    )

                x0 = np.array(
                    [np.log(self._length_scale), np.log(np.sqrt(lo_e * hi_e))]
                )
                bounds_lb = [
                    (np.log(float(lo_l)), np.log(float(hi_l))),
                    (np.log(lo_e), np.log(hi_e)),
                ]
                res = minimize(neg_lml_3, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
                return sigma2_fixed, float(np.exp(res.x[0])), float(np.exp(res.x[1]))

            else:
                # Case 4: 3-D over (log_sigma2, log_l, log_eps)
                lo_s, hi_s = self._constant_value_bounds

                def neg_lml_4(params: np.ndarray) -> float:
                    sigma2 = float(np.exp(params[0]))
                    length_scale = float(np.exp(params[1]))
                    eps = float(np.exp(params[2]))
                    tmp_op = FFTKernelOp(
                        ny, nx, dy, dx, sigma2, length_scale, self._workers
                    )
                    lam = tmp_op.eigenvalues_2d.ravel()
                    le = lam + eps
                    return 0.5 * float(
                        m * np.sum(np.log(le)) + np.sum(power_flat / le) / n
                    )

                x0 = np.array(
                    [
                        np.log(self._constant_value),
                        np.log(self._length_scale),
                        np.log(np.sqrt(lo_e * hi_e)),
                    ]
                )
                bounds_lb = [
                    (np.log(float(lo_s)), np.log(float(hi_s))),
                    (np.log(float(lo_l)), np.log(float(hi_l))),
                    (np.log(lo_e), np.log(hi_e)),
                ]
                res = minimize(neg_lml_4, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
                return (
                    float(np.exp(res.x[0])),
                    float(np.exp(res.x[1])),
                    float(np.exp(res.x[2])),
                )

    def _residualize_cube(
        self,
        raw_cube: np.ndarray,
        ny: int,
        nx: int,
        dy: float,
        dx: float,
    ) -> tuple[np.ndarray, float, float, float]:
        """Core per-cube residualization.

        Parameters
        ----------
        raw_cube : np.ndarray, shape (ny, nx, m)
            Raw raster data (may contain NaN).

        Returns
        -------
        res_cube : np.ndarray, shape (ny, nx, m)
        sigma2 : float
        length_scale : float
        eps : float
        """
        cube = raw_cube - np.nanmean(raw_cube, axis=(0, 1), keepdims=True)
        cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

        power, n, m = self._yhat_power(cube)
        sigma2, length_scale, eps = self._optimize(power, n, m, ny, nx, dy, dx)

        # Apply residual op with optimized hyperparameters
        signal_op = FFTKernelOp(
            ny=ny,
            nx=nx,
            dy=dy,
            dx=dx,
            constant_value=sigma2,
            length_scale=length_scale,
            workers=self._workers,
        )
        res = signal_op.residuals(cube.reshape(n, m), eps)
        return res.reshape(ny, nx, m), sigma2, length_scale, eps

    # ------------------------------------------------------------------
    # KernelGPR public interface
    # ------------------------------------------------------------------

    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP and return spatial residuals.

        Auto-detects the regular grid from ``coords``, rasterizes ``Y``,
        optimizes hyperparameters via spectral LML, then de-rasterizes.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, 2)``. Spatial coordinates forming a regular grid.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor
            Shape ``(n_spots, m)``. Spatial residuals. NaN rows are preserved.
        """
        Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)
        _, m = Y2d.shape[0], Y2d.shape[1]

        ny, nx, dy, dx, rows, cols = self._detect_grid(coords)

        # Rasterize (NaN-fill gaps)
        cube = np.full((ny, nx, m), np.nan, dtype=float)
        cube[rows, cols, :] = Y2d.numpy()

        res_cube, _, _, _ = self._residualize_cube(cube, ny, nx, dy, dx)

        # De-rasterize and restore NaN rows
        out = torch.from_numpy(res_cube[rows, cols, :]).float()
        is_nan = torch.isnan(Y2d).any(1)
        if is_nan.any():
            out[is_nan] = float("nan")

        return out.squeeze(1) if Y.dim() == 1 else out

    def fit_residuals_batch(
        self,
        coords: torch.Tensor,
        Y_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Residualize a list of targets, amortizing grid detection and unit spectrum.

        When :attr:`signal_bounds_fixed` is True the unit-spectrum
        eigenvalues are shared; only eps is re-optimised per target (1-D
        search).  Otherwise each target gets full 2-D optimisation.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, 2)``. Spatial coordinates forming a regular grid.
        Y_list : list of torch.Tensor
            Target matrices, each shape ``(n_spots, m_i)``.

        Returns
        -------
        list of torch.Tensor
            Residual matrices, each shape ``(n_spots, m_i)``.
        """
        ny, nx, dy, dx, rows, cols = self._detect_grid(coords)
        # No pre-computed unit_op: _optimize handles caching internally for
        # fixed-ls cases; cases 3 & 4 rebuild per optimizer step anyway.

        results = []
        for Y in Y_list:
            Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)
            _, m = Y2d.shape[0], Y2d.shape[1]

            cube = np.full((ny, nx, m), np.nan, dtype=float)
            cube[rows, cols, :] = Y2d.numpy()
            cube_c = cube - np.nanmean(cube, axis=(0, 1), keepdims=True)
            cube_c = np.nan_to_num(cube_c, nan=0.0, posinf=0.0, neginf=0.0)

            power, n_cells, m_ch = self._yhat_power(cube_c)
            sigma2, length_scale, eps = self._optimize(
                power, n_cells, m_ch, ny, nx, dy, dx
            )

            signal_op = FFTKernelOp(
                ny=ny,
                nx=nx,
                dy=dy,
                dx=dx,
                constant_value=sigma2,
                length_scale=length_scale,
                workers=self._workers,
            )
            res = signal_op.residuals(cube_c.reshape(n_cells, m), eps)
            res_cube = res.reshape(ny, nx, m)

            out = torch.from_numpy(res_cube[rows, cols, :]).float()
            is_nan = torch.isnan(Y2d).any(1)
            if is_nan.any():
                out[is_nan] = float("nan")
            results.append(out.squeeze(1) if Y.dim() == 1 else out)

        return results

    def fit_residuals_cube(
        self,
        data_cube: np.ndarray,
        spacing: tuple = (1.0, 1.0),
        epsilon: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Residualize a raster cube in O(N log N).

        Kept for backward compatibility with :class:`SplisosmFFT`.

        Parameters
        ----------
        data_cube : np.ndarray, shape (ny, nx) or (ny, nx, m)
        spacing : (dy, dx)
            Physical grid spacing.
        epsilon : float or None
            If given, skip optimization and use this value directly.

        Returns
        -------
        residuals : np.ndarray, same shape as data_cube
        epsilon : float
            Regularization level used.
        """
        cube = np.asarray(data_cube, dtype=float)
        scalar = cube.ndim == 2
        if scalar:
            cube = cube[..., np.newaxis]

        ny, nx, m = cube.shape
        dy, dx = float(spacing[0]), float(spacing[1])

        if epsilon is not None:
            # Skip optimization — use provided epsilon with current sigma^2
            cube_c = cube - np.nanmean(cube, axis=(0, 1), keepdims=True)
            cube_c = np.nan_to_num(cube_c, nan=0.0, posinf=0.0, neginf=0.0)
            op = FFTKernelOp(
                ny=ny,
                nx=nx,
                dy=dy,
                dx=dx,
                constant_value=self._constant_value,
                length_scale=self._length_scale,
                workers=self._workers,
            )
            res = op.residuals(cube_c.reshape(ny * nx, m), epsilon)
            res_cube = res.reshape(ny, nx, m)
            self._ny, self._nx, self._dy, self._dx = ny, nx, dy, dx
        else:
            res_cube, _, _, epsilon = self._residualize_cube(cube, ny, nx, dy, dx)

        return (res_cube[..., 0] if scalar else res_cube), float(epsilon)

    def get_kernel_op(self, coords: Optional[torch.Tensor] = None) -> "FFTKernelOp":
        """Return the cached unit-spectrum FFTKernelOp.

        Raises
        ------
        RuntimeError
            If called before any residualization.
        """
        if self._unit_op is None:
            raise RuntimeError(
                "No FFTKernelOp available. Call fit_residuals or fit_residuals_cube first."
            )
        return self._unit_op


# ---------------------------------------------------------------------------
# NUFFT GPR backend (irregular 2-D coordinates, RBF kernel)
# ---------------------------------------------------------------------------


class NUFFTKernelGPR(KernelGPR):
    """FINUFFT GP residualizer for irregular 2-D coordinates.

    The backend keeps the RBF kernel implicit: FINUFFT evaluates kernel-vector
    products and conjugate gradients solve ``(K + epsilon * I)`` systems. Regular
    grids with matching ``n_modes`` use the same spectral likelihood as
    :class:`FFTKernelGPR`; irregular grids use a leading-eigenpair summary plus
    ``tr(K)`` / ``tr(K**2)`` tail correction for marginal likelihood fitting.

    Parameters
    ----------
    constant_value : float
        Initial RBF signal variance.
    constant_value_bounds : tuple or "fixed"
        Bounds for the signal variance. ``"fixed"`` pins it.
    length_scale : float
        Initial RBF length scale in coordinate units.
    length_scale_bounds : tuple or "fixed"
        Bounds for the length scale. ``"fixed"`` pins it.
    epsilon_bounds : tuple[float, float]
        Search bounds for the white-noise level.
    n_modes : int, tuple[int, int], or None
        Fourier mode grid.  ``None`` chooses the full effective grid from the
        observed point count and aspect ratio.
    max_auto_modes : int or None
        Optional cap on automatic ``n_modes``. Ignored when ``n_modes`` is
        explicit.
    nufft_eps : float
        FINUFFT requested precision.
    nufft_opts : dict or None
        Extra FINUFFT options, such as ``{"upsampfac": 2.0}``. Use
        ``workers`` for thread count.
    lml_approx_rank : int or None
        Leading eigenpair count for irregular-grid marginal likelihoods.
        Memory is ``O(n_spots * lml_approx_rank)``. Ignored on compatible
        regular grids.  ``None`` forces exact dense eigendecomposition for
        small diagnostics.
    lml_exact_max_n : int
        Maximum ``n_spots`` for exact dense eigendecomposition.
    eigsh_tol : float
        Relative tolerance for ``eigsh`` in the irregular-grid eigensummary
        path.
    period_margin : float
        Fractional padding around the coordinate bounding box before periodic
        wrapping.
    cg_rtol : float
        Relative tolerance for conjugate-gradient solves.
    cg_maxiter : int or None
        Maximum CG iterations.
    workers : int or None
        FINUFFT thread count passed as ``nthreads``.  ``None`` uses one
        FINUFFT thread.

    Notes
    -----
    Requires ``finufft`` (optional dependency)::

        pip install gpytorch
    
    Results should agree exactly with :class:`FFTKernelGPR` on compatible regular grids.  
    Compared with :class:`SklearnKernelGPR`, differences near the bounding-box edges are
    expected because this backend uses periodic boundary conditions.
    """

    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
        length_scale: float = 1.0,
        length_scale_bounds: Union[tuple, str] = "fixed",
        epsilon_bounds: tuple = (1e-5, 1e1),
        n_modes: Optional[int | tuple[int, int]] = None,
        max_auto_modes: Optional[int] = None,
        nufft_eps: float = 1e-6,
        nufft_opts: Optional[dict[str, Any]] = None,
        lml_approx_rank: Optional[int] = 256,
        lml_exact_max_n: int = 512,
        eigsh_tol: float = 1e-4,
        period_margin: float = 0.5,
        cg_rtol: float = 1e-5,
        cg_maxiter: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> None:
        _require_finufft()
        if length_scale_bounds != "fixed":
            if not (
                isinstance(length_scale_bounds, (tuple, list))
                and len(length_scale_bounds) == 2
                and float(length_scale_bounds[0]) > 0
                and float(length_scale_bounds[1]) > float(length_scale_bounds[0])
            ):
                raise ValueError(
                    "length_scale_bounds must be 'fixed' or a 2-tuple of positive "
                    f"floats (lo, hi) with lo < hi; got {length_scale_bounds!r}."
                )
        if constant_value_bounds != "fixed":
            if not (
                isinstance(constant_value_bounds, (tuple, list))
                and len(constant_value_bounds) == 2
                and float(constant_value_bounds[0]) > 0
                and float(constant_value_bounds[1]) > float(constant_value_bounds[0])
            ):
                raise ValueError(
                    "constant_value_bounds must be 'fixed' or a 2-tuple of positive "
                    f"floats (lo, hi) with lo < hi; got {constant_value_bounds!r}."
                )

        self._constant_value = float(constant_value)
        self._constant_value_bounds = constant_value_bounds
        self._length_scale = float(length_scale)
        self._length_scale_bounds = length_scale_bounds
        self._epsilon_bounds = tuple(epsilon_bounds)
        self._n_modes = n_modes
        self._max_auto_modes = max_auto_modes
        self._nufft_eps = float(nufft_eps)
        self._nufft_opts = dict(nufft_opts or {})
        self._lml_approx_rank = lml_approx_rank
        self._lml_exact_max_n = int(lml_exact_max_n)
        self._eigsh_tol = float(eigsh_tol)
        self._period_margin = float(period_margin)
        self._cg_rtol = float(cg_rtol)
        self._cg_maxiter = cg_maxiter
        self._workers = workers

        self._basis_cache: Optional[tuple[tuple, NUFFTKernelOp]] = None
        self._eig_cache: Optional[tuple[tuple, np.ndarray, np.ndarray]] = None
        self._trace2_cache: Optional[tuple[tuple, float]] = None

    @property
    def signal_bounds_fixed(self) -> bool:
        """True when both signal variance and length scale are fixed."""
        return (
            self._constant_value_bounds == "fixed"
            and self._length_scale_bounds == "fixed"
        )

    def _mode_length_scale(self) -> float:
        """Length scale used to choose the automatic Fourier mode grid."""
        if self._length_scale_bounds == "fixed":
            return self._length_scale
        return min(self._length_scale, float(self._length_scale_bounds[0]))

    def _coords_cache_key(self, coords: torch.Tensor | np.ndarray) -> tuple:
        """Compact cache key for repeated calls on the same coordinate set."""
        xy = _as_numpy_coords_2d(coords)
        return (
            xy.shape,
            tuple(np.round(xy.min(axis=0), 12)),
            tuple(np.round(xy.max(axis=0), 12)),
            float(np.round(xy.sum(), 12)),
            float(np.round(np.square(xy).sum(), 12)),
            self._n_modes,
            self._max_auto_modes,
            tuple(sorted((k, repr(v)) for k, v in self._nufft_opts.items())),
            self._period_margin,
            self._mode_length_scale(),
        )

    def _make_op(
        self,
        coords: torch.Tensor | np.ndarray,
        constant_value: float,
        length_scale: float,
        n_modes: Optional[tuple[int, int]] = None,
    ) -> NUFFTKernelOp:
        """Build a NUFFT kernel operator with shared constructor settings."""
        return NUFFTKernelOp(
            coords=coords,
            constant_value=constant_value,
            length_scale=length_scale,
            n_modes=self._n_modes if n_modes is None else n_modes,
            max_auto_modes=self._max_auto_modes,
            nufft_eps=self._nufft_eps,
            nufft_opts=self._nufft_opts,
            period_margin=self._period_margin,
            cg_rtol=self._cg_rtol,
            cg_maxiter=self._cg_maxiter,
            workers=self._workers,
        )

    def _get_basis_op(self, coords: torch.Tensor | np.ndarray) -> NUFFTKernelOp:
        """Build/cache a unit-variance operator that fixes coords and modes."""
        key = self._coords_cache_key(coords)
        cache = self._basis_cache
        if cache is not None:
            cached_key, cached_op = cache
            if cached_key == key:
                return cached_op

        basis_op = self._make_op(
            coords,
            constant_value=1.0,
            length_scale=self._mode_length_scale(),
            n_modes=None,
        )
        self._basis_cache = (key, basis_op)
        return basis_op

    @staticmethod
    def _spectral_neg_lml(
        lam: np.ndarray,
        power: np.ndarray,
        total_ss: float,
        n: int,
        m: int,
        eps: float,
    ) -> float:
        """Approximate negative log marginal likelihood from NUFFT spectrum."""
        lam = np.asarray(lam, dtype=float).ravel()
        power = np.asarray(power, dtype=float).ravel()

        if lam.size > n:
            keep = np.argpartition(lam, -n)[-n:]
            lam = lam[keep]
            power = power[keep]

        le = lam + eps
        explained_ss = float(np.sum(power)) / n
        missing_ss = max(float(total_ss) - explained_ss, 0.0)

        logdet = float(np.sum(np.log(le)))
        if lam.size < n:
            logdet += (n - lam.size) * float(np.log(eps))

        quad = float(np.sum(power / le)) / n + missing_ss / eps
        return 0.5 * float(m * logdet + quad)

    def _optimize_spectral(
        self,
        basis_op: NUFFTKernelOp,
        power: np.ndarray,
        total_ss: float,
        n: int,
        m: int,
        coords: torch.Tensor | np.ndarray,
    ) -> tuple[float, float, float]:
        """Optimize ``(sigma², length_scale, eps)`` via NUFFT spectral LML."""
        lo_e, hi_e = self._epsilon_bounds
        ls_fixed = self._length_scale_bounds == "fixed"
        cv_fixed = self._constant_value_bounds == "fixed"
        n_modes = basis_op.n_modes

        if ls_fixed:
            weights0 = basis_op.spectral_weights.ravel()

            if cv_fixed:
                lam_fixed = self._constant_value * n * weights0

                def neg_lml_1d(log_eps: float) -> float:
                    return self._spectral_neg_lml(
                        lam_fixed, power, total_ss, n, m, float(np.exp(log_eps))
                    )

                res = minimize_scalar(
                    neg_lml_1d, bounds=(np.log(lo_e), np.log(hi_e)), method="bounded"
                )
                return self._constant_value, self._length_scale, float(np.exp(res.x))

            lo_s, hi_s = self._constant_value_bounds

            def neg_lml_2d(params: np.ndarray) -> float:
                sigma2 = float(np.exp(params[0]))
                eps = float(np.exp(params[1]))
                return self._spectral_neg_lml(
                    sigma2 * n * weights0, power, total_ss, n, m, eps
                )

            x0 = np.array([np.log(self._constant_value), np.log(np.sqrt(lo_e * hi_e))])
            bounds_lb = [
                (np.log(float(lo_s)), np.log(float(hi_s))),
                (np.log(lo_e), np.log(hi_e)),
            ]
            res = minimize(neg_lml_2d, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
            return float(np.exp(res.x[0])), self._length_scale, float(np.exp(res.x[1]))

        lo_l, hi_l = self._length_scale_bounds

        def _weights_for_length_scale(length_scale: float) -> np.ndarray:
            tmp = self._make_op(
                coords,
                constant_value=1.0,
                length_scale=length_scale,
                n_modes=n_modes,
            )
            return tmp.spectral_weights.ravel()

        if cv_fixed:
            sigma2_fixed = self._constant_value

            def neg_lml_3(params: np.ndarray) -> float:
                length_scale = float(np.exp(params[0]))
                eps = float(np.exp(params[1]))
                weights = _weights_for_length_scale(length_scale)
                return self._spectral_neg_lml(
                    sigma2_fixed * n * weights, power, total_ss, n, m, eps
                )

            x0 = np.array([np.log(self._length_scale), np.log(np.sqrt(lo_e * hi_e))])
            bounds_lb = [
                (np.log(float(lo_l)), np.log(float(hi_l))),
                (np.log(lo_e), np.log(hi_e)),
            ]
            res = minimize(neg_lml_3, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
            return sigma2_fixed, float(np.exp(res.x[0])), float(np.exp(res.x[1]))

        lo_s, hi_s = self._constant_value_bounds

        def neg_lml_4(params: np.ndarray) -> float:
            sigma2 = float(np.exp(params[0]))
            length_scale = float(np.exp(params[1]))
            eps = float(np.exp(params[2]))
            weights = _weights_for_length_scale(length_scale)
            return self._spectral_neg_lml(
                sigma2 * n * weights, power, total_ss, n, m, eps
            )

        x0 = np.array(
            [
                np.log(self._constant_value),
                np.log(self._length_scale),
                np.log(np.sqrt(lo_e * hi_e)),
            ]
        )
        bounds_lb = [
            (np.log(float(lo_s)), np.log(float(hi_s))),
            (np.log(float(lo_l)), np.log(float(hi_l))),
            (np.log(lo_e), np.log(hi_e)),
        ]
        res = minimize(neg_lml_4, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
        return float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(np.exp(res.x[2]))

    def _is_regular_fft_compatible(
        self,
        coords: torch.Tensor | np.ndarray,
        basis_op: NUFFTKernelOp,
    ) -> bool:
        """Return whether NUFFT spectral LML is exact for the sampled grid."""
        if self._period_margin != 0:
            return False
        xy = _as_numpy_coords_2d(coords)
        meta = _regular_grid_shape_spacing(xy)
        if meta is None:
            return False
        ny, nx, _, _ = meta
        return basis_op.n_modes == (ny, nx)

    @staticmethod
    def _eigen_neg_lml(
        evals: np.ndarray,
        power: np.ndarray,
        total_ss: float,
        trace: float,
        trace2: Optional[float],
        n: int,
        m: int,
        sigma2: float,
        eps: float,
    ) -> float:
        """Approximate negative LML from implicit-kernel eigenpairs."""
        evals = np.asarray(evals, dtype=float)
        power = np.asarray(power, dtype=float)
        den = sigma2 * evals + eps
        logdet = float(np.sum(np.log(den)))
        quad = float(np.sum(power / den))

        tail_count = n - evals.size
        if tail_count > 0:
            tail_trace = max(float(trace) - float(np.sum(evals)), 0.0)
            residual_ss = max(float(total_ss) - float(np.sum(power)), 0.0)
            if trace2 is None or tail_trace <= 0.0:
                tail_eig = tail_trace / tail_count
                tail_den = sigma2 * tail_eig + eps
                logdet += tail_count * float(np.log(tail_den))
                quad += residual_ss / tail_den
            else:
                tail_trace2 = max(float(trace2) - float(np.sum(evals**2)), 0.0)
                min_tail_trace2 = tail_trace**2 / tail_count
                tail_trace2 = max(tail_trace2, min_tail_trace2)
                eff_rank = min(
                    float(tail_count),
                    max(1.0, tail_trace**2 / max(tail_trace2, 1e-15)),
                )
                tail_eig = tail_trace / eff_rank
                tail_den = sigma2 * tail_eig + eps
                zero_count = float(tail_count) - eff_rank
                logdet += eff_rank * float(np.log(tail_den))
                logdet += zero_count * float(np.log(eps))
                # Allocate unexplained target power uniformly across the omitted
                # eigenspace. The eigenvalue distribution still uses the
                # trace/trace(K^2)-matched effective rank above.
                quad += (residual_ss / tail_count) * (
                    eff_rank / tail_den + zero_count / eps
                )

        return 0.5 * float(m * logdet + quad)

    def _eig_cache_key(
        self,
        coords: torch.Tensor | np.ndarray,
        length_scale: float,
        n_modes: tuple[int, int],
    ) -> tuple:
        """Compact cache key for implicit eigendecompositions."""
        return (
            self._coords_cache_key(coords),
            float(np.round(length_scale, 12)),
            n_modes,
            self._lml_approx_rank,
            self._lml_exact_max_n,
            self._eigsh_tol,
        )

    def _get_lml_eigendecomp(
        self,
        coords: torch.Tensor | np.ndarray,
        length_scale: float,
        n_modes: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return leading eigenpairs of the unit-variance NUFFT kernel."""
        key = self._eig_cache_key(coords, length_scale, n_modes)
        if self._eig_cache is not None:
            cached_key, cached_vals, cached_vecs = self._eig_cache
            if cached_key == key:
                return cached_vals, cached_vecs

        unit_op = self._make_op(
            coords,
            constant_value=1.0,
            length_scale=length_scale,
            n_modes=n_modes,
        )
        n = unit_op.n
        exact = (
            self._lml_approx_rank is None
            or n <= self._lml_exact_max_n
            or self._lml_approx_rank >= n
        )
        if exact:
            K = unit_op.matvec(np.eye(n))
            K = 0.5 * (K + K.T)
            vals, vecs = np.linalg.eigh(K)
        else:
            rank = max(1, min(int(self._lml_approx_rank), n - 2))
            linop = LinearOperator(
                shape=(n, n),
                matvec=unit_op.matvec,
                matmat=unit_op.matvec,
                dtype=np.float64,
            )
            vals, vecs = eigsh(linop, k=rank, which="LA", tol=self._eigsh_tol)

        order = np.argsort(vals)[::-1]
        vals = np.maximum(np.asarray(vals[order], dtype=float), 0.0)
        vecs = np.asarray(vecs[:, order], dtype=float)
        self._eig_cache = (key, vals, vecs)
        return vals, vecs

    def _trace2_cache_key(
        self,
        coords: torch.Tensor | np.ndarray,
        length_scale: float,
        n_modes: tuple[int, int],
    ) -> tuple:
        """Compact cache key for unit-kernel ``trace(K^2)`` estimates."""
        return (
            self._coords_cache_key(coords),
            float(np.round(length_scale, 12)),
            n_modes,
        )

    def _get_lml_trace2(
        self,
        coords: torch.Tensor | np.ndarray,
        length_scale: float,
        n_modes: tuple[int, int],
    ) -> float:
        """Return ``trace(K^2)`` for the unit-variance periodic RBF kernel.

        For the RBF kernel, ``K(x, x')**2`` is another RBF kernel with length
        scale ``length_scale / sqrt(2)``. Summing its row sums gives
        ``sum_ij K_ij**2 = trace(K^2)`` without materializing K.
        """
        key = self._trace2_cache_key(coords, length_scale, n_modes)
        if self._trace2_cache is not None:
            cached_key, cached_trace2 = self._trace2_cache
            if cached_key == key:
                return cached_trace2

        sq_op = self._make_op(
            coords,
            constant_value=1.0,
            length_scale=length_scale / np.sqrt(2.0),
            n_modes=n_modes,
        )
        trace2 = float(np.sum(sq_op.matvec(np.ones(sq_op.n, dtype=np.float64))))
        trace2 = min(max(trace2, float(sq_op.n)), float(sq_op.n) ** 2)
        self._trace2_cache = (key, trace2)
        return trace2

    def _optimize_eigen(
        self,
        coords: torch.Tensor | np.ndarray,
        Y_centered: np.ndarray,
        n_modes: tuple[int, int],
    ) -> tuple[float, float, float]:
        """Optimize hyperparameters using NUFFT matvec eigensummaries."""
        lo_e, hi_e = self._epsilon_bounds
        ls_fixed = self._length_scale_bounds == "fixed"
        cv_fixed = self._constant_value_bounds == "fixed"
        n, m = Y_centered.shape
        total_ss = float(np.sum(Y_centered**2))
        trace = float(n)

        def summary(length_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
            evals, vecs = self._get_lml_eigendecomp(coords, length_scale, n_modes)
            coeff = vecs.T @ Y_centered
            power = np.sum(coeff**2, axis=1)
            trace2 = self._get_lml_trace2(coords, length_scale, n_modes)
            return evals, power, trace2

        if ls_fixed:
            evals0, power0, trace20 = summary(self._length_scale)

            if cv_fixed:

                def neg_lml_1d(log_eps: float) -> float:
                    return self._eigen_neg_lml(
                        evals0,
                        power0,
                        total_ss,
                        trace,
                        trace20,
                        n,
                        m,
                        self._constant_value,
                        float(np.exp(log_eps)),
                    )

                res = minimize_scalar(
                    neg_lml_1d, bounds=(np.log(lo_e), np.log(hi_e)), method="bounded"
                )
                return self._constant_value, self._length_scale, float(np.exp(res.x))

            lo_s, hi_s = self._constant_value_bounds

            def neg_lml_2d(params: np.ndarray) -> float:
                return self._eigen_neg_lml(
                    evals0,
                    power0,
                    total_ss,
                    trace,
                    trace20,
                    n,
                    m,
                    float(np.exp(params[0])),
                    float(np.exp(params[1])),
                )

            x0 = np.array([np.log(self._constant_value), np.log(np.sqrt(lo_e * hi_e))])
            bounds_lb = [
                (np.log(float(lo_s)), np.log(float(hi_s))),
                (np.log(lo_e), np.log(hi_e)),
            ]
            res = minimize(neg_lml_2d, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
            return float(np.exp(res.x[0])), self._length_scale, float(np.exp(res.x[1]))

        lo_l, hi_l = self._length_scale_bounds

        if cv_fixed:

            def neg_lml_3(params: np.ndarray) -> float:
                length_scale = float(np.exp(params[0]))
                evals, power, trace2 = summary(length_scale)
                return self._eigen_neg_lml(
                    evals,
                    power,
                    total_ss,
                    trace,
                    trace2,
                    n,
                    m,
                    self._constant_value,
                    float(np.exp(params[1])),
                )

            x0 = np.array([np.log(self._length_scale), np.log(np.sqrt(lo_e * hi_e))])
            bounds_lb = [
                (np.log(float(lo_l)), np.log(float(hi_l))),
                (np.log(lo_e), np.log(hi_e)),
            ]
            res = minimize(neg_lml_3, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
            return (
                self._constant_value,
                float(np.exp(res.x[0])),
                float(np.exp(res.x[1])),
            )

        lo_s, hi_s = self._constant_value_bounds

        def neg_lml_4(params: np.ndarray) -> float:
            length_scale = float(np.exp(params[1]))
            evals, power, trace2 = summary(length_scale)
            return self._eigen_neg_lml(
                evals,
                power,
                total_ss,
                trace,
                trace2,
                n,
                m,
                float(np.exp(params[0])),
                float(np.exp(params[2])),
            )

        x0 = np.array(
            [
                np.log(self._constant_value),
                np.log(self._length_scale),
                np.log(np.sqrt(lo_e * hi_e)),
            ]
        )
        bounds_lb = [
            (np.log(float(lo_s)), np.log(float(hi_s))),
            (np.log(float(lo_l)), np.log(float(hi_l))),
            (np.log(lo_e), np.log(hi_e)),
        ]
        res = minimize(neg_lml_4, x0=x0, method="L-BFGS-B", bounds=bounds_lb)
        return float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(np.exp(res.x[2]))

    def _fit_no_nan(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
        basis_op: Optional[NUFFTKernelOp] = None,
    ) -> torch.Tensor:
        """Inner residualization assuming no NaN rows."""
        Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)
        Y_np = Y2d.detach().cpu().numpy().astype(np.float64, copy=False)
        basis = basis_op if basis_op is not None else self._get_basis_op(coords)
        regular_fft_compatible = self._is_regular_fft_compatible(coords, basis)
        Y_fit = (
            Y_np - np.mean(Y_np, axis=0, keepdims=True)
            if regular_fft_compatible
            else Y_np
        )
        if regular_fft_compatible:
            coeff = basis.mode_coefficients(Y_fit)
            power = (np.abs(coeff) ** 2).sum(axis=0)
            sigma2, length_scale, eps = self._optimize_spectral(
                basis,
                power,
                total_ss=float(np.sum(Y_fit**2)),
                n=Y_fit.shape[0],
                m=Y_fit.shape[1],
                coords=coords,
            )
        else:
            sigma2, length_scale, eps = self._optimize_eigen(
                coords, Y_fit, basis.n_modes
            )
        signal_op = self._make_op(
            coords,
            constant_value=sigma2,
            length_scale=length_scale,
            n_modes=basis.n_modes,
        )
        res = signal_op.residuals(Y_fit, eps)
        out = torch.as_tensor(res, dtype=Y2d.dtype, device=Y2d.device)
        return out.squeeze(1) if Y.dim() == 1 else out

    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP residuals for one target matrix on irregular 2-D coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, 2)``. Normalized spatial coordinates.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor
            Shape ``(n_spots, m)``. Spatial residuals. NaN rows from ``Y`` are
            preserved as NaN.
        """
        Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)
        is_nan = torch.isnan(Y2d).any(1)
        if is_nan.any():
            coords_c = coords[~is_nan]
            Y_c = Y2d[~is_nan]
            res_clean = self._fit_no_nan(coords_c, Y_c)
            out = torch.full_like(Y2d, float("nan"))
            out[~is_nan] = res_clean
            return out.squeeze(1) if Y.dim() == 1 else out
        return self._fit_no_nan(coords, Y)

    def fit_residuals_batch(
        self,
        coords: torch.Tensor,
        Y_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Residualize multiple targets while reusing coordinates and mode grid."""
        basis = self._get_basis_op(coords)
        results = []
        for Y in Y_list:
            Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)
            is_nan = torch.isnan(Y2d).any(1)
            if is_nan.any():
                results.append(self.fit_residuals(coords, Y))
            else:
                results.append(self._fit_no_nan(coords, Y, basis_op=basis))
        return results

    def get_kernel_op(self, coords: torch.Tensor) -> NUFFTKernelOp:
        """Return a NUFFT kernel operator using the current fixed parameters."""
        return self._make_op(
            coords,
            constant_value=self._constant_value,
            length_scale=self._length_scale,
            n_modes=None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _filter_gpr_kwargs(backend: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep kwargs relevant to ``backend`` while rejecting unknown keys."""
    unknown = set(kwargs) - _KNOWN_GPR_KWARGS
    if unknown:
        raise ValueError(
            "Unsupported GPR configuration key(s): " + ", ".join(sorted(unknown)) + "."
        )

    if backend == "sklearn":
        allowed = _SKLEARN_GPR_KWARGS
    elif backend == "gpytorch":
        allowed = _GPYTORCH_GPR_KWARGS
    elif backend == "fft":
        allowed = _FFT_GPR_KWARGS
    elif backend in {"nufft", "finufft"}:
        allowed = _NUFFT_GPR_KWARGS
    else:
        allowed = set()
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_kernel_gpr(
    backend: Literal["sklearn", "gpytorch", "fft", "nufft", "finufft"] = "sklearn",
    **kwargs: Any,
) -> KernelGPR:
    """Construct a GP residualizer from a backend name.

    Parameters
    ----------
    backend : {"sklearn", "gpytorch", "fft", "nufft", "finufft"}
        Backend to use.
    **kwargs
        GP backend configuration. Common keys are ``constant_value``,
        ``constant_value_bounds``, ``length_scale``, and
        ``length_scale_bounds``.  ``n_inducing`` is used by ``"sklearn"`` and
        ``"gpytorch"`` only.  NUFFT-specific keys include ``n_modes``,
        ``max_auto_modes``, ``lml_approx_rank``, ``period_margin``, and
        conjugate-gradient / FINUFFT controls.  Backend-irrelevant known keys
        from :data:`_DEFAULT_GPR_CONFIGS` are ignored.

    Returns
    -------
    KernelGPR
        GPR residualizer instance for the specified backend.

    Raises
    ------
    ValueError
        If backend is not one of the supported options.
        If an unknown configuration key is supplied.
    """
    if backend == "sklearn":
        return SklearnKernelGPR(**_filter_gpr_kwargs(backend, kwargs))
    if backend == "gpytorch":
        return GPyTorchKernelGPR(**_filter_gpr_kwargs(backend, kwargs))
    if backend == "fft":
        return FFTKernelGPR(**_filter_gpr_kwargs(backend, kwargs))
    if backend in {"nufft", "finufft"}:
        return NUFFTKernelGPR(**_filter_gpr_kwargs(backend, kwargs))
    raise ValueError(
        f"Unknown backend '{backend}'. Choose from 'sklearn', 'gpytorch', "
        "'fft', 'nufft', or 'finufft'."
    )
