"""Kernel Gaussian Process Regression backends for SPLISOSM spatial residualization.

This module provides an abstract interface and concrete implementations for
kernel-based spatial residualization used by the SPLISOSM differential-usage
tests.  The design deliberately decouples what the kernel computes from how
the n x n matrix is (or is not) materialized, so that large-dataset backends can
be swapped in without changing the public calling interface.

Kernel-operator hierarchy
--------------------------
``SpatialKernelOp`` (abstract)
    ``DenseKernelOp``     - stores K explicitly; Cholesky solve; O(n^2) memory.
    ``FFTKernelOp``       - defined in ``hyptest_fft``; O(N log N); no matrix.
    (planned) ``LanczosKernelOp`` - CG/Lanczos for large sparse kernels.

GPR-residualizer hierarchy
---------------------------
``KernelGPR`` (abstract)
    ``SklearnKernelGPR``  - sklearn backend; dense; suitable for n <= ~10,000.
    ``GPyTorchKernelGPR`` - GPyTorch backend (optional dep); lazy tensors.

``SplisosmNP`` currently constructs ``DenseKernelOp`` / ``SklearnKernelGPR``.
Future large-dataset support is unlocked by substituting an implicit
``SpatialKernelOp`` subclass without changing the public API.

Public API
----------
``SpatialKernelOp``, ``DenseKernelOp``
``KernelGPR``, ``SklearnKernelGPR``, ``GPyTorchKernelGPR``
``make_kernel_gpr``
``linear_hsic_test``
``get_kernel_regression_residual_op``
``fit_kernel_gpr``     (backward-compat wrapper)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from splisosm.likelihood import liu_sf

__all__ = [
    # Kernel linear operators
    "SpatialKernelOp",
    "DenseKernelOp",
    # GPR residualizers
    "KernelGPR",
    "SklearnKernelGPR",
    "GPyTorchKernelGPR",
    "make_kernel_gpr",
    # HSIC utilities
    "linear_hsic_test",
    # Lower-level helpers (also exported for tests / advanced use)
    "get_kernel_regression_residual_op",
    "build_rbf_kernel",
    # Backward-compat wrapper
    "fit_kernel_gpr",
]

# ---------------------------------------------------------------------------
# Kernel linear operator abstractions
# ---------------------------------------------------------------------------


class SpatialKernelOp(ABC):
    """Abstract spatial kernel linear operator.

    Provides kernel-vector products and linear system solves without requiring
    the full n x n kernel matrix to be materialised in memory.

    Concrete subclasses
    -------------------
    ``DenseKernelOp``
        Stores K explicitly as a ``torch.Tensor``; solves via Cholesky.
        Suitable for n <= ~10,000.
    ``FFTKernelOp`` (``splisosm.hyptest_fft``)
        Operates entirely in the spectral domain; O(N log N) per operation;
        no n x n matrix formed at any point.
    (planned) ``LanczosKernelOp``
        Conjugate-gradient / Lanczos for large or sparse kernels where only
        matrix-vector products are available.

    Extending ``SplisosmNP`` to large datasets
    ------------------------------------------
    ``SplisosmNP`` currently builds ``DenseKernelOp`` instances and passes them
    to ``SklearnKernelGPR``.  To support n > 10,000, replace the operator with
    an implicit subclass (e.g. ``LanczosKernelOp``) and pair it with an
    iterative ``KernelGPR`` that only calls ``matvec``; the rest of the
    pipeline is unchanged.

    Attributes
    ----------
    n : int
        Number of data points.

    Methods
    -------
    matvec(v)
        Compute K @ v.
    solve(v, epsilon)
        Solve (K + epsilon * I) u = v.
    residuals(v, epsilon)
        Apply the kernel regression residual operator.
    eigenvalues(k)
        Return eigenvalues of K in descending order.
    trace()
        Return trace(K).
    square_trace()
        Return trace(K**2).
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of data points."""

    @abstractmethod
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute ``K @ v``.

        Parameters
        ----------
        v : torch.Tensor, shape (n,) or (n, m)
            Input vector or matrix.

        Returns
        -------
        torch.Tensor, shape (n,) or (n, m)
            Result of K @ v.
        """

    @abstractmethod
    def solve(self, v: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Solve ``(K + epsilon * I) u = v`` and return ``u``.

        Parameters
        ----------
        v : torch.Tensor, shape (n, m)
            Right-hand side vector or matrix.
        epsilon : float
            Regularization / noise level (> 0).

        Returns
        -------
        torch.Tensor, shape (n, m)
            Solution u.
        """

    def residuals(self, v: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Apply the kernel regression residual operator.

        Computes ``epsilon * (K + epsilon * I)**(-1) @ v``, i.e. the part
        of ``v`` that is not explained by the GP mean.

        Parameters
        ----------
        v : torch.Tensor, shape (n, m)
            Input vector or matrix.
        epsilon : float
            Regularization / noise level (> 0).

        Returns
        -------
        torch.Tensor, shape (n, m)
            Residual vector or matrix.
        """
        return epsilon * self.solve(v, epsilon)

    @abstractmethod
    def eigenvalues(self, k: Optional[int] = None) -> torch.Tensor:
        """Return eigenvalues of K in descending order.

        Parameters
        ----------
        k : int or None
            Number of leading eigenvalues to return. None returns all.

        Returns
        -------
        torch.Tensor, shape (k,) or (n,)
            Eigenvalues in descending order.
        """

    def trace(self) -> float:
        """Return ``trace(K)`` (sum of eigenvalues)."""
        return float(self.eigenvalues().sum())

    def square_trace(self) -> float:
        """Return ``trace(K**2)``."""
        evals = self.eigenvalues()
        return float((evals**2).sum())


class DenseKernelOp(SpatialKernelOp):
    """Dense kernel matrix wrapped as a ``SpatialKernelOp``.

    Stores the full n x n kernel matrix and solves linear systems via
    Cholesky factorization.  Suitable for n <= ~10,000.

    Parameters
    ----------
    K : torch.Tensor, shape (n, n)
        Symmetric positive semi-definite kernel matrix.

    Attributes
    ----------
    n : int
        Number of data points (inferred from K).
    """

    def __init__(self, K: torch.Tensor) -> None:
        self._K = 0.5 * (K + K.T)  # symmetrize for numerical stability
        self._chol: Optional[torch.Tensor] = None
        self._eps_cached: Optional[float] = None

    @property
    def n(self) -> int:
        return self._K.shape[0]

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self._K @ v

    def _ensure_chol(self, epsilon: float) -> None:
        if self._chol is None or self._eps_cached != epsilon:
            reg = self._K + epsilon * torch.eye(self._K.shape[0], dtype=self._K.dtype)
            self._chol = torch.linalg.cholesky(reg)
            self._eps_cached = epsilon

    def solve(self, v: torch.Tensor, epsilon: float) -> torch.Tensor:
        self._ensure_chol(epsilon)
        squeeze = v.dim() == 1
        v2 = v.unsqueeze(1) if squeeze else v
        out = torch.cholesky_solve(v2, self._chol)
        return out.squeeze(1) if squeeze else out

    def eigenvalues(self, k: Optional[int] = None) -> torch.Tensor:
        evals = torch.linalg.eigvalsh(self._K)  # ascending
        evals = evals.flip(0)  # descending
        if k is not None:
            evals = evals[:k]
        return evals

    def eigh(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(eigenvalues, eigenvectors)`` in ascending order.

        Useful for spectral operations such as the fast epsilon search in
        :func:`_kernel_residuals_from_eigdecomp`.

        Returns
        -------
        eigenvalues : torch.Tensor, shape (n,)
            Eigenvalues in ascending order.
        eigenvectors : torch.Tensor, shape (n, n)
            Corresponding orthonormal eigenvectors.
        """
        return torch.linalg.eigh(self._K)


# ---------------------------------------------------------------------------
# HSIC utilities
# ---------------------------------------------------------------------------


def linear_hsic_test(
    X: torch.Tensor, Y: torch.Tensor, centering: bool = True
) -> tuple[float, float]:
    """The linear HSIC test (multivariate RV coefficient).

    Equivalent to a multivariate extension of Pearson correlation.

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_x)
    Y : torch.Tensor, shape (n_samples, n_y)
    centering : bool
        Whether to mean-centre X and Y.

    Returns
    -------
    hsic : float
        HSIC test statistic (scaled by 1 / (n - 1)**2).
    pvalue : float
        P-value from the asymptotic chi-squared mixture distribution.
    """
    is_nan = torch.isnan(X).any(1) | torch.isnan(Y).any(1)
    X = X[~is_nan]
    Y = Y[~is_nan]

    if centering:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

    n = X.shape[0]
    eigv_th = 1e-5

    hsic_scaled = torch.norm(Y.T @ X, p="fro").pow(2)

    lambda_x = torch.linalg.eigvalsh(X.T @ X)
    lambda_x = lambda_x[lambda_x > eigv_th]
    lambda_y = torch.linalg.eigvalsh(Y.T @ Y)
    lambda_y = lambda_y[lambda_y > eigv_th]

    lambda_xy = (lambda_x.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1)
    pval = liu_sf((hsic_scaled * n).numpy(), lambda_xy.numpy())

    return float(hsic_scaled / (n - 1) ** 2), pval


# ---------------------------------------------------------------------------
# Lower-level kernel helpers
# ---------------------------------------------------------------------------


def build_rbf_kernel(
    X: torch.Tensor,
    constant_value: float,
    length_scale: float,
) -> torch.Tensor:
    """Build a ``ConstantKernel x RBF`` kernel matrix.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        Normalized input coordinates.
    constant_value : float
        Signal variance.
    length_scale : float
        RBF length scale.

    Returns
    -------
    torch.Tensor, shape (n, n)
        Symmetric positive semi-definite kernel matrix.
    """
    kernel = C(constant_value, "fixed") * RBF(length_scale, "fixed")
    return torch.from_numpy(kernel(X.numpy())).float()


def get_kernel_regression_residual_op(Kx: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute the regularized kernel regression residual operator.

    Returns ``R = epsilon * (K + epsilon * I)**(-1)`` as a dense matrix.

    Parameters
    ----------
    Kx : torch.Tensor, shape (n, n)
        Symmetric positive semi-definite kernel matrix.
    epsilon : float
        Regularization / noise level (> 0).

    Returns
    -------
    Rx : torch.Tensor, shape (n, n)
        Residual operator ``R`` such that ``R @ Y`` gives spatial residuals of ``Y``.
    """
    Kx = 0.5 * (Kx + Kx.T)
    return epsilon * torch.linalg.inv(Kx + epsilon * torch.eye(Kx.shape[0]))


def _kernel_residuals_from_eigdecomp(
    eigvecs: torch.Tensor,
    eigvals: torch.Tensor,
    Y: torch.Tensor,
    epsilon_bounds: tuple[float, float] = (1e-5, 1e1),
) -> torch.Tensor:
    """Fast kernel-regression residuals using a precomputed eigendecomposition.

    Finds the optimal white-noise ``epsilon`` per target via 1-D
    log-marginal-likelihood maximization (no matrix factorization), then
    applies ``R = epsilon * (K + epsilon * I)**(-1)`` spectrally.

    Complexity is O(n^2) per call once eigendecomposition is precomputed,
    compared to O(n^3) for a full GP fit.

    Parameters
    ----------
    eigvecs : torch.Tensor, shape (n, n)
        Eigenvectors of the base kernel, ascending eigenvalue order.
    eigvals : torch.Tensor, shape (n,)
        Eigenvalues, ascending order.
    Y : torch.Tensor, shape (n, m)
        Target data (no NaN values).
    epsilon_bounds : tuple[float, float]
        Log-space search bounds for the noise level.

    Returns
    -------
    torch.Tensor, shape (n, m)
        Spatial residuals of Y.
    """
    eigvals_np = eigvals.numpy()
    alpha = eigvecs.T @ Y  # (n, m)
    alpha_sq = (alpha**2).sum(dim=1).numpy()  # sum over output dims

    def neg_lml(log_eps: float) -> float:
        eps = float(np.exp(log_eps))
        lam_eps = eigvals_np + eps
        return 0.5 * (np.sum(np.log(lam_eps)) + np.sum(alpha_sq / lam_eps))

    result = minimize_scalar(
        neg_lml,
        bounds=(np.log(epsilon_bounds[0]), np.log(epsilon_bounds[1])),
        method="bounded",
    )
    epsilon = float(np.exp(result.x))

    scale = epsilon / (eigvals + epsilon)  # (n,)
    return eigvecs @ (alpha * scale.unsqueeze(1))  # (n, m)


# ---------------------------------------------------------------------------
# Abstract KernelGPR base class
# ---------------------------------------------------------------------------


class KernelGPR(ABC):
    """Abstract GPR residualizer: learns a spatial kernel and returns residuals.

    Subclasses determine how hyperparameters are optimized (MLE, variational,
    fixed) and whether the kernel matrix is materialized (dense) or kept
    implicit.

    All subclasses expose the same ``fit_residuals`` interface so that the
    DU-test pipeline in ``SplisosmNP`` and ``SplisosmFFT`` is backend-agnostic.

    Methods
    -------
    fit_residuals(coords, Y)
        Fit GP and return spatial residuals for a single target matrix.
    fit_residuals_batch(coords, Y_list)
        Fit residuals for multiple targets sharing the same coordinates.
    get_kernel_op(coords)
        Build the kernel operator for given coordinates.
    """

    @abstractmethod
    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP to (coords, Y) and return spatial residuals.

        Parameters
        ----------
        coords : torch.Tensor, shape (n, d)
            Spatial coordinates (should be pre-normalized by the caller).
        Y : torch.Tensor, shape (n, m)
            Target values (may contain NaN rows, which are handled internally).

        Returns
        -------
        residuals : torch.Tensor, shape (n, m)
            Residuals Y - GP_smooth(Y | coords). NaN rows in Y are preserved as NaN.
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
        coords : torch.Tensor, shape (n, d)
            Spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape (n, m_i).

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape (n, m_i).
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
        coords : torch.Tensor, shape (n, d)
            Spatial coordinates.

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

_DEFAULT_GPR_CONFIGS = {
    "covariate": {
        "constant_value": 1.0,
        "constant_value_bounds": (1e-3, 1e3),
        "length_scale": 1.0,
        "length_scale_bounds": "fixed",
    },
    "isoform": {
        "constant_value": 1e-3,
        "constant_value_bounds": "fixed",
        "length_scale": 1.0,
        "length_scale_bounds": "fixed",
    },
}


class SklearnKernelGPR(KernelGPR):
    """sklearn-based GPR residualizer.

    Uses ``GaussianProcessRegressor`` with a ``ConstantKernel x RBF +
    WhiteKernel`` to learn spatial hyperparameters and residualize targets.

    When both signal bounds are ``"fixed"`` (:attr:`signal_bounds_fixed`),
    the base kernel matrix is the same for every target sharing the same
    coordinates.  Calling :meth:`precompute_shared_kernel` once before a
    loop over many targets reduces per-target cost from O(n^3) GP fitting
    to O(n^2) spectral operations (see :func:`_kernel_residuals_from_eigdecomp`).

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

    Attributes
    ----------
    signal_bounds_fixed : bool
        True when both constant_value and length_scale bounds are fixed.

    Notes
    -----
    This backend always materialises a dense n x n kernel matrix.  For
    n > ~10,000, consider ``GPyTorchKernelGPR`` with inducing points.
    """

    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
        length_scale: float = 1.0,
        length_scale_bounds: Union[tuple, str] = "fixed",
    ) -> None:
        self._constant_value = constant_value
        self._constant_value_bounds = constant_value_bounds
        self._length_scale = length_scale
        self._length_scale_bounds = length_scale_bounds

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
            ``constant_value_bounds``, ``length_scale``, ``length_scale_bounds``.

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
        O(n^2) rather than O(n^3).

        Has no effect (and logs a warning) when ``signal_bounds_fixed`` is
        False.

        Parameters
        ----------
        coords : torch.Tensor, shape (n, d)
            Normalized spatial coordinates, shape (n, d).

        Returns
        -------
        None
        """
        if not self.signal_bounds_fixed:
            warnings.warn(
                "precompute_shared_kernel() has no effect when signal bounds "
                "are not fixed; the full GP will still be fitted per target.",
                stacklevel=2,
            )
            return
        K = build_rbf_kernel(coords, self._constant_value, self._length_scale)
        K = 0.5 * (K + K.T)
        eigvals, eigvecs = torch.linalg.eigh(K)  # ascending
        self._shared_eigvals = eigvals
        self._shared_eigvecs = eigvecs

    def get_kernel_op(self, coords: torch.Tensor) -> DenseKernelOp:
        """Return a ``DenseKernelOp`` built from the current (fixed) params.

        Parameters
        ----------
        coords : torch.Tensor, shape (n, d)
            Normalized spatial coordinates.

        Returns
        -------
        DenseKernelOp
            Kernel operator for the given coordinates.
        """
        K = build_rbf_kernel(coords, self._constant_value, self._length_scale)
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
        back to a full sklearn GP fit.

        NaN rows in ``Y`` are excluded from fitting and reinserted as NaN
        in the output.

        Parameters
        ----------
        coords : torch.Tensor, shape (n, d)
            Normalized spatial coordinates.
        Y : torch.Tensor, shape (n, m)
            Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor, shape (n, m)
            Spatial residuals. NaN rows from Y are preserved as NaN.
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

    def _fit_no_nan(self, coords: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Inner residualization without NaN handling."""
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
        coords : torch.Tensor, shape (n, d)
            Normalized spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape (n, m_i).

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape (n, m_i).
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


class GPyTorchKernelGPR(KernelGPR):
    """GPyTorch-based GP residualizer.

    Uses GPyTorch's lazy-tensor infrastructure so that the n x n kernel
    matrix is never explicitly materialised; all computations operate via
    kernel-vector products.

    For n > ~5,000, pass ``n_inducing > 0`` to use sparse variational GP
    (SVGP), reducing complexity from O(n^3) to O(m^2 n) where m = n_inducing.

    Parameters
    ----------
    n_inducing : int or None
        Number of inducing points for SVGP.  ``None`` (default) uses exact GP.
    n_iter : int
        Optimiser iterations for hyperparameter learning.
    lr : float
        Adam learning rate.
    device : str
        PyTorch device (``"cpu"`` or ``"cuda"``).

    Notes
    -----
    Requires ``gpytorch`` (optional dependency)::

        pip install gpytorch

    The exact-GP path is already more memory-efficient than the sklearn
    backend because GPyTorch uses Lanczos-based CG rather than dense Cholesky.
    The sparse-SVGP path (``n_inducing > 0``) will be implemented in a future
    release.  The n x n matrix is never formed.
    """

    def __init__(
        self,
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
        self.n_inducing = n_inducing
        self.n_iter = n_iter
        self.lr = lr
        self.device = device

    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP residuals using GPyTorch.

        Parameters
        ----------
        coords : torch.Tensor, shape (n, d)
            Normalized spatial coordinates.
        Y : torch.Tensor, shape (n, m)
            Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor, shape (n, m)
            Spatial residuals.

        Notes
        -----
        The full (exact-GP) implementation is planned for the next
        release.  The sparse-SVGP path (``n_inducing > 0``) will follow.
        """
        raise NotImplementedError(
            "GPyTorchKernelGPR.fit_residuals() is not yet implemented.  "
            "Use SklearnKernelGPR for now, or contribute the GPyTorch "
            "backend via a pull request."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_kernel_gpr(
    backend: Literal["sklearn", "gpytorch", "fft"] = "sklearn",
    **kwargs: Any,
) -> KernelGPR:
    """Construct a KernelGPR from a backend name.

    Parameters
    ----------
    backend : {"sklearn", "gpytorch", "fft"}
        Backend to use.  ``"fft"`` requires passing a ``kernel`` keyword
        argument of type :class:`~splisosm.hyptest_fft.FFTKernel`.
    **kwargs
        Passed to the backend constructor.

    Returns
    -------
    KernelGPR
        GPR residualizer instance for the specified backend.

    Raises
    ------
    ValueError
        If backend is not one of the supported options.
    """
    if backend == "sklearn":
        return SklearnKernelGPR(**kwargs)
    if backend == "gpytorch":
        return GPyTorchKernelGPR(**kwargs)
    if backend == "fft":
        from splisosm.hyptest_fft import FFTKernelGPR  # lazy to avoid circular import

        return FFTKernelGPR(**kwargs)
    raise ValueError(
        f"Unknown backend '{backend}'. Choose from 'sklearn', 'gpytorch', 'fft'."
    )


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------


def fit_kernel_gpr(
    X: torch.Tensor,
    Y: torch.Tensor,
    normalize_x: bool = True,
    normalize_y: bool = True,
    return_residuals: bool = True,
    constant_value: float = 1.0,
    constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
    length_scale: float = 1.0,
    length_scale_bounds: Union[tuple, str] = (1e-2, 1e2),
) -> Union[tuple[torch.Tensor, float], torch.Tensor]:
    """Fit a Gaussian process regression and optionally return residuals.

    .. deprecated::
        Use :class:`SklearnKernelGPR` directly.  This wrapper is kept for
        backward compatibility and will be removed in a future release.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        Input spatial coordinates.
    Y : torch.Tensor, shape (n, m)
        Target values.
    normalize_x : bool
        Whether to z-score normalize X before fitting.
    normalize_y : bool
        Whether to z-score normalize Y before fitting.
    return_residuals : bool
        If True, return residuals; if False, return ``(Kxy, epsilon)``.
    constant_value : float
        Initial signal amplitude.
    constant_value_bounds : tuple or ``"fixed"``
        Search bounds for the signal amplitude.
    length_scale : float
        Initial RBF length scale.
    length_scale_bounds : tuple or ``"fixed"``
        Search bounds for the length scale.

    Returns
    -------
    tuple[torch.Tensor, float] or torch.Tensor
        If ``return_residuals`` is False: ``(Kxy, epsilon)`` kernel matrix and
        noise level.  If ``return_residuals`` is True: residual tensor of shape
        (n, m).
    """
    n_orig = Y.shape[0]
    is_nan = torch.isnan(Y).any(1)
    X_ = X[~is_nan]
    Y_ = Y[~is_nan]

    if normalize_x:
        X_ = (X_ - X_.mean(0)) / X_.std(0)
        X_[torch.isinf(X_)] = 0.0
    if normalize_y:
        Y_ = (Y_ - Y_.mean(0)) / Y_.std(0)
        Y_[torch.isinf(Y_)] = 0.0

    x_np = X_.numpy()
    kernel = C(constant_value, constant_value_bounds) * RBF(
        length_scale, length_scale_bounds
    ) + WhiteKernel(0.1, (1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_np, Y_.numpy())

    Kxy = torch.from_numpy(gp.kernel_.k1(x_np, x_np)).float()
    epsilon = float(np.exp(gp.kernel_.theta[-1]))

    if not return_residuals:
        return Kxy, epsilon

    Kxy = 0.5 * (Kxy + Kxy.T)
    Rx = epsilon * torch.linalg.inv(Kxy + epsilon * torch.eye(Kxy.shape[0]))
    res = Rx @ Y_

    if n_orig == Y_.shape[0]:
        return res

    out = torch.full((n_orig, res.shape[1]), float("nan"))
    out[~is_nan] = res
    return out
