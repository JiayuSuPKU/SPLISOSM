"""Dense sklearn GP residualization backend."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from splisosm.gpr.base import KernelGPR
from splisosm.gpr.operators import DenseKernelOp

__all__ = [
    "SklearnKernelGPR",
    "_build_rbf_kernel",
    "_build_rbf_cross_kernel",
    "_kernel_residuals_from_eigdecomp",
]


def _build_rbf_kernel(
    X: torch.Tensor,
    constant_value: float,
    length_scale: float,
) -> torch.Tensor:
    """Build a ``ConstantKernel x RBF`` kernel matrix.

    Parameters
    ----------
    X : torch.Tensor, shape (n_obs, d)
        Normalized input coordinates.
    constant_value : float
        Signal variance.
    length_scale : float
        RBF length scale.

    Returns
    -------
    torch.Tensor, shape (n_obs, n_obs)
        Symmetric positive semi-definite kernel matrix.
    """
    kernel = C(constant_value, "fixed") * RBF(length_scale, "fixed")
    return torch.from_numpy(kernel(X.numpy())).float()


def _build_rbf_cross_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    constant_value: float,
    length_scale: float,
) -> torch.Tensor:
    """Cross-covariance matrix K(X1, X2).

    Parameters
    ----------
    X1 : torch.Tensor, shape (n1, d)
    X2 : torch.Tensor, shape (n2, d)
    constant_value : float
        Signal variance.
    length_scale : float
        RBF length scale.

    Returns
    -------
    torch.Tensor, shape (n1, n2)
    """
    kernel = C(constant_value, "fixed") * RBF(length_scale, "fixed")
    return torch.from_numpy(kernel(X1.numpy(), X2.numpy())).float()


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
    eigvecs : torch.Tensor, shape (n_obs, n_obs)
        Eigenvectors of the base kernel, ascending eigenvalue order.
    eigvals : torch.Tensor, shape (n_obs,)
        Eigenvalues, ascending order.
    Y : torch.Tensor, shape (n_obs, m)
        Target data (no NaN values).
    epsilon_bounds : tuple[float, float]
        Log-space search bounds for the noise level.

    Returns
    -------
    torch.Tensor, shape (n_obs, m)
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
