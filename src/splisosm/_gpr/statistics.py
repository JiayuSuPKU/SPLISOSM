"""HSIC and dense RBF helper functions for GPR backends."""

from __future__ import annotations

import numpy as np
import scipy.sparse
import torch
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF

from splisosm.likelihood import liu_sf

__all__ = [
    "linear_hsic_test",
    "_build_rbf_kernel",
    "_build_rbf_cross_kernel",
    "_kernel_residuals_from_eigdecomp",
]

# ---------------------------------------------------------------------------
# HSIC utilities
# ---------------------------------------------------------------------------


def linear_hsic_test(
    X: "torch.Tensor | scipy.sparse.spmatrix",
    Y: "torch.Tensor | scipy.sparse.spmatrix",
    centering: bool = True,
) -> tuple[float, float]:
    """The linear HSIC test (multivariate RV coefficient).

    Equivalent to a multivariate extension of Pearson correlation.

    Supports sparse inputs for memory and speed efficiency:

    * **Sparse X** (scipy sparse matrix, shape ``(n, p)``): the cross-product
      ``Y_c.T @ X_c`` is computed via a sparse matrix multiply
      (``X.T.dot(Y_c)``).  Because ``Y`` is mean-centred, the ``X`` centering
      correction reduces to zero, so only the original sparse ``X`` is needed.
      ``X_c.T @ X_c`` is obtained as ``X.T @ X  -  n * mean_X ⊗ mean_X``,
      keeping the first term sparse.
    * **Sparse Y** (scipy sparse or torch sparse COO): densified once upfront
      before any computation.

    Parameters
    ----------
    X : torch.Tensor or scipy.sparse.spmatrix, shape (n_samples, n_x)
    Y : torch.Tensor or scipy.sparse.spmatrix, shape (n_samples, n_y)
    centering : bool
        Whether to mean-centre X and Y before computing the statistic.

    Returns
    -------
    hsic : float
        HSIC test statistic (scaled by 1 / (n - 1)**2).
    pvalue : float
        P-value from the asymptotic chi-squared mixture distribution.
    """
    eigv_th = 1e-5

    # ── Normalise Y to a dense float32 tensor ─────────────────────────────
    if scipy.sparse.issparse(Y):
        Y = torch.from_numpy(np.asarray(Y.todense(), dtype=np.float32))
    elif isinstance(Y, torch.Tensor) and Y.is_sparse:
        Y = Y.to_dense().float()
    elif isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y.astype(np.float32))
    else:
        Y = Y.float()

    X_is_sparse = scipy.sparse.issparse(X)

    # ── NaN filtering ─────────────────────────────────────────────────────
    is_nan_y = torch.isnan(Y).any(1)
    if X_is_sparse:
        # Sparse matrices contain no NaN by construction.
        is_nan = is_nan_y
    else:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        else:
            X = X.float()
        is_nan = is_nan_y | torch.isnan(X).any(1)

    if is_nan.any():
        keep = ~is_nan
        Y = Y[keep]
        if X_is_sparse:
            X = X[keep.numpy()]
        else:
            X = X[keep]

    n = Y.shape[0]

    # ── Sparse-X path ─────────────────────────────────────────────────────
    if X_is_sparse:
        X_sp = X.tocsr()
        if centering:
            Y = Y - Y.mean(0)

        # Key identity: when Y is centred, Y_c.T @ X_c == Y_c.T @ X
        # because the correction term Y_c.sum(0) == 0 vanishes.
        # So we can use the original (non-centred) sparse X directly.
        YcTX = torch.from_numpy(
            X_sp.T.dot(Y.numpy()).astype(np.float32)
        ).T  # (n_y, n_x)
        hsic_scaled = YcTX.pow(2).sum()

        # X_c.T @ X_c = X.T @ X - n * mean_X.outer(mean_X)
        X_mean = np.asarray(X_sp.mean(axis=0), dtype=np.float32).ravel()
        XcTXc = torch.from_numpy(
            X_sp.T.dot(X_sp).toarray().astype(np.float32) - n * np.outer(X_mean, X_mean)
        )
        lambda_x = torch.linalg.eigvalsh(XcTXc)
        lambda_x = lambda_x[lambda_x > eigv_th]
        lambda_y = torch.linalg.eigvalsh(Y.T @ Y)
        lambda_y = lambda_y[lambda_y > eigv_th]

    # ── Dense-X path (original) ───────────────────────────────────────────
    else:
        if centering:
            X = X - X.mean(0)
            Y = Y - Y.mean(0)

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
