"""Kernel Gaussian Process Regression backends for SPLISOSM spatial residualization.

This module provides an abstract interface and concrete implementations for
kernel-based conditional independence testing (via spatial residualization)
used by the SPLISOSM differential-usage tests.

GPR-residualizer hierarchy
---------------------------
``KernelGPR`` (abstract):

    - ``SklearnKernelGPR``  - sklearn backend; dense; suitable for n <= ~10,000.
    - ``GPyTorchKernelGPR`` - GPyTorch backend (optional dep); lazy tensors.
    - ``FFTKernelGPR``      - FFT-based kernel; suitable for regular grids.
    - ``NUFFTKernelGPR``    - FINUFFT-based kernel; suitable for irregular 2-D coordinates.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
import os
import sys
from typing import Any, Literal, Optional, Union

import numpy as np
import scipy.fft
import scipy.sparse
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator, cg, eigsh

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from splisosm.likelihood import liu_sf

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

    Extending ``SplisosmNP`` to large datasets
    ------------------------------------------
    ``SplisosmNP`` currently builds ``DenseKernelOp`` instances and passes them
    to ``SklearnKernelGPR``.  To support n > 10,000, replace the operator with
    an implicit subclass (e.g. ``LanczosKernelOp``) and pair it with an
    iterative ``KernelGPR`` that only calls ``matvec``; the rest of the
    pipeline is unchanged.
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
        v : torch.Tensor, shape (n_obs,) or (n_obs, m)
            Input vector or matrix.

        Returns
        -------
        torch.Tensor, shape (n_obs,) or (n_obs, m)
            Result of K @ v.
        """

    @abstractmethod
    def solve(self, v: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Solve ``(K + epsilon * I) u = v`` and return ``u``.

        Parameters
        ----------
        v : torch.Tensor, shape (n_obs, m)
            Right-hand side vector or matrix.
        epsilon : float
            Regularization / noise level (> 0).

        Returns
        -------
        torch.Tensor, shape (n_obs, m)
            Solution u.
        """

    def residuals(self, v: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Apply the kernel regression residual operator.

        Computes ``epsilon * (K + epsilon * I)**(-1) @ v``, i.e. the part
        of ``v`` that is not explained by the GP mean.

        Parameters
        ----------
        v : torch.Tensor, shape (n_obs, m)
            Input vector or matrix.
        epsilon : float
            Regularization / noise level (> 0).

        Returns
        -------
        torch.Tensor, shape (n_obs, m)
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
        torch.Tensor, shape (k,) or (n_obs,)
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
    K : torch.Tensor, shape (n_obs, n_obs)
        Symmetric positive semi-definite kernel matrix.
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
        eigenvalues : torch.Tensor, shape (n_obs,)
            Eigenvalues in ascending order.
        eigenvectors : torch.Tensor, shape (n_obs, n_obs)
            Corresponding orthonormal eigenvectors.
        """
        return torch.linalg.eigh(self._K)


# ---------------------------------------------------------------------------
# FFT kernel operator (RBF, regular grid)
# ---------------------------------------------------------------------------


class FFTKernelOp(SpatialKernelOp):
    """Implicit RBF kernel linear operator on a periodic 2-D grid via FFT.

    Eigenvalues are computed as ``FFT2(K_row)`` where
    ``K_row[i, j] = sigma^2 * exp(-(di^2 + dj^2) / (2 * l^2))`` with
    periodic (torus) distances — no ``/n_grid`` normalization.  This
    matches the convention used by :class:`DenseKernelOp` and
    :class:`SklearnKernelGPR`.

    All linear-algebra operations are O(N log N) via FFT; the n x n kernel
    matrix is never formed.

    Parameters
    ----------
    ny, nx : int
        Grid shape.
    dy, dx : float
        Physical spacing between neighboring grid cells (used to compute
        torus distances).
    constant_value : float
        RBF signal variance sigma^2.
    length_scale : float
        RBF length scale (in the same units as dy/dx).
    workers : int or None
        Number of ``scipy.fft`` workers.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        dy: float,
        dx: float,
        constant_value: float,
        length_scale: float,
        workers: Optional[int] = None,
    ) -> None:
        self._ny = int(ny)
        self._nx = int(nx)
        self._dy = float(dy)
        self._dx = float(dx)
        self._constant_value = float(constant_value)
        self._length_scale = float(length_scale)
        self._workers = workers
        self._eigenvalues_2d: Optional[np.ndarray] = None  # lazy

    @property
    def ny(self) -> int:
        """Grid height."""
        return self._ny

    @property
    def nx(self) -> int:
        """Grid width."""
        return self._nx

    @property
    def n(self) -> int:
        """Total grid cells ny * nx."""
        return self._ny * self._nx

    @property
    def eigenvalues_2d(self) -> np.ndarray:
        """2-D eigenvalue array, shape (ny, nx). Computed lazily."""
        if self._eigenvalues_2d is None:
            y = np.arange(self._ny, dtype=float) * self._dy
            x = np.arange(self._nx, dtype=float) * self._dx
            # Periodic (torus) distances from origin
            y = np.minimum(y, self._ny * self._dy - y)
            x = np.minimum(x, self._nx * self._dx - x)
            yy, xx = np.meshgrid(y, x, indexing="ij")
            K_row = self._constant_value * np.exp(
                -(yy**2 + xx**2) / (2.0 * self._length_scale**2)
            )
            lam = np.real(scipy.fft.fft2(K_row, workers=self._workers))
            # Clip negligible numerical negatives (RBF kernel is PSD)
            self._eigenvalues_2d = np.maximum(lam, 0.0)
        return self._eigenvalues_2d

    def _to_cube(self, v: np.ndarray) -> tuple[np.ndarray, bool]:
        """Reshape flat/2-D input to (ny, nx, m) cube."""
        was_1d = v.ndim == 1
        if was_1d:
            return v.reshape(self._ny, self._nx)[..., np.newaxis], True
        return v.reshape(self._ny, self._nx, v.shape[1]), False

    def _from_cube(self, cube: np.ndarray, was_1d: bool) -> np.ndarray:
        if was_1d:
            return cube[..., 0].ravel()
        return cube.reshape(self.n, -1)

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute ``K @ v`` in O(N log N).

        Parameters
        ----------
        v : np.ndarray, shape (n_obs,) or (n_obs, m)

        Returns
        -------
        np.ndarray, same shape as v.
        """
        cube, was_1d = self._to_cube(np.asarray(v, dtype=float))
        v_hat = scipy.fft.fft2(cube, axes=(0, 1), workers=self._workers)
        kv_hat = v_hat * self.eigenvalues_2d[:, :, np.newaxis]
        kv = np.real(scipy.fft.ifft2(kv_hat, axes=(0, 1), workers=self._workers))
        return self._from_cube(kv, was_1d)

    def solve(self, v: np.ndarray, epsilon: float) -> np.ndarray:
        """Solve ``(K + epsilon * I) u = v`` in O(N log N).

        Parameters
        ----------
        v : np.ndarray, shape (n_obs,) or (n_obs, m)
        epsilon : float
            Regularization level (> 0).

        Returns
        -------
        np.ndarray, same shape as v.
        """
        cube, was_1d = self._to_cube(np.asarray(v, dtype=float))
        v_hat = scipy.fft.fft2(cube, axes=(0, 1), workers=self._workers)
        scale = 1.0 / (self.eigenvalues_2d[:, :, np.newaxis] + epsilon)
        u = np.real(scipy.fft.ifft2(scale * v_hat, axes=(0, 1), workers=self._workers))
        return self._from_cube(u, was_1d)

    def residuals(self, v: np.ndarray, epsilon: float) -> np.ndarray:
        """Apply ``epsilon * (K + epsilon * I)^{-1} @ v`` in O(N log N).

        Parameters
        ----------
        v : np.ndarray, shape (n_obs,) or (n_obs, m)
        epsilon : float

        Returns
        -------
        np.ndarray, same shape as v.
        """
        cube, was_1d = self._to_cube(np.asarray(v, dtype=float))
        v_hat = scipy.fft.fft2(cube, axes=(0, 1), workers=self._workers)
        scale = epsilon / (self.eigenvalues_2d[:, :, np.newaxis] + epsilon)
        r = np.real(scipy.fft.ifft2(scale * v_hat, axes=(0, 1), workers=self._workers))
        return self._from_cube(r, was_1d)

    def eigenvalues(self, k: Optional[int] = None) -> np.ndarray:
        """Return eigenvalues in descending order.

        Parameters
        ----------
        k : int or None
            Number of leading eigenvalues. ``None`` returns all.

        Returns
        -------
        np.ndarray, shape (k,) or (n_obs,).
        """
        evals = np.sort(self.eigenvalues_2d.ravel())[::-1]
        return evals[:k] if k is not None else evals

    def trace(self) -> float:
        """Return ``trace(K)``."""
        return float(self.eigenvalues_2d.sum())

    def square_trace(self) -> float:
        """Return ``trace(K^2)``."""
        return float((self.eigenvalues_2d**2).sum())


# ---------------------------------------------------------------------------
# NUFFT kernel operator (RBF, irregular 2-D coordinates)
# ---------------------------------------------------------------------------

_NUFFT_MACOS_THREAD_WARNED = False


def _require_finufft() -> Any:
    """Import FINUFFT lazily with a backend-specific error message."""
    if sys.platform == "darwin":
        # PyTorch and FINUFFT wheels can load separate libomp.dylib copies on
        # macOS conda environments.  FINUFFT aborts at the first transform
        # unless this compatibility switch is present before OpenMP init.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import finufft
    except ImportError as exc:
        raise ImportError(
            "NUFFTKernelGPR requires the optional 'finufft' package. "
            "Install it with `pip install finufft` or SPLISOSM's GP extra."
        ) from exc
    return finufft


def _safe_finufft_workers(workers: Optional[int]) -> int:
    """Return a FINUFFT thread count that is safe with PyTorch on macOS."""
    global _NUFFT_MACOS_THREAD_WARNED
    if workers is not None and int(workers) < 1:
        raise ValueError("workers must be None or a positive integer.")
    if sys.platform == "darwin" and workers not in (None, 1):
        if not _NUFFT_MACOS_THREAD_WARNED:
            warnings.warn(
                "FINUFFT with nthreads > 1 can segfault on macOS when PyTorch "
                "and other OpenMP-linked packages are loaded. Falling back to "
                "one FINUFFT thread; use process-level parallelism instead.",
                RuntimeWarning,
                stacklevel=3,
            )
            _NUFFT_MACOS_THREAD_WARNED = True
        return 1
    return 1 if workers is None else int(workers)


def _as_numpy_coords_2d(coords: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return coordinates as a float64 ``(n, 2)`` NumPy array."""
    if isinstance(coords, torch.Tensor):
        arr = coords.detach().cpu().numpy()
    else:
        arr = np.asarray(coords)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "NUFFTKernelGPR requires 2-D coordinates with shape (n_obs, 2); "
            f"got shape {arr.shape}."
        )
    return arr


def _regular_grid_shape_spacing(
    coords: np.ndarray,
) -> Optional[tuple[int, int, float, float]]:
    """Return ``(ny, nx, dy, dx)`` when coordinates form a complete grid."""
    uy = np.unique(coords[:, 0])
    ux = np.unique(coords[:, 1])
    ny, nx = len(uy), len(ux)
    if ny * nx != coords.shape[0]:
        return None

    dy = float(np.diff(uy).mean()) if ny > 1 else 1.0
    dx = float(np.diff(ux).mean()) if nx > 1 else 1.0
    if ny > 1 and not np.allclose(np.diff(uy), dy, rtol=1e-4, atol=1e-8):
        return None
    if nx > 1 and not np.allclose(np.diff(ux), dx, rtol=1e-4, atol=1e-8):
        return None
    return ny, nx, dy, dx


def _auto_nufft_modes(
    n_obs: int,
    span: np.ndarray,
    period: np.ndarray,
    max_auto_modes: Optional[int] = None,
) -> tuple[int, int]:
    """Choose a full effective Fourier grid from point count and aspect ratio."""
    if max_auto_modes is not None and int(max_auto_modes) < 16:
        raise ValueError("max_auto_modes must be None or at least 16.")

    span = np.maximum(np.asarray(span, dtype=float), 1e-12)
    period = np.maximum(np.asarray(period, dtype=float), 1e-12)
    cell = np.sqrt(float(np.prod(span)) / max(int(n_obs), 16))
    modes = np.maximum(4, np.ceil(period / cell).astype(int))

    if max_auto_modes is not None:
        max_modes = int(max_auto_modes)
        mode_product = int(modes[0] * modes[1])
        if mode_product > max_modes:
            scale = np.sqrt(max_modes / mode_product)
            modes = np.maximum(4, np.floor(modes * scale).astype(int))
            # Rounding both axes down can still leave the product just above the
            # requested cap. Trim the larger axis while preserving the minimum
            # valid FINUFFT mode shape.
            while int(modes[0] * modes[1]) > max_modes:
                axis = int(np.argmax(modes))
                modes[axis] -= 1
                if modes[axis] < 4:
                    raise ValueError(
                        "max_auto_modes is too small for a valid 2-D NUFFT grid."
                    )

    return int(modes[0]), int(modes[1])


def _normalize_nufft_modes(
    n_modes: Optional[int | tuple[int, int]],
    n_obs: int,
    span: np.ndarray,
    period: np.ndarray,
    max_auto_modes: Optional[int],
) -> tuple[int, int]:
    """Validate user-supplied NUFFT mode shape or choose one automatically."""
    if n_modes is None:
        return _auto_nufft_modes(n_obs, span, period, max_auto_modes)
    if isinstance(n_modes, (int, np.integer)):
        modes = (int(n_modes), int(n_modes))
    else:
        if len(n_modes) != 2:
            raise ValueError("n_modes must be an int or a length-2 tuple.")
        modes = (int(n_modes[0]), int(n_modes[1]))
    if modes[0] < 4 or modes[1] < 4:
        raise ValueError("n_modes entries must be at least 4.")
    return modes


class NUFFTKernelOp(SpatialKernelOp):
    """Implicit periodic RBF kernel on irregular 2-D coordinates via FINUFFT.

    The operator approximates a squared-exponential kernel with periodic
    boundary conditions over the observed coordinate bounding box.  Matrix-vector
    products use two FINUFFT calls:

    ``nonuniform points -> Fourier modes -> nonuniform points``.

    The dense ``n_obs x n_obs`` kernel matrix is never formed.  Linear solves use
    conjugate gradients against the implicit operator.

    Parameters
    ----------
    coords : torch.Tensor or np.ndarray, shape (n_obs, 2)
        Spatial coordinates.  Units match ``length_scale``.
    constant_value : float
        RBF signal variance sigma².
    length_scale : float
        RBF length scale in the same coordinate units.
    n_modes : int, tuple[int, int], or None
        Fourier mode grid.  ``None`` chooses the full effective grid from the
        observed coordinate density, with aspect ratio matching the padded
        periodic box.  For example, roughly square data with 1M spots and no
        padding uses about ``(1000, 1000)`` modes before FINUFFT's internal
        oversampling.
    max_auto_modes : int or None
        Optional cap on the total automatically chosen Fourier modes.  Ignored
        when ``n_modes`` is explicit.  ``None`` uses the full effective grid.
    nufft_eps : float
        FINUFFT requested precision.
    nufft_opts : dict or None
        Extra FINUFFT options, such as ``{"upsampfac": 2.0}``. Thread count is
        controlled by ``workers`` instead of ``nufft_opts["nthreads"]``.
    period_margin : float
        Fractional padding added to each side of the coordinate bounding box
        before wrapping onto ``[-pi, pi)``.
    cg_rtol : float
        Relative tolerance for conjugate-gradient solves.
    cg_maxiter : int or None
        Maximum CG iterations.
    workers : int or None
        FINUFFT thread count passed as ``nthreads``.  ``None`` uses one
        FINUFFT thread, which avoids OpenMP runtime conflicts on common macOS
        PyTorch/conda stacks.
    """

    def __init__(
        self,
        coords: torch.Tensor | np.ndarray,
        constant_value: float,
        length_scale: float,
        n_modes: Optional[int | tuple[int, int]] = None,
        max_auto_modes: Optional[int] = None,
        nufft_eps: float = 1e-6,
        nufft_opts: Optional[dict[str, Any]] = None,
        period_margin: float = 0.5,
        cg_rtol: float = 1e-5,
        cg_maxiter: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> None:
        self._finufft = _require_finufft()
        xy = _as_numpy_coords_2d(coords)
        self._coords = xy
        self._constant_value = float(constant_value)
        self._length_scale = float(length_scale)
        self._nufft_eps = float(nufft_eps)
        self._nufft_opts = dict(nufft_opts or {})
        self._period_margin = float(period_margin)
        self._cg_rtol = float(cg_rtol)
        self._cg_maxiter = cg_maxiter
        self._workers = _safe_finufft_workers(workers)

        if self._constant_value <= 0:
            raise ValueError("constant_value must be positive.")
        if self._length_scale <= 0:
            raise ValueError("length_scale must be positive.")
        if self._period_margin < 0:
            raise ValueError("period_margin must be non-negative.")

        span = np.ptp(xy, axis=0)
        span = np.where(span <= 0.0, 1.0, span)
        grid_meta = _regular_grid_shape_spacing(xy)
        if grid_meta is None:
            base_period = span
        else:
            ny, nx, dy, dx = grid_meta
            base_period = np.array(
                [ny * dy if ny > 1 else span[0], nx * dx if nx > 1 else span[1]],
                dtype=float,
            )

        self._span = span
        lo = xy.min(axis=0) - self._period_margin * span
        self._period = base_period + 2.0 * self._period_margin * span
        theta = 2.0 * np.pi * (xy - lo) / self._period - np.pi
        self._theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi
        self._theta_x = np.ascontiguousarray(self._theta[:, 0], dtype=np.float64)
        self._theta_y = np.ascontiguousarray(self._theta[:, 1], dtype=np.float64)

        self._n_modes = _normalize_nufft_modes(
            n_modes, xy.shape[0], self._span, self._period, max_auto_modes
        )
        self._weights = self._compute_spectral_weights(
            self._constant_value, self._length_scale
        )

    @property
    def n(self) -> int:
        """Number of observations."""
        return self._coords.shape[0]

    @property
    def n_modes(self) -> tuple[int, int]:
        """Fourier mode grid shape."""
        return self._n_modes

    @property
    def spectral_weights(self) -> np.ndarray:
        """Fourier weights whose sum equals ``constant_value``."""
        return self._weights

    @property
    def eigenvalue_proxy(self) -> np.ndarray:
        """FFT-style spectrum proxy for the periodic Fourier kernel."""
        return self.n * self._weights.ravel()

    def _compute_spectral_weights(
        self, constant_value: float, length_scale: float
    ) -> np.ndarray:
        """Return finite-grid periodic RBF Fourier weights.

        On a complete regular grid with ``n_modes == grid_shape`` and zero
        padding these are exactly ``FFTKernelOp`` eigenvalues divided by the
        number of grid cells, in FINUFFT's centered mode order.
        """
        dy, dx = self._period[0] / self._n_modes[0], self._period[1] / self._n_modes[1]
        y = np.arange(self._n_modes[0], dtype=float) * dy
        x = np.arange(self._n_modes[1], dtype=float) * dx
        y = np.minimum(y, self._period[0] - y)
        x = np.minimum(x, self._period[1] - x)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        row = float(constant_value) * np.exp(
            -(yy**2 + xx**2) / (2.0 * float(length_scale) ** 2)
        )
        lam = np.real(scipy.fft.fft2(row, workers=self._workers))
        lam = np.maximum(lam, 0.0)
        weights = scipy.fft.fftshift(lam) / float(np.prod(self._n_modes))
        return weights.astype(np.float64, copy=False)

    def _finufft_kwargs(self) -> dict[str, Any]:
        opts = dict(self._nufft_opts)
        opts["nthreads"] = self._workers
        return opts

    def _as_2d_array(self, v: torch.Tensor | np.ndarray) -> tuple[np.ndarray, bool]:
        """Normalize an input vector/matrix to ``(n_obs, m)``."""
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)
        arr = np.asarray(arr, dtype=np.float64)
        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] != self.n:
            raise ValueError(
                f"Input must have shape ({self.n},) or ({self.n}, m); "
                f"got {arr.shape}."
            )
        return arr, was_1d

    def mode_coefficients(self, v: torch.Tensor | np.ndarray) -> np.ndarray:
        """Apply the type-1 NUFFT to one or more real-valued columns."""
        arr, _ = self._as_2d_array(v)
        strengths = np.ascontiguousarray(arr.T.astype(np.complex128))
        coeff = self._finufft.nufft2d1(
            self._theta_x,
            self._theta_y,
            strengths,
            self._n_modes,
            eps=self._nufft_eps,
            isign=-1,
            **self._finufft_kwargs(),
        )
        if coeff.ndim == 2:
            coeff = coeff[np.newaxis, :, :]
        return coeff

    def evaluate_modes(self, coeff: np.ndarray) -> np.ndarray:
        """Apply the type-2 NUFFT and return values as ``(n_obs, m)``."""
        coeff = np.asarray(coeff, dtype=np.complex128)
        if coeff.ndim == 2:
            coeff = coeff[np.newaxis, :, :]
        vals = self._finufft.nufft2d2(
            self._theta_x,
            self._theta_y,
            np.ascontiguousarray(coeff),
            eps=self._nufft_eps,
            isign=1,
            **self._finufft_kwargs(),
        )
        if vals.ndim == 1:
            vals = vals[np.newaxis, :]
        return vals.T

    def matvec(self, v: torch.Tensor | np.ndarray) -> np.ndarray:
        """Compute ``K @ v`` without materializing ``K``."""
        _, was_1d = self._as_2d_array(v)
        coeff = self.mode_coefficients(v)
        out = self.evaluate_modes(coeff * self._weights[np.newaxis, :, :])
        out = np.real(out)
        return out[:, 0] if was_1d else out

    def solve(self, v: torch.Tensor | np.ndarray, epsilon: float) -> np.ndarray:
        """Solve ``(K + epsilon * I) u = v`` with conjugate gradients."""
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        rhs, was_1d = self._as_2d_array(v)

        def _matvec(x: np.ndarray) -> np.ndarray:
            return self.matvec(x) + epsilon * x

        def _matmat(x: np.ndarray) -> np.ndarray:
            return self.matvec(x) + epsilon * x

        op = LinearOperator(
            shape=(self.n, self.n),
            matvec=_matvec,
            matmat=_matmat,
            dtype=np.float64,
        )

        out = np.empty_like(rhs, dtype=np.float64)
        for j in range(rhs.shape[1]):
            sol, info = cg(
                op,
                rhs[:, j],
                rtol=self._cg_rtol,
                atol=0.0,
                maxiter=self._cg_maxiter,
            )
            if info != 0:
                warnings.warn(
                    "NUFFT conjugate-gradient solve did not fully converge "
                    f"(info={info}); residuals may be approximate.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            out[:, j] = sol

        return out[:, 0] if was_1d else out

    def residuals(self, v: torch.Tensor | np.ndarray, epsilon: float) -> np.ndarray:
        """Apply ``epsilon * (K + epsilon * I)^{-1} @ v``."""
        return float(epsilon) * self.solve(v, epsilon)

    def eigenvalues(self, k: Optional[int] = None) -> np.ndarray:
        """Return an FFT-style spectral proxy, sorted descending.

        Irregular coordinates do not diagonalize in the Fourier basis, so these
        values are useful only as a rough low-rank spectrum summary.  They
        should not be interpreted as exact eigenvalues of the sampled kernel
        matrix.
        """
        evals = np.sort(self.eigenvalue_proxy)[::-1]
        if evals.size < self.n:
            evals = np.pad(evals, (0, self.n - evals.size))
        return evals[:k] if k is not None else evals

    def trace(self) -> float:
        """Return the exact trace implied by the normalized kernel diagonal."""
        return float(self.n * self._constant_value)

    def square_trace(self) -> float:
        """Return a spectral proxy for ``trace(K**2)``."""
        return float(np.sum(self.eigenvalues() ** 2))


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
        coords : torch.Tensor, shape (n_obs, d)
            Spatial coordinates (should be pre-normalized by the caller).
        Y : torch.Tensor, shape (n_obs, m)
            Target values (may contain NaN rows, which are handled internally).

        Returns
        -------
        residuals : torch.Tensor, shape (n_obs, m)
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
        coords : torch.Tensor, shape (n_obs, d)
            Spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape (n_obs, m_i).

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape (n_obs, m_i).
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
        coords : torch.Tensor, shape (n_obs, d)
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
    n_inducing : int or None
        Maximum number of observations used for hyperparameter fitting.
        When ``n_obs <= n_inducing`` (or ``n_inducing`` is ``None``), all
        observations are used for an exact GP fit.  When ``n_obs > n_inducing``,
        a randomly sampled **subset-of-data** of ``n_inducing`` points is used
        for hyperparameter search, then the fitted kernel is applied to predict
        on all observations.  This is **not** an inducing-point (FITC/VFE)
        approximation — the subset is only used to determine hyperparameters.
        Setting ``n_inducing=None`` disables the subset shortcut and always
        uses the full dataset; a warning is issued when ``n_obs > 10_000``.

    Notes
    -----
    This backend always materialises a dense n x n kernel matrix (up to
    ``n_inducing`` x ``n_inducing`` when the subset path is taken).  For
    very large n, consider ``GPyTorchKernelGPR`` with FITC inducing points
    or ``FFTKernelGPR`` for regular raster grids.
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
        O(n^2) rather than O(n^3).

        Has no effect (and logs a warning) when ``signal_bounds_fixed`` is
        False.

        Parameters
        ----------
        coords : torch.Tensor, shape (n_obs, d)
            Normalized spatial coordinates, shape (n_obs, d).

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
        coords : torch.Tensor, shape (n_obs, d)
            Normalized spatial coordinates.

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
        back to a full sklearn GP fit when n_obs <= n_inducing, or an
        inducing-point approximation when n_obs > n_inducing.

        NaN rows in ``Y`` are excluded from fitting and reinserted as NaN
        in the output.

        Parameters
        ----------
        coords : torch.Tensor, shape (n_obs, d)
            Normalized spatial coordinates.
        Y : torch.Tensor, shape (n_obs, m)
            Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor, shape (n_obs, m)
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
        coords : torch.Tensor, shape (n_obs, d)
            Normalized spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape (n_obs, m_i).

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape (n_obs, m_i).
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
    """GPyTorch-based GP residualizer.

    Uses GPyTorch's lazy-tensor infrastructure so that all linear-system
    solves during training use Lanczos-based conjugate gradients rather
    than dense Cholesky.  Hyperparameters are optimized with Adam on the
    marginal log-likelihood.

    For most datasets (n <= ~10,000) the exact-GP path is used.  Passing
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
        memory scales as O(n x n_inducing) rather than O(n²). Set to None to
        use exact GP.
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
    This matches the sklearn backend behavior and amortizes O(n^3) GP
    fitting across outputs.

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
        coords : torch.Tensor, shape (n_obs, d)
            Normalized spatial coordinates.
        Y : torch.Tensor, shape (n_obs, m)
            Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor, shape (n_obs, m)
            Spatial residuals ``epsilon * (K + epsilon * I)**(-1) @ Y``.
            NaN rows from Y are preserved as NaN.
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
    """FFT-based GPR residualizer for regular (ny, nx) grids using an RBF kernel.

    Accepts the same hyperparameter interface as :class:`SklearnKernelGPR`.
    Kernel eigenvalues are computed via :class:`FFTKernelOp`; all linear-algebra
    is O(N log N) with no n x n matrix formed.

    Notes
    -----
    **Hyperparameter optimization strategy**

    The spectral log-marginal-likelihood (LML) for Y (n = nx * ny observations, m channels)
    is::

        -2 LML = m * sum_k log(lambda_k + eps)
                 + (1/n) * sum_k ||Y_hat_k||^2 / (lambda_k + eps)

    where ``lambda_k = sigma^2 * s_k(l)`` are the kernel eigenvalues with
    ``sigma`` being the constant value (signal variance) and ``s_k(l)``
    the unit-spectrum eigenvalues at length scale ``l``, and
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
        Initial signal variance sigma^2.
    constant_value_bounds : tuple or "fixed"
        Bounds for sigma^2.  ``"fixed"`` pins sigma^2.
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
        coords : torch.Tensor, shape (n_obs, 2)
            Spatial coordinates forming a regular grid.
        Y : torch.Tensor, shape (n_obs, m)
            Target values (may contain NaN rows).

        Returns
        -------
        torch.Tensor, shape (n_obs, m)
            Spatial residuals. NaN rows are preserved.
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
        coords : torch.Tensor, shape (n_obs, 2)
        Y_list : list of torch.Tensor, each (n_obs, m_i)

        Returns
        -------
        list of torch.Tensor, each (n_obs, m_i)
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
    """FINUFFT-based GPR residualizer for irregular 2-D coordinates.

    The backend keeps the RBF kernel implicit: FINUFFT evaluates kernel-vector
    products and conjugate gradients solve ``(K + epsilon I)`` systems.  Regular
    grids with matching ``n_modes`` use the same spectral likelihood as
    :class:`FFTKernelGPR`; irregular grids use a leading-eigenpair summary plus
    trace/trace(K^2) tail correction for marginal-likelihood fitting.

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
        Memory is ``O(n_obs * lml_approx_rank)``.  Ignored on compatible
        regular grids.  ``None`` forces exact dense eigendecomposition for
        small diagnostics.
    lml_exact_max_n : int
        Maximum ``n_obs`` for exact dense eigendecomposition.
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
    Requires the optional ``finufft`` package.  Results should agree exactly
    with :class:`FFTKernelGPR` on compatible regular grids.  Compared with
    :class:`SklearnKernelGPR`, differences near the bounding-box edges are
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
        coords : torch.Tensor, shape (n_obs, 2)
            Normalized spatial coordinates.
        Y : torch.Tensor, shape (n_obs, m)
            Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor, shape (n_obs, m)
            Spatial residuals. NaN rows from ``Y`` are preserved as NaN.
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
    """Construct a KernelGPR from a backend name.

    Parameters
    ----------
    backend : {"sklearn", "gpytorch", "fft", "nufft", "finufft"}
        Backend to use.
    **kwargs
        GPR configuration.  Common keys are ``constant_value``,
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

    .. deprecated:: 1.0.4
        Use :class:`SklearnKernelGPR` directly.  This wrapper is kept for
        backward compatibility and will be removed in a future release.

    Parameters
    ----------
    X : torch.Tensor, shape (n_obs, d)
        Input spatial coordinates.
    Y : torch.Tensor, shape (n_obs, m)
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
        (n_obs, m).
    """
    n_orig = Y.shape[0]
    is_nan = torch.isnan(Y).any(1)
    X_ = X[~is_nan]
    Y_ = Y[~is_nan]

    if normalize_x:
        X_ = (X_ - X_.mean(0)) / X_.std(0).clamp(min=1e-8)
    if normalize_y:
        Y_ = (Y_ - Y_.mean(0)) / Y_.std(0).clamp(min=1e-8)

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
