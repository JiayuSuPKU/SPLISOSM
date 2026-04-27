"""Implicit spatial kernel operators for GPR backends."""

from __future__ import annotations

import os
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import scipy.fft
import torch
from scipy.sparse.linalg import LinearOperator, cg

__all__ = [
    "SpatialKernelOp",
    "DenseKernelOp",
    "FFTKernelOp",
    "NUFFTKernelOp",
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
    ``FFTKernelOp`` (``splisosm.hyptest.fft``)
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
