"""FFT GP residualization backend for regular grids."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.fft
import torch
from scipy.optimize import minimize, minimize_scalar

from splisosm.gpr.base import KernelGPR
from splisosm.gpr.operators import FFTKernelOp

__all__ = ["FFTKernelGPR"]


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
