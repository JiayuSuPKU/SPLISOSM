"""FINUFFT GP residualization backend for irregular 2-D coordinates."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

from splisosm.gpr.base import KernelGPR
from splisosm.gpr.operators import (
    NUFFTKernelOp,
    _as_numpy_coords_2d,
    _regular_grid_shape_spacing,
    _require_finufft,
)

__all__ = ["NUFFTKernelGPR"]


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

        pip install "splisosm[gp]"

    Results should match :class:`FFTKernelGPR` on compatible regular grids up
    to numerical tolerance. Compared with :class:`SklearnKernelGPR`,
    differences near the bounding-box edges are expected because this backend
    uses periodic boundary conditions.
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
