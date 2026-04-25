"""Cumulant-based null approximations for HSIC spatial variability tests."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
import torch
from scipy.stats import chi2 as _chi2_dist

from splisosm.kernel import FFTKernel, IdentityKernel, Kernel, SpatialCovKernel
from splisosm.likelihood import liu_sf_from_cumulants

_DELTA = 1e-10
_EIGVAL_THRESHOLD = 1e-8


def _cumulants_from_eigenvalues(
    eigvals: np.ndarray | torch.Tensor,
    threshold: float = _EIGVAL_THRESHOLD,
    max_power: Literal[2, 4] = 4,
) -> dict[int, float]:
    """Return ``trace(A**p)`` cumulants from nonzero PSD eigenvalues."""
    if isinstance(eigvals, torch.Tensor):
        vals = eigvals.detach().cpu().numpy()
    else:
        vals = np.asarray(eigvals)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > threshold)]
    return {p: float(np.sum(vals**p)) for p in range(1, max_power + 1)}


def _kernel_has_exact_c12(kernel: Kernel) -> bool:
    """Return whether ``trace`` / ``square_trace`` are known exact or analytic."""
    if isinstance(kernel, SpatialCovKernel):
        # Dense CAR stores the realised covariance and traces are exact.
        # Implicit CAR stores only the precision; its trace methods are
        # Hutchinson estimators, so callers should use their configured probe
        # budget instead of treating these as analytic substitutions.
        return kernel.K_sp is not None
    return isinstance(kernel, (FFTKernel, IdentityKernel)) or (
        hasattr(kernel, "trace") and hasattr(kernel, "square_trace")
    )


def _exact_c12(kernel: Kernel) -> Optional[dict[int, float]]:
    """Return exact first two cumulants when that path is available."""
    if not _kernel_has_exact_c12(kernel):
        return None
    try:
        return {1: float(kernel.trace()), 2: float(kernel.square_trace())}
    except (AttributeError, TypeError, ValueError, NotImplementedError, RuntimeError):
        return None


def _normalize_hsic_null_method(null_method: str, allow_perm: bool = False) -> str:
    """Return the canonical HSIC null method name, warning on legacy aliases."""
    if null_method == "eig":
        warnings.warn(
            "null_method='eig' is deprecated; use 'liu' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "liu"
    if null_method in ("clt", "trace"):
        warnings.warn(
            f"null_method='{null_method}' is deprecated; using 'welch' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "welch"

    valid = {"liu", "welch"}
    if allow_perm:
        valid.add("perm")
    if null_method not in valid:
        opts = sorted(valid | {"eig", "clt", "trace"})
        raise ValueError(f"Invalid null method. Must be one of {opts}.")
    return null_method


def _feature_cumulants_from_data(y: np.ndarray | torch.Tensor) -> dict[int, float]:
    """Return cumulants of the small feature Gram matrix ``Y.T @ Y``."""
    if isinstance(y, torch.Tensor):
        gram = y.detach().cpu().T @ y.detach().cpu()
        try:
            eigvals = torch.linalg.eigvalsh(gram)
        except torch._C._LinAlgError:
            jitter = 1e-6 * torch.eye(gram.shape[0], dtype=gram.dtype)
            eigvals = torch.linalg.eigvalsh(gram + jitter)
        return _cumulants_from_eigenvalues(eigvals, threshold=1e-5)

    arr = np.asarray(y, dtype=float)
    gram_np = arr.T @ arr
    try:
        eigvals_np = np.linalg.eigvalsh(gram_np)
    except np.linalg.LinAlgError:
        eigvals_np = np.linalg.eigvalsh(
            gram_np + 1e-6 * np.eye(gram_np.shape[0], dtype=float)
        )
    return _cumulants_from_eigenvalues(eigvals_np)


def _hutchinson_cumulants(
    kernel: Kernel,
    n_probes: int = 60,
    rng_seed: int = 0,
    max_power: Literal[2, 4] = 4,
) -> dict[int, float]:
    """Estimate ``trace(K**p)`` with Rademacher probes.

    ``max_power=2`` estimates only ``c1`` and ``c2`` and uses one kernel
    application per probe.  ``max_power=4`` also estimates ``c3`` and ``c4`` and
    uses two applications per probe.  ``c1`` and ``c2`` are replaced
    automatically only for backends with known exact/analytic trace methods.
    Dense CAR kernels have exact traces because they store ``K_sp``; implicit
    CAR kernels keep the probe estimates because they store only the sparse
    precision and their trace methods are themselves stochastic estimators.

    Parameters
    ----------
    kernel
        SPLISOSM kernel exposing :meth:`Kx`.
    n_probes
        Number of iid Rademacher probe vectors.
    rng_seed
        Seed for reproducible probe draws.
    max_power
        Highest cumulant power to return. Must be ``2`` or ``4``.

    Returns
    -------
    dict
        Mapping ``{1: c1, ..., max_power: c_max_power}``.
    """
    if max_power not in (2, 4):
        raise ValueError("`max_power` must be 2 or 4.")
    if n_probes < 1:
        raise ValueError("`n_probes` must be >= 1.")

    if isinstance(kernel, FFTKernel):
        return _cumulants_from_eigenvalues(kernel.eigenvalues(), max_power=max_power)

    if isinstance(kernel, IdentityKernel):
        return _cumulants_from_eigenvalues(kernel.eigenvalues(), max_power=max_power)

    if not hasattr(kernel, "n") or not hasattr(kernel, "Kx"):
        try:
            return _cumulants_from_eigenvalues(
                kernel.eigenvalues(),
                max_power=max_power,
            )
        except (AttributeError, ValueError, NotImplementedError, RuntimeError) as exc:
            raise AttributeError(
                "Hutchinson cumulants require a kernel with `n` and `Kx`, "
                "or a fallback `eigenvalues()` method."
            ) from exc

    n = int(kernel.n)
    rng = np.random.default_rng(rng_seed)
    probes = rng.choice(np.array([-1.0, 1.0]), size=(n, int(n_probes)))

    def _apply(x: np.ndarray) -> np.ndarray:
        return np.asarray(kernel.Kx(x), dtype=float)

    u = _apply(probes)
    cumulants = {
        1: float(np.mean(np.sum(probes * u, axis=0))),
        2: float(np.mean(np.sum(u * u, axis=0))),
    }

    if max_power == 4:
        w = _apply(u)
        cumulants[3] = float(np.mean(np.sum(u * w, axis=0)))
        cumulants[4] = float(np.mean(np.sum(w * w, axis=0)))

    exact = _exact_c12(kernel)
    if exact is not None:
        cumulants.update(exact)

    return cumulants


def _kernel_cumulants(
    kernel: Kernel,
    *,
    approx_rank: Optional[int] = None,
    n_probes: Optional[int] = None,
    rng_seed: int = 0,
    max_power: Literal[2, 4] = 4,
    prefer_eigenvalues: bool = True,
) -> dict[int, float]:
    """Return kernel cumulants from eigenvalues or Hutchinson probes."""
    if n_probes is not None:
        return _hutchinson_cumulants(
            kernel,
            n_probes=int(n_probes),
            rng_seed=rng_seed,
            max_power=max_power,
        )

    if max_power == 2 and not prefer_eigenvalues:
        exact = _exact_c12(kernel)
        if exact is not None:
            return exact

    if prefer_eigenvalues:
        try:
            return _cumulants_from_eigenvalues(
                kernel.eigenvalues(k=approx_rank),
                max_power=max_power,
            )
        except (ValueError, NotImplementedError, RuntimeError):
            pass

    return _hutchinson_cumulants(
        kernel,
        n_probes=60,
        rng_seed=rng_seed,
        max_power=max_power,
    )


def _kernel_cumulants_for_null(
    kernel: Kernel,
    *,
    null_method: Literal["liu", "welch"],
    n_spots: int,
    null_configs: Optional[dict[str, Any]] = None,
    dense_threshold: int = 5000,
) -> tuple[dict[int, float], Optional[int]]:
    """Resolve SV null config and return kernel cumulants plus stat rank.

    The returned rank is non-``None`` only when ``null_method="liu"`` uses an
    explicit low-rank spatial spectrum; callers should use that same rank for
    the observed statistic.
    """
    configs = null_configs or {}
    rng_seed = int(configs.get("rng_seed", 0))

    if null_method == "welch":
        n_probes = configs.get("n_probes")
        if _kernel_has_exact_c12(kernel):
            n_probes = None
        elif n_probes is None:
            n_probes = 60
        return (
            _kernel_cumulants(
                kernel,
                n_probes=None if n_probes is None else int(n_probes),
                rng_seed=rng_seed,
                max_power=2,
                prefer_eigenvalues=False,
            ),
            None,
        )

    if null_method != "liu":
        raise ValueError(f"Unsupported null_method for cumulants: {null_method!r}.")

    dense_threshold = int(dense_threshold)
    explicit_approx = "approx_rank" in configs
    approx_rank = configs.get("approx_rank")
    n_probes = configs.get("n_probes")
    kernel_approx_rank = None

    if approx_rank is not None:
        approx_rank = int(approx_rank)
        if approx_rank < 1:
            raise ValueError("null_configs['approx_rank'] must be positive.")
        if approx_rank < n_spots:
            kernel_approx_rank = approx_rank
        elif n_spots > dense_threshold:
            warnings.warn(
                "null_configs['approx_rank'] >= n_spots requests a full "
                "large-kernel spectrum; using Hutchinson Rademacher probes instead.",
                UserWarning,
                stacklevel=2,
            )
            n_probes = 60 if n_probes is None else n_probes
    elif n_spots > dense_threshold:
        if explicit_approx:
            warnings.warn(
                "Full eigenvalue cumulants for large implicit kernels are "
                "expensive; using Hutchinson Rademacher probes instead. "
                "Set null_configs['n_probes'] to control the probe budget.",
                UserWarning,
                stacklevel=2,
            )
        n_probes = 60 if n_probes is None else n_probes

    return (
        _kernel_cumulants(
            kernel,
            approx_rank=kernel_approx_rank,
            n_probes=None if n_probes is None else int(n_probes),
            rng_seed=rng_seed,
            max_power=4,
        ),
        kernel_approx_rank,
    )


def _dense_kernel_cumulants(K: np.ndarray | torch.Tensor) -> dict[int, float]:
    """Return cumulants for a dense kernel matrix."""
    if isinstance(K, torch.Tensor):
        eigvals = torch.linalg.eigvalsh(K)
    else:
        eigvals = np.linalg.eigvalsh(np.asarray(K, dtype=float))
    return _cumulants_from_eigenvalues(eigvals)


def _hsic_mixture_cumulants(
    kernel_cumulants: dict[int, float],
    feature_cumulants: dict[int, float],
    n_spots: int,
    max_power: Literal[2, 4] = 4,
) -> dict[int, float]:
    """Cumulants of the HSIC chi-squared mixture for ``hsic_scaled``."""
    denom = float(max(int(n_spots) - 1, 1))
    return {
        p: float(kernel_cumulants[p] * feature_cumulants[p] / (denom**p))
        for p in range(1, max_power + 1)
    }


def _hsic_null_mean_var(
    kernel_cumulants: dict[int, float],
    feature_cumulants: dict[int, float],
    n_spots: int,
) -> tuple[float, float]:
    """Return the first two null moments for ``hsic_scaled``."""
    c = _hsic_mixture_cumulants(
        kernel_cumulants,
        feature_cumulants,
        n_spots,
        max_power=2,
    )
    mean = c[1]
    var = 2.0 * c[2]
    return float(mean), float(max(var, 0.0))


def _hsic_liu_pvalue(
    hsic_scaled: float,
    kernel_cumulants: dict[int, float],
    feature_cumulants: dict[int, float],
    n_spots: int,
) -> float:
    """Compute a Liu p-value for an HSIC spatial-variability statistic."""
    c = _hsic_mixture_cumulants(kernel_cumulants, feature_cumulants, n_spots)
    if c[2] <= _DELTA:
        return 1.0
    pvalue = liu_sf_from_cumulants(float(hsic_scaled), c)
    return float(np.clip(np.asarray(pvalue).item(), 0.0, 1.0))


def _hsic_welch_pvalue(
    hsic_scaled: float,
    kernel_cumulants: dict[int, float],
    feature_cumulants: dict[int, float],
    n_spots: int,
) -> float:
    """Compute a Welch-Satterthwaite p-value from HSIC null moments."""
    mean_null, var_null = _hsic_null_mean_var(
        kernel_cumulants,
        feature_cumulants,
        n_spots,
    )
    if var_null <= _DELTA or mean_null <= _DELTA:
        return 0.0 if float(hsic_scaled) > mean_null + _DELTA else 1.0

    scale_g = var_null / (2.0 * mean_null)
    df_h = 2.0 * mean_null**2 / var_null
    return float(_chi2_dist.sf(float(hsic_scaled) / scale_g, df=df_h))
