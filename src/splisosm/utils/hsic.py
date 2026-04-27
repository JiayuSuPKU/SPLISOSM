"""HSIC test statistics and null approximations shared by SPLISOSM."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import numpy as np
import scipy.sparse
import torch
from numpy.typing import ArrayLike
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import ncx2

from splisosm.kernel import (
    FFTKernel,
    IdentityKernel,
    Kernel,
    SpatialCovKernel,
    _MaskedSpatialKernel,
)

__all__ = ["linear_hsic_test", "liu_sf", "liu_sf_from_cumulants"]

_DELTA = 1e-10
_EIGVAL_THRESHOLD = 1e-8


def liu_sf(
    t: ArrayLike,
    lambs: ArrayLike,
    dofs: Optional[ArrayLike] = None,
    deltas: Optional[ArrayLike] = None,
    kurtosis: bool = False,
) -> np.ndarray:
    """Compute p-values for weighted chi-squared sums using Liu's approximation.

    Parameters
    ----------
    t
        Observed test statistic value(s).
    lambs
        Mixture weights.
    dofs
        Degrees of freedom for each component. Defaults to one.
    deltas
        Noncentrality parameters for each component. Defaults to zero.
    kurtosis
        If ``True``, use the kurtosis-matching edge-case branch.

    Returns
    -------
    numpy.ndarray
        Survival probabilities ``Pr(X > t)``.
    """
    coeffs = _liu_prepare(
        lambs,
        dofs=dofs,
        deltas=deltas,
        kurtosis=kurtosis,
    )
    return _liu_apply(t, coeffs)


def liu_sf_from_cumulants(
    t: ArrayLike,
    cumulants: dict[int, float],
    kurtosis: bool = False,
) -> np.ndarray:
    """Compute Liu p-values from chi-squared-mixture cumulants."""
    return _liu_apply(
        t,
        _liu_prepare_from_cumulants(
            cumulants,
            kurtosis=kurtosis,
        ),
    )


def _liu_prepare(
    lambs: ArrayLike,
    dofs: Optional[ArrayLike] = None,
    deltas: Optional[ArrayLike] = None,
    kurtosis: bool = False,
) -> dict[str, float]:
    """Precompute Liu shifted-chi-squared coefficients from eigenvalues."""
    if dofs is None:
        dofs = np.ones_like(lambs)
    if deltas is None:
        deltas = np.zeros_like(lambs)

    lambs = np.asarray(lambs, float)
    dofs = np.asarray(dofs, float)
    deltas = np.asarray(deltas, float)
    lambs_power = {i: lambs**i for i in range(1, 5)}

    c = {
        i: float(np.sum(lambs_power[i] * dofs) + i * np.sum(lambs_power[i] * deltas))
        for i in range(1, 5)
    }
    return _liu_prepare_from_cumulants(c, kurtosis=kurtosis)


def _liu_prepare_from_cumulants(
    cumulants: dict[int, float],
    kurtosis: bool = False,
) -> dict[str, float]:
    """Precompute Liu shifted-chi-squared coefficients from cumulants."""
    c = {i: float(cumulants.get(i, 0.0)) for i in range(1, 5)}

    if c[2] <= _DELTA or not np.isfinite(c[2]):
        return {
            "mu_q": float(c[1]),
            "sigma_q": 0.0,
            "mu_x": 1.0,
            "sigma_x": np.sqrt(2.0),
            "dof_x": 1.0,
            "delta_x": 0.0,
        }

    s1 = c[3] / (np.sqrt(c[2]) ** 3 + _DELTA)
    s2 = c[4] / (c[2] ** 2 + _DELTA)

    s12 = s1**2
    if s12 > s2:
        denom = s1 - np.sqrt(s12 - s2)
        if abs(denom) < _DELTA:
            delta_x = 0.0
            dof_x = 1 / (s2 + _DELTA)
        else:
            a = 1 / denom
            delta_x = s1 * a**3 - a**2
            dof_x = a**2 - 2 * delta_x
    else:
        delta_x = 0
        if kurtosis:
            a = 1 / np.sqrt(s2)
            dof_x = 1 / s2
        else:
            a = 1 / (s1 + _DELTA)
            dof_x = 1 / (s12 + _DELTA)

    dof_x = max(float(dof_x), _DELTA)
    delta_x = max(float(delta_x), 0.0)

    var_q = 2.0 * c[2]
    mu_x = dof_x + delta_x
    sigma_x = np.sqrt(2 * (dof_x + 2 * delta_x))

    return {
        "mu_q": float(c[1]),
        "sigma_q": float(np.sqrt(max(var_q, 0.0))),
        "mu_x": float(mu_x),
        "sigma_x": float(sigma_x),
        "dof_x": float(dof_x),
        "delta_x": float(delta_x),
    }


def _liu_apply(t: ArrayLike, coeffs: dict[str, float]) -> np.ndarray:
    """Apply cached Liu coefficients to one or more statistics."""
    t = np.asarray(t, float)
    if coeffs["sigma_q"] <= _DELTA:
        return np.where(t > coeffs["mu_q"] + _DELTA, 0.0, 1.0)
    t_star = (t - coeffs["mu_q"]) / (coeffs["sigma_q"] + _DELTA)
    tfinal = t_star * coeffs["sigma_x"] + coeffs["mu_x"]
    return ncx2.sf(tfinal, coeffs["dof_x"], max(coeffs["delta_x"], 1e-9))


def _to_dense_float_tensor(
    x: torch.Tensor | np.ndarray | scipy.sparse.spmatrix,
) -> torch.Tensor:
    """Return a dense float tensor, densifying sparse inputs once."""
    if scipy.sparse.issparse(x):
        return torch.from_numpy(np.asarray(x.toarray(), dtype=np.float32))
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        return x.float()
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _normalise_linear_hsic_inputs(
    X: torch.Tensor | np.ndarray | scipy.sparse.spmatrix,
    Y: torch.Tensor | np.ndarray | scipy.sparse.spmatrix,
) -> tuple[torch.Tensor | scipy.sparse.spmatrix, torch.Tensor, bool]:
    """Normalize HSIC inputs and preserve SciPy sparsity for X."""
    Y_tensor = _to_dense_float_tensor(Y)
    X_is_sparse = scipy.sparse.issparse(X)
    if X_is_sparse:
        return X, Y_tensor.detach().cpu(), True

    X_tensor = _to_dense_float_tensor(X)
    if Y_tensor.device != X_tensor.device:
        Y_tensor = Y_tensor.to(X_tensor.device)
    return X_tensor, Y_tensor, False


def _drop_linear_hsic_nan_rows(
    X: torch.Tensor | scipy.sparse.spmatrix,
    Y: torch.Tensor,
    *,
    X_is_sparse: bool,
) -> tuple[torch.Tensor | scipy.sparse.spmatrix, torch.Tensor]:
    """Drop rows with NaNs from dense Y and dense X when present."""
    is_nan_y = torch.isnan(Y).any(1)
    if X_is_sparse:
        is_nan = is_nan_y
    else:
        is_nan = is_nan_y | torch.isnan(X).any(1)

    if not bool(is_nan.any()):
        return X, Y

    keep = ~is_nan
    Y = Y[keep]
    if X_is_sparse:
        X = X[keep.detach().cpu().numpy()]
    else:
        X = X[keep]
    return X, Y


def _linear_hsic_sparse_x_components(
    X: scipy.sparse.spmatrix,
    Y: torch.Tensor,
    *,
    centering: bool,
    eigv_th: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return statistic and null eigenvalues for sparse-X linear HSIC."""
    X_sp = X.tocsr()
    if centering:
        Y = Y - Y.mean(0)

    # When Y is centred, Y_c.T @ X_c == Y_c.T @ X because
    # the correction term Y_c.sum(0) vanishes.
    YcTX = torch.from_numpy(X_sp.T.dot(Y.numpy()).astype(np.float32)).T
    hsic_scaled = YcTX.pow(2).sum()

    XTX = X_sp.T.dot(X_sp).toarray().astype(np.float32)
    if centering:
        X_mean = np.asarray(X_sp.mean(axis=0), dtype=np.float32).ravel()
        XTX = XTX - Y.shape[0] * np.outer(X_mean, X_mean)

    lambda_x = torch.linalg.eigvalsh(torch.from_numpy(XTX))
    lambda_y = torch.linalg.eigvalsh(Y.T @ Y)
    return (
        hsic_scaled,
        lambda_x[lambda_x > eigv_th],
        lambda_y[lambda_y > eigv_th],
    )


def _linear_hsic_dense_components(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    centering: bool,
    eigv_th: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return statistic and null eigenvalues for dense linear HSIC."""
    if centering:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

    hsic_scaled = torch.norm(Y.T @ X, p="fro").pow(2)
    lambda_x = torch.linalg.eigvalsh(X.T @ X)
    lambda_y = torch.linalg.eigvalsh(Y.T @ Y)
    return (
        hsic_scaled,
        lambda_x[lambda_x > eigv_th],
        lambda_y[lambda_y > eigv_th],
    )


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

    X, Y, X_is_sparse = _normalise_linear_hsic_inputs(X, Y)
    X, Y = _drop_linear_hsic_nan_rows(X, Y, X_is_sparse=X_is_sparse)
    n = Y.shape[0]

    if X_is_sparse:
        hsic_scaled, lambda_x, lambda_y = _linear_hsic_sparse_x_components(
            X,
            Y,
            centering=centering,
            eigv_th=eigv_th,
        )
    else:
        hsic_scaled, lambda_x, lambda_y = _linear_hsic_dense_components(
            X,
            Y,
            centering=centering,
            eigv_th=eigv_th,
        )

    lambda_xy = (lambda_x.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1)
    pval = liu_sf(
        (hsic_scaled * n).detach().cpu().numpy(),
        lambda_xy.detach().cpu().numpy(),
    )

    return float(hsic_scaled / (n - 1) ** 2), float(pval)


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
    if isinstance(kernel, _MaskedSpatialKernel):
        # The masked kernel can compute exact traces only by applying the
        # operator to a dense identity block. For per-gene NaN masks we want
        # the configured Hutchinson budget instead.
        return False
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


def _feature_cumulants_from_data(
    y: np.ndarray | torch.Tensor | scipy.sparse.spmatrix,
    *,
    centered: bool = True,
) -> dict[int, float]:
    """Return cumulants of the small feature Gram matrix ``Y.T @ Y``."""
    if scipy.sparse.issparse(y):
        y_csc = y.tocsc(copy=False)
        gram = y_csc.T @ y_csc
        gram_np = gram.toarray() if scipy.sparse.issparse(gram) else np.asarray(gram)
        if not centered:
            sums = np.asarray(y_csc.sum(axis=0)).ravel().astype(float)
            gram_np = gram_np - np.outer(sums, sums) / float(y_csc.shape[0])
        gram_np = np.asarray(gram_np, dtype=float)
        try:
            eigvals_np = np.linalg.eigvalsh(gram_np)
        except np.linalg.LinAlgError:
            eigvals_np = np.linalg.eigvalsh(
                gram_np + 1e-6 * np.eye(gram_np.shape[0], dtype=float)
            )
        return _cumulants_from_eigenvalues(eigvals_np)

    if isinstance(y, torch.Tensor):
        y_cpu = y.detach().cpu()
        if y_cpu.layout != torch.strided:
            if y_cpu.layout != torch.sparse_coo and hasattr(y_cpu, "to_sparse_coo"):
                y_cpu = y_cpu.to_sparse_coo()
            y_cpu = y_cpu.coalesce()
            idx = y_cpu.indices().numpy()
            vals = y_cpu.values().numpy()
            y_sp = scipy.sparse.coo_matrix(
                (vals, (idx[0], idx[1])),
                shape=tuple(y_cpu.shape),
            ).tocsc()
            return _feature_cumulants_from_data(y_sp, centered=centered)
        if not centered:
            y_cpu = y_cpu - y_cpu.mean(dim=0, keepdim=True)
        gram = y_cpu.T @ y_cpu
        try:
            eigvals = torch.linalg.eigvalsh(gram)
        except torch._C._LinAlgError:
            jitter = 1e-6 * torch.eye(gram.shape[0], dtype=gram.dtype)
            eigvals = torch.linalg.eigvalsh(gram + jitter)
        return _cumulants_from_eigenvalues(eigvals, threshold=1e-5)

    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if not centered:
        arr = arr - arr.mean(axis=0, keepdims=True)
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
    """Estimate ``trace(K**p)`` with Rademacher probes."""
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
    """Resolve SV null config and return kernel cumulants plus statistic rank."""
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
