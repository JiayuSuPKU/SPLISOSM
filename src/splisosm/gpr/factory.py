"""Factory helpers for spatial GP residualization backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from splisosm.gpr.config import (
    _FFT_GPR_KWARGS,
    _GPYTORCH_GPR_KWARGS,
    _KNOWN_GPR_KWARGS,
    _NUFFT_GPR_KWARGS,
    _SKLEARN_GPR_KWARGS,
)

if TYPE_CHECKING:
    from splisosm.gpr.base import KernelGPR

__all__ = ["make_kernel_gpr"]


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


def _backend_class(backend: str) -> type["KernelGPR"]:
    """Import and return only the backend class selected by the caller."""
    if backend == "sklearn":
        from splisosm.gpr.sklearn import SklearnKernelGPR

        return SklearnKernelGPR
    if backend == "gpytorch":
        from splisosm.gpr.gpytorch import GPyTorchKernelGPR

        return GPyTorchKernelGPR
    if backend == "fft":
        from splisosm.gpr.fft import FFTKernelGPR

        return FFTKernelGPR
    if backend in {"nufft", "finufft"}:
        from splisosm.gpr.nufft import NUFFTKernelGPR

        return NUFFTKernelGPR
    raise ValueError(
        f"Unknown backend '{backend}'. Choose from 'sklearn', 'gpytorch', "
        "'fft', 'nufft', or 'finufft'."
    )


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
    cls = _backend_class(backend)
    return cls(**_filter_gpr_kwargs(backend, kwargs))
