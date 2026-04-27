"""Spatial Gaussian-process residualization backends.

These classes implement the GP backends used by
``SplisosmNP.test_differential_usage(method="hsic-gp")`` and
``SplisosmFFT.test_differential_usage(method="hsic-gp")``. Most users select
them through the ``gpr_backend`` and ``gpr_configs`` arguments on those methods;
direct class access is useful when inspecting backend-specific configuration
or residualizing custom response matrices.
"""

from __future__ import annotations

from splisosm.gpr.config import _DEFAULT_GPR_CONFIGS
from splisosm.gpr.factory import make_kernel_gpr

_LAZY_EXPORTS = {
    "KernelGPR": ("splisosm.gpr.base", "KernelGPR"),
    "SklearnKernelGPR": ("splisosm.gpr.sklearn", "SklearnKernelGPR"),
    "GPyTorchKernelGPR": ("splisosm.gpr.gpytorch", "GPyTorchKernelGPR"),
    "FFTKernelGPR": ("splisosm.gpr.fft", "FFTKernelGPR"),
    "NUFFTKernelGPR": ("splisosm.gpr.nufft", "NUFFTKernelGPR"),
    "SpatialKernelOp": ("splisosm.gpr.operators", "SpatialKernelOp"),
    "DenseKernelOp": ("splisosm.gpr.operators", "DenseKernelOp"),
    "FFTKernelOp": ("splisosm.gpr.operators", "FFTKernelOp"),
    "NUFFTKernelOp": ("splisosm.gpr.operators", "NUFFTKernelOp"),
    "_build_rbf_kernel": ("splisosm.gpr.sklearn", "_build_rbf_kernel"),
    "_build_rbf_cross_kernel": (
        "splisosm.gpr.sklearn",
        "_build_rbf_cross_kernel",
    ),
    "_kernel_residuals_from_eigdecomp": (
        "splisosm.gpr.sklearn",
        "_kernel_residuals_from_eigdecomp",
    ),
}

__all__ = [
    "SpatialKernelOp",
    "DenseKernelOp",
    "FFTKernelOp",
    "NUFFTKernelOp",
    "KernelGPR",
    "SklearnKernelGPR",
    "GPyTorchKernelGPR",
    "FFTKernelGPR",
    "NUFFTKernelGPR",
    "make_kernel_gpr",
    "_DEFAULT_GPR_CONFIGS",
    "_build_rbf_kernel",
    "_build_rbf_cross_kernel",
    "_kernel_residuals_from_eigdecomp",
]


def __getattr__(name: str) -> object:
    """Lazily import backend classes and helper functions on first access."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        attr = getattr(import_module(module_name), attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public names for interactive completion."""
    return sorted(set(globals()) | set(__all__))
