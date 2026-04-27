"""Internal Gaussian-process residualization helpers."""

from __future__ import annotations

from splisosm._gpr.backends import (
    FFTKernelGPR,
    GPyTorchKernelGPR,
    KernelGPR,
    NUFFTKernelGPR,
    SklearnKernelGPR,
    _DEFAULT_GPR_CONFIGS,
    make_kernel_gpr,
)
from splisosm._gpr.operators import (
    DenseKernelOp,
    FFTKernelOp,
    NUFFTKernelOp,
    SpatialKernelOp,
)
from splisosm._gpr.statistics import (
    _build_rbf_cross_kernel,
    _build_rbf_kernel,
    _kernel_residuals_from_eigdecomp,
    linear_hsic_test,
)

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
    "linear_hsic_test",
    "_DEFAULT_GPR_CONFIGS",
    "_build_rbf_kernel",
    "_build_rbf_cross_kernel",
    "_kernel_residuals_from_eigdecomp",
]
