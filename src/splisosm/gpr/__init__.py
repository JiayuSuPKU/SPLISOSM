"""Spatial Gaussian-process residualization backends.

These classes implement the GP backends used by
``SplisosmNP.test_differential_usage(method="hsic-gp")`` and
``SplisosmFFT.test_differential_usage(method="hsic-gp")``. Most users select
them through the ``gpr_backend`` and ``gpr_configs`` arguments on those methods;
direct class access is useful when inspecting backend-specific configuration
or residualizing custom response matrices.
"""

from __future__ import annotations

from splisosm.gpr.backends import (
    FFTKernelGPR,
    GPyTorchKernelGPR,
    KernelGPR,
    NUFFTKernelGPR,
    SklearnKernelGPR,
    _DEFAULT_GPR_CONFIGS,
    make_kernel_gpr,
)
from splisosm.gpr.operators import (
    DenseKernelOp,
    FFTKernelOp,
    NUFFTKernelOp,
    SpatialKernelOp,
)
from splisosm.gpr.statistics import (
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
