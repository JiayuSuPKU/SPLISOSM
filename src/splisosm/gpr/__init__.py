"""Spatial Gaussian-process residualization backends.

These classes implement the GP backends used by
``SplisosmNP.test_differential_usage(method="hsic-gp")`` and
``SplisosmFFT.test_differential_usage(method="hsic-gp")``. Most users select
them through the ``gpr_backend`` and ``gpr_configs`` arguments on those methods;
direct class access is useful when inspecting backend-specific configuration
or residualizing custom response matrices.
"""

from __future__ import annotations

from splisosm.gpr.base import KernelGPR
from splisosm.gpr.config import _DEFAULT_GPR_CONFIGS
from splisosm.gpr.factory import make_kernel_gpr
from splisosm.gpr.fft import FFTKernelGPR
from splisosm.gpr.gpytorch import GPyTorchKernelGPR
from splisosm.gpr.nufft import NUFFTKernelGPR
from splisosm.gpr.operators import (
    DenseKernelOp,
    FFTKernelOp,
    NUFFTKernelOp,
    SpatialKernelOp,
)
from splisosm.gpr.sklearn import (
    SklearnKernelGPR,
    _build_rbf_cross_kernel,
    _build_rbf_kernel,
    _kernel_residuals_from_eigdecomp,
)
from splisosm.utils.hsic import linear_hsic_test

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
