"""Spatial Gaussian-process residualization backends.

These classes implement the GP backends used by
``SplisosmNP.test_differential_usage(method="hsic-gp")`` and
``SplisosmFFT.test_differential_usage(method="hsic-gp")``. Most users select
them through the ``gpr_backend`` and ``gpr_configs`` arguments on those methods;
direct class access is useful when inspecting backend-specific configuration
or residualizing custom response matrices.
"""

from __future__ import annotations

from splisosm._gpr import (
    FFTKernelGPR,
    GPyTorchKernelGPR,
    KernelGPR,
    NUFFTKernelGPR,
    SklearnKernelGPR,
    make_kernel_gpr,
)

__all__ = [
    "KernelGPR",
    "SklearnKernelGPR",
    "GPyTorchKernelGPR",
    "FFTKernelGPR",
    "NUFFTKernelGPR",
    "make_kernel_gpr",
]
