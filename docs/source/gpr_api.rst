GP Regression Backends
======================

SPLISOSM uses spatial Gaussian-process (GP) residualization for conditional
differential-usage tests with ``method="hsic-gp"``. In routine workflows,
choose a backend through ``gpr_backend`` and tune it through ``gpr_configs`` on
:meth:`splisosm.SplisosmNP.test_differential_usage` or
:meth:`splisosm.SplisosmFFT.test_differential_usage`.

This page documents the concrete backend classes for users who need
backend-specific options or direct residualization calls. Shapes use
``n_spots`` for spatial locations and ``m`` for response columns.

Backend selection guide:

- ``sklearn``: dense reference backend for small to moderate datasets.
- ``gpytorch``: exact or FITC sparse GP with optional GPU support.
- ``fft``: regular raster grids, used by :class:`splisosm.SplisosmFFT`.
- ``nufft`` / ``finufft``: irregular 2-D coordinates without a dense kernel.

Backend Classes
---------------

.. autosummary::
   :nosignatures:

   splisosm.gpr.KernelGPR
   splisosm.gpr.SklearnKernelGPR
   splisosm.gpr.GPyTorchKernelGPR
   splisosm.gpr.FFTKernelGPR
   splisosm.gpr.NUFFTKernelGPR

.. autoclass:: splisosm.gpr.KernelGPR
   :members:
   :show-inheritance:

.. autoclass:: splisosm.gpr.SklearnKernelGPR
   :members:
   :show-inheritance:

.. autoclass:: splisosm.gpr.GPyTorchKernelGPR
   :members:
   :show-inheritance:

.. autoclass:: splisosm.gpr.FFTKernelGPR
   :members:
   :show-inheritance:

.. autoclass:: splisosm.gpr.NUFFTKernelGPR
   :members:
   :show-inheritance:

Factory Helper
--------------

.. autofunction:: splisosm.gpr.make_kernel_gpr
