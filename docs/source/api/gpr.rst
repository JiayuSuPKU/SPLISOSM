GP Regression Backends
======================

SPLISOSM uses spatial Gaussian-process (GP) residualization for conditional
differential-usage tests with ``method="hsic-gp"``. In routine workflows,
choose a backend through ``gpr_backend`` and tune it through ``gpr_configs`` on
:meth:`splisosm.SplisosmNP.test_differential_usage` or
:meth:`splisosm.SplisosmFFT.test_differential_usage`.

This page documents the lower-level operator and residualizer classes for users
who need backend-specific options or direct residualization calls. Shapes use
``n_spots`` for spatial locations and ``m`` for response columns.

Backend selection guide:

- ``sklearn``: dense reference backend for small to moderate datasets.
- ``gpytorch``: exact or FITC sparse GP with optional GPU support.
- ``fft``: regular raster grids, used by :class:`splisosm.SplisosmFFT`.
- ``nufft`` / ``finufft``: large irregular 2-D coordinates without a dense kernel.

For NUFFT, ``lml_approx_rank`` controls only the approximate log marginal
likelihood used during hyperparameter fitting on irregular coordinates. It is
not a low-rank spatial test. The NUFFT matvec still uses the selected Fourier
grid; larger ranks improve GP hyperparameter accuracy but use more memory.

Kernel operators
----------------

Kernel operators provide matrix-vector products, solves, residualization, and
spectral summaries without requiring each backend to expose a dense kernel
matrix.

.. autosummary::
   :nosignatures:

   splisosm.gpr.SpatialKernelOp
   splisosm.gpr.DenseKernelOp
   splisosm.gpr.FFTKernelOp
   splisosm.gpr.NUFFTKernelOp

.. autoclass:: splisosm.gpr.SpatialKernelOp
   :no-members:
   :no-special-members:

   .. automethod:: matvec
   .. automethod:: solve
   .. automethod:: residuals
   .. automethod:: eigenvalues
   .. automethod:: trace
   .. automethod:: square_trace

.. autoclass:: splisosm.gpr.DenseKernelOp
   :no-members:
   :no-special-members:

   .. automethod:: matvec
   .. automethod:: solve
   .. automethod:: residuals
   .. automethod:: eigenvalues

.. autoclass:: splisosm.gpr.FFTKernelOp
   :no-members:
   :no-special-members:

   .. automethod:: matvec
   .. automethod:: solve
   .. automethod:: residuals
   .. automethod:: eigenvalues
   .. automethod:: trace
   .. automethod:: square_trace

.. autoclass:: splisosm.gpr.NUFFTKernelOp
   :no-members:
   :no-special-members:

   .. automethod:: matvec
   .. automethod:: solve
   .. automethod:: residuals
   .. automethod:: eigenvalues
   .. automethod:: trace
   .. automethod:: square_trace

Residualizer classes
--------------------

.. autosummary::
   :nosignatures:

   splisosm.gpr.KernelGPR
   splisosm.gpr.SklearnKernelGPR
   splisosm.gpr.GPyTorchKernelGPR
   splisosm.gpr.FFTKernelGPR
   splisosm.gpr.NUFFTKernelGPR

.. autoclass:: splisosm.gpr.KernelGPR
   :no-members:
   :no-special-members:

   .. automethod:: fit_residuals
   .. automethod:: fit_residuals_batch
   .. automethod:: get_kernel_op

.. autoclass:: splisosm.gpr.SklearnKernelGPR
   :no-members:
   :no-special-members:

   .. automethod:: from_config
   .. automethod:: precompute_shared_kernel
   .. automethod:: fit_residuals
   .. automethod:: fit_residuals_batch
   .. automethod:: get_kernel_op

.. autoclass:: splisosm.gpr.GPyTorchKernelGPR
   :no-members:
   :no-special-members:

   .. automethod:: fit_residuals

.. autoclass:: splisosm.gpr.FFTKernelGPR
   :no-members:
   :no-special-members:

   .. automethod:: fit_residuals
   .. automethod:: fit_residuals_batch
   .. automethod:: fit_residuals_cube
   .. automethod:: get_kernel_op

.. autoclass:: splisosm.gpr.NUFFTKernelGPR
   :no-members:
   :no-special-members:

   .. automethod:: fit_residuals
   .. automethod:: fit_residuals_batch
   .. automethod:: get_kernel_op

Factory helper
--------------

.. autofunction:: splisosm.gpr.make_kernel_gpr
