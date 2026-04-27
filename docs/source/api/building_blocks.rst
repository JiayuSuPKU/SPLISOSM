Building Blocks
===============

These lower-level classes and helpers are useful for method development,
diagnostics, and reproducing implementation details discussed in
:doc:`../methods`.

Kernels
-------

.. autosummary::
   :nosignatures:

   splisosm.kernel.SpatialCovKernel
   splisosm.kernel.FFTKernel

.. autoclass:: splisosm.kernel.SpatialCovKernel
   :no-members:
   :no-special-members:

   .. automethod:: from_coordinates
   .. automethod:: Kx
   .. automethod:: xtKx
   .. automethod:: xtKx_exact
   .. automethod:: trace
   .. automethod:: square_trace
   .. automethod:: realization

.. autoclass:: splisosm.kernel.FFTKernel
   :no-members:
   :no-special-members:

   .. automethod:: Kx
   .. automethod:: xtKx
   .. automethod:: eigenvalues
   .. automethod:: trace
   .. automethod:: square_trace

GLMM internals
--------------

.. autosummary::
   :nosignatures:

   splisosm.glmm.MultinomGLM
   splisosm.glmm.MultinomGLMM

.. autoclass:: splisosm.glmm.MultinomGLM
   :no-members:
   :no-special-members:

   .. automethod:: setup_data
   .. automethod:: forward
   .. automethod:: fit
   .. automethod:: get_isoform_ratio

.. autoclass:: splisosm.glmm.MultinomGLMM
   :no-members:
   :no-special-members:

   .. automethod:: setup_data
   .. automethod:: forward
   .. automethod:: fit
   .. automethod:: get_isoform_ratio


Simulation
----------

.. autosummary::
   :nosignatures:

   splisosm.utils.simulation.simulate_isoform_counts
   splisosm.utils.simulation.simulate_isoform_counts_single_gene

.. autofunction:: splisosm.utils.simulation.simulate_isoform_counts
.. autofunction:: splisosm.utils.simulation.simulate_isoform_counts_single_gene
