API Reference
=============

This page lists the stable user-facing API. Most workflows use one of the
three model classes, then call ``setup_data()``, a test method, and
``get_formatted_test_results()``. GP residualization backends used by
``method="hsic-gp"`` are documented separately in :doc:`gpr_api`.
The implementation modules are ``splisosm.hyptest.np``,
``splisosm.hyptest.fft``, and ``splisosm.hyptest.glmm``; the top-level
imports below remain the recommended user-facing entry points.

Model Classes
-------------

.. autosummary::
   :nosignatures:

   splisosm.SplisosmNP
   splisosm.SplisosmFFT
   splisosm.SplisosmGLMM

.. autoclass:: splisosm.SplisosmNP
   :members: setup_data, test_spatial_variability, test_differential_usage, get_formatted_test_results
   :inherited-members:

.. autoclass:: splisosm.SplisosmFFT
   :members: setup_data, test_spatial_variability, test_differential_usage, get_formatted_test_results
   :inherited-members:

.. autoclass:: splisosm.SplisosmGLMM
   :members: setup_data, test_spatial_variability, test_differential_usage, get_formatted_test_results, get_fitted_models, get_gene_model
   :inherited-members:

Helper Functions
----------------

These helpers support data preparation, standalone tests, and result
post-processing outside the model classes. They are grouped by purpose in the
public modules below.

Data Preparation
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   splisosm.utils.preprocessing.prepare_inputs_from_anndata
   splisosm.utils.preprocessing.counts_to_ratios
   splisosm.utils.preprocessing.add_ratio_layer
   splisosm.utils.preprocessing.compute_feature_summaries
   splisosm.utils.preprocessing.auto_chunk_size

.. autofunction:: splisosm.utils.preprocessing.prepare_inputs_from_anndata
.. autofunction:: splisosm.utils.preprocessing.counts_to_ratios
.. autofunction:: splisosm.utils.preprocessing.add_ratio_layer
.. autofunction:: splisosm.utils.preprocessing.compute_feature_summaries
.. autofunction:: splisosm.utils.preprocessing.auto_chunk_size

Standalone Tests
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   splisosm.utils.stats.run_hsic_gc
   splisosm.utils.stats.run_sparkx
   splisosm.utils.stats.false_discovery_control

.. autofunction:: splisosm.utils.stats.run_hsic_gc
.. autofunction:: splisosm.utils.stats.run_sparkx
.. autofunction:: splisosm.utils.stats.false_discovery_control

HSIC And Liu Helpers
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   splisosm.utils.hsic.linear_hsic_test
   splisosm.utils.hsic.liu_sf
   splisosm.utils.hsic.liu_sf_from_cumulants

.. autofunction:: splisosm.utils.hsic.linear_hsic_test
.. autofunction:: splisosm.utils.hsic.liu_sf
.. autofunction:: splisosm.utils.hsic.liu_sf_from_cumulants

SpatialData Loaders
-------------------

.. autosummary::
   :nosignatures:

   splisosm.io.load_visium_sp_meta
   splisosm.io.load_visium_probe
   splisosm.io.load_visiumhd_probe
   splisosm.io.load_xenium_codeword

.. autofunction:: splisosm.io.load_visium_sp_meta
.. autofunction:: splisosm.io.load_visium_probe
.. autofunction:: splisosm.io.load_visiumhd_probe
.. autofunction:: splisosm.io.load_xenium_codeword

Advanced Building Blocks
------------------------

These lower-level classes and helpers are useful for method development,
diagnostics, and reproducing implementation details discussed in
:doc:`methods`.

.. autosummary::
   :nosignatures:

   splisosm.kernel.SpatialCovKernel
   splisosm.kernel.FFTKernel
   splisosm.glmm.MultinomGLM
   splisosm.glmm.MultinomGLMM
   splisosm.glmm.log_prob_fastmult
   splisosm.glmm.log_prob_fastmvn
   splisosm.utils.simulation.simulate_isoform_counts
   splisosm.utils.simulation.simulate_isoform_counts_single_gene

.. autoclass:: splisosm.kernel.SpatialCovKernel
   :members: from_coordinates, Kx, xtKx, xtKx_exact, trace, square_trace, realization
   :show-inheritance:

.. autoclass:: splisosm.kernel.FFTKernel
   :members: Kx, xtKx, eigenvalues, trace, square_trace
   :show-inheritance:

.. autoclass:: splisosm.glmm.MultinomGLM
   :members: fit
   :show-inheritance:

.. autoclass:: splisosm.glmm.MultinomGLMM
   :members: fit
   :show-inheritance:

.. autofunction:: splisosm.glmm.log_prob_fastmult
.. autofunction:: splisosm.glmm.log_prob_fastmvn
.. autofunction:: splisosm.utils.simulation.simulate_isoform_counts
.. autofunction:: splisosm.utils.simulation.simulate_isoform_counts_single_gene
