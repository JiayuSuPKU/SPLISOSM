API Reference
=============

This page lists the stable user-facing API. Most workflows use one of the
three model classes, then call ``setup_data()``, a test method, and
``get_formatted_test_results()``. GP residualization backends used by
``method="hsic-gp"`` are documented separately in :doc:`gpr_api`.

Model Classes
-------------

.. autosummary::
   :nosignatures:

   splisosm.SplisosmNP
   splisosm.SplisosmFFT
   splisosm.SplisosmGLMM

.. autoclass:: splisosm.SplisosmNP
   :members: setup_data, test_spatial_variability, test_differential_usage, get_formatted_test_results
   :show-inheritance:

.. autoclass:: splisosm.SplisosmFFT
   :members: setup_data, test_spatial_variability, test_differential_usage, get_formatted_test_results
   :show-inheritance:

.. autoclass:: splisosm.SplisosmGLMM
   :members: setup_data, test_spatial_variability, test_differential_usage, get_formatted_test_results, get_fitted_models, get_gene_model
   :show-inheritance:

Utility Functions
-----------------

These helpers support data preparation, standalone tests, and result
post-processing outside the model classes.

Data preparation
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   splisosm.utils.prepare_inputs_from_anndata
   splisosm.utils.counts_to_ratios
   splisosm.utils.add_ratio_layer
   splisosm.utils.compute_feature_summaries
   splisosm.utils.auto_chunk_size

.. autofunction:: splisosm.utils.prepare_inputs_from_anndata
.. autofunction:: splisosm.utils.counts_to_ratios
.. autofunction:: splisosm.utils.add_ratio_layer
.. autofunction:: splisosm.utils.compute_feature_summaries
.. autofunction:: splisosm.utils.auto_chunk_size

Standalone Tests
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   splisosm.utils.run_hsic_gc
   splisosm.utils.run_sparkx
   splisosm.utils.false_discovery_control

.. autofunction:: splisosm.utils.run_hsic_gc
.. autofunction:: splisosm.utils.run_sparkx
.. autofunction:: splisosm.utils.false_discovery_control

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
