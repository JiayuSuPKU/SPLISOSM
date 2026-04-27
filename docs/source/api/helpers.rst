Helper Functions
================

These helpers support data preparation, standalone tests, and result
post-processing outside the model classes.

Data preparation
----------------

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

Standalone tests
----------------

.. autosummary::
   :nosignatures:

   splisosm.utils.stats.run_hsic_gc
   splisosm.utils.stats.run_sparkx
   splisosm.utils.stats.false_discovery_control

.. autofunction:: splisosm.utils.stats.run_hsic_gc
.. autofunction:: splisosm.utils.stats.run_sparkx
.. autofunction:: splisosm.utils.stats.false_discovery_control

HSIC and Liu helpers
--------------------

.. autosummary::
   :nosignatures:

   splisosm.utils.hsic.linear_hsic_test
   splisosm.utils.hsic.liu_sf
   splisosm.utils.hsic.liu_sf_from_cumulants

.. autofunction:: splisosm.utils.hsic.linear_hsic_test
.. autofunction:: splisosm.utils.hsic.liu_sf
.. autofunction:: splisosm.utils.hsic.liu_sf_from_cumulants
