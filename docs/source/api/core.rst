Core Classes
============

Use these top-level classes for complete SPLISOSM workflows. The recommended
imports are ``from splisosm import SplisosmNP, SplisosmFFT, SplisosmGLMM``.

.. autosummary::
   :nosignatures:

   splisosm.SplisosmNP
   splisosm.SplisosmFFT
   splisosm.SplisosmGLMM

.. autoclass:: splisosm.SplisosmNP
   :no-members:
   :no-special-members:

   .. autoproperty:: filtered_adata
   .. automethod:: setup_data
   .. automethod:: test_spatial_variability
   .. automethod:: test_differential_usage
   .. automethod:: get_formatted_test_results
   .. automethod:: extract_feature_summary

.. autoclass:: splisosm.SplisosmFFT
   :no-members:
   :no-special-members:

   .. automethod:: setup_data
   .. automethod:: test_spatial_variability
   .. automethod:: test_differential_usage
   .. automethod:: get_formatted_test_results
   .. automethod:: extract_feature_summary

.. autoclass:: splisosm.SplisosmGLMM
   :no-members:
   :no-special-members:

   .. autoproperty:: filtered_adata
   .. automethod:: setup_data
   .. automethod:: fit
   .. automethod:: test_spatial_variability
   .. automethod:: test_differential_usage
   .. automethod:: get_formatted_test_results
   .. automethod:: extract_feature_summary
   .. automethod:: get_fitted_models
   .. automethod:: get_gene_model
   .. automethod:: get_training_summary
   .. automethod:: get_fitted_ratios_anndata
   .. automethod:: save
   .. automethod:: load
