Quick Start
===========

This guide provides step-by-step instructions on how to run SPLISOSM to test for spatial variability (SV) and differential isoform usage (DU) using both non-parametric and parametric methods.

Inputs
------------------

SPLISOSM requires the following inputs:

- ``data``: A list containing per-gene isoform quantification for ``n_gene`` genes. Each element is a ``(n_spot, n_iso)`` tensor, where ``n_spot`` is the number of spatial spots and ``n_iso`` is the number of isoforms (or TREND events) for that gene.
- ``coordinates``: A ``(n_spot, 2)`` tensor containing the spatial coordinates for each spot.

For differential usage (DU) testing, an additional input is required:

- ``covariates``: A ``(n_spot, n_covariate)`` tensor of spatial variables (e.g., RBP expression levels) for association testing.

If your isoform quantification data is stored in an AnnData object (for instance, by following the :doc:`steps in the preprocessing documentation <txquant>`), you can use the helper function :func:`splisosm.utils.extract_counts_n_ratios` to prepare the inputs for SPLISOSM.

.. code-block:: python

   from splisosm.utils import extract_counts_n_ratios

   adata = ...  # AnnData object of (n_spot, n_isoform) with adata.var['gene_symbol'] indicating gene assignment
   data, ratio_data, gene_names, iso_names = extract_counts_n_ratios(
       adata, layer='counts', group_iso_by='gene_symbol'
   )
   coordinates = adata.obs.loc[:, ['array_row', 'array_col']] # spatial coordinates

Outputs
------------------

- **SV tests**: A pandas DataFrame containing per-gene test statistics and p-values.
- **DU tests**: A pandas DataFrame containing test statistics and p-values for each gene-covariate pair.

Example data
---------------

A small demo dataset of Visium-ONT mouse olfactory bulb (SiT-MOB) is available for download `from Dropbox (~100Mb) <https://www.dropbox.com/scl/fo/dmuobtbof54jl4ht9zbjo/ALVIIEp-Ua5yYUPO8QxlIZ8?rlkey=q9o3jisd25ef5hwfqnsqdbf3i&st=vxhgokzw&dl=0>`_ to test the package functionality.

- ``mob_ont_filtered_1107.h5ad``: An AnnData object with isoform quantification results.
- ``mob_visium_rbp_1107.h5ad``: An AnnData object with short-read-based RBP gene expression.

.. code-block:: python

   import scanpy as sc
   from splisosm.utils import extract_counts_n_ratios

   adata_ont = sc.read("mob_ont_filtered_1107.h5ad")
   adata_rbp = sc.read("mob_visium_rbp_1107.h5ad")

   # prepare per gene isoform tensor list
   # data[i] = (n_spot, n_iso) tensor for gene i
   # assuming isoforms (adata_ont.var_names) are grouped by adata_ont.var['gene_symbol']
   data, _, gene_names, _ = extract_counts_n_ratios(
       adata_ont, layer='counts', group_iso_by='gene_symbol'
   )

   # spatial coordinates
   coordinates = adata_ont.obs.loc[:, ['array_row', 'array_col']]

   # prepare covariates for differential usage testing
   adata_rbp = adata_rbp[adata_ont.obs_names, :].copy()  # align the RBP data with the isoform data
   # focus on spatially variably expressed RBPs only
   # adata_rbp.var['is_visium_sve'] = adata_rbp.var['pvalue_adj_sparkx'] < 0.01
   covariates = adata_rbp[:, adata_rbp.var['is_visium_sve']].layers['log1p'].toarray()
   covariate_names = adata_rbp.var.loc[adata_rbp.var['is_visium_sve'], 'features']


Testing for spatial variability (SV)
----------------------------------------

SPLISOSM uses the Hilbert-Schmidt Independence Criterion (HSIC) to test for statistical independence between isoform expression patterns and spatial coordinates. Specifically, SPLISOSM provides the following SV tests, which assess variability in three different aspects of expression:

1.  **HSIC-GC (Gene Counts)**: Tests for SV in total gene expression. This test can also be called directly via :func:`splisosm.utils.run_hsic_gc`. It is designed as a drop-in replacement for `SPARK-X <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02404-0>`_, optimizing statistical power through an improved spatial kernel design.
2.  **HSIC-IR (Isoform Ratios)**: Tests for SV in relative isoform usage.
3.  **HSIC-IC (Isoform Counts)**: Tests for SV in individual isoform expression. Biologically, this variability is a joint result of changes in total gene expression and relative isoform usage. In practice, the results from HSIC-IC and HSIC-GC are often similar.

.. code-block:: python

   from splisosm import SplisosmNP

   # minimal input data, extracted from AnnData as shown above
   data = ...  # list of length n_gene, each a (n_spot, n_iso) tensor
   coordinates = ...  # (n_spot, 2), spatial coordinates
   gene_names = ...  # list of length n_gene, gene names

   # initialize the model
   model_np = SplisosmNP()
   model_np.setup_data(data, coordinates, gene_names=gene_names)

   # per-gene test for spatial variability
   # method can be 'hsic-ir' (isoform ratio), 'hsic-ic' (isoform counts)
   # 'hsic-gc' (gene counts), 'spark-x' (gene counts)
   model_np.test_spatial_variability(
       method="hsic-ir",
       ratio_transformation='none',  # only for 'hsic-ir': 'none', 'clr', 'ilr', 'alr', 'radial'
       nan_filling='mean',           # 'mean' (global mean) or 'none' (ignore NaN spots)
   )

   # extract the per-gene test statistics
   df_sv_res = model_np.get_formatted_test_results(test_type='sv')


Testing for differential isoform usage (DU)
----------------------------------------------

SPLISOSM implements conditional association tests to identify genes whose isoform usage patterns are associated with specific covariates (e.g., RBP expression), while accounting for spatial correlation. These tests are available in both non-parametric and parametric forms.

Non-parametric DU test
~~~~~~~~~~~~~~~~~~~~~~~~
The non-parametric test is based on the HSIC kernel independence test and relies on Gaussian process regression for spatial conditioning.

.. code-block:: python

   from splisosm.hyptest_np import SplisosmNP

   # minimal input data
   # after running the SV test, keep only SVS genes for DU testing
   data_svs = ...  # list of length n_gene, each a (n_spot, n_iso) tensor
   gene_svs_names = ...  # list of length n_gene, gene names
   coordinates = ...  # (n_spot, 2), spatial coordinates

   # covariates for differential usage testing
   # e.g. spatial domains, RBP expression, etc.
   covariates = ...  # (n_spot, n_factor), design matrix of covariates
   covariate_names = ...  # list of length n_factor, covariate names

   # initialize the model with covariates
   model_np = SplisosmNP()
   model_np.setup_data(
       data_svs,
       coordinates,
       design_mtx=covariates,
       gene_names=gene_svs_names,
       covariate_names=covariate_names
   )

   # run the conditional HSIC test for differential usage
   model_np.test_differential_usage(
       method="hsic-gp",  # options: 'hsic', 'hsic-knn', 'hsic-gp', 't-fisher', 't-tippett'
       ratio_transformation='none', nan_filling='mean',
       hsic_eps=1e-3,      # regularization for kernel regression (for 'hsic'); None => unconditional HSIC
       gp_configs=None,    # dict for GP regression config (for 'hsic-gp')
       print_progress=True, return_results=False
   )

   # extract per gene-factor pair test statistics
   df_du_res = model_np.get_formatted_test_results(test_type='du')


Parametric DU test (GLMM)
~~~~~~~~~~~~~~~~~~~~~~~~~

The parametric test is based on a Generalized Linear Mixed Model (GLMM), where the spatial correlation is modeled as a random effect term following a Gaussian random field. Models are fitted using the `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ framework. The marginal likelihood, which integrates out the random effect, is approximated using the Laplace approximation at the mode.

.. code-block:: python

   from splisosm.hyptest_glmm import SplisosmGLMM

   # parametric model fitting
   model_p = SplisosmGLMM(
       model_type='glmm-full',  # 'glmm-full', 'glmm-null', 'glm'
       share_variance=True,
       var_parameterization_sigma_theta=True,
       var_fix_sigma=False,
       var_prior_model="none",
       var_prior_model_params={},
       init_ratio="observed",
       fitting_method="joint_gd",
       fitting_configs={'max_epochs': -1}
   )
   model_p.setup_data(
       data_svs,                # list length n_genes, each (n_spots, n_isos) tensor
       coordinates,             # (n_spots, 2)
       design_mtx=covariates,   # (n_spots, n_covariates)
       gene_names=gene_svs_names,
       covariate_names=covariate_names,
       group_gene_by_n_iso=True,
   )
   model_p.fit(
       n_jobs=2,
       batch_size=20,
       quiet=True, print_progress=True,
       with_design_mtx=False,   # fit without covariates for score DU test
       from_null=False, refit_null=True,  # for LLR test
       random_seed=None
   )
   model_p.save("model_p.pkl")
   per_gene_glmm_models = model_p.get_fitted_models()

   # differential usage testing
   model_p.test_differential_usage(method="score", print_progress=True, return_results=False)

   # extract per gene-factor pair test statistics
   df_du_res = model_p.get_formatted_test_results(test_type='du')
