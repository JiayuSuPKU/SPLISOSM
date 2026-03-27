Quick Start
===========

This guide walks through running SPLISOSM for spatial variability (SV) and differential isoform usage (DU) testing.

Choosing a model class
----------------------

SPLISOSM provides three model classes:

- :class:`~splisosm.SplisosmNP` — Non-parametric HSIC tests with low-rank kernel approximation. Works on both regular and irregular geometries (e.g., segmented spatial data or even single-cell data).
- :class:`~splisosm.SplisosmGLMM` — Parametric GLMM-based DU test with spatial random effects.
- :class:`~splisosm.SplisosmFFT` — FFT-accelerated HSIC tests on **regular grids** (Visium HD, Xenium binned data). Shares the same statistical model as :class:`~splisosm.SplisosmNP`. Recommended for large datasets for fast computation and memory efficiency.

Inputs and outputs
------------------

:class:`~splisosm.SplisosmNP` and :class:`~splisosm.SplisosmGLMM` accept isoform-level quantification as either an ``AnnData`` object or as raw tensors.
See :doc:`Feature Quantification page <txquant>` for guidance on preparing input data for different platforms.

.. code-block:: python

   model.setup_data(
       adata,                    # AnnData of shape (n_spots, n_isoforms)
       spatial_key="spatial",    # key in adata.obsm for spatial coordinates
       layer="counts",           # layer containing raw isoform counts
       group_iso_by="gene_symbol",  # adata.var column grouping isoforms by gene
       gene_names="gene_symbol",  # adata.var column for gene names
       min_counts=10,
       min_bin_pct=0.01,
       design_mtx=covariates,        # (n_spots, n_factors) optional for DU testing
       covariate_names=covariate_names,  # list of covariate names
   )

.. note::
    For versions of SPLISOSM prior to v1.0.4, `setup_data` takes the legacy arguments 
    ``data``, ``coordinates``, which can be computed from `adata` via :func:`splisosm.utils.extract_counts_n_ratios`.
    Please check the API documentation :func:`SplisosmNP.setup_data` for details.

For :class:`~splisosm.SplisosmFFT`, pass a ``SpatialData`` object with isoform-level counts to ``setup_data``.

.. code-block:: python

   model.setup_data(
       sdata,                    # SpatialData object
       bins="Visium_HD_Mouse_Brain_square_016um",  # SpatialData bin element name
       table_name="square_016um", # adata containing isoform-level counts
       col_key="array_col",
       row_key="array_row",
       layer="counts",
       group_iso_by="gene_ids",  # adata.var column grouping isoforms by gene
       gene_names="gene_name",    # adata.var column for gene names
       min_counts=10,
       min_bin_pct=0.01,
   )


All model classes share the same output convention:

- **SV results**: a ``DataFrame`` with per-gene test statistics and p-values.
- **DU results**: a ``DataFrame`` with test statistics and p-values for each gene-covariate pair.


Example data
------------

A demo Visium-ONT mouse olfactory bulb dataset (SiT-MOB) is available `from Dropbox (~100 MB) <https://www.dropbox.com/scl/fo/dmuobtbof54jl4ht9zbjo/ALVIIEp-Ua5yYUPO8QxlIZ8?rlkey=q9o3jisd25ef5hwfqnsqdbf3i&st=vxhgokzw&dl=0>`_:

- ``mob_ont_filtered_1107.h5ad``: isoform quantification (AnnData).
- ``mob_visium_rbp_1107.h5ad``: short-read RBP gene expression (AnnData).

.. code-block:: python

   import scanpy as sc

   adata_ont = sc.read("mob_ont_filtered_1107.h5ad")
   adata_rbp = sc.read("mob_visium_rbp_1107.h5ad")

   # Align RBP data and extract spatially variable RBPs as covariates
   adata_rbp = adata_rbp[adata_ont.obs_names, :].copy()
   covariates = adata_rbp[:, adata_rbp.var['is_visium_sve']].layers['log1p'].toarray()
   covariate_names = adata_rbp.var.loc[adata_rbp.var['is_visium_sve'], 'features']


Testing for spatial variability (SV)
-------------------------------------

SPLISOSM tests for statistical independence between isoform expression and spatial location using the Hilbert-Schmidt Independence Criterion (HSIC). Three SV tests are available:

1. **HSIC-IR** — tests SV of relative isoform usage (spatially variably processed genes, SVP).
2. **HSIC-GC** — tests SV of total gene expression (spatially variably expressed genes, SVE). Drop-in replacement for `SPARK-X <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02404-0>`_ with improved power.
3. **HSIC-IC** — tests SV of individual isoform counts; reflects joint changes in expression and usage.

**SplisosmNP** (any geometries — AnnData input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from splisosm import SplisosmNP

   model = SplisosmNP()
   model.setup_data(
       adata=adata_ont,
       spatial_key="spatial",
       layer="counts",
       group_iso_by="gene_symbol",
       approx_rank=100,   # lower rank speeds up computation for large datasets
       min_counts=10,
       min_bin_pct=0.01,
   )
   model.test_spatial_variability(
       method="hsic-ir",
       ratio_transformation="none",  # 'none', 'clr', 'ilr', 'alr', 'radial'
       nan_filling="mean",
   )
   df_sv_res = model.get_formatted_test_results(test_type="sv")

**SplisosmFFT** (regular grids only — binned Visium HD, Xenium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from splisosm import SplisosmFFT

   model = SplisosmFFT(neighbor_degree=1, rho=0.99)
   model.setup_data(
       sdata=sdata,
       bins="Visium_HD_Mouse_Brain_square_016um",  # SpatialData bin element name
       table_name="square_016um",
       col_key="array_col",
       row_key="array_row",
       layer="counts",
       group_iso_by="gene_ids",
       gene_names="gene_name",
       min_counts=10,
       min_bin_pct=0.01,
   )
   model.test_spatial_variability(method="hsic-ir", n_jobs=-1)
   df_sv_res = model.get_formatted_test_results(test_type="sv")

Testing for differential isoform usage (DU)
--------------------------------------------

DU tests identify genes whose isoform usage is associated with a covariate (e.g., spatial domain, RBP expression), conditioned on spatial correlation. Two approaches are available.

**SplisosmNP** (non-parametric, HSIC-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses conditional HSIC with Gaussian process (GP) spatial detrending:

.. code-block:: python

   from splisosm import SplisosmNP

   # Focus on SVP genes from the SV step
   svp_genes = df_sv_res.loc[df_sv_res['pvalue_adj'] < 0.05, 'gene'].tolist()
   adata_svp = adata_ont[:, adata_ont.var['gene_symbol'].isin(svp_genes)].copy()

   # Run DU test for each covariate (e.g., RBP expression)
   model = SplisosmNP()
   model.setup_data(
       adata=adata_svp,
       spatial_key="spatial",
       layer="counts",
       design_mtx=covariates,        # (n_spots, n_factors)
       group_iso_by="gene_symbol",
       covariate_names=covariate_names,
       approx_rank=None
   )
   model.test_differential_usage(
       method="hsic-gp",  # 'hsic', 'hsic-knn', 'hsic-gp', 't-fisher', 't-tippett'
       ratio_transformation="none",
       nan_filling="mean",
       print_progress=True,
   )
   df_du_res = model.get_formatted_test_results(test_type="du")


**SplisosmGLMM** (parametric, GLM/GLMM-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fits a multinomial GLMM with a Gaussian random field spatial random effect. The marginal likelihood is approximated via Laplace at the mode.

.. code-block:: python

   from splisosm import SplisosmGLMM

   model_p = SplisosmGLMM(
       model_type="glmm-full",  # 'glmm-full', 'glmm-null', 'glm'
       fitting_method="joint_gd",
   )
   model_p.setup_data(
       adata=adata_svp,
       spatial_key="spatial",
       layer="counts",
       design_mtx=covariates,        # (n_spots, n_factors)
       group_iso_by="gene_symbol",
       covariate_names=covariate_names,
       group_gene_by_n_iso=True,
   )
   model_p.fit(
       n_jobs=2, batch_size=20,
       with_design_mtx=False,  # fit null model first for score test
       refit_null=True,
       print_progress=True,
   )
   model_p.test_differential_usage(method="score", print_progress=True)
   df_du_res = model_p.get_formatted_test_results(test_type="du")
