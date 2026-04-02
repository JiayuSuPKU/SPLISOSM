Quick Start
===========

This guide walks through running SPLISOSM for spatial variability (SV) and differential isoform usage (DU) testing.

Choosing a model class
----------------------

SPLISOSM provides three model classes:

- :class:`~splisosm.SplisosmNP` — Non-parametric HSIC tests with a sparse CAR spatial kernel. Works on both regular and irregular geometries (e.g., segmented spatial data or even single-cell data).
- :class:`~splisosm.SplisosmFFT` — Same tests as :class:`~splisosm.SplisosmNP` but with FFT-accelerated implementation for data on **regular grids** (e.g., Visium HD, Xenium binned data).
- :class:`~splisosm.SplisosmGLMM` — Parametric Multinomial-based DU test with spatial random effects. The GLM model is similar to conventional (bulk) differential splicing analysis. The GLMM model is extended to handle cross-spot variability.

Use the decision tree below to pick the right class for your dataset.
For methodological details of each test, see :doc:`Statistical Methods <methods>`.

.. code-block:: text

   Is your data on a REGULAR GRID (Visium HD, Xenium binned, etc.)?
   ├── YES  →  SplisosmFFT  (fastest, requires SpatialData input)
   └── NO  (Slide-seq, single-cell segmented data, etc.)
       │
       Are you interested in parametric model fitting (effect sizes, covariates)?
       ├── YES  →  SplisosmGLMM (not calibrated for SV, but can be used for DU testing)
       └── NO  →  SplisosmNP  (the most general, supports both SV and DU testing)

Summary of class features:

.. list-table::
   :header-rows: 1
   :widths: 15 8 8 15 15 15 24

   * - Model class
     - SV
     - DU
     - Input type
     - Geometry
     - Speed
     - Low-rank approximation
   * - :class:`~splisosm.SplisosmNP`
     - ✓
     - ✓
     - AnnData
     - any
     - fast
     - optional via ``null_configs={"approx_rank": k}`` for SV, and ``gpr_configs={"covariate": {"n_inducing": m}}`` for DU.
   * - :class:`~splisosm.SplisosmFFT`
     - ✓
     - ✓
     - SpatialData
     - grid only
     - fastest
     - N/A (FFT-based; no eigendecomposition)
   * - :class:`~splisosm.SplisosmGLMM`
     - ✗
     - ✓
     - AnnData
     - any
     - slow
     - optional via ``approx_rank=k`` at ``SplisosmGLMM()`` construction time.

Inputs and outputs
------------------

:class:`~splisosm.SplisosmNP` and :class:`~splisosm.SplisosmGLMM` accept isoform-level quantification as either an ``AnnData`` object or as raw tensors.

.. code-block:: python

   model.setup_data(
       adata,                        # AnnData of shape (n_spots, n_isoforms)
       spatial_key="spatial",        # key in adata.obsm for spatial coordinates
       layer="counts",               # layer containing raw isoform counts
       group_iso_by="gene_symbol",   # adata.var column grouping isoforms by gene
       gene_names="gene_symbol",     # adata.var column for gene names
       # ----- Differential usage testing
       design_mtx=covariates,           # (n_spots, n_factors)
       covariate_names=covariate_names, # list of covariate names
       # ----- Spot filtering options (for removing disconnected tissue fragments, etc.)
       min_component_size=1,         # minimal size of a connected spatial k-NN graph component to keep
       # ----- Feature filtering options (applied after spot filtering)
       min_counts=10,                # minimal total counts per isoform to keep
       min_bin_pct=0.01,             # minimal proportion of non-zero expression spots per isoform to keep
       filter_single_iso_genes=True, # remove genes with only one isoform (no usage variation)
   )

.. note::
    Since ``v1.1.0``, the ``approx_rank`` argument was moved out of ``setup_data``:

    * **SplisosmNP** — pass ``null_configs={"approx_rank": k}`` to
      :meth:`~splisosm.SplisosmNP.test_spatial_variability` to cap the eigenvalue
      rank used for the HSIC null distribution.
    * **SplisosmGLMM** — pass ``approx_rank=k`` to :class:`~splisosm.SplisosmGLMM`
      at construction time to control the low-rank spatial kernel approximation.
    * **SplisosmFFT** uses FFT-based convolutions instead of eigendecomposition and
      does not support ``approx_rank`` or ``null_configs``.

For :class:`~splisosm.SplisosmFFT`, pass a ``SpatialData`` object with isoform-level counts to ``setup_data``.

.. code-block:: python

   model.setup_data(
       sdata,                     # SpatialData object
       bins="Visium_HD_Mouse_Brain_square_016um",  # SpatialData bin element name
       table_name="square_016um", # adata containing isoform-level counts
       col_key="array_col",       # adata.obs column for array column indices
       row_key="array_row",       # adata.obs column for array row indices
       layer="counts",            # layer containing raw isoform counts
       group_iso_by="gene_ids",   # adata.var column grouping isoforms by gene
       gene_names="gene_name",    # adata.var column for gene names
       min_counts=10,
       min_bin_pct=0.01,
   )

See :doc:`Feature Quantification page <txquant>` for guidance on preparing input data from different spatial transcriptomics platforms. 
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
       min_counts=10,
       min_bin_pct=0.01,
   )
   model.test_spatial_variability(
       method="hsic-ir",
       ratio_transformation="none",
       nan_filling="mean",
       # null_method: 'eig' (default, Liu's method), 'trace' (normal approx)
       # null_configs: optional dict; e.g. {"approx_rank": 20} to cap eigenvalues
       null_method="eig",
   )
   df_sv_res = model.get_formatted_test_results(test_type="sv")

**SplisosmFFT** (regular grids only — SpatialData input)
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

DU tests identify genes whose isoform usage is associated with a covariate (e.g., spatial domain, RBP expression), conditioned on spatial correlation.

**SplisosmNP** (non-parametric, HSIC-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses Gaussian process regression (GPR) to remove spatial autocorrelation.

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
       group_iso_by="gene_symbol",
       design_mtx=covariates,        # (n_spots, n_factors)
       covariate_names=covariate_names, # list of RBP names
   )
   model.test_differential_usage(
       method="hsic-gp",  # 'hsic', 'hsic-gp', 't-fisher', 't-tippett'
       ratio_transformation="none",
       nan_filling="mean",
       print_progress=True,
   )
   df_du_res = model.get_formatted_test_results(test_type="du")


**SplisosmFFT** (Same as SplisosmNP but regular grids only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same test as :class:`~splisosm.SplisosmNP` but with FFT-accelerated implementation of Gaussian process regression for regular grid data.

.. code-block:: python

   from splisosm import SplisosmFFT

   model = SplisosmFFT(neighbor_degree=1, rho=0.99)
   model_du_fft.setup_data(
       sdata=sdata,
       bins=test_bins_element, # the 16um bin for rasterization
       table_name="square_016um_svp", # probe counts from SVP genes
       # design_mtx : can be stored as a separate table in the same SpatialData object
       design_mtx="square_016um_rbp_sve",
       col_key="array_col",
       row_key="array_row",
       layer="counts",
       group_iso_by="gene_ids",
       gene_names="gene_name",
       min_counts=10,
       min_bin_pct=0.01,
    )
   model.test_differential_usage(method="hsic-gp", n_jobs=-1, print_progress=True)
   df_du_res = model.get_formatted_test_results(test_type="du")


**SplisosmGLMM** (parametric, GLM/GLMM-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fits a multinomial GLMM with a Gaussian random field spatial random effect. The marginal likelihood is approximated via Laplace at the mode.

.. code-block:: python

   from splisosm import SplisosmGLMM

   model_p = SplisosmGLMM(
       model_type="glmm-full",   # 'glmm-full' | 'glmm-null' | 'glm'
       fitting_method="joint_gd",
       device="cpu",             # 'cpu' | 'cuda' (NVIDIA GPU) | 'mps' (Apple Silicon)
       approx_rank=None,         # None = full rank; int = low-rank approximation
   )
   model_p.setup_data(
       adata=adata_svp,
       spatial_key="spatial",
       layer="counts",
       group_iso_by="gene_symbol",
       group_gene_by_n_iso=True,    # required for batch_size > 1
       design_mtx=covariates,       # (n_spots, n_factors) array or obs column name
       covariate_names=covariate_names,
   )
   model_p.fit(
       n_jobs=2, batch_size=20,
       with_design_mtx=False,       # False → score test (recommended); True → Wald test
       refit_null=True,
       print_progress=True,
   )
   model_p.test_differential_usage(method="score", print_progress=True)
   df_du_res = model_p.get_formatted_test_results(test_type="du")

Performance tuning and configuration reference
------------------------------------------------

Each model class has its own set of tuning knobs.  All options have sensible defaults;
adjust them only when you hit performance or accuracy limits.


.. raw:: html

   <details>
   <summary><strong>SplisosmNP / run_hsic_gc — configuration options (click to expand)</strong></summary>
   <br>

.. list-table::
   :header-rows: 1
   :widths: 26 36 14 24

   * - Feature
     - How to set
     - Default
     - Trade-off
   * - **SV null approximation**
     - ``null_method=`` in ``test_spatial_variability`` / ``run_hsic_gc``
     - ``"eig"`` (Liu's chi-square mixture)
     - ``"trace"`` (moment-matching normal) avoids eigendecomposition
       entirely — fastest for very large *n* but p-values may be slightly
       less calibrated in the tails.  ``"perm"`` uses permutation; slowest
       but assumption-free.
   * - **SV null low-rank** (when ``null_method="eig"``)
     - ``null_configs={"approx_rank": k}`` in ``test_spatial_variability`` /
       ``run_hsic_gc``
     - Auto: full rank when ``n_spots ≤ 5000``, else ``⌈4√n⌉``
     - Restricts test statistic and null computation to the top-*k* eigenvalues; 
       significant speedup for large *n* with potential power loss for high-frequency patterns 
       (usually rare in real spatial data).
   * - **Conditional DU test** (``method="hsic-gp"``)
     - ``residualize=`` in ``test_differential_usage``
     - ``"cov_only"``
     - ``"cov_only"`` removes spatial effects from covariates only — fastest and
       recommended in most cases.  ``"both"`` additionally residualizes
       isoform ratios, which is more conservative and can be more robust in case 
       the covariate residualization is incomplete.
   * - **GPR backend** (DU, ``method="hsic-gp"``)
     - ``gpr_backend=`` in ``test_differential_usage``
     - ``"sklearn"``
     - ``"gpytorch"`` uses the FITC sparse-GP approximation with GPU support.  
        Tends to be slower than ``"sklearn"`` on CPU due to overhead.
   * - **GPR inducing points** (DU, ``method="hsic-gp"``)
     - ``gpr_configs={"covariate": {"n_inducing": M}}`` in
       ``test_differential_usage``
     - ``5000`` (subset-of-data for sklearn; FITC for gpytorch)
     - Reduces GP fitting cost from O(*n*³) to O(*nM*²).
       Accuracy degrades if *M* is too small.
       Set to ``None`` for exact GP (warns when ``n_spots > 10000``).
   * - **Other GPR configuration**
     - ``gpr_configs=`` in ``test_differential_usage``
     - See :func:`test_differential_usage <splisosm.hyptest_np.SplisosmNP.test_differential_usage>` docstring for details
     - Adjust ``constant_value_bounds`` or ``length_scale_bounds`` to tune the hyperparameter
       searching ranges.

.. raw:: html

   </details>

.. raw:: html

   <details>
   <summary><strong>SplisosmFFT — configuration options (click to expand)</strong></summary>
   <br>

.. list-table::
   :header-rows: 1
   :widths: 26 36 14 24

   * - Feature
     - How to set
     - Default
     - Trade-off
   * - **FFT thread count**
     - ``SplisosmFFT(workers=N)``
     - ``None`` (all available CPUs, via ``scipy.fft``)
     - Controls the number of threads used for the FFT-based spatial
       convolutions.  Set to ``1`` to disable multi-threading (e.g. when
       already parallelising at the gene level with ``n_jobs``).
   * - **SV parallel jobs**
     - ``n_jobs=`` in ``test_spatial_variability``
     - ``-1`` (all CPUs)
     - Number of joblib workers for gene-wise HSIC computation.  Set to
       ``1`` to disable parallelism (useful when ``workers > 1`` already
       saturates the CPU).
   * - **DU parallel jobs**
     - ``n_jobs=`` in ``test_differential_usage``
     - ``-1`` (all CPUs)
     - Number of joblib workers for (gene, covariate) pair computation.
       Same trade-off as for ``test_spatial_variability``.
   * - **Conditional DU test** (``method="hsic-gp"``)
     - ``residualize=`` in ``test_differential_usage``
     - ``"cov_only"``
     - Same semantics as :class:`~splisosm.SplisosmNP`: ``"cov_only"``
       residualizes covariates only; ``"both"`` also residualizes isoform
       ratios.
   * - **Other GPR configuration**
     - ``gpr_configs=`` in ``test_differential_usage``
     - See :func:`test_differential_usage <splisosm.hyptest_fft.SplisosmFFT.test_differential_usage>` docstring for details
     - Adjust ``constant_value_bounds`` or ``length_scale_bounds`` to tune the hyperparameter
       searching ranges.

.. raw:: html

   </details>


.. raw:: html

   <details>
   <summary><strong>SplisosmGLMM — configuration options (click to expand)</strong></summary>
   <br>

.. list-table::
   :header-rows: 1
   :widths: 26 32 14 28

   * - Feature
     - How to set
     - Default
     - Trade-off
   * - **GPU acceleration**
     - ``SplisosmGLMM(device=...)``
     - ``"cpu"``
     - Use ``"cuda"`` (NVIDIA) or ``"mps"`` (Apple Silicon) for a large
       per-gene speed-up.  Parallel fitting (``n_jobs > 1``) is
       auto-disabled on non-CPU devices because fork-based workers cannot
       share GPU context.
   * - **Low-rank spatial kernel**
     - ``SplisosmGLMM(approx_rank=k)``
     - Auto: full rank when ``n_spots ≤ 5 000``, else ``⌈4√n⌉``
     - Fewer eigenvectors → faster fitting and lower memory (with accuracy loss).
       Pass ``None`` to force full rank regardless of ``n_spots``.
   * - **Fitting method**
     - ``SplisosmGLMM(fitting_method=...)``
     - ``"joint_gd"``
     - ``"joint_gd"`` maximises the joint likelihood by gradient descent —
       fastest.  ``"marginal_gd"`` / ``"marginal_newton"`` integrate out the
       random effect via second-order Laplace approximation and give more accurate
       variance posteriors at but are significantly more expensive.
   * - **Parallel gene fitting** (CPU only)
     - ``fit(n_jobs=N, batch_size=B)``
     - ``n_jobs=1``, ``batch_size=1``
     - Near-linear speed-up with ``n_jobs`` on multi-core CPUs; 
       disabled automatically when ``device != "cpu"``.
       ``batch_size > 1`` requires ``group_gene_by_n_iso=True`` in
       ``setup_data``, which batches genes with the same number of isoforms together 
       to utilize PyTorch's efficient batched operations.
   * - **Model complexity**
     - ``SplisosmGLMM(model_type=...)``
     - ``"glmm-full"``
     - ``"glm"`` fits a GLM without any random effect — fastest but ignores
       spatial autocorrelation and may inflate FDR.  Use for quick
       screening or as a baseline.
   * - **Variance sharing across isoforms**
     - ``SplisosmGLMM(share_variance=False)``
     - ``True`` (one shared σ)
     - ``False`` fits one variance component per isoform — more flexible
       but slower and prone to overfitting for genes with many isoforms.
   * - **Other GLMM configuration**
     - ``SplisosmGLMM(...)``
     - See :class:`SplisosmGLMM <splisosm.hyptest_glmm.SplisosmGLMM>` docstring for details
     - Adjust ``var_fix_sigma`` to disable total variance estimation and fix σ to a constant value.

.. raw:: html

   </details>
