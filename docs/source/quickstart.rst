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
     - optional via ``null_configs={"n_probes": m}`` / ``{"approx_rank": k}`` for SV, and ``gpr_configs={"covariate": {"n_inducing": m}}`` for DU.
   * - :class:`~splisosm.SplisosmFFT`
     - ✓
     - ✓
     - SpatialData
     - grid only
     - fastest
     - N/A (FFT-based; no eigendecomposition)
   * - :class:`~splisosm.SplisosmGLMM`
     - ⚠️
     - ✓
     - AnnData
     - any
     - slow
     - optional via ``approx_rank=k`` at ``SplisosmGLMM()`` construction time.

.. note::

   **SplisosmNP vs SplisosmFFT:** same test, different assumptions, potentially different results.

.. raw:: html

   <details>
   <summary>Show details</summary>

   While classes compute the same HSIC test statistic and use the same spectrum-based null, 
   results can still differ because SplisosmFFT (i) assumes <b>periodic boundaries</b> 
   (the grid wraps around, so edge bins become neighbours) and
   (ii) operates on the <b>full raster grid</b>, zero-padding unobserved bins. On the other hand,
   SplisosmNP works only on the observed spots with a k-NN graph that has no wrap-around.  
   The discrepancy is small when the grid is densely observed and 
   the tissue is far from the slide edges (i.e. boundaries are already all zero).

.. raw:: html

   </details>

Inputs and outputs
------------------

Expected input data format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmNP` and :class:`~splisosm.SplisosmGLMM` accept an
`AnnData <https://anndata.readthedocs.io/en/latest/index.html>`_ of shape ``(n_spots, n_isoforms)``:

- ``.layers[<layer>]`` — raw isoform counts.
- ``.var`` — isoform metadata with at least a ``<group_iso_by>`` column
  (e.g. ``"gene_symbol"``, ``"gene_ids"``) assigning each isoform to a gene.
- Spatial information, either (or both) of:

  - ``.obsm[<spatial_key>]`` — an ``(n_spots, n_dim)`` coordinate array
    (≥2 dimensions; Scanpy/Squidpy use ``"spatial"``).
  - ``.obsp[<adj_key>]`` — a pre-built ``(n_spots, n_spots)`` adjacency matrix
    (e.g. ``"connectivities"`` from `scanpy.pp.neighbors <https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html>`_).

:class:`~splisosm.SplisosmFFT` instead takes a `SpatialData <https://spatialdata.scverse.org/en/latest/index.html>`_ 
object whose table follows the same conventions, plus rasterisation metadata:

- ``sdata.tables[<table_name>]`` — an AnnData of shape
  ``(n_bins, n_isoforms)`` with the ``.layers``/``.var`` fields above plus
  ``.obs[[<row_key>, <col_key>]]`` (e.g. ``"array_row"`` / ``"array_col"``) for
  grid rasterisation.
- ``sdata.shapes[<bins>]`` — the grid of bins where each row represents a bin, and the ``"geometry"`` column contains the bin geometries. 
  See the `Intro to SpatialData tutorial <https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/intro.html#shapes>`_ for details.
  Internally, the count table is converted into an image-like square grid using `spatialdata.rasterize_bins <https://spatialdata.scverse.org/en/latest/api/operations.html#spatialdata.rasterize_bins>`_,
  with zero-padding at unobserved locations.

See the :doc:`Feature Quantification page <txquant>` for platform-specific
loaders and preprocessing recipes, and
:func:`splisosm.utils.prepare_inputs_from_anndata` for parsing details.

Setting up a model
~~~~~~~~~~~~~~~~~~

A typical :class:`~splisosm.SplisosmNP` / :class:`~splisosm.SplisosmGLMM` call:

.. code-block:: python

   model.setup_data(
       adata,                           # AnnData of shape (n_spots, n_isoforms)
       spatial_key="spatial",           # adata.obsm key for coordinates
       adj_key=None,                    # or adata.obsp key — see below
       layer="counts",                  # raw isoform counts layer
       group_iso_by="gene_symbol",      # adata.var column grouping isoforms by gene
       gene_names="gene_symbol",        # adata.var column for gene display names
       # ----- Differential usage testing (optional)
       design_mtx=covariates,           # (n_spots, n_factors)
       covariate_names=covariate_names, # list of covariate names
       # ----- Spot filtering
       min_component_size=1,            # set > 1 to drop small disconnected tissue fragments
       # ----- Feature filtering (applied after spot filtering)
       min_counts=10,                   # min total counts per isoform to keep
       min_bin_pct=0.01,                # min fraction of spots expressing the isoform
       filter_single_iso_genes=True,    # SplisosmNP/FFT only; GLMM always requires ≥2
   )

For :class:`~splisosm.SplisosmFFT`, the grid metadata replaces ``spatial_key``:

.. code-block:: python

   model.setup_data(
       sdata,                      # SpatialData object
       bins="Visium_HD_Mouse_Brain_square_016um",  # sdata.shapes key (the grid of bins)
       table_name="square_016um",  # sdata.tables key (the isoform AnnData)
       col_key="array_col",        # column in sdata.tables[table_name].obs for column indices
       row_key="array_row",        # column in sdata.tables[table_name].obs for row indices
       layer="counts",
       group_iso_by="gene_ids",
       gene_names="gene_name",
       min_counts=10,
       min_bin_pct=0.01,
   )

Non-spatial single-cell data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmNP` and :class:`~splisosm.SplisosmGLMM` work on single-cell data without spatial information.
In this case, the tests find and explain variability along a pre-computed graph whose
adjacency matrix is supplied via ``adj_key`` (e.g., ``.obsp["connectivities"]`` from ``scanpy.pp.neighbors``).

.. code-block:: python

   import scanpy as sc

   sc.pp.neighbors(adata)   # writes adata.obsp["connectivities"]

   model.setup_data(
       adata,
       adj_key="connectivities",
       layer="counts",
       group_iso_by="gene_symbol",
   )
   model.test_spatial_variability(method="hsic-ir")   # works without coordinates

Alternatively, pass a low-dimensional embedding (e.g. PCA/UMAP) as pseudo-coordinates
via ``adata.obsm[spatial_key]``; :class:`~splisosm.SplisosmNP` natively supports
arrays of any dimensionality ≥ 2. Interpret such "spatial" patterns with care,
and avoid circularity if the isoform data was itself used to compute the embedding.

.. note::

   The conditional DU test ``test_differential_usage(method="hsic-gp")`` fits
   a Gaussian process using raw spatial coordinates. It will raise a targeted
   ``ValueError`` when called on an adjacency-only setup.
   Use ``method="hsic"`` (unconditional) or supply an embedding via ``spatial_key`` instead.

Outputs
~~~~~~~

All classes share the same output convention, accessed via
:meth:`~splisosm.SplisosmNP.get_formatted_test_results`:

- ``"sv"`` — per-gene SV test statistics and p-values (one row per gene).
- ``"du"`` — DU test statistics and p-values (one row per gene-covariate pair).

Both include a Benjamini-Hochberg adjusted ``pvalue_adj`` column. For DU tests, the adjustment is performed per covariate across tested genes.

.. note::
    Since ``v1.1.0``, the ``approx_rank`` argument was moved out of ``setup_data``:

    * **SplisosmNP** — pass ``null_configs={"n_probes": m}`` to control
      Hutchinson cumulant probes, or ``null_configs={"approx_rank": k}`` to cap
      the eigenvalue rank used for the HSIC null distribution.
      The same ``n_probes`` setting controls Welch Hutchinson trace probes
      when the CAR covariance is implicit.
    * **SplisosmGLMM** — pass ``approx_rank=k`` to :class:`~splisosm.SplisosmGLMM`
      at construction time to control the low-rank spatial kernel approximation.
    * **SplisosmFFT** uses FFT-based convolutions instead of eigendecomposition.


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
      adata_ont, spatial_key="spatial", 
      layer="counts", group_iso_by="gene_symbol"
   )
   model.test_spatial_variability(method="hsic-ir")
   df_sv_res = model.get_formatted_test_results("sv")

.. raw:: html

   <details>
   <summary>Show full call with all defaults</summary>
   <br>

.. code-block:: python

   model = SplisosmNP(
       k_neighbors=4,        # k-NN graph degree for CAR spatial kernel
       rho=0.99,             # spatial autocorrelation strength (0 < rho < 1)
       standardize_cov=True, # set kernel diagonal to 1 (to downweight outliers)
   )
   model.setup_data(
       adata=adata_ont,
       spatial_key="spatial",    # key in adata.obsm for 2-D coordinates
       adj_key=None,             # key in adata.obsp if using precomputed adjacency
       layer="counts",
       group_iso_by="gene_symbol",
       min_counts=10,            # min total counts per isoform to keep
       min_bin_pct=0.0,          # min fraction of spots expressing the isoform
       filter_single_iso_genes=True,
       min_component_size=1,     # set > 1 to drop small tissue fragments
       skip_spatial_kernel=False,  # True → use IdentityKernel (DU-only mode)
   )
   model.test_spatial_variability(
       method="hsic-ir",          # 'hsic-ir' | 'hsic-gc' | 'hsic-ic' | 'spark-x'
       ratio_transformation="none",  # 'none' | 'clr' | 'ilr' | 'alr'
       nan_filling="mean",           # if 'none', use only non-zero spots per gene (slow)
       null_method="liu",            # 'liu' (default) | 'welch' (scaled chi²) | 'perm'
       null_configs=None,            # e.g. {"n_probes": 60} or {"approx_rank": 20}
       n_jobs=-1,                    # gene-level parallelism; -1 = all CPUs
       print_progress=True,
   )
   df_sv_res = model.get_formatted_test_results("sv")

.. raw:: html

   </details>

**SplisosmFFT** (regular grids only — SpatialData input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from splisosm import SplisosmFFT

   model = SplisosmFFT()
   model.setup_data(
       sdata,
       bins="Visium_HD_Mouse_Brain_square_016um",
       table_name="square_016um",
       col_key="array_col",
       row_key="array_row",
       layer="counts",
       group_iso_by="gene_ids",
   )
   model.test_spatial_variability(method="hsic-ir")
   df_sv_res = model.get_formatted_test_results("sv")

.. raw:: html

   <details>
   <summary>Show full call with all defaults</summary>
   <br>

.. code-block:: python

   model = SplisosmFFT(
       neighbor_degree=1,   # grid adjacency degree (1 = 4-connectivity, 2 = 8-connectivity)
       rho=0.99,            # spatial autocorrelation strength
       workers=None,        # scipy.fft thread count; None = all available CPUs
   )
   model.setup_data(
       sdata,
       bins="Visium_HD_Mouse_Brain_square_016um",
       table_name="square_016um",
       col_key="array_col",
       row_key="array_row",
       layer="counts",
       group_iso_by="gene_ids",
       gene_names="gene_name",  # adata.var column to change gene display names in results
       min_counts=10,
       min_bin_pct=0.0,
       filter_single_iso_genes=True,
   )
   model.test_spatial_variability(
       method="hsic-ir",    # 'hsic-ir' | 'hsic-gc' | 'hsic-ic'
       ratio_transformation="none",  # 'none' | 'clr' | 'ilr' | 'alr'
       n_jobs=-1,           # gene-level parallelism; -1 = all CPUs
       print_progress=True,
   )
   df_sv_res = model.get_formatted_test_results("sv")

.. raw:: html

   </details>

Testing for differential isoform usage (DU)
--------------------------------------------

DU tests identify genes whose isoform usage is associated with a covariate (e.g., spatial domain, RBP expression), conditioned on spatial correlation.

**SplisosmNP** (non-parametric, HSIC-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses Gaussian process regression (GPR) to remove spatial autocorrelation.

.. code-block:: python

   from splisosm import SplisosmNP

   # Focus on SVP genes from the SV step
   svp_genes = df_sv_res.query("pvalue_adj < 0.05")['gene'].tolist()
   adata_svp = adata_ont[:, adata_ont.var['gene_symbol'].isin(svp_genes)].copy()

   model = SplisosmNP()
   model.setup_data(
       adata_svp, spatial_key="spatial",
       layer="counts", group_iso_by="gene_symbol",
       design_mtx=covariates, covariate_names=covariate_names,
       skip_spatial_kernel=True,  # DU-only; skip expensive CAR kernel construction
   )
   model.test_differential_usage(method="hsic-gp")
   df_du_res = model.get_formatted_test_results("du")

.. raw:: html

   <details>
   <summary>Show full call with all defaults</summary>
   <br>

.. code-block:: python

   model = SplisosmNP()
   model.setup_data(
       adata=adata_svp,
       spatial_key="spatial",
       adj_key=None,                    # key in adata.obsp if using precomputed adjacency
       layer="counts",
       group_iso_by="gene_symbol",
       design_mtx=covariates,           # (n_spots, n_factors) array/tensor/DataFrame
       covariate_names=covariate_names, # list of covariate display names
       skip_spatial_kernel=True,        # skip CAR kernel — not needed for DU
       min_counts=10,
       min_bin_pct=0.0,
       filter_single_iso_genes=True,
       min_component_size=1,     # set > 1 to drop small tissue fragments
   )
   model.test_differential_usage(
       method="hsic-gp",              # 'hsic' | 'hsic-gp' | 't-fisher' | 't-tippett'
       ratio_transformation="none",   # 'none' | 'clr' | 'ilr' | 'alr'
       nan_filling="mean",            # if 'none', use only non-zero spots per gene (slow)
       residualize="cov_only",        # 'cov_only' (faster) | 'both' (more conservative)
       gpr_backend="sklearn",         # 'sklearn' | 'gpytorch' (FITC sparse GP with GPU)
       gpr_configs=None,              # e.g. {"covariate": {"n_inducing": 500}}
       n_jobs=-1,                     # gene-level parallelism; -1 = all CPUs
       print_progress=True,
   )
   df_du_res = model.get_formatted_test_results("du")

.. raw:: html

   </details>


**SplisosmFFT** (same tests but for regular grids only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same test as :class:`~splisosm.SplisosmNP` but with FFT-accelerated GPR for regular grid data.

.. code-block:: python

   from splisosm import SplisosmFFT

   model = SplisosmFFT()
   model.setup_data(
       sdata,
       bins="Visium_HD_Mouse_Brain_square_016um",
       table_name="square_016um_svp",
       design_mtx="square_016um_rbp_sve",  # design stored as a separate sdata table
       col_key="array_col", row_key="array_row",
       layer="counts", group_iso_by="gene_ids",
   )
   model.test_differential_usage(method="hsic-gp")
   df_du_res = model.get_formatted_test_results("du")

.. raw:: html

   <details>
   <summary>Show full call with all defaults</summary>
   <br>

.. code-block:: python

   model = SplisosmFFT(
       neighbor_degree=1,
       rho=0.99,
       workers=None,   # scipy.fft thread count; None = all CPUs
   )
   model.setup_data(
       sdata,
       bins="Visium_HD_Mouse_Brain_square_016um",
       table_name="square_016um_svp",
       design_mtx="square_016um_rbp_sve",
       col_key="array_col",
       row_key="array_row",
       layer="counts",
       group_iso_by="gene_ids",
       gene_names="gene_name",
       min_counts=10,
       min_bin_pct=0.0,
       filter_single_iso_genes=True,
   )
   model.test_differential_usage(
       method="hsic-gp",              # 'hsic' | 'hsic-gp' | 't-fisher' | 't-tippett'
       ratio_transformation="none",   # 'none' | 'clr' | 'ilr' | 'alr'
       residualize="cov_only",        # 'cov_only' | 'both'
       gpr_configs=None,              # e.g. {"covariate": {"length_scale_bounds": "fixed"}}
       n_jobs=-1,                     # gene-level parallelism
       print_progress=True,
   )
   df_du_res = model.get_formatted_test_results("du")

.. raw:: html

   </details>


**SplisosmGLMM** (parametric, GLM/GLMM-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fits a multinomial GLMM with a Gaussian random field spatial random effect.
Set ``var_fix_sigma=False`` to learn the total variance jointly.
With the default settings (``var_fix_sigma=True``, ``max_epochs=500``), SV and DU
hypothesis tests are **conservative** (better false positive control but
reduced power).

.. code-block:: python

   from splisosm import SplisosmGLMM

   # DU testing (score test conditioned on spatial random effects)
   model = SplisosmGLMM(model_type="glmm-full")
   model.setup_data(
       adata_svp, spatial_key="spatial",
       layer="counts", group_iso_by="gene_symbol",
       group_gene_by_n_iso=True,
       design_mtx=covariates, covariate_names=covariate_names,
   )
   model.fit(with_design_mtx=False)
   model.test_differential_usage(method="score")
   df_du_res = model.get_formatted_test_results("du")

.. raw:: html

   <details>
   <summary>Show full call with all defaults</summary>
   <br>

.. code-block:: python

   model = SplisosmGLMM(
       model_type="glmm-full",      # 'glmm-full' | 'glmm-null' | 'glm'
       var_fix_sigma=True,          # True → conservative tests; False → more power but may inflate FPR
       init_ratio="uniform",        # 'uniform' | 'observed'
       fitting_method="joint_gd",   # 'joint_gd' | 'joint_newton' | 'marginal_gd' | 'marginal_newton'
       fitting_configs={"max_epochs": 500},
       device="cpu",                # 'cpu' | 'cuda' (NVIDIA) | 'mps' (Apple Silicon)
       # approx_rank: omit for auto (default) | None (force full rank) | int (fixed low-rank)
   )
   model.setup_data(
       adata=adata_svp,
       spatial_key="spatial",
       adj_key=None,                   # key in adata.obsp if using precomputed adjacency
       layer="counts",
       group_iso_by="gene_symbol",
       group_gene_by_n_iso=True,       # required for batch_size > 1
       design_mtx=covariates,
       covariate_names=covariate_names,
       min_counts=10,            # min total counts per isoform to keep
       min_bin_pct=0.0,          # min fraction of spots expressing the isoform
       min_component_size=1,     # set > 1 to drop small tissue fragments
   )
   model.fit(
       n_jobs=1,               # parallel gene fitting (CPU only; auto-disabled on GPU)
       batch_size=1,           # genes per batch; > 1 requires group_gene_by_n_iso=True
       with_design_mtx=False,  # False → score test (recommended); True → Wald test
       from_null=False,        # if True, initialize from glmm-null model 
       print_progress=True,
   )
   model.test_differential_usage(
       method="score",   # 'score' | 'wald' (not recommended)
       print_progress=True,
   )
   df_du_res = model.get_formatted_test_results("du")

.. raw:: html

   </details>

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
   * - **Ratio transformation**
     - ``ratio_transformation=`` in ``test_spatial_variability`` / ``test_differential_usage``
     - ``"none"`` (no compositional constraint)
     - log-based transformations (``"clr"``, ``"ilr"``, ``"alr"``) require pseudocounts to handle zero ratios; 
       ``"radial"`` is not calibrated. Performance of ``"none"`` is better in most cases.
   * - **SV null approximation**
     - ``null_method=`` in ``test_spatial_variability`` / ``run_hsic_gc``
     - ``"liu"`` (Liu's chi-square mixture)
     - ``"welch"`` (Welch–Satterthwaite scaled chi-squared) avoids higher
       cumulants and eigendecomposition — fastest for very large *n*.  ``"perm"``
       uses permutation; slowest but assumption-free.  Deprecated aliases are
       mapped automatically: ``"eig"`` → ``"liu"``, ``"clt"`` / ``"trace"`` →
       ``"welch"``.
   * - **SV null low-rank** (when ``null_method="liu"``)
     - ``null_configs={"approx_rank": k}`` in ``test_spatial_variability`` /
       ``run_hsic_gc``
     - Optional; large implicit kernels otherwise use Hutchinson cumulants
     - Restricts test statistic and null computation to the top-*k* eigenvalues; 
       significant speedup for large *n* with potential power loss for high-frequency patterns 
       (usually rare in real spatial data).
   * - **SV probes**
     - ``null_configs={"n_probes": m}``
     - Optional
     - ``n_probes`` controls Liu cumulant probes and Welch Hutchinson traces
       for implicit CAR kernels.
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
   * - **Parallel jobs**
     - ``n_jobs=`` in ``test_spatial_variability`` / ``test_differential_usage``
     - ``-1`` (all CPUs)
     - Number of joblib workers for gene-wise computation.
       Uses ``prefer="threads"`` to avoid pickling large shared objects.
       When ``gpr_backend="gpytorch"`` with ``device != "cpu"``, parallelism
       is automatically disabled (CUDA not thread-safe).
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
   * - **Ratio transformation**
     - ``ratio_transformation=`` in ``test_spatial_variability`` / ``test_differential_usage``
     - ``"none"`` (no compositional constraint)
     - log-based transformations (``"clr"``, ``"ilr"``, ``"alr"``) require pseudocounts to handle zero ratios; 
       ``"radial"`` is not calibrated. Performance of ``"none"`` is better in most cases.
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
     - ``"auto"``: full rank when ``n_spots ≤ 5000``, else ``⌈4√n⌉``
     - Fewer eigenvectors → faster fitting and lower memory (with accuracy loss).
       Pass ``None`` to force full rank regardless of ``n_spots``;
       pass an integer to set a fixed rank.
   * - **Fitting method**
     - ``SplisosmGLMM(fitting_method=...)``
     - ``"joint_gd"``
     - ``"joint_gd"`` maximises the joint likelihood by gradient descent —
       fastest.  ``"joint_newton"`` adds a Newton step for the variance
       parameters.  ``"marginal_gd"`` / ``"marginal_newton"`` integrate out the
       random effect via second-order Laplace approximation and give more accurate
       variance posteriors but are significantly more expensive
       (O(n³) Cholesky per epoch; see warning for ``n_spots > 300``).
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
   * - **Fix total variance**
     - ``SplisosmGLMM(var_fix_sigma=...)``
     - ``True`` (σ frozen at Fano-factor estimate)
     - When ``True``, only the spatial proportion θ is learned, producing
       conservative tests (near-zero FPR).  Set to ``False`` to learn σ
       jointly — may increase SV power but inflates FPR for both SV and
       DU tests.
   * - **Isoform ratio initialization**
     - ``SplisosmGLMM(init_ratio=...)``
     - ``"uniform"``
     - ``"uniform"`` initialises isoform ratios to equal proportions — fast
       convergence.  ``"observed"`` initialises from raw count ratios — slower
       but may improve ratio estimates for high-count data.

.. raw:: html

   </details>
