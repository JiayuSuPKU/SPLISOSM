Feature Quantification
====================================

This page describes how to obtain probe/peak/isoform-level quantification for various spatial transcriptomics (ST) platforms as input to SPLISOSM. 
To check if you already have a compatible ``AnnData`` or ``SpatialData`` object, see :ref:`quickstart:Expected input data format`.

Platform overview
-----------------

The table below summarizes the supported ST platforms, the type of isoform feature used, and the recommended quantification approach.

.. list-table::
   :header-rows: 1
   :widths: 22 28 22 28

   * - Platform type
     - Example platforms
     - Feature
     - Quantification approach
   * - **Long-read ST**
     - ONT + Visium (e.g., SiT :cite:`lebrigand2023spatial`)
     - Full-length transcript isoform
     - Any long-read aligner/quantifier (e.g., `IsoQuant <https://ablab.github.io/IsoQuant/>`_, `kallisto <https://pachterlab.github.io/kallisto/>`_)
   * - **Short-read 3\' end**
     - 10x Visium/Visium HD (fresh-frozen), Slide-seqV2
     - 3\' end diversity (TREND) event (peak)
     - `Sierra <https://github.com/VCCRI/Sierra>`__ for *de novo* peak calling
   * - **Short-read targeted**
     - 10x Visium v2 CytAssist (FFPE), 10x Visium HD (FFPE), 10x Flex
     - Exon/junction probe
     - Space Ranger output directly (``raw_probe_bc_matrix.h5``)
   * - **In situ targeted**
     - 10x Xenium Prime 5K
     - Exon/junction probe (codeword)
     - Xenium Ranger output directly (``transcripts.zarr.zip``)

Expected data format
--------------------

See :ref:`quickstart:Expected input data format` in the Quick Start for the expected
``AnnData`` / ``SpatialData`` layout. The rest of this page covers how to
*produce* such inputs from each platform's raw output.

For :class:`~splisosm.SplisosmFFT`, counts are internally rasterised into square
bins via `spatialdata.rasterize_bins
<https://spatialdata.scverse.org/en/latest/api/operations.html#spatialdata.rasterize_bins>`_
(unobserved bins are zero-padded). The ``SpatialData`` table must therefore also
carry ``.uns['spatialdata_attrs']`` with ``spatial_key`` / ``row_key`` /
``col_key`` entries — see the `SpatialData tutorial on table metadata
<https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html#table-metadata-annotation-targets>`_.


Long-read ST data
-----------------

SPLISOSM is compatible with any long-read quantification tool that produces an isoform-by-spot count matrix. Popular options include `IsoQuant <https://ablab.github.io/IsoQuant/>`_ and `kallisto <https://pachterlab.github.io/kallisto/>`_.

.. note::
   Detection power scales linearly with sequencing depth (:ref:`FAQ <faq:umi-depth>`). Any processing choice that increases captured UMIs per spot will improve results.

Tested platforms:

- 10x Visium + ONT (SiT :cite:`lebrigand2023spatial`)
- 10x Visium HD + ONT (the EPI2ME dataset)

:doc:`The Visium HD ONT tutorial <tutorials/visiumhd_ont>` shows how to load a preprocessed ONT + Visium HD dataset and run SPLISOSM. 
The dataset is publicly accessible and can be downloaded from `EPI2ME <https://epi2me.nanoporetech.com/visium_hd_2025.06/>`_. 
Transcript assignment and quantification were performed using ``minimap2``, ``Stringtie``, and ``FLAMES``. See the `EPI2ME workflow documentation <https://epi2me.nanoporetech.com/epi2me-docs/workflows/wf-single-cell/#transcript-assignment>`_ for details.
The generated ``SpatialData`` object has the following structure (example):

.. code-block:: text

  SpatialData object
  ├── Images
  │     ├── '_hires_image': DataArray[cyx] (3, 3000, 3200)
  │     └── '_lowres_image': DataArray[cyx] (3, 563, 600)
  ├── Shapes
  │     ├── '_square_002um': GeoDataFrame shape: (7857218, 1) (2D shapes)
  │     ├── '_square_008um': GeoDataFrame shape: (492663, 1) (2D shapes)
  │     └── '_square_016um': GeoDataFrame shape: (123658, 1) (2D shapes)
  └── Tables
        ├── 'square_002um': AnnData (7857218, 48001)
        ├── 'square_008um': AnnData (492663, 48001)
        └── 'square_016um': AnnData (123658, 48001)
  with coordinate systems:
      ▸ '', with elements:
          _hires_image (Images), _lowres_image (Images), _square_002um (Shapes), _square_008um (Shapes), _square_016um (Shapes)
      ▸ '_downscaled_hires', with elements:
          _hires_image (Images), _square_002um (Shapes), _square_008um (Shapes), _square_016um (Shapes)
      ▸ '_downscaled_lowres', with elements:
          _lowres_image (Images), _square_002um (Shapes), _square_008um (Shapes), _square_016um (Shapes)


Short-read 3' end ST data
--------------------------

We recommend using `Sierra <https://github.com/VCCRI/Sierra/>`__ to call 3' end diversity (TREND) events *de novo* from Space Ranger BAM files. 
See the `Sierra vignette <https://github.com/VCCRI/Sierra/wiki/Sierra-Vignette>`_ for installation and usage details.

Tested platforms: 

- 10x Visium 3' (fresh-frozen)
- Slide-seqV2
- 10x Visium HD 3' (fresh-frozen)

.. warning::
   When processing multiple samples, avoid Sierra's ``MergePeakCoordinates`` function — it can produce overlapping peak definitions.

.. note::
  If you encounter issues when running Sierra's ``CountPeaks``, the following workaround quantification workflow may be helpful:

10x Visium HD 3' data (custom quantification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For 10x Visium HD 3' data, we have prepared a hybrid quantification workflow where Sierra is used for peak calling but not counting.
See the attached bash script (`scripts/visiumhd_3p_trend_quant.sh <https://github.com/JiayuSuPKU/SPLISOSM/blob/main/scripts/visiumhd_3p_trend_quant.sh>`_).

.. raw:: html

   <details>
   <summary><strong>Step-by-step workflow explained (click to expand)</strong></summary>
   <br>


0. Run ``spaceranger count`` (>=v4.0) with ``create-bam=true`` to get the BAM file.
1. Run Sierra ``FindPeaks`` (plus ``AnnotatePeaksFromGTF``) on the Space Ranger BAM.
2. Convert peak definitions to SAF/BED and run ``featureCounts -R BAM`` to add peak-level tags using `Subread <https://github.com/ShiLab-Bioinformatics/subread>`_.
3. Count UMIs per peak per barcode with ``umi_tools count``. 
4. Build a 10x-compatible feature-barcode matrix and save as ``raw_probe_bc_matrix.h5`` under the output directory ``outs/binned_outputs/square_002um``.
5. Load peak-level data into ``SpatialData`` with :func:`splisosm.io.load_visiumhd_probe`.

.. raw:: html

   </details>

The generated ``SpatialData`` object has the following structure (example):

.. code-block:: text

  SpatialData object
  ├── Images
  │     ├── '_hires_image': DataArray[cyx] (3, 5492, 6000)
  │     └── '_lowres_image': DataArray[cyx] (3, 549, 600)
  ├── Shapes
  │     ├── '_cell_segmentations': GeoDataFrame shape: (84031, 2) (2D shapes)
  │     ├── '_square_002um': GeoDataFrame shape: (5998466, 1) (2D shapes)
  │     ├── '_square_008um': GeoDataFrame shape: (376419, 1) (2D shapes)
  │     └── '_square_016um': GeoDataFrame shape: (94592, 1) (2D shapes)
  └── Tables
        ├── 'cell_segmentations': AnnData (84031, 19575)
        ├── 'square_002um': AnnData (5998466, 19575)
        ├── 'square_008um': AnnData (376419, 19575)
        └── 'square_016um': AnnData (94592, 19575)
  with coordinate systems:
      ▸ '', with elements:
          _hires_image (Images), _lowres_image (Images), _cell_segmentations (Shapes), _square_002um (Shapes), _square_008um (Shapes), _square_016um (Shapes)
      ▸ '_downscaled_hires', with elements:
          _hires_image (Images), _cell_segmentations (Shapes), _square_002um (Shapes), _square_008um (Shapes), _square_016um (Shapes)
      ▸ '_downscaled_lowres', with elements:
          _lowres_image (Images), _cell_segmentations (Shapes), _square_002um (Shapes), _square_008um (Shapes), _square_016um (Shapes)

For downstream analysis of TREND spatial patterns, see the :doc:`Visium HD 3' tutorial <tutorials/visiumhd_3prime>`.

10x Visium / Slide-seqV2 3' data (Sierra quantification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Running Sierra on Space Ranger BAM**

.. code-block:: r

   # load the R package 
   library(Sierra)
   FindPeaks(
     output.file = '${peak_file}',
     gtf.file = '${gtf_file}', # SpaceRanger reference gtf file
     bamfile = '${bam_file}',  # bam file from SpaceRanger
     junctions.file = '${junc_file}', # junctions bed files extracted using regtools junctions extract
   # optional arguments for retaining low-abundance peaks
   #   min.jcutoff.prop = 0.0,
   #   min.cov.prop = 0.0,
   #   min.peak.prop = 0.0
   )

   CountPeaks(
     peak.sites.file = '${peak_file}',
     gtf.file = '${gtf_file}',
     bamfile = '${bam_file}',
     whitelist.file = '${whitelist_file}', # barcodes.tsv file from SpaceRanger
     output.dir = '${output_dir}',
   )

.. raw:: html

   <details>
   <summary><strong>Converting Sierra output to AnnData (click to expand)</strong></summary>
   <br>

.. code-block:: python

    import scanpy as sc
    import pandas as pd
    from splisosm.io import load_visium_sp_meta

    sierra_out_dir = "path/to/sierra/output"  # 'output.dir' in CountPeaks
    sp_meta_dir = "path/to/visium/spatial/metadata"  # 'spatial' directory in 10X Visium data

    # load the Sierra outputs as an AnnData object
    adata = sc.read(
        f"{sierra_out_dir}/matrix.mtx.gz",
        cache_compression='cache_compression',
    ).T

    # load TREND peak metadata
    peaks = pd.read_csv(
        f"{sierra_out_dir}/sitenames.tsv.gz",
        header=None,
        sep="\t",
    )
    df_var = peaks[0].str.split(':', expand=True)
    df_var.columns = ['gene_symbol', 'chr', 'position', 'strand']
    df_var.index = peaks[0].values

    # load spatial barcode metadata
    barcodes = pd.read_csv(f"{sierra_out_dir}/barcodes.tsv.gz", header=None)

    # add metadata to the AnnData object
    adata.var_names = peaks[0].values
    adata.obs_names = barcodes[0].values
    adata.var = df_var
    adata.var['gene_id'] = adata.var['gene_symbol']

    # load Visium spatial metadata
    adata = load_visium_sp_meta(adata, f"{sp_meta_dir}/", library_id='adata_peak')
    adata = adata[adata.obs['in_tissue'].astype(bool), :].copy()


.. raw:: html

   </details>


.. raw:: html

   <details>
   <summary><strong>Custom AnnData filtering (click to expand)</strong></summary>
   <br>

SPLISOSM compares all events associated with the same gene and is agnostic to their specific structure. Filter out low-abundance features before testing to reduce computation and improve power.


.. code-block:: python

    # filter out lowly expressed peaks
    sc.pp.filter_genes(adata, min_cells=0.01 * adata.shape[0])

    # extract gene symbols and peak ids
    df_iso_meta = adata.var.copy()  # gene_symbol, chr, position, strand, gene_id
    df_iso_meta['peak_id'] = adata.var_names

    # prepare gene-level metadata
    df_gene_meta = df_iso_meta.groupby('gene_symbol').size().reset_index(name='n_peak')
    df_gene_meta = df_gene_meta.set_index('gene_symbol')

    print(f"Number of spots: {adata.shape[0]}")
    print(f"Number of genes before QC: {df_gene_meta.shape[0]}")
    print(f"Number of peaks before QC: {adata.shape[1]}")
    print(f"Average number of peaks per gene before QC: {adata.shape[1] / df_gene_meta.shape[0]}")

    # calculate the total counts per gene
    mapping_matrix = pd.get_dummies(df_iso_meta['gene_symbol'])
    mapping_matrix = mapping_matrix.loc[df_iso_meta.index, df_gene_meta.index]
    isog_counts = adata[:, mapping_matrix.index].layers['counts'] @ mapping_matrix

    # calculate mean and sd of total gene counts
    df_gene_meta['pct_spot_on'] = (isog_counts > 0).mean(axis=0)
    df_gene_meta['count_avg'] = isog_counts.mean(axis=0)
    df_gene_meta['count_std'] = isog_counts.std(axis=0)

    # filter out lowly expressed genes
    _gene_keep = df_gene_meta['pct_spot_on'] > 0.01
    # _gene_keep = (df_gene_meta['count_avg'] > 0.5) & _gene_keep

    # filter out genes with single isoform
    _gene_keep = (df_gene_meta['n_peak'] > 1) & _gene_keep

    # filter for isoforms
    _iso_keep = df_iso_meta['gene_symbol'].isin(df_gene_meta.index[_gene_keep])

    # update feature meta
    df_gene_meta = df_gene_meta.loc[_gene_keep, :]
    adata = adata[:, _iso_keep]
    adata.var = df_iso_meta.loc[_iso_keep, :].copy()

    print(f"Number of genes after QC: {sum(_gene_keep)}")
    print(f"Number of peaks after QC: {sum(_iso_keep)}")
    print(f"Average number of peaks per gene after QC: {sum(_iso_keep) / sum(_gene_keep)}")

.. raw:: html

   </details>


Short-read targeted ST data
----------------------------

10x Visium and Visium HD FFPE kits use fixed pan-genome probes for read enrichment. SPLISOSM treats probe-level counts as the isoform-level quantification — probes targeting the same gene are grouped and tested jointly.

Tested platforms:

- 10x Visium FFPE (v2 CytAssist Spatial Gene Expression)
- 10x Visium HD FFPE

10x Visium HD FFPE
^^^^^^^^^^^^^^^^^^^

Given Space Ranger output, the following code creates a ``SpatialData`` object with probe-level counts (from ``raw_probe_bc_matrix.h5``):

.. code-block:: python

   from splisosm.io import load_visiumhd_probe

   sdata = load_visiumhd_probe(
     path=visium_hd_outs,
     bin_sizes=[2, 8, 16],
     filtered_counts_file=True,
     load_all_images=False,
     var_names_make_unique=True,
     counts_layer_name="counts",
   )

.. note::

    :func:`splisosm.io.load_visiumhd_probe` uses the ``barcode_mappings.parquet`` file, 
    which contains the spatial mapping information of 2um barcodes to coarser bins and segmented cells. 
    See `10x documentation <https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/segmented-outputs>`_ for details.
    If you don't have this file, please re-run the
    `Space Ranger count pipeline <https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/running-pipelines/space-ranger-count>`_
    with Space Ranger v4.0+.

The generated ``SpatialData`` object has the following structure (example):

.. code-block:: text

  SpatialData object
  ├── Images
  │     ├── 'Visium_HD_Mouse_Brain_full_image': DataTree[cyx] (3, 23947, 18872), (3, 11973, 9436), (3, 5986, 4718), (3, 2993, 2359), (3, 1496, 1179)
  │     ├── 'Visium_HD_Mouse_Brain_hires_image': DataArray[cyx] (3, 6000, 4729)
  │     └── 'Visium_HD_Mouse_Brain_lowres_image': DataArray[cyx] (3, 600, 473)
  ├── Shapes
  │     ├── 'Visium_HD_Mouse_Brain_cell_segmentations': GeoDataFrame shape: (40222, 2) (2D shapes)
  │     ├── 'Visium_HD_Mouse_Brain_square_002um': GeoDataFrame shape: (6296688, 1) (2D shapes)
  │     ├── 'Visium_HD_Mouse_Brain_square_008um': GeoDataFrame shape: (393543, 1) (2D shapes)
  │     └── 'Visium_HD_Mouse_Brain_square_016um': GeoDataFrame shape: (98917, 1) (2D shapes)
  └── Tables
        ├── 'cell_segmentations': AnnData (40222, 55538)
        ├── 'square_002um': AnnData (6296688, 55538)
        ├── 'square_008um': AnnData (393543, 55538)
        └── 'square_016um': AnnData (98917, 55538)
  with coordinate systems:
      ▸ 'Visium_HD_Mouse_Brain', with elements:
          Visium_HD_Mouse_Brain_full_image (Images), Visium_HD_Mouse_Brain_hires_image (Images), Visium_HD_Mouse_Brain_lowres_image (Images), Visium_HD_Mouse_Brain_cell_segmentations (Shapes), Visium_HD_Mouse_Brain_square_002um (Shapes), Visium_HD_Mouse_Brain_square_008um (Shapes), Visium_HD_Mouse_Brain_square_016um (Shapes)
      ▸ 'Visium_HD_Mouse_Brain_downscaled_hires', with elements:
          Visium_HD_Mouse_Brain_hires_image (Images), Visium_HD_Mouse_Brain_cell_segmentations (Shapes), Visium_HD_Mouse_Brain_square_002um (Shapes), Visium_HD_Mouse_Brain_square_008um (Shapes), Visium_HD_Mouse_Brain_square_016um (Shapes)
      ▸ 'Visium_HD_Mouse_Brain_downscaled_lowres', with elements:
          Visium_HD_Mouse_Brain_lowres_image (Images), Visium_HD_Mouse_Brain_cell_segmentations (Shapes), Visium_HD_Mouse_Brain_square_002um (Shapes), Visium_HD_Mouse_Brain_square_008um (Shapes), Visium_HD_Mouse_Brain_square_016um (Shapes)

For downstream analysis, see the :doc:`Visium HD FFPE tutorial <tutorials/visiumhd_ffpe>`.


10x Visium FFPE (v2 CytAssist)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard-resolution Visium CytAssist FFPE uses a probe-based capture workflow.
Space Ranger (**v3.0+**) outputs a ``raw_probe_bc_matrix.h5`` file containing per-probe, per-barcode counts
that SPLISOSM uses directly for probe usage testing.

**Running Space Ranger count**

See the `10x documentation <https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/running-pipelines/probe-based-assay-count-cytassist-gex>`_ for detailed instructions.
Example Space Ranger command:

.. code-block:: bash

   spaceranger count \
       --id="CytAssist_FFPE_Mouse_Brain_Rep1" \
       --transcriptome=refdata-gex-mm10-2020-A \
       --probe-set=CytAssist_FFPE_Mouse_Brain_Rep1_probe_set.csv \
       --fastqs=path/to/fastqs \
       --cytaimage=CytAssist_FFPE_Mouse_Brain_Rep1_image.tif \
       --image=CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.tif \
       --loupe-alignment=CytAssist_FFPE_Mouse_Brain_Rep1_alignment_file.json \
       --slide=V42A20-353 \
       --area=A1 \
       --create-bam=false \
       --localcores=16 \
       --localmem=64


**Loading as AnnData/SpatialData**

.. code-block:: python

   from splisosm.io import load_visium_probe

   adata = load_visium_probe(
       "CytAssist_FFPE_Mouse_Brain_Rep1/outs",
       counts_file="raw_probe_bc_matrix.h5",  # probe-level counts (default)
       counts_layer_name="counts",
   )

The loaded ``AnnData`` has the following structure:

.. code-block:: text

   AnnData object with n_obs x n_vars = 4992 x 21178
       obs: 'filtered_barcodes', 'in_tissue', 'array_row', 'array_col'
       var: 'gene_ids', 'probe_ids', 'feature_types', 'filtered_probes', 'gene_name', 'genome', 'probe_region'
       uns: 'spatial'
       obsm: 'spatial'
       layers: 'counts'

Each row is a Visium spot (barcode) and each column is an individual probe.
The ``gene_ids`` column in ``.var`` groups probes by gene — pass ``group_iso_by="gene_ids"``
and ``gene_names="gene_name"`` to :meth:`~splisosm.SplisosmNP.setup_data`.

To load as a ``SpatialData`` object (e.g., for :class:`~splisosm.SplisosmFFT`), pass ``return_type="spatialdata"``:

.. code-block:: python

   sdata = load_visium_probe(
       "CytAssist_FFPE_Mouse_Brain_Rep1/outs",
       return_type="spatialdata",
   )

.. note::
   On the `Mouse Brain Coronal Section 1 (FFPE) <https://www.10xgenomics.com/datasets/mouse-brain-coronal-section-1-ffpe-2-standard>`_ dataset,
   281 genes with multiple probes passed filtering (``min_counts=10, filter_single_iso_genes=True``),
   of which 83 were identified as spatially variably processed (SVP, HSIC-IR) at FDR < 0.01.

For downstream analysis comparing ``SplisosmNP`` (AnnData) and ``SplisosmFFT`` (SpatialData) spatial variability tests using this dataset,
see the :doc:`Visium FFPE tutorial <tutorials/visium_ffpe>`.


In situ targeted ST data
-------------------------

For imaging-based platforms with exon- or junction-specific probes (e.g., `10x Xenium Prime 5K <https://www.10xgenomics.com/products/xenium-5k-panel>`_), SPLISOSM uses codeword-level counts as isoform proxies. 
Data can be analyzed at single-cell resolution (segmented cells) or on spatially binned spots.

Tested platform:

- 10x Xenium Prime 5K

Given Xenium Ranger output, the following code creates a binned ``SpatialData`` object with codeword-level counts (from ``transcripts.zarr.zip``):

.. code-block:: python

   from splisosm.io import load_xenium_codeword

   sdata = load_xenium_codeword(
     path=xenium_ranger_outs,
     spatial_resolutions=[8.0, 16.0], # None or [] if you only want segmented cell-level data
     quality_threshold=20.0,
     chunk_batch_size=64,
     counts_layer_name="counts",
     build_cell_codeword_table=True,
     create_square_shapes=True,
   )

.. note::

    If your Xenium Output Bundle was generated by older Xenium Ranger versions before v3.1.0, please re-run the
    `Xenium Ranger relabel pipeline <https://www.10xgenomics.com/support/software/xenium-ranger/latest/analysis/running-pipelines/XR-relabel>`_
    to get an updated ``transcripts.zarr.zip`` file.

The generated ``SpatialData`` object has the following structure (example):

.. code-block:: text

  SpatialData object
  ├── Images
  │     └── 'morphology_focus': DataTree[cyx] (4, 23912, 34154), (4, 11956, 17077), (4, 5978, 8538), (4, 2989, 4269), (4, 1494, 2134)
  ├── Labels
  │     ├── 'cell_labels': DataTree[yx] (23912, 34154), (11956, 17077), (5978, 8538), (2989, 4269), (1494, 2134)
  │     └── 'nucleus_labels': DataTree[yx] (23912, 34154), (11956, 17077), (5978, 8538), (2989, 4269), (1494, 2134)
  ├── Points
  │     └── 'transcripts': DataFrame with shape: (<Delayed>, 13) (3D points)
  ├── Shapes
  │     ├── 'cell_boundaries': GeoDataFrame shape: (63173, 1) (2D shapes)
  │     ├── 'nucleus_boundaries': GeoDataFrame shape: (63036, 1) (2D shapes)
  │     ├── 'square_008um_bins': GeoDataFrame shape: (576580, 1) (2D shapes)
  │     └── 'square_016um_bins': GeoDataFrame shape: (144372, 1) (2D shapes)
  └── Tables
        ├── 'square_008um': AnnData (576580, 11163)
        ├── 'square_016um': AnnData (144372, 11163)
        ├── 'table': AnnData (63173, 5006)
        └── 'table_codeword': AnnData (63173, 11163)
  with coordinate systems:
      ▸ 'global', with elements:
          morphology_focus (Images), cell_labels (Labels), nucleus_labels (Labels), transcripts (Points), cell_boundaries (Shapes), nucleus_boundaries (Shapes), square_008um_bins (Shapes), square_016um_bins (Shapes)

For downstream analysis, see the :doc:`Xenium Prime 5K tutorial <tutorials/xenium_prime_5k>`.