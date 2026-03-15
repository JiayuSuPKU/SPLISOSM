Isoform Quantification for ST Data
====================================

This page describes how to obtain isoform-level quantifications for various spatial transcriptomics (ST) platforms as input to SPLISOSM. 
If you already have a compatible ``AnnData`` or ``SpatialData`` object, skip ahead to :ref:`the expected data format <txquant:expected data format>`.

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
     - 3\' end diversity (TREND) event / peak
     - `Sierra <https://github.com/VCCRI/Sierra>`__ for *de novo* peak calling
   * - **Short-read targeted**
     - 10x Visium/Visium HD (FFPE), 10x Flex
     - Exon/junction probe
     - Space Ranger output directly (``raw_probe_bc_matrix.h5``)
   * - **In situ targeted**
     - 10x Xenium Prime 5K
     - Exon/junction probe (codeword)
     - Xenium Ranger output directly (``transcripts.zarr.zip``)

Expected data format
--------------------

SPLISOSM expects isoform-level data in an ``AnnData`` of shape ``(n_spots, n_isoforms)``:

- ``adata.layers['counts']``: raw isoform counts.
- ``adata.var``: isoform metadata with at least:

  - ``gene_symbol/gene_id``: gene assignment for each isoform.
  - A unique feature identifier in ``adata.var_names`` (e.g., transcript ID, peak ID, codeword ID).

- Spatial coordinates in one of:

  - ``adata.obsm['spatial']``: preferred; a ``(n_spots, 2)`` array with ``[x (col), y (row)]`` columns, compatible with Scanpy/Squidpy conventions.
  - ``adata.obs[['array_row', 'array_col']]``: legacy pixel-coordinate format.

  See :func:`splisosm.utils.prepare_inputs_from_anndata` for parsing details.

The spot-by-isoform ``AnnData`` can also be a table of a ``SpatialData`` object. In such cases, we will typically run 
`spatialdata.rasterize_bins <https://spatialdata.scverse.org/en/latest/api/operations.html#spatialdata.rasterize_bins>`_ to 
rasterize counts into square bins of varying sizes, which can speed up computation. See :func:`~splisosm.SplisosmFFT.setup_data` for details.

Long-read ST data
-----------------

SPLISOSM is compatible with any long-read quantification tool that produces an isoform-by-spot count matrix. Popular options include `IsoQuant <https://ablab.github.io/IsoQuant/>`_ and `kallisto <https://pachterlab.github.io/kallisto/>`_.

.. note::
   Detection power scales linearly with sequencing depth (:ref:`FAQ <faq:umi-depth>`). Any processing choice that increases captured UMIs per spot will improve results.

Short-read 3' end ST data
--------------------------

Use `Sierra <https://github.com/VCCRI/Sierra/>`__ to call 3' end diversity (TREND) events *de novo* from Space Ranger BAM files. See the `Sierra vignette <https://github.com/VCCRI/Sierra/wiki/Sierra-Vignette>`_ for installation and usage details.

Tested platforms: 10x Visium (fresh-frozen), Slide-seqV2.

.. warning::
   When processing multiple samples, avoid Sierra's ``MergePeakCoordinates`` function — it can produce overlapping peak definitions.

.. note::
   There is an known issue with running Sierra's ``CountPeaks`` on Visium HD 3\' data. We are working on a fix.

Running Sierra on Space Ranger BAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: r

   # load the R package 
   library(Sierra)
   FindPeaks(
     output.file = '${peak_file}',
     gtf.file = '${gtf_file}', # SpaceRanger reference gtf file
     bamfile = '${bam_file}',  # bam file from SpaceRanger
     junctions.file = '${junc_file}', # junctions bed files extracted using regtools junctions extract
   # optional arguments for retainning low-abundance peaks
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


Converting Sierra output to AnnData
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import scanpy as sc
    import pandas as pd
    from splisosm.utils import load_visium_sp_meta

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


Short-read targeted ST data
----------------------------

10x Visium HD FFPE uses a fixed pan-genome probe for read enrichment. SPLISOSM treats probe-level counts as the isoform-level quantification — probes targeting the same gene are grouped and tested jointly.

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
       ├── 'cell_segmentations': AnnData (40222, 19070)
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

See the :doc:`Visium HD FFPE tutorial <tutorials/visiumhd_ffpe>` for a complete step-by-step workflow.


In situ targeted ST data
-------------------------

For imaging-based platforms with exon- or junction-specific probes (e.g., `10x Xenium Prime 5K <https://www.10xgenomics.com/products/xenium-5k-panel>`_), SPLISOSM uses codeword-level counts as isoform proxies. 
Data can be analysed at single-cell resolution (segmented cells) or on spatially binned spots.

Given Xenium Ranger output, the following code creates a binned ``SpatialData`` object with codeword-level counts (from ``transcripts.zarr.zip``):

.. code-block:: python

   from splisosm.io import load_xenium_codeword

   sdata = load_xenium_codeword(
     path=xenium_ranger_outs,
     spatial_resolutions=[8.0, 16.0],
     quality_threshold=20.0,
     n_jobs=-1,
     chunk_batch_size=64,
     counts_layer_name="counts",
     create_square_shapes=True,
   )

See the :doc:`Xenium Prime 5K tutorial <tutorials/xenium_prime_5k>` for a complete step-by-step workflow.

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
     └── 'table': AnnData (63173, 5006)
   with coordinate systems:
     ▸ 'global', with elements:
       morphology_focus (Images), cell_labels (Labels), nucleus_labels (Labels), transcripts (Points), cell_boundaries (Shapes), nucleus_boundaries (Shapes), square_008um_bins (Shapes), square_016um_bins (Shapes)


