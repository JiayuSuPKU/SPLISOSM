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
     - `Sierra <https://github.com/VCCRI/Sierra>`_ for *de novo* peak calling
   * - **Short-read targeted**
     - 10x Visium/Visium HD (FFPE), 10x Flex
     - Exon/junction probe
     - Space Ranger output directly
   * - **In situ targeted**
     - 10x Xenium Prime 5K
     - Exon/junction probe (codeword)
     - Custom extraction script (see below)

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

Long-read ST data
-----------------

SPLISOSM is compatible with any long-read quantification tool that produces an isoform-by-spot count matrix. Popular options include `IsoQuant <https://ablab.github.io/IsoQuant/>`_ and `kallisto <https://pachterlab.github.io/kallisto/>`_.

.. note::
   Detection power scales linearly with sequencing depth (:ref:`FAQ <faq:umi-depth>`). Any processing choice that increases captured UMIs per spot will improve results.

Short-read 3' end ST data
--------------------------

Use `Sierra <https://github.com/VCCRI/Sierra/tree/master>`_ to call 3' end diversity (TREND) events *de novo* from Space Ranger BAM files. See the `Sierra vignette <https://github.com/VCCRI/Sierra/wiki/Sierra-Vignette>`_ for installation and usage details.

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

   import scanpy as sc
   from splisosm.io import load_visiumhd_probe

   sdata = load_visiumhd_probe(
     path=visium_hd_outs,
     bin_sizes=[2, 8, 16],
     filtered_counts_file=True,
     load_all_images=False,
     var_names_make_unique=True,
     counts_layer_name="counts",
   )

See the :doc:`Visium HD FFPE tutorial <tutorials/visiumhd_ffpe>` for a complete step-by-step workflow.


In situ targeted ST data
-------------------------

For imaging-based platforms with exon- or junction-specific probes (e.g., `10x Xenium Prime 5K <https://www.10xgenomics.com/products/xenium-5k-panel>`_), SPLISOSM uses codeword-level counts as isoform proxies. Data can be analysed at single-cell resolution (segmented cells) or on spatially binned spots.

We provide a `helper script <https://github.com/JiayuSuPKU/SPLISOSM/blob/main/scripts/extract_xenium_codeword_dist.py>`_ to extract codeword counts from the ``transcripts.zarr.zip`` output of Xenium Ranger (tested on v3.1.1) and bin them into spots of a user-defined size.

.. code-block:: zsh

   # download the 'transcripts.zarr.zip' file from 10x Xenium data
   $ wget <Xenium_transcripts_zarr_url> -O transcripts.zarr.zip

   # run the helper script to extract codeword counts and bin into spots
   # estimated runtime: ~15 minutes for a full Xenium 5K dataset, 64GB RAM recommended
   $ python scripts/extract_xenium_codeword_dist.py \
       --data_dir <where_transcripts_zarr_zip_is> \
       --res_dir <output_directory> \
       --spatial_resolution 20 # specify the desired spot size in microns
       --n_jobs 16 # number of parallel threads

   # the output file '<output_directory>/codeword_quant_res_20um.h5ad' is an AnnData object of (n_spot, n_codeword)
   # and can be used as input to SPLISOSM


