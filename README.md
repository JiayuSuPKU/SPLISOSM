# SPLISOSM - Spatial Isoform Statistical Modeling

![overview](splisosm_overview.png)

SPLISOSM (<u>SP</u>atia<u>L</u> <u>ISO</u>form <u>S</u>tatistical <u>M</u>odeling) is a Python package for analyzing isoform usage patterns in spatial transcriptomics data. It employs optimized spatial and isoform kernels to capture complex dependencies between spatial locations and transcript usage, maximizing statistical power while maintaining well-calibrated, permutation-free test statistics even with extremely sparse data. The differential usage tests are conditioned on spatial locations to reduce false positives due to spatial autocorrelation.

SPLISOSM accommodates isoform quantification results of both full-length isoforms from long-read sequencing and local transcript diversity events from short-read sequencing platforms including 10X Visium and Slide-seqV2. 

*This repository is under active development. Please check back later for detailed documentations, tutorials and end-to-end analysis examples using public long-read and short-read datasets.*

## Installation
Via GitHub (latest version):
```bash
pip install git+https://github.com/JiayuSuPKU/SPLISOSM.git#egg=splisosm
```
Via PyPI (stable version):
```bash
pip install splisosm
```

## Quick start
The basic unit of SPLISOSM analysis is the isoform-level event, which can be a full-length transcript from long-read sequencing or a local variable structure from short-read 3'end sequencing. Given spatial isoform quantification results, SPLISOSM runs two types of tests:
1. **Spatial variability (SV)**: Find spatially variable transcript usage (HSIC-IR), transcript expression (HSIC-IC) or gene expression (HSIC-GC). 
2. **Differential usage (DU)**: Testing the *conditional* association between transcript usage and spatial covariates such as spatial domains and expression of potential regulators like RNA binding proteins (RBPs).

Both SV and DU tests are *multivariate* performed at the gene level (i.e. one test per gene or per gene-covariate pair).

SPLISOSM takes the following inputs:
* `data`: list of per-gene isoform quantification for 'n_gene' genes. Each element is a (n_spot, n_iso) tensor, where 'n_spot' is the number of spatial spots and 'n_iso' is the number of isoforms (or TREND events) of that given gene. 
* `coordinates`: spatial coordinates. (n_spot, 2).

And additionally for differential usage testing:

* `covariates`: spatial variable such as RBP expression to test for association. (n_spot, n_covariate).

See the [Isoform quantification and data preparation](#isoform-quantification-and-data-preparation) section below for tips on isoform quantification and data preparation. Given an AnnData object with isoform-level quantification results, inputs for SPLISOSM can be extracted using the `extract_counts_n_ratios` function from `splisosm.utils`.

The output of the SV test is a data frame of per-gene test statistics and p-values, and the output of the DU test is a data frame of per-gene-covariate-pair test statistics and p-values.

### Example data
A small demo dataset of Visium-ONT mouse olfactory bulb (SiT-MOB) can be downloaded [from here via Dropbox (~100Mb)](https://www.dropbox.com/scl/fo/dmuobtbof54jl4ht9zbjo/ALVIIEp-Ua5yYUPO8QxlIZ8?rlkey=q9o3jisd25ef5hwfqnsqdbf3i&st=vxhgokzw&dl=0) to test the package functionality.
* `mob_ont_filtered_1107.h5ad`: AnnData object with isoform quantification results.
* `mob_visium_rbp_1107.h5ad`: AnnData object with short-read-based RBP gene expression.

```python
import scanpy as sc
from splisosm.utils import extract_counts_n_ratios

adata_ont = sc.read("mob_ont_filtered_1107.h5ad")
adata_rbp = sc.read("mob_visium_rbp_1107.h5ad")

# prepare per gene isoform tensor list
# data[i] = (n_spot, n_iso) tensor for gene i
# assuming isoforms (adata_ont.var_names) are grouped by adata_ont.var['gene_symbol']
data, _, gene_name_list, _ = extract_counts_n_ratios(
    adata_ont, layer = 'counts', group_iso_by = 'gene_symbol'
)

# spatial coordinates
coords = adata_ont.obs.loc[:, ['array_row', 'array_col']]

# prepare covariates for differential usage testing
adata_rbp = adata_rbp[adata_ont.obs_names, :].copy() # align the RBP data with the isoform data
# focus on spatially variably expressed RBPs only
# adata_rbp.var['is_visium_sve'] = adata_rbp.var['pvalue_adj_sparkx'] < 0.01
covariates = adata_rbp[:, adata_rbp.var['is_visium_sve']].layers['log1p'].toarray()
covariate_names = adata_rbp.var.loc[adata_rbp.var['is_visium_sve'], 'features']
```

### Testing for spatial variability (SV)
SPLISOSM uses Hilbert-Schmidt Independence Criterion (HSIC) to test for kernel independence between isoform quantities and spatial coordinates. Specifically, we have the following SV tests for variability in three aspects:
1. **HSIC-IR** for *isoform relative ratio*.
2. **HSIC-GC** for *total gene expression*. This test is similar to [SPARK-X](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02404-0) but with a different spatial kernel design and improved statistical power.
3. **HSIC-IC** for *isoform count*. Biologically, isoform expression variability is the joint result of variability in total gene expression and in isoform usage. In practice the results of HSIC-IC and HSIC-GC are usually similar.

```python
from splisosm import SplisosmNP

# minimal input data
data = ... # list of length n_gene, each a (n_spot, n_iso) tensor
coordinates = ..., # (n_spot, 2), spatial coordinates
gene_names = ... # list of length n_gene, gene names

# initialize the model
model_np = SplisosmNP()
model_np.setup_data(data, coordinates, gene_names = gene_names)

# per-gene test for spatial variability
# method can be 'hsic-ir' (isoform ratio), 'hsic-ic' (isoform counts)
# 'hsic-gc' (gene counts), 'spark-x' (gene counts)
model_np.test_spatial_variability(
    method = "hsic-ir",
    ratio_transformation = 'none', # only applicable to 'hsic-ir', can be 'none', 'clr', 'ilr', 'alr', 'radial'
    nan_filling = 'mean', # how to fill NaN values in the data, can be 'mean' (global mean), 'none' (ignoring NaN spots)
)

# extract the per-gene test statistics
df_sv_res = model.get_formatted_test_results(test_type = 'sv')
```

### Testing for differential isoform usage (DU)
<!-- #### Motivating example: Marker event discovery
One common utilities of DU testing is to identify "marker events" for categorical groups, e.g. cell-type-specific marker genes. In the context of spatial splicing, we can also define "marker isoform switching events", which is to test whether the isoform preference of a gene changes in different spatial domains. Naively, this can be done by combining isoform-level pairwise t-tests across isoforms. -->
<!-- 
```python
from splisosm import SplisosmNP

# minimal input data
# after running the SV test, keep only SVS genes for DU testing
data_svs = ... # list of length n_genes, each a (n_spots, n_isos) tensor
coordinates = ..., # (n_spots, 2), spatial coordinates
gene_names = ... # list of length n_genes, gene names

# here we also need the one-hot encoding of spatial domain membership
# covariates[i, j] = 1 if spot i belongs to domain j
covariates = ... # (n_spots, n_regions)
covariate_names = ... # list of length n_regions, domain names


# initialize the model with covariates
model_np = SplisosmNP()
model_np.setup_data(
    data_svs,
    coordinates,
    design_mtx = covariates,
    gene_names = gene_names,
    covariate_names = covariate_names
)

# run the pairwise t-test (one domain vs. all others) for each isoform,
# then combine the p-values using either Fisher's or Tippett's method to get gene-level p-values
model_np.test_differential_usage(
    method = "t-fisher", # can be "t-fisher", "t-tippett". 
    ratio_transformation = 'none', nan_filling = 'mean', # isoform ratio computing options
)

# per gene-factor pair test statistics
df_du_res = model.get_formatted_test_results(test_type = 'du')
``` -->

<!-- However, t-test does not account for spatial autocorrelation and thus has inflated false positives (i.e. isoform usage ratio can form similar patterns by chance). This would be more problematic if the spatial domains are defined using unsupervised clustering. The double dipping problem. 

. This problem would be more prominent if 
To address this, we can use the HSIC-based DU test, which is more powerful and robust to noise.

#### DU test with continuous covariates


. To address this, we can use the HSIC-based DU test, which is more powerful and robust to noise.

 associated with spatial covariates. SPLISOSM provides both non-parametric and parametric tests for DU. -->

<!-- ### Non-parametric testing using SplisosmNP -->

We implemented conditional association tests for isoform usage in both parametric and non-parametric settings. 

The non-parametric test is again based on the HSIC kernel independence test and relies on Gaussian process regression for spatial conditioning. 

```python
# from splisosm.hyptest import IsoSDE # will be deprecated in the future
from splisosm.hyptest_np import SplisosmNP

# minimal input data
# after running the SV test, keep only SVS genes for DU testing
data_svs = ... # list of length n_gene, each a (n_spot, n_iso) tensor
coordinates = ..., # (n_spot, 2), spatial coordinates
gene_names = ... # list of length n_gene, gene names

# covariates for differential usage testing
# e.g. spatial domains, RBP expression, etc.
covariates = ... # (n_spot, n_factor), design matrix of covariates
covariate_names = ... # list of length n_factor, covariate names

# initialize the model with covariates
model_np = SplisosmNP()
model_np.setup_data(
    data_svs,
    coordinates,
    design_mtx = covariates,
    gene_names = gene_names,
    covariate_names = covariate_names
)

# run the conditional HSIC test for differential usage
model_np.test_differential_usage(
    method = "hsic-gp", # can be "hsic", "hsic-knn", "hsic-gp", "t-fisher", "t-tippett". See the function docstring for details.
    ratio_transformation = 'none', nan_filling = 'mean', # same as above
    hsic_eps = 1e-3, # regularization parameter kernel regression, only applicable to 'hsic'. If set to None, will be the unconditional HSIC test.
    gp_configs = None, # dictionary of configs for the Gaussian process regression, only applicable to 'hsic-gp'
    print_progress = True, return_results = False
)

# extract per gene-factor pair test statistics
df_du_res = model.get_formatted_test_results(test_type = 'du')
```

<!-- ### Parametric testing using SplisosmGLMM -->
The parametric test is based on a generalized linear mixed model (GLMM) with the spatial random effect term following a Gaussian random field. The model is fitted using the [PyTorch Lightning](https://www.pytorchlightning.ai/) framework with marginal likelihood (i.e. integrating out the random effect) approximated by the Laplace's method at the mode.

```python
# from splisosm.hyptest import IsoSDE # will be deprecated in the future
from splisosm.hyptest_glmm import SplisosmGLMM

# parametric model fitting
model_p = SplisosmGLMM(
    model_type = 'glmm-full', # can be 'glmm-full', 'glmm-null', 'glm'
    share_variance: bool = True,
    var_parameterization_sigma_theta: bool = True,
    var_fix_sigma: bool = False,
    var_prior_model: str = "none",
    var_prior_model_params: dict = {},
    init_ratio: str = "observed",
    fitting_method: str = "joint_gd",
    fitting_configs: dict = {'max_epochs': -1}
)
model_p.setup_data(
    data, # list of length n_genes, each element is a (n_spots, n_isos) tensor
    coordinates, # (n_spots, 2), 2D array/tensor of spatial coordinates
    design_mtx = None, # (n_spots, n_covariates), 2D array/tensor of covariates
    gene_names = None, # list of length n_genes, gene names
    covariate_names = None # list of length n_covariates, covariate names
    group_gene_by_n_iso = False, # whether to group genes by the number of isoforms for batch processing
)
model_p.fit(
    n_jobs = 1, # number of cores to use
    batch_size = 1, # number of genes with the same number of isoforms to process in parallel per core
    quiet=True, print_progress=True,
    with_design_mtx = False, # fit the model without covariates for the score DU test
    from_null = False, refit_null = True, # for LLR test
    random_seed = None
)
model_p.save("model_p.pkl") # save the fitted model
per_gene_glmm_models = model_p.get_fitted_models() # list of length n_genes, each element is a fitted model

# differential usage testing
model_p.test_differential_usage(
    method = "score", # can be "wald", "score",
    print_progress = True, return_results = False
)

# extract per gene-factor pair test statistics
df_du_res = model_p.get_formatted_test_results(test_type = 'du')
```

### Isoform quantification and data preparation
If you have long-read spatial transcriptomics data, feel free to pick your favorite isoform quantification tools (e.g. [IsoQuant](https://ablab.github.io/IsoQuant/)) to get the isoform-level quantification results. 

If you have 3'end short-read spatial transcriptomics data, we recommend using [Sierra](https://github.com/VCCRI/Sierra/tree/master) to extract transcriptome 3'end diversity (TREND) events **de novo**. Please refer to the [Sierra documentation](https://github.com/VCCRI/Sierra/wiki/Sierra-Vignette) for detailed instructions on how to run Sierra. For multiple samples, we do NOT recommend running Sierra's `MergePeakCoordinates` function as it sometimes creates overlapping peaks. Note that the 10X Visium FFPE protocol uses targeted gene panels and thus does not provide isoform-specific information.

SPLISOSM is agnostic to isoform/event structure and will compare all events associated with the same gene. For computational efficiency, we recommend filtering out low-abundance isoforms/events before running SPLISOSM. 

<details>
<summary>Here is an example of Sierra run with bam from SpaceRanger</summary>

```r
library(Sierra)
FindPeaks(
  output.file = '${peak_file}', 
  gtf.file = '${gtf_file}', # SpaceRanger gtf file
  bamfile = '${bam_file}', # bam file from SpaceRanger
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
```
</details>


<details>
<summary>Here is an example of data preprocessing using scanpy and anndata</summary>

```python
import scanpy as sc
import pandas as pd

from splisosm.utils import load_visium_sp_meta

sierra_out_dir = "path/to/sierra/output" # 'output.dir' in CountPeaks
sp_meta_dir = "path/to/visium/spatial/metadata" # 'spatial' directory in 10X Visium data

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
df_var = peaks[0].str.split(':', expand = True)
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

# filter out lowly expressed peaks
sc.pp.filter_genes(adata, min_cells=0.01 * adata.shape[0])

# extract gene symbols and peak ids
df_iso_meta = adata.var.copy() # gene_symbol, chr, position, strand, gene_id
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
df_gene_meta['pct_spot_on'] = (isog_counts > 0).mean(axis = 0)
df_gene_meta['count_avg'] = isog_counts.mean(axis = 0)
df_gene_meta['count_std'] = isog_counts.std(axis = 0)

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
```

</details>

## Citation
Su, Jiayu, et al. "A computational framework for mapping isoform landscape and regulatory mechanisms from spatial transcriptomics data." bioRxiv (2025): 2025-05. [link to preprint](https://www.biorxiv.org/content/10.1101/2025.05.02.651907v1)


<!-- ## Documentation
See the [documentation](TODO) for more details on the statistical models and the testing procedures. -->
