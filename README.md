# SPLISOSM - Spatial Isoform Statistical Modeling

SPLISOSM (<u>SP</u>atia<u>L</u> <u>ISO</u>form <u>S</u>tatistical <u>M</u>odeling) is a Python package
for analyzing RNA isoform patterns in spatial transcriptomics (ST) data. It employs multivariate kernel association tests to detect (i) *spatial variability* in
isoform usage across spatial locations, and (ii) *differential association* between isoform usage and spatial covariates such as region annotation and RNA binding protein (RBP) expression.

![overview](docs/img/splisosm_overview.png)

## Documentation
https://splisosm.readthedocs.io/

*The documentation is under active development. Please check back later for more tutorials and end-to-end analysis examples using public long-read and short-read datasets.*

## Installation
Via GitHub (latest version):
```bash
pip install git+https://github.com/JiayuSuPKU/SPLISOSM.git#egg=splisosm
```
Via PyPI (stable version):
```bash
pip install splisosm
```

## Citation
Su, Jiayu, et al. "A computational framework for mapping isoform landscape and regulatory mechanisms from spatial transcriptomics data." bioRxiv (2025): 2025-05. [link to preprint](https://www.biorxiv.org/content/10.1101/2025.05.02.651907v2)

