Installation
============

PyPI
-----

SPLISOSM can be installed via pip:

.. code-block:: zsh

  # from PyPI (stable version)
  $ pip install splisosm

  # or from GitHub (latest version)
  $ pip install git+https://github.com/JiayuSuPKU/SPLISOSM.git#egg=splisosm


Minimal software dependencies:

.. code-block:: text

  torch
  scipy
  scikit-learn
  pandas
  tqdm
  matplotlib
  anndata


Additional dependencies
-----------------------

`SplisosmFFT` requires the `spatialdata` package to be installed, and `SplisosmNP`'s differential usage test optionally supports `gpytorch` and `finufft` backends for Gaussian process regression.
To install these additional dependencies, you can use the following command:

.. code-block:: zsh

  $ pip install "splisosm[sdata, gp]"

.. note::

  `finufft` multi-threading on MacOS is currently suppressed due to a known OMP parallelization issue when `torch`and `finufft` are both installed. 
  The root cause is duplicated `libomp.dylib` installations (shipped along with `torch`).

For spatially variable gene expression testing, we also provide a Python wrapper, :func:`splisosm.utils.run_sparkx`, for `SPARK-X <https://xzhoulab.github.io/SPARK/04_installation/>`_.
To use this functionality, ensure that the conda environment where SPLISOSM is installed has R (>=4.0) available in ``PATH``, and that the R package `SPARK-X <https://xzhoulab.github.io/SPARK/04_installation/>`_ is properly installed.

.. code-block:: zsh

  # install R and the R package SPARK-X in the conda environment
  $ conda install -c conda-forge r-base r-devtools
  $ Rscript -e "devtools::install_github('xzhoulab/SPARK')"

  # install the rpy2 package to interface R from python
  $ pip install rpy2
  $ pip install splisosm # if not already installed

  # test whether SPARK-X is correctly configured
  $ python -c "import numpy as np; from splisosm.utils import run_sparkx; run_sparkx(np.random.randn(10,5), np.random.rand(10,2))"

.. note::

  Our new gene-level spatial variability test, *HSIC-GC*, is available as a standalone function (:func:`splisosm.utils.run_hsic_gc`).
  It can be used as a drop-in replacement for SPARK-X, where we optimize the spatial kernel to achieve higher statistical power while maintaining computational efficiency.

.. code-block:: python

   from splisosm.utils import run_hsic_gc
   import numpy as np

   # gene expression matrix: (n_spot, n_gene)
   gene_counts = np.random.randn(100, 50)

   # spatial coordinates: (n_spot, 2)
   coordinates = np.random.rand(100, 2)

   # run HSIC-GC test (default: Liu's cumulant approximation for the null)
   test_results = run_hsic_gc(gene_counts, coordinates)
   print(test_results['statistic'])  # test statistics, (n_gene,)
   print(test_results['pvalue'])     # p-values, (n_gene,)

   # ── AnnData mode ─────────────────────────────────────────────────
   # adata.X or adata.layers[layer] must be a (n_spots, n_genes) count
   # matrix (dense or scipy sparse); adata.obsm[spatial_key] provides
   # spatial coordinates.
   test_results = run_hsic_gc(
       adata=adata,           # AnnData of shape (n_spots, n_genes)
       layer=None,            # None → use adata.X; str → use adata.layers[layer]
       spatial_key="spatial", # key in adata.obsm for spatial coordinates
       min_counts=1,          # optional: drop genes with fewer total counts
       min_bin_pct=0.05,      # optional: drop genes expressed in < 5% of spots
   )
