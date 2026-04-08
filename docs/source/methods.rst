Statistical Methods
====================

This page describes the statistical framework underlying SPLISOSM's spatial variability (SV) and differential isoform usage (DU) tests.
For a full derivation and theoretical justification, please refer to the Supplementary Notes of the `SPLISOSM paper <https://www.nature.com/articles/s41587-025-02965-6>`_ :cite:`su2026mapping`.

Overview
--------

SPLISOSM treats every gene as a *multivariate* object: a gene with :math:`p` isoforms at :math:`n` spatial locations is represented by an :math:`n \times p` matrix.
The objective of the SV test is to determine whether the *multivariate* :math:`p`-dimensional vector varies across the :math:`n` spatial locations, 
while the DU test is to determine whether variation in this vector is associated with other spatial covariates (e.g., spatial domains, RBP expression), 
potentially after conditioning on spatial autocorrelation.

The main statistical tool for both tests is a kernel-based measure of association called the **Hilbert-Schmidt Independence Criterion (HSIC)** :cite:`gretton2005measuring`. 
Intuitively,

- The **spatial kernel** :math:`K \in \mathbb{R}^{n \times n}` encodes the spatial structure of the tissue.
- The **response kernel** :math:`L \in \mathbb{R}^{n \times n}` encodes similarity between isoform profiles of different spots/cells.
- HSIC measures whether spots that are spatially close (large :math:`K_{ij}`) also tend to have similar isoform profiles (large :math:`L_{ij}`).

Spatial Kernel: CAR Model
--------------------------

SPLISOSM uses a **Conditional Autoregressive (CAR)** model :cite:`su2023smoother` to define spatial covariance.
Given a k-mutual-nearest-neighbor adjacency matrix :math:`W` built from spot coordinates, the CAR precision matrix is

.. math::

   M = I - \rho D^{-1/2} W D^{-1/2},

where :math:`D = \mathrm{diag}(\sum_j W_{ij})` is the degree matrix and :math:`\rho \in (0, 1)` is the spatial autocorrelation coefficient.
The spatial covariance (kernel) matrix is :math:`K = M^{-1}`, standardised to unit marginal variance.

We choose the CAR kernel for the following properties:

- **Sparse precision**: only the k-NN graph is stored explicitly; :math:`K` is never formed for large datasets (implicit LU-solve mode when :math:`n > 5000`).
- **Irregular geometries**: the k-NN graph is translation-invariant and naturally handles non-grid layouts (Visium spots, single cells, segmented tissue).
- **Graph Fourier basis**: low-rank approximation of :math:`K` is the well-studied graph Fourier basis ordered by spatial frequency.
- **Polynomial spectrum decay**: the eigenvalues of :math:`K` decay polynomially, which yields higher power than exponential-decay kernels for mix-frequency patterns.

For general theory and the equivalence of spatial variability testing methods, see our recent work on the consistent and scalable detection of spatial patterns :cite:`su2026consistent`.


Spatial Variability (SV) Tests
--------------------------------

Three SV tests are available, differing only in how the response matrix :math:`Y \in \mathbb{R}^{n \times p}` is constructed.

.. _sv-test-types:

Test types
~~~~~~~~~~

Let :math:`c_{ig}` denote the raw count of isoform :math:`i` at spot :math:`g`, and :math:`r_{ig} = c_{ig} / \sum_i c_{ig}` the corresponding usage ratio.

.. list-table::
   :header-rows: 1
   :widths: 15 30 30 25

   * - Test
     - Response :math:`Y`
     - Null hypothesis
     - Typical use case
   * - **HSIC-IR**
     - Centred isoform usage ratios :math:`r_{ig}` (optionally log-ratio transformed)
     - Isoform *usage* is spatially uniform
     - Identify **SVP** genes (spatially variable RNA processing)
   * - **HSIC-GC**
     - Centred total gene count :math:`\sum_i c_{ig}` (single column)
     - Gene *expression* is spatially uniform
     - Identify **SVE** genes; drop-in for `SPARK-X <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02404-0>`_
   * - **HSIC-IC**
     - Centred raw isoform counts :math:`c_{ig}`
     - Isoform *counts* are spatially uniform
     - Reflects joint changes in expression and processing

**HSIC-IR** is the recommended test for discovering spatially variable RNA processing (SVP genes).
**HSIC-GC** is equivalent to a gene-level spatial variability test and can serve as a drop-in replacement for SPARK-X with improved statistical power.
**HSIC-IC** tests the joint null; significance can arise from either differential expression or differential processing. **HSIC-IC does not test each isoform individually and instead yields a single test statistic per gene.**
In practice, the results of **HSIC-IC** and **HSIC-GC** are often similar, as changes in overall gene expression is the main driver of isoform expression changes for many genes. 

Ratio transformations
~~~~~~~~~~~~~~~~~~~~~

For HSIC-IR, the usage ratios are optionally transformed before computing the HSIC statistic.
The ``ratio_transformation`` argument controls this:

- ``'none'`` (default): raw proportions :math:`r_{ig}`, mean-centred per isoform.
- ``'clr'``: centred log-ratio :math:`\log(r_{ig}) - \frac{1}{p}\sum_i \log(r_{ig})`.
- ``'ilr'``: isometric log-ratio (orthonormal Helmert contrast in log-simplex).
- ``'alr'``: additive log-ratio relative to the last isoform.
- ``'radial'``: radial transformation :math:`r_{ig} / \|r_g\|` :cite:`park2022kernel`. Empirically, it is not calibrated and thus not recommended.

Due to excessive sparsity, log-ratio-based transformations require pseudo-counts to avoid zero ratios, 
which may lead to unwanted artefacts. Our empirical results suggest that the untransformed ratios (``'none'``) are well-calibrated, robust and often more powerful for SVP detection.


Test statistic
~~~~~~~~~~~~~~

SPLISOSM uses a **quadratic form** V-statistic, which computes HSIC with a linear kernel on :math:`Y`:

.. math::

   \widehat{\mathrm{HSIC}} = \frac{1}{(n-1)^2} \mathrm{tr}(Y^\top K Y),

where :math:`K` is the (standardised, optionally double-centred) CAR kernel.
This equals the sum of squared spatial autocorrelations of each linear combination of isoform profiles.
Since we have

.. math::

   \mathrm{tr}(Y^\top K Y) = \mathrm{tr}(K \cdot Y Y^\top),

the test statistic can also be viewed as a weighted sum of spatial autocorrelations of the principal components of :math:`Y`.

Null distribution
~~~~~~~~~~~~~~~~~~

To compute p-values, we need to compute the distribution of :math:`\widehat{\mathrm{HSIC}}` under the null hypothesis of no spatial variability.
Three methods are available via the ``null_method`` argument to :meth:`~splisosm.SplisosmNP.test_spatial_variability`.

.. _null-eig:

**1. Liu's approximation of chi-square mixture (default)**: ``null_method='eig'`` 

Denote eigenvalues :math:`\lambda_1^K \geq \cdots \geq \lambda_n^K` of :math:`K` and :math:`\lambda_1^Y \geq \cdots \geq \lambda_p^Y` of :math:`Y^\top Y`.
Under the null, the test statistic :math:`(n-1)^2 \widehat{\mathrm{HSIC}}` asymptotically follows a distribution that can be expressed
as a weighted sum of independent :math:`\chi^2_{1}` variables,

.. math::

   (n-1)^2 \widehat{\mathrm{HSIC}} \;\overset{d}{\approx}\; \sum_{k,\ell} \lambda_k^K \lambda_\ell^Y \, \chi^2_{1,(k,\ell)},

where the double sum runs over all pairs of spatial and response eigenvalues.
:cite:`liu2009new` propose an efficient three-moment matching scheme (skewness matching) to approximate the tail probability of this mixture with a scaled and shifted chi-squared variable.
See :func:`splisosm.likelihood.liu_sf` for implementation details.

.. note::

   - Full eigen-decomposition of :math:`K` is :math:`O(n^3)` and is not feasible for large datasets.  
     For :math:`n > 5000`, we approximate the kernel via a low-rank approximation using the top-:math:`r` eigenvalues/vectors.  
     The approximation is controlled by ``approx_rank`` in ``null_configs``, with default :math:`\lceil 4\sqrt{n} \rceil` when :math:`n > 5000`.
   - When ``nan_filling='mean'`` (default), The spatial eigenvalues are cached after the first gene and reused for all subsequent genes.

.. _null-trace:

**2. Moment-matching normal approximation**: ``null_method='trace'``

Alternatively, we may use the first two moments of the null distribution to compute p-values (i.e., via Central Limit Theorem), 
which requires only :math:`\mathrm{tr}(K)` and :math:`\mathrm{tr}(K^2)`:

.. math::

   \mu_0 &= \frac{1}{n-1}\mathrm{tr}(K)\,\mathrm{tr}(Y^\top Y), \\[4pt]
   \sigma_0^2 &= \frac{1}{(n-1)^2}2\,\mathrm{tr}(K^2)\,\mathrm{tr}((Y^\top Y)^2).

The p-value is :math:`\Phi^c\!\left(\frac{\widehat{\mathrm{HSIC}} - \mu_0}{\sigma_0}\right)`, where :math:`\Phi^c` is the standard normal survival function. 
Note that for non-Gaussian data :math:`Y`, the null variance is off by a kurtosis factor, which we omit for brevity.

.. note::

   - Since the CLT approximation requires no eigen-decomposition, it is the fastest and most scalable option while being slightly less accurate for heavy-tailed null.
   - As the sample size increases, the approximation becomes more accurate, and the test is generally well-calibrated for large :math:`n`.
   - For implicit kernels (where only the sparse precision matrix :math:`M=K^{-1}` is stored), :math:`\mathrm{tr}(K)` and :math:`\mathrm{tr}(K^2)` are estimated via the **Hutchinson stochastic trace estimator** using 30 Rademacher probing vectors.

.. _null-perm:

**3. Batched permutation test**: ``null_method='perm'``

Generates a null distribution by repeatedly permuting the rows of :math:`Y` (breaking the spatial structure) and recomputing :math:`\mathrm{tr}(Y_\pi^\top K Y_\pi)` for each permutation :math:`\pi`.

To avoid :math:`O(n^2)` memory, SPLISOSM batches :math:`B` permutations into a single :math:`(n, Bp)` matrix and calls :meth:`~splisosm.kernel.SpatialCovKernel.xtKx` once per batch.
Per-permutation traces are recovered as diagonal blocks of the :math:`(Bp, Bp)` result matrix without materialising the full kernel.

Configuration via ``null_configs``:

- ``n_perms_per_gene`` (default 1000): total permutations.
- ``perm_batch_size`` (default 50): permutations per :meth:`xtKx` call.

FFT acceleration for regular grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmFFT` exploits the **translation-invariance** of regular grids (Visium HD, Xenium binned data) to accelerate the SV test via the 2-D Fast Fourier Transform.
Specifically, it reduces the kernel eigen-decomposition required for the Liu's method from :math:`O(n^3)` to :math:`O(n \log n)`, and the quadratic form computation from :math:`O(n^2 p)` to :math:`O(n p \log n)`.

On a regular :math:`H \times W` grid, the CAR precision matrix :math:`M` is block-circulant.
Its eigenvalues — and hence the spatial kernel eigenvalues — are the DFT of the first row of :math:`M`, computable in :math:`O(HW \log HW)` time:

.. math::

   \lambda_{(h,w)}^K = \frac{1}{1 - \rho \, \hat{W}_{hw} / d},

where :math:`\hat{W}_{hw}` is the 2-D DFT of the adjacency weight kernel and :math:`d` is the degree.

The quadratic form :math:`\mathrm{tr}(Y^\top K Y)` is then computed as a *pointwise* product in Fourier space:

.. math::

   \mathrm{tr}(Y^\top K Y) = \frac{1}{n} \sum_{h,w} \lambda_{(h,w)}^K \|\hat{Y}_{hw}\|_F^2,

where :math:`\hat{Y}_{hw}` is the 2-D DFT of the isoform count image.
This reduces the quadratic-form computation from :math:`O(n^2 p)` to :math:`O(n p \log n)`.

Furthermore, the spatial eigenvalues :math:`\{\lambda_{(h,w)}^K\}` are shared across all genes (computed once), so the cost of the eigenvalue null is also :math:`O(1)` per gene in terms of kernel operations.


Differential Isoform Usage (DU) Tests
---------------------------------------

DU tests ask whether isoform usage is associated with a covariate (e.g., spatial domain label, RBP expression), potentially **conditioned on** spatial autocorrelation.

Unconditional test (``method='hsic'``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unconditional test directly applies HSIC between the centred covariate vector :math:`Z \in \mathbb{R}^n` and the isoform ratio matrix :math:`Y`:

.. math::

   \widehat{\mathrm{HSIC}}_{\text{uncond}} = \frac{1}{(n-1)^2} \mathrm{tr}(Y^\top K_Z Y),

where :math:`K_Z = ZZ^\top / \|Z\|^2` is a rank-1 linear kernel on the covariate.
This is equivalent to the multivariate `RV coefficient <https://en.wikipedia.org/wiki/RV_coefficient>`_ between :math:`Z` and :math:`Y`.
When :math:`Z` is binary, this is also equivalent to a two-step procedure where two-sample T-test is first applied to each isoform separately, and the resulting p-values are combined 
(e.g., via Fisher's method ``method='t-fisher'``).

**Limitation:** If both isoform usage and the covariate are spatially autocorrelated (which is common), the unconditional test can be anti-conservative, meaning that
it will report false positive associations that merely reflect shared spatial structure.

Conditional test (``method='hsic-gp'``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given spatial coordinates :math:`X`, the conditional test assesses the association between :math:`Z | X` and :math:`Y | X`.
This is a difficult problem because in most spatial data, each spatial location is observed only once (i.e., the above conditionals are point mass).
To condition on spatial autocorrelation, we adopt a **residualisation** approach :cite:`zhang2012kernel`.

Specifically, SPLISOSM first **residualises** the covariate against a Gaussian Process (GP) spatial model, then tests the independence of the residuals against isoform usage:

1. **Fit a GP** to the covariate :math:`z` using spatial coordinates :math:`x`:

   .. math::

      z = f(x) + \varepsilon, \quad f \sim \mathcal{GP}(0, k_\theta),

   where :math:`k_\theta` is a 'Constant x RBF + WhiteNoise' kernel (See :class:`splisosm.kernel_gpr.SklearnKernelGPR`). A sparse inducing-point approximation is used for large datasets.

2. **Compute covariate residuals** :math:`\tilde{Z} = Z - \hat{f}(X)`, capturing the part of covariate variation *not explained by* spatial position.

3. **Test** :math:`\widehat{\mathrm{HSIC}}(\tilde{Z},\, Y)` using the linear covariate kernel :math:`K_{\tilde{Z}} = \tilde{Z}\tilde{Z}^\top / \|\tilde{Z}\|^2` and a similar linear kernel for :math:`Y`.
   Since both kernels are low-rank, the null distribution can be efficiently computed via the eigenvalue method.

.. note::

   Theoretically, the conditional association between :math:`Z` and :math:`Y` given spatial position :math:`X` requires both residualisations
   :math:`Z | X` or :math:`Y | X` to be performed. However, :math:`\tilde{Z} \perp\!\!\!\perp Y` implies :math:`Z \perp\!\!\!\perp Y | X` (but not vice versa), 
   so any significant dependency found by residualising only the covariate is also a true positive for the conditional association.

In practice, we found the covariate-only residualisation delivers massive computational savings and remains well-calibrated.
For flexibility, we provide optional control via the ``residualize`` argument:

- ``residualize='cov_only'`` (default): only :math:`Z` the spatial covariate is residualised; :math:`Y` the isoform ratio matrix is used as-is.
- ``residualize='both'``: both :math:`Z` and :math:`Y` are GP-residualised before testing.

.. note::

   The GP fitting step is the dominant computational cost of the DU test.
   Two backends are supported:

   - ``gpr_backend='sklearn'`` (default): uses ``GaussianProcessRegressor`` from sklearn with an RBF kernel and subsampled Nyström approximation.
   - ``gpr_backend='gpytorch'``: Exact or sparse GP with ``n_inducing`` inducing points.

   For very large datasets, pass ``gpr_configs={"covariate": {"n_inducing": 1000}}`` to control the number of inducing points for both backends.

FFT-accelerated conditional DU test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmFFT` accelerates the conditional DU test (``method='hsic-gp'``)
by replacing the dense GP with an **FFT-based Gaussian process**.

On a regular grid the GP covariance kernel :math:`k_\theta(x, x')` is stationary, so
the Gram matrix :math:`K_\theta` is block-circulant.  All operations required for GP
fitting and prediction, including matrix-vector products, log-determinants, and gradient
computations, can be performed via 2-D FFTs:

.. math::

   K_\theta \cdot v = \mathcal{F}^{-1}\!\bigl(\hat{k}_\theta \odot \mathcal{F}(v)\bigr),
   \quad
   \log|K_\theta| = \sum_{h,w} \log \hat{k}_{\theta,(h,w)},

where :math:`\hat{k}_\theta` is the 2-D DFT of the kernel's first row.
This reduces the per-step GP cost from :math:`O(n^3)` (dense Cholesky) or
:math:`O(nM^2)` (inducing-point) to :math:`O(n \log n)` per L-BFGS iteration,
with no approximation error for stationary kernels on the grid.

The GP hyperparameters (signal variance, length scale, noise variance) are optimised by
maximising the marginal log-likelihood using L-BFGS with gradients computed in Fourier
space.  Covariate residuals :math:`\tilde{Z} = Z - \hat{f}(X)` are then obtained via
the FFT-based posterior mean.

After residualisation, the HSIC test itself proceeds identically to the
:class:`~splisosm.SplisosmNP` path: linear-kernel HSIC via :math:`\mathrm{tr}(Y^\top K_{\tilde{Z}} Y)=\|\tilde{Z}^\top Y\|^2`,
and p-value via the FFT-derived spatial kernel spectrum.


Parametric test: SplisosmGLMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmGLMM` provides a parametric alternative based on a **multinomial GLMM** with a Gaussian random field (GRF) spatial random effect:

.. math::

   Y_i \mid \phi_i \sim \mathrm{Multinomial}(N_i,\, \mathrm{softmax}(\eta_i)), \quad
   \eta_i = X_i \beta + u_i, \quad
   u \sim \mathcal{N}(0,\, \sigma^2 K),

where :math:`X_i` is the row of covariates for spot :math:`i`, :math:`\beta` are fixed effects, and :math:`u` is a spatially correlated random effect with covariance :math:`\sigma^2 K`.

The marginal likelihood is approximated via **Laplace's method** at the mode of the random effects.
DU testing uses a **score test** comparing the null model (no fixed-effect covariates) to the full model, which avoids fitting the full model for each covariate.

Compared to :class:`~splisosm.SplisosmNP`, the GLMM approach:

- Makes **stronger distributional assumptions** (multinomial counts), which may or may not be well-satisfied in practice.  
  See :cite:`su2026consistent` for a theoretical analysis of the effect of link function on test power.
- Can be more **interpretable** (when fitting with effect sizes :math:`\hat\beta` and standard errors).
- Is more **computationally intensive** (per-gene numerical optimisation).

Multiple testing correction
----------------------------

All SV and DU tests apply **Benjamini-Hochberg (BH) FDR correction** :cite:`benjamini1995controlling` across genes (for SV) or gene-covariate pairs (**for DU, each covariate factor is tested and adjusted separately**).
The adjusted p-values are stored in the ``pvalue_adj`` column of the results DataFrame returned by :meth:`~splisosm.SplisosmNP.get_formatted_test_results`.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 20 22

   * - Test
     - Class
     - Response :math:`Y`
     - Conditioning
     - Null method
   * - HSIC-IR (SVP)
     - SplisosmNP / FFT
     - Isoform ratios
     - None
     - ``eig`` / ``trace`` / ``perm``
   * - HSIC-GC (SVE)
     - SplisosmNP / FFT
     - Gene counts
     - None
     - ``eig`` / ``trace`` / ``perm``
   * - HSIC-IC
     - SplisosmNP / FFT
     - Isoform counts
     - None
     - ``eig`` / ``trace`` / ``perm``
   * - DU (unconditional)
     - SplisosmNP / FFT
     - Isoform ratios
     - None
     - ``eig``
   * - DU (GP-conditional)
     - SplisosmNP / FFT
     - Isoform ratios
     - GP spatial residualisation
     - ``eig``
   * - DU (GLM score)
     - SplisosmGLMM
     - Isoform counts
     - None
     - Chi-squared (score)
   * - DU (GLMM score)
     - SplisosmGLMM
     - Isoform counts
     - GRF random effect
     - Chi-squared (score)
