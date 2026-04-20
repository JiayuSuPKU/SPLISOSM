Statistical Methods
====================

This page describes the statistical framework underlying SPLISOSM's spatial variability (SV) and differential isoform usage (DU) tests.
For a full derivation and theoretical justification, please refer to the Supplementary Notes of the `SPLISOSM paper <https://www.nature.com/articles/s41587-025-02965-6>`_ :cite:`su2026mapping`,
and of :cite:`su2026consistent`.

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
- HSIC measures whether spots that are spatially close (large :math:`K_{ii'}`) also tend to have similar isoform profiles (large :math:`L_{ii'}`).

Spatial Kernel: CAR Model
--------------------------

SPLISOSM uses a **Conditional Autoregressive (CAR)** model :cite:`su2023smoother` to define spatial covariance.
Given a k-mutual-nearest-neighbor adjacency matrix :math:`W` built from spot coordinates, the CAR precision matrix is

.. math::

   M = I - \rho D^{-1/2} W D^{-1/2},

where :math:`D = \mathrm{diag}(\sum_{i'} W_{ii'})` is the degree matrix and :math:`\rho \in (0, 1)` is the spatial autocorrelation coefficient.
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

Let :math:`c_{ij}` denote the raw count at spot :math:`i \in \{1, \ldots, n\}` of isoform :math:`j \in \{1, \ldots, p\}`, and :math:`r_{ij} = c_{ij} / \sum_{j'} c_{ij'}` the corresponding isoform usage ratio.

.. list-table::
   :header-rows: 1
   :widths: 15 30 30 25

   * - Test
     - Response :math:`Y`
     - Null hypothesis
     - Typical use case
   * - **HSIC-IR**
     - Centred isoform usage ratios :math:`r_{ij}` (optionally log-ratio transformed)
     - Isoform *usage* is spatially uniform
     - Identify **SVP** genes (spatially variable RNA processing)
   * - **HSIC-GC**
     - Centred total gene count :math:`\sum_j c_{ij}` (single column)
     - Gene *expression* is spatially uniform
     - Identify **SVE** genes; drop-in for `SPARK-X <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02404-0>`_
   * - **HSIC-IC**
     - Centred raw isoform counts :math:`c_{ij}`
     - Isoform *counts* are spatially uniform
     - Reflects joint changes in expression and processing

**HSIC-IR** is the recommended test for discovering spatially variable RNA processing (SVP genes).
**HSIC-GC** is equivalent to a gene-level spatial variability test and can serve as a drop-in replacement for SPARK-X with improved statistical power.
**HSIC-IC** tests the joint null; significance can arise from either differential expression or differential processing. **HSIC-IC does not test each isoform individually and instead yields a single test statistic per gene.**
In practice, the results of **HSIC-IC** and **HSIC-GC** are often similar, as changes in overall gene expression are the main driver of isoform expression changes for many genes.

Ratio transformations
~~~~~~~~~~~~~~~~~~~~~

For HSIC-IR, the usage ratios are optionally transformed before computing the HSIC statistic.
The ``ratio_transformation`` argument controls this:

- ``'none'`` (default): raw proportions :math:`r_{ij}`, mean-centred per isoform.
- ``'clr'``: centred log-ratio :math:`\log(r_{ij}) - \frac{1}{p}\sum_{j'} \log(r_{ij'})`.
- ``'ilr'``: isometric log-ratio (orthonormal Helmert contrast in log-simplex).
- ``'alr'``: additive log-ratio relative to the last isoform.
- ``'radial'``: radial transformation :math:`r_{ij} / \|r_i\|` :cite:`park2022kernel`, where :math:`r_i = (r_{i1}, \ldots, r_{ip})` is the isoform-ratio vector at spot :math:`i`. Empirically, it is not calibrated and thus not recommended.

Due to excessive sparsity, log-ratio-based transformations require pseudo-counts to avoid zero ratios, 
which may lead to unwanted artefacts. Our empirical results suggest that the untransformed ratios (``'none'``) are well-calibrated, robust and often more powerful for SVP detection.


Test statistic
~~~~~~~~~~~~~~

SPLISOSM uses a **quadratic form** V-statistic, which computes HSIC with a linear kernel on :math:`Y`:

.. math::

   \widehat{\mathrm{HSIC}} = \frac{1}{(n-1)^2} \mathrm{tr}(Y^\top K Y),

where :math:`K` is the (**double-centred**, optionally standardised) CAR kernel.
This equals the sum of squared spatial autocorrelations of each linear combination of isoform profiles.
Since we have

.. math::

   \mathrm{tr}(Y^\top K Y) = \mathrm{tr}(K \cdot Y Y^\top),

the test statistic can also be viewed as a weighted sum of spatial autocorrelations of the principal components of :math:`Y`.

Null distribution
~~~~~~~~~~~~~~~~~~

To compute p-values, we need to compute the distribution of :math:`\widehat{\mathrm{HSIC}}` under the null hypothesis of no spatial variability.
Four methods are available via the ``null_method`` argument to :meth:`~splisosm.SplisosmNP.test_spatial_variability`.

.. _null-eig:

**1. Liu's approximation of chi-square mixture (default)**: ``null_method='eig'``

Let :math:`Q = \mathrm{tr}(Y^\top K Y) = (n-1)^2 \widehat{\mathrm{HSIC}}` denote the unnormalised HSIC V-statistic,
with :math:`\lambda_1^K \geq \cdots \geq \lambda_n^K` the eigenvalues of :math:`K` and
:math:`\lambda_1^Y \geq \cdots \geq \lambda_p^Y` those of :math:`Y^\top Y`.
Under the null, :math:`Q` asymptotically follows a weighted sum of independent :math:`\chi^2_{1}` variables,

.. math::

   Q \;\overset{d}{\approx}\; \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{p} \lambda_i^K \, \lambda_j^Y \; Z_{ij}, \qquad Z_{ij} \overset{\text{iid}}{\sim} \chi^2_{1},

where the double sum runs over all pairs of spatial and response eigenvalues.
:cite:`liu2009new` propose an efficient three-moment matching scheme (skewness matching) to approximate the tail probability of this mixture with a scaled and shifted chi-squared variable.
See :func:`splisosm.likelihood.liu_sf` for implementation details.

.. note::

   - Full eigen-decomposition of :math:`K` is :math:`O(n^3)` and is not feasible for large datasets.  
     For :math:`n > 5000`, we approximate the kernel via a low-rank approximation using the top-:math:`r` eigenvalues/vectors.  
     The approximation is controlled by ``approx_rank`` in ``null_configs``, with default :math:`\lceil 4\sqrt{n} \rceil` when :math:`n > 5000`.
   - When ``nan_filling='mean'`` (default), the spatial eigenvalues are cached after the first gene and reused for all subsequent genes.

.. _null-clt:

**2. Moment-matching normal approximation (CLT)**: ``null_method='clt'``

.. note::

   The previous name ``null_method='trace'`` is retained as a deprecated
   alias and will emit a ``DeprecationWarning``.  The canonical name is
   now ``'clt'``, which better disambiguates this moment-matching normal
   approximation from :ref:`welch <null-welch>` (the Welch–Satterthwaite
   scaled chi-squared that shares the same matrix traces).

Alternatively, we may use the first two moments of the null distribution to compute p-values (i.e., via Central Limit Theorem),
which requires only :math:`\mathrm{tr}(K)` and :math:`\mathrm{tr}(K^2)`.
Taking moments of the :math:`\chi^2_1` mixture above gives

.. math::

   \mu_0 &= \mathbb{E}[Q] = \frac{1}{n}\,\mathrm{tr}(K)\,\mathrm{tr}(Y^\top Y), \\[4pt]
   \sigma_0^2 &= \mathrm{Var}(Q) = \frac{2}{n^2}\,\mathrm{tr}(K^2)\,\mathrm{tr}\!\bigl((Y^\top Y)^2\bigr).

The p-value is :math:`\Phi^c\!\left(\frac{Q - \mu_0}{\sigma_0}\right)`, where :math:`\Phi^c` is the standard normal survival function.
Note that for non-Gaussian data :math:`Y`, the null variance is off by a kurtosis factor, which we omit for brevity.

.. note::

   - Since the CLT approximation requires no eigen-decomposition, it is the fastest and most scalable option while being slightly less accurate for a heavy-tailed null.
   - As the sample size increases, the approximation becomes more accurate, and the test is generally well-calibrated for large :math:`n`.
   - For implicit kernels (where only the sparse precision matrix :math:`M=K^{-1}` is stored), :math:`\mathrm{tr}(K)` and :math:`\mathrm{tr}(K^2)` are estimated via the **Hutchinson stochastic trace estimator** using 30 Rademacher probing vectors.

.. _null-welch:

**3. Welch-Satterthwaite scaled chi-squared approximation**: ``null_method='welch'``

With all positive eigenvalues, the chi-squared mixture null can also be approximated by the `Welch-Satterthwaite method <https://en.wikipedia.org/wiki/Welch%E2%80%93Satterthwaite_equation>`_, 
using one scaled chi-squared variable :math:`g \, \chi^2_h` with scale parameter :math:`g` and degrees of freedom :math:`h`. 
The parameters are chosen to match the first two moments of the null :math:`(\mu_0, \sigma_0^2)` (same as in the :ref:`clt <null-clt>` method).

.. math::

   g = \frac{\sigma_0^2}{2\mu_0}, \qquad h = \frac{2\mu_0^2}{\sigma_0^2},

and the p-value is :math:`\mathbb{P}\!\left(\chi^2_h \geq Q/g\right)`.

.. note::

   - Same cost as ``null_method='clt'`` (only :math:`\mathrm{tr}(K)` and
     :math:`\mathrm{tr}(K^2)` are needed), but typically more accurate in the
     right tail because the null under H₀ is a weighted sum of
     :math:`\chi^2_1` variables rather than Gaussian.  In practice its
     p-values are close to the :ref:`eig <null-eig>` (Liu) reference while
     remaining as cheap as the :ref:`clt <null-clt>` method.
   - Recommended when ``null_method='eig'`` is too slow for the dataset
     (e.g., very large :math:`n` with no FFT grid) but a reliable right-tail
     calibration is still needed.

.. _null-perm:

**4. Batched permutation test**: ``null_method='perm'``

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

On a regular :math:`H \times W` grid with :math:`n = HW` spots, uniform degree :math:`d` and periodic boundaries, the CAR precision :math:`M = I - (\rho/d)\,W` is block-circulant.
Its eigenvalues (and hence those of :math:`K = M^{-1}`) are the 2-D DFT of its first row, computable in :math:`O(n \log n)` time:

.. math::

   \lambda_{(h,w)}^K = \frac{1}{1 - \rho \, \hat{W}_{hw} / d}, \qquad h = 0, \ldots, H-1,\; w = 0, \ldots, W-1,

where :math:`\hat{W}_{hw} = (\mathcal{F}\,w)_{hw}` is the unnormalised 2-D DFT of the first row of the adjacency matrix :math:`W`.

The quadratic form :math:`\mathrm{tr}(Y^\top K Y)` then reduces to a *pointwise* product in Fourier space.
Reshape each isoform image :math:`y_j \in \mathbb{R}^n` to :math:`H \times W` and let
:math:`\hat{Y}_{hw} = \bigl((\mathcal{F} y_1)_{hw},\, \ldots,\, (\mathcal{F} y_p)_{hw}\bigr) \in \mathbb{C}^{p}`
denote the vector of 2-D DFT coefficients at frequency :math:`(h, w)` stacked across all :math:`p` isoforms. Then

.. math::

   \mathrm{tr}(Y^\top K Y) \;=\; \frac{1}{n} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} \lambda_{(h,w)}^K \; \|\hat{Y}_{hw}\|_2^2,

where the leading :math:`1/n` comes from the unnormalised DFT convention (``scipy.fft.fft2``) that satisfies :math:`\mathcal{F}^{-1} = (1/n)\,\mathcal{F}^{*}`.
This reduces the quadratic-form computation from :math:`O(n^2 p)` to :math:`O(n p \log n)`.

Furthermore, the spatial eigenvalues :math:`\{\lambda_{(h,w)}^K\}` are shared across all genes (computed once), so the cost of the eigenvalue null is also :math:`O(1)` per gene in terms of kernel operations.

.. note::

   For irregularly spaced 2D and 3D coordinates, it is possible to compute the test statistic and its null using the `non-uniform FFT (NUFFT) <https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform>`_ approach, 
   which also scales as :math:`O(n \log n)`. We are working on an implementation of this method for future releases (depending on personal bandwidth).

Differential Isoform Usage (DU) Tests
---------------------------------------

DU tests ask whether isoform usage is associated with a covariate (e.g., spatial domain label, RBP expression), potentially **conditioned on** spatial autocorrelation.

Unconditional test (``method='hsic'``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unconditional test directly applies HSIC between the centred covariate vector :math:`Z \in \mathbb{R}^n` and the isoform ratio matrix :math:`Y`:

.. math::

   \widehat{\mathrm{HSIC}}_{\text{uncond}} = \frac{1}{(n-1)^2} \mathrm{tr}(Y^\top K_Z Y),

where :math:`K_Z = ZZ^\top` is a rank-1 linear kernel on the covariate, so that
:math:`\mathrm{tr}(Y^\top K_Z Y) = \|Z^\top Y\|^2`.
Up to a scalar, the statistic is equivalent to the multivariate
`RV coefficient <https://en.wikipedia.org/wiki/RV_coefficient>`_ between :math:`Z` and :math:`Y`.
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

3. **Test** :math:`\widehat{\mathrm{HSIC}}(\tilde{Z},\, Y)` using the rank-1 linear covariate kernel :math:`K_{\tilde{Z}} = \tilde{Z}\tilde{Z}^\top` and a similar linear kernel :math:`K_Y = Y Y^\top` for the response.
   Since both kernels are low-rank, the null distribution can be efficiently computed via the eigenvalue method.

.. note::

   Theoretically, a fully equivalent test of conditional independence :math:`Z \perp\!\!\!\perp Y \mid X` would residualise **both** :math:`Z \mid X` and :math:`Y \mid X` against the spatial GP.
   However, :math:`\tilde{Z} \perp\!\!\!\perp Y` implies :math:`Z \perp\!\!\!\perp Y \mid X` (but not vice versa),
   so any significant dependency found by residualising only the covariate is also a true positive for the conditional association.

In practice, we found that the covariate-only residualisation delivers massive computational savings and remains well-calibrated.
For flexibility, we provide optional control via the ``residualize`` argument:

- ``residualize='cov_only'`` (default): only :math:`Z` the spatial covariate is residualised; :math:`Y` the isoform ratio matrix is used as-is.
- ``residualize='both'``: both :math:`Z` and :math:`Y` are GP-residualised before testing.

.. note::

   The GP fitting step is the dominant computational cost of the DU test.
   Two backends are supported:

   - ``gpr_backend='sklearn'`` (default): uses ``GaussianProcessRegressor`` from sklearn with an RBF kernel and subsampled Nyström approximation.
   - ``gpr_backend='gpytorch'``: Exact or sparse GP with ``n_inducing`` inducing points.

   For very large datasets, pass ``gpr_configs={"covariate": {"n_inducing": 1000}}`` to control the number of inducing points for both backends.
   It is possible to further scale up GP fitting using the `non-uniform FFT (NUFFT) <https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform>`_ approach. 
   We are working on an implementation of this method for future releases (depending on personal bandwidth).


FFT-accelerated conditional DU test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmFFT` accelerates the conditional DU test (``method='hsic-gp'``)
by replacing the dense GP with an **FFT-based Gaussian process**.

On a regular grid with periodic boundaries the GP covariance kernel :math:`k_\theta(x, x')` is stationary, so
the Gram matrix :math:`K_\theta` is block-circulant with eigenvalues
:math:`\hat{k}_{\theta,(h,w)} = (\mathcal{F}\,k_\theta)_{hw}` (2-D DFT).
All operations required for GP fitting and prediction — matrix-vector products, log-determinants,
and gradient computations — can be performed in the spectral domain:

.. math::

   K_\theta \cdot v \;=\; \tfrac{1}{n}\, \mathcal{F}^{*}\!\bigl(\hat{k}_\theta \odot \mathcal{F}(v)\bigr),
   \qquad
   \log |K_\theta| \;=\; \sum_{h,w} \log \hat{k}_{\theta,(h,w)},

where the :math:`1/n` in the matrix-vector product reflects the unnormalised DFT convention
:math:`\mathcal{F}^{-1} = (1/n)\,\mathcal{F}^{*}`, and the log-determinant identity follows from
:math:`\hat{k}_{\theta,(h,w)} > 0` for a positive-definite stationary kernel.
This reduces the per-step GP cost from :math:`O(n^3)` (dense Cholesky) or
:math:`O(nM^2)` (inducing-point) to :math:`O(n \log n)` per L-BFGS iteration,
with no approximation error for stationary kernels on the grid.

The GP hyperparameters (signal variance, length scale, noise variance) are optimised by
maximising the marginal log-likelihood using L-BFGS with gradients computed in Fourier
space.  Covariate residuals :math:`\tilde{Z} = Z - \hat{f}(X)` are then obtained via
the FFT-based posterior mean.

After residualisation, the HSIC test itself proceeds identically to the
:class:`~splisosm.SplisosmNP` path: linear-kernel HSIC via :math:`\mathrm{tr}(Y^\top K_{\tilde{Z}} Y)=\|\tilde{Z}^\top Y\|^2`,
with p-values from Liu's chi-squared mixture null using the eigenvalues of :math:`K_{\tilde{Z}} = \tilde{Z}\tilde{Z}^\top` (rank-1, giving a single nonzero eigenvalue :math:`\|\tilde{Z}\|^2`) and :math:`K_Y = Y Y^\top`.


Parametric test: SplisosmGLMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~splisosm.SplisosmGLMM` provides a parametric alternative based on a **multinomial GLMM** with a Gaussian random field (GRF) random effect.
Using a reference-category multinomial-logit link with :math:`q = p - 1` free dimensions per spot, the model is

.. math::

   Y_i \mid \eta_i &\sim \mathrm{Multinomial}(N_i,\, \mathrm{softmax}([\eta_i, 0])), \quad i = 1, \ldots, n, \\
   \eta_i &= X_i \beta + b + u_i \;\in\; \mathbb{R}^{q}, \\
   \mathrm{vec}(U) &\sim \mathcal{N}\!\bigl(0,\; \sigma^2 \, \Sigma \otimes I_q \bigr), \quad
   \Sigma \;=\; \theta\, K \;+\; (1-\theta)\, I_n,

where :math:`X_i` is the row of covariates for spot :math:`i`, :math:`\beta \in \mathbb{R}^{d \times q}` are fixed effects, :math:`b \in \mathbb{R}^{q}` is an intercept,
and :math:`U = (u_1, \ldots, u_n)^\top \in \mathbb{R}^{n \times q}` stacks the per-spot random effects. The covariance mixes the CAR spatial kernel :math:`K` with an i.i.d. component via :math:`\theta \in [0, 1]`, and assumes independence across the :math:`q` logit dimensions (:math:`\otimes I_q`).

.. note::

   The SV test (H₀: :math:`\theta = 0`) is implemented as a likelihood ratio test (LRT) in :func:`~splisosm.SplisosmGLMM.test_spatial_variability`. 
   However, it is not well-calibrated due to technical challenges in model fitting. 
   The equivalent score test version also takes a quadratic form similar to the HSIC test statistic but with spot-specific adjustments. 
   See :cite:`su2026consistent` for detailed analysis.


For DU testing, we use a **score test** comparing coefficient gradients at the null model (no fixed-effect covariates, :math:`\beta = 0`), which avoids fitting the full model for each covariate.
However, it still requires estimating nuisance parameters (intercept :math:`b`, total variance :math:`\sigma^2`, and spatial variance proportion :math:`\theta`).
To compute the maximum likelihood estimates, we approximate the marginal likelihood via **Laplace's method** at the mode of the random effects.
See :class:`splisosm.model.MultinomGLMM` for implementation details.

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
     - ``eig`` [#fft]_ / ``clt`` / ``welch`` / ``perm``
   * - HSIC-GC (SVE)
     - SplisosmNP / FFT
     - Gene counts
     - None
     - ``eig`` [#fft]_ / ``clt`` / ``welch`` / ``perm``
   * - HSIC-IC
     - SplisosmNP / FFT
     - Isoform counts
     - None
     - ``eig`` [#fft]_ / ``clt`` / ``welch`` / ``perm``
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

.. [#fft] :class:`~splisosm.SplisosmFFT` does not take a ``null_method`` parameter; ``'eig'`` is the default and only option for FFT-accelerated tests, since the full kernel spectrum is efficiently computed via FFT.
