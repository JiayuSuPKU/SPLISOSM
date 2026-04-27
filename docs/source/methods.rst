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
- **Graph Fourier basis**: eigenvectors of :math:`K` form a graph Fourier basis ordered by spatial frequency.
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
Three methods are available via the ``null_method`` argument to :meth:`~splisosm.SplisosmNP.test_spatial_variability`.

.. _null-liu:

**1. Liu's approximation of chi-square mixture (default)**: ``null_method='liu'``

Let :math:`Q = \mathrm{tr}(Y^\top K Y) = (n-1)^2 \widehat{\mathrm{HSIC}}` denote the unnormalised HSIC V-statistic,
write :math:`m=n-1` for the dimension of the centred spot space,
with :math:`\lambda_1^K \geq \cdots \geq \lambda_n^K` the eigenvalues of :math:`K` and
:math:`\lambda_1^Y \geq \cdots \geq \lambda_p^Y` those of :math:`Y^\top Y`.
Under the null, :math:`Q` asymptotically follows a weighted sum of independent :math:`\chi^2_{1}` variables,

.. math::

   Q \;\overset{d}{\approx}\; \frac{1}{n-1} \sum_{i=1}^{n} \sum_{j=1}^{p} \lambda_i^K \, \lambda_j^Y \; Z_{ij}, \qquad Z_{ij} \overset{\text{iid}}{\sim} \chi^2_{1},

where the double sum runs over all pairs of spatial and response eigenvalues.
SPLISOSM evaluates :cite:t:`liu2009new` from the first four cumulants
:math:`c_r = \mathrm{tr}(K^r)\mathrm{tr}((Y^\top Y)^r)/(n-1)^r`, so it does not need to materialize the full pairwise eigenvalue product.
See :func:`splisosm.utils.hsic.liu_sf_from_cumulants` for implementation details.

.. note::

   - The previous name ``null_method='eig'`` is retained as a deprecated alias
     and will emit a ``DeprecationWarning``.
   - Full eigen-decomposition of :math:`K` is :math:`O(n^3)` and is not feasible for large datasets.
     For implicit CAR kernels with no realised dense covariance, SPLISOSM estimates :math:`\mathrm{tr}(K^r)` with Hutchinson Rademacher probes by default.
     Use ``null_configs={"n_probes": m}`` to control that budget.
   - Since v1.2.0, large :class:`~splisosm.SplisosmNP` SV tests use full-rank
     cumulants by default rather than the v1.1.x low-rank spatial
     approximation. This preserves sensitivity to both global and local spatial
     patterns. If the analysis should intentionally emphasize global smooth
     patterns, prefer increasing the CAR smoothness parameter (for example
     ``rho=0.999``) instead of truncating the spatial rank.
   - When ``nan_filling='mean'`` (default), the spatial cumulants are cached once and reused for all subsequent genes.
   - When ``nan_filling='none'`` for HSIC-IR, SPLISOSM drops zero-coverage
     spots per gene and applies a masked implicit spatial kernel.  This avoids
     dense parent-kernel realization, but per-gene masked cumulants must still
     be estimated and the path is slower than mean filling.

.. _null-welch:

**2. Welch-Satterthwaite scaled chi-squared approximation**: ``null_method='welch'``

Alternatively, we may use only the first two moments of the null distribution,
which requires :math:`\mathrm{tr}(K)` and :math:`\mathrm{tr}(K^2)` but not
higher cumulants or a full eigendecomposition.  Taking moments of the
:math:`\chi^2_1` mixture above gives

.. math::

   \mu_0 &= \mathbb{E}[Q] = \frac{1}{m}\,\mathrm{tr}(K)\,\mathrm{tr}(Y^\top Y), \\[4pt]
   \sigma_0^2 &= \mathrm{Var}(Q) = \frac{2}{m^2}\,\mathrm{tr}(K^2)\,\mathrm{tr}\!\bigl((Y^\top Y)^2\bigr).

With all positive eigenvalues, the chi-squared mixture null can be approximated by the `Welch-Satterthwaite method <https://en.wikipedia.org/wiki/Welch%E2%80%93Satterthwaite_equation>`_,
using one scaled chi-squared variable :math:`g \, \chi^2_h` with scale parameter :math:`g` and degrees of freedom :math:`h`. 
The parameters are chosen to match the first two moments of the null :math:`(\mu_0, \sigma_0^2)`.

.. math::

   g = \frac{\sigma_0^2}{2\mu_0}, \qquad h = \frac{2\mu_0^2}{\sigma_0^2},

and the p-value is :math:`\mathbb{P}\!\left(\chi^2_h \geq Q/g\right)`.

.. note::

   - The retired ``null_method='clt'`` and ``null_method='trace'`` names are
     retained as deprecated aliases and automatically use this Welch
     approximation.
   - Using only the first two cumulants :math:`\mathrm{tr}(K)` and :math:`\mathrm{tr}(K^2)`,
     Welch is typically less accurate in the right tail (small p-values) than
     the :ref:`Liu approximation <null-liu>`, which uses four cumulants.
     In practice the difference is often small.

.. _null-perm:

**3. Batched permutation test**: ``null_method='perm'``

Generates a null distribution by repeatedly permuting the rows of :math:`Y` (breaking the spatial structure) and recomputing :math:`\mathrm{tr}(Y_\pi^\top K Y_\pi)` for each permutation :math:`\pi`.
P-values use the finite-permutation correction
:math:`(1 + \#\{Q_\pi \ge Q_{\mathrm{obs}}\}) / (B + 1)`.

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

   For irregular coordinates, use :class:`~splisosm.SplisosmNP` for SV tests.
   The FINUFFT backend described below is currently used for GP residualization
   in conditional DU tests, not for the SV spatial kernel.

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

   where :math:`k_\theta` is a ``Constant x RBF + WhiteNoise`` kernel. SPLISOSM provides dense sklearn, GPyTorch, FFT, and FINUFFT-backed NUFFT GP backends for different data geometries and scales.

2. **Compute covariate residuals** :math:`\tilde{Z} = Z - \hat{f}(X)`, capturing the part of covariate variation *not explained by* spatial position.

3. **Test** :math:`\widehat{\mathrm{HSIC}}(\tilde{Z},\, Y)` using the rank-1 linear covariate kernel :math:`K_{\tilde{Z}} = \tilde{Z}\tilde{Z}^\top` and a similar linear kernel :math:`K_Y = Y Y^\top` for the response.
   Since both kernels are low-rank, the Liu null can be computed from the
   nonzero covariate and response eigenvalues.

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
   Three backend families are supported:

   - ``gpr_backend='sklearn'`` (default): dense sklearn ``GaussianProcessRegressor`` with optional subset-of-data hyperparameter fitting.
   - ``gpr_backend='gpytorch'``: exact or FITC sparse GP with ``n_inducing`` inducing points and optional GPU support.
   - ``gpr_backend='nufft'`` / ``'finufft'``: FINUFFT-backed implicit RBF grid-kernel for irregular 2-D coordinates, recommended for large-scale spatial data.

   For sklearn and gpytorch, pass ``gpr_configs={"covariate": {"n_inducing": 1000}}`` to control the subset or inducing-point budget.
   For the NUFFT backend, ``max_auto_modes`` caps the automatically inferred full effective grid, and ``lml_approx_rank`` controls the irregular-grid likelihood approximation used during hyperparameter fitting.
   These options are passed through ``gpr_configs``; see :meth:`splisosm.SplisosmNP.test_differential_usage` for the user-facing configuration table and :doc:`api/gpr` for backend class details.

NUFFT backend for irregular-coordinate GP residualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``gpr_backend='nufft'`` / ``'finufft'`` path targets irregular 2-D coordinates where dense GP matrices are too expensive. Coordinates are affinely mapped into a periodic box :math:`[-\pi,\pi)^2`, and the stationary RBF covariance is represented on a Fourier mode set :math:`\Omega` as

.. math::

   k_\theta(x_i, x_j)
   \approx
   \sigma_f^2
   \sum_{\omega \in \Omega}
   a_\theta(\omega)
   \exp\{i\omega^\top(t_i - t_j)\}
   \;+\; \sigma_\varepsilon^2 \mathbf{1}_{i=j},

where :math:`t_i` are the mapped coordinates and :math:`a_\theta(\omega)` are non-negative spectral weights determined by the RBF length scale and Fourier grid spacing. With ``n_modes=None``, SPLISOSM uses the full effective Fourier grid inferred from the point count and coordinate aspect ratio; FINUFFT's internal oversampling is controlled separately via ``nufft_opts``.

With this representation, we can compute the matrix-vector product in :math:`O(n \log n)` time by

.. math::

   K_s v \;\approx\; F_X^* \left(a_\theta \odot F_X v\right),

where :math:`F_X v = \sum_i v_i \exp(-i\omega^\top t_i)` is a type-1 NUFFT and :math:`F_X^*` is the corresponding type-2 NUFFT. GP residualization solves

.. math::

   (K_s + \sigma_\varepsilon^2 I)\alpha = z,
   \qquad
   \tilde z = \sigma_\varepsilon^2 \alpha,

using conjugate gradients, because :math:`K_s` is only available through NUFFT matvecs.

On compatible regular grids, the same spectral representation reduces to the FFT GP path because the Fourier modes are exact eigenvectors. On irregular coordinates the nonuniform Fourier features are not orthogonal eigenvectors, so hyperparameter fitting approximates the log marginal likelihood with leading NUFFT eigensummaries plus trace and trace-square tail corrections. The ``lml_approx_rank`` parameter controls this eigensummary rank; it affects hyperparameter fitting accuracy and memory, not the Fourier grid used for each matvec. Memory scales as :math:`O(nr)` for rank ``r = lml_approx_rank`` (about 512 MB for 1M spots and ``r=64`` in float64), while the NUFFT matvecs avoid forming the :math:`n \times n` dense GP matrix.


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

   The SV test (H₀: :math:`\theta = 0`) is implemented as a likelihood ratio test (LRT) in :meth:`~splisosm.SplisosmGLMM.test_spatial_variability`.
   However, it is not well-calibrated due to technical challenges in model fitting. 
   The equivalent score test version also takes a quadratic form similar to the HSIC test statistic but with spot-specific adjustments. 
   See :cite:`su2026consistent` for detailed analysis.


For DU testing, we use a **score test** comparing coefficient gradients at the null model (no fixed-effect covariates, :math:`\beta = 0`), which avoids fitting the full model for each covariate.
However, it still requires estimating nuisance parameters (intercept :math:`b`, total variance :math:`\sigma^2`, and spatial variance proportion :math:`\theta`).
To compute the maximum likelihood estimates, we approximate the marginal likelihood via **Laplace's method** at the mode of the random effects.
See :class:`splisosm.glmm.MultinomGLMM` for implementation details.

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
     - ``liu`` [#fft]_ / ``welch`` / ``perm``
   * - HSIC-GC (SVE)
     - SplisosmNP / FFT
     - Gene counts
     - None
     - ``liu`` [#fft]_ / ``welch`` / ``perm``
   * - HSIC-IC
     - SplisosmNP / FFT
     - Isoform counts
     - None
     - ``liu`` [#fft]_ / ``welch`` / ``perm``
   * - DU (unconditional)
     - SplisosmNP / FFT
     - Isoform ratios
     - None
     - ``liu``
   * - DU (GP-conditional)
     - SplisosmNP / FFT
     - Isoform ratios
     - GP spatial residualisation
     - ``liu``
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

.. [#fft] :class:`~splisosm.SplisosmFFT` does not take a ``null_method`` parameter; ``'liu'`` is the default and only option for FFT-accelerated tests, since the full kernel spectrum is efficiently computed via FFT.
