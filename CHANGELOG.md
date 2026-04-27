# Changelog

## v1.2.0rc1 (2026-04-27, preview)

Preview release for v1.2.0. This release candidate is intended for GitHub
testing and is not uploaded to PyPI.

```bash
pip install "splisosm[sdata,gp] @ git+https://github.com/JiayuSuPKU/SPLISOSM.git@v1.2.0rc1"
```

### Behavioral changes

**`SplisosmNP` SV default: Liu + low-rank -> Liu + full-rank cumulants**

- `SplisosmNP.test_spatial_variability()` and `run_hsic_gc()` still default to
`null_method="liu"` (previously named "eig"), but now estimate cumulants for Liu's approximation from **full-rank** spatial-kernel, avoiding eigen-decomposition and low-rank approximations. 
- Dense/spectral kernels
use exact traces when cheap; implicit kernels use Hutchinson Rademacher trace
estimates. Use `null_configs={"n_probes": m}` to tune this stochastic trace
budget. This is not a low-rank approximation.
- To recover the previous low-rank behavior, set `null_configs={"approx_rank": k}`.

This changes the statistic and p-values for large datasets (`n>5,000`) that
previously used automatic rank truncation. FDR hit counts may
change compared with v1.1.1, although gene rankings are usually similar. Specifically, the old low-rank path can look more powerful because it prioritizes low-frequency structure at
the cost of zero sensitivity to local variation. To emphasize
global patterns in v1.2.0, prefer a smoother full-rank kernel, for example
`rho=0.999`, rather than returning to rank truncation.

Permutation p-values now use `(1 + # null >= observed) / (B + 1)`. GLMM SV
permutation nulls are kept per gene instead of being pooled across genes.

### Performance and memory

Expect significant runtime and memory improvements for large datasets due to:

- Memory-aware automatic feature chunking to reduce overhead and improve speed.
- Sparse-preserving algrebra for kernel operations.
- Joblib-based parallelism for the stand-alone `run_hsic_gc()` function.

Estimated impact compared with v1.1.1; exact gains depend on sparsity, isoform
counts per gene, `n_jobs`, and the spatial kernel.

- **`SplisosmNP` SV**:
  - v1.1.1's default large-data Liu path cached a rank
    `k = ceil(4 * sqrt(n_spots))` eigensummary. Storing both eigenvectors and
    the weighted low-rank factor costs about `2 * n_spots * k * 4` bytes:
    roughly 1.0 GB at 100K spots and 32 GB at 1M spots, before eigensolver
    work arrays.
  - v1.2.0's default `n_probes=60` cumulant path uses about
    `3 * n_spots * n_probes * 8` bytes for batched probe/result arrays:
    roughly 144 MB at 100K spots and 1.4 GB at 1M spots, plus sparse
    graph/precision storage. That is about 7x lower memory at 100K spots and
    more than 20x lower at 1M spots for the null-calibration state.
  - Null setup replaces thousands of Lanczos eigenvectors at million-spot scale
    with `2 * n_probes` kernel applications. The observed statistic is now
    full-rank, so per-gene work can be heavier than the old low-rank shortcut,
    but sparse reductions and 32-column chunks reduce dispatch and solver
    overhead. Kernel calls drop from one per gene to about
    `ceil(total_response_columns / 32)`: up to 32x fewer calls for `hsic-gc`
    and usually about 8-16x fewer calls for 2-4 isoforms per gene.
- **`SplisosmFFT` SV**:
  - v1.1.1 formed a per-gene product spectrum
    `lambda_spatial x lambda_response`. This temporary costs about
    `8 * n_grid * rank(response)` bytes per gene: about 24 MB at a 1M-cell grid
    for a 3-dimensional response, or 80 MB for a 10-dimensional response.
    v1.2.0 uses cumulants instead, so the null calculation is effectively
    constant memory per gene after the FFT spectrum is cached.
  - FFT spatial statistics are packed by response channel. The number of FFT
    kernel calls drops from one per gene to about
    `ceil(total_response_channels / 32)`, with the same 32x (`hsic-gc`) or
    8-16x (typical multi-isoform genes) call-count reduction. The automatic
    chunk cap keeps live FFT work arrays under the 2 GiB per-worker budget; at
    a 1M-cell grid and 32 channels the estimate is about 1.5 GB.

### New features

- FINUFFT-backed NUFFT GP backend for
  `SplisosmNP.test_differential_usage(method="hsic-gp")` on large irregular
  2-D coordinates. Faster, more memory-efficient, and more accurate than the default Sklearn backend. Use `gpr_backend="nufft"` or `"finufft"`.
- New NUFFT controls: `n_modes`, `max_auto_modes`, and
  `gpr_configs={"covariate": {"lml_approx_rank": r}}`.
- Sparse-aware, response-column chunked SV tests for `SplisosmNP`,
  `SplisosmFFT`, and `run_hsic_gc`; `chunk_size="auto"` caps NP and FFT chunks
  at 32 response columns/channels.
- `run_hsic_gc()` now accepts `n_jobs`.
- `hsic-ir` with `nan_filling="none"` uses a masked implicit spatial kernel
  instead of materializing dense per-gene spatial submatrices.
- Deprecated null aliases are routed automatically: `eig` -> `liu`, and
  `clt` / `trace` -> `welch`.

### Fixes

- GLMM SV permutation calibration now compares each gene with its own
  permutation null.
- FFT DU t-tests reject constant or all-NaN binary covariates during
  validation.
- FFT `n_jobs=0` now receives the shared input-validation error.
- Sparse linear HSIC keeps null eigenvalues consistent with `centering=False`.

### Docs and API

- Quickstart, FAQ, installation, methods, README, API pages, and tutorial text
  now describe the full-rank cumulant SV default and clarify that `n_probes`
  is trace-estimation control, not low-rank approximation.
- Added NUFFT GP methods/API documentation, including `lml_approx_rank`.
- Reorganized API docs into Core API and Advanced Options.
- Moved the package to a `src/splisosm` layout and grouped advanced helpers
  under `splisosm.gpr`, `splisosm.utils`, `splisosm.io`, `splisosm.glmm`, and
  `splisosm.hyptest`. Main imports such as
  `from splisosm import SplisosmNP, SplisosmFFT, SplisosmGLMM` are unchanged.
- Tutorial notebooks were refreshed for the preview release.

### Testing

- Added and extended tests for cumulant Liu p-values, SV chunking and sparse
  paths, `run_hsic_gc` parallelism, NUFFT GP agreement, public API imports, and
  removal of old internal import paths.
- Sphinx docs, local links, package build, and tutorial outputs were refreshed
  for the preview release.

## v1.1.1 (2026-04-20)

### Bug fixes

**`SplisosmFFT` — missing kernel double-centring on FFT SV tests**

Prior to v1.1.1, `SplisosmFFT.setup_data()` built the internal `FFTKernel` without passing `centering=True`, so the periodic CAR kernel
retained its DC eigenvalue $\lambda^K_{(0,0)} = 1/(1 - \rho)$ (≈ 100 at the default `rho=0.99`). 
This violates the standard HSIC double-centring convention and inflates p-values because the constant mode is included in the null mixture.

Impact of the fix (`centering=True` is now passed explicitly):

- **Test statistic** (`tr(Y^T K Y)`): **unchanged** because of existing column-centring.
- **Gene ranking**: **unchanged** (a monotone transformation of test statistics).
- **P-values**: systematically *smaller* (more significant) after the fix.
- **FDR hits**: expect *more* genes to pass a fixed BH threshold after upgrading; the previous v1.1.0 results were conservative.
  From tutorials: 
  - `Visium FFPE`: 68 → 74 SVP genes (FDR < 0.01).
  - `Visium HD FFPE`: 192 → 196 SVP genes (FDR < 0.01).
  - `Visium HD 3'`: 501 → 506 SVP genes (FDR < 0.01).
  - `Visium HD ONT`: 784 → 790 SVP genes (FDR < 0.01).
  - `Xenium Prime 5K binned`: 2144 → 2158 SVP genes (FDR < 0.01).

**`SplisosmNP` — per-gene null mismatch for `hsic-ir` + `nan_filling='none'`**

In this branch the worker builds a per-gene double-centred kernel submatrix `K_sp_gene` (dropping spots whose isoform ratios are NaN),
but the `trace` / `welch` null reused `tr(K_sp)` / `tr(K_sp²)` from the *global* kernel, and the `perm` null applied the global `K_sp` to the
NaN-filtered `y_batch` (causing a shape mismatch and runtime error whenever filtering actually removed spots). All three null methods now
reference `K_sp_gene` consistently.

### Other bug fixes with no user-visible numerical change

- `SpatialCovKernel` implicit (LU-solve) path for `n > 5000`:
  - `_hutchinson_trace` unconditionally returned `tr(HKH)` / `tr((HKH)²)`; now branches on the `_centering` flag so
    `centering=False` correctly returns `tr(K)` / `tr(K²)`.
  - `xtKx_exact` returned `x^T K x` regardless of `_centering`; now applies `H` on both sides when `centering=True`.

  All current call sites build the implicit kernel with `centering=True` and pre-column-centre the input, so dense-mode and implicit-mode
  results continue to agree; these are latent API consistency fixes.
- BED probe filtering in `load_visium_probe`: tightened substring matching to avoid spurious hits across gene name prefixes.

### Renames (back-compat preserved)

- **`null_method='trace'` → `null_method='clt'`** in `SplisosmNP.test_spatial_variability` and `splisosm.utils.run_hsic_gc`.
  The previous name `'trace'` conflated the moment-matching normal (Central Limit Theorem) approximation with the Welch–Satterthwaite
  `'welch'` path, which also uses the matrix traces `tr(K)` / `tr(K²)`.
  `'trace'` is still accepted and returns identical results, but emits a `DeprecationWarning`; please update call sites to `'clt'`.

### New features

- **`null_method='welch'`** for `SplisosmNP.test_spatial_variability` and `splisosm.utils.run_hsic_gc`. 
  Uses Welch–Satterthwaite moment matching (`g·χ²_h` with `g = Var/2E`, `h = 2E²/Var`) from the same `tr(K)` and `tr(K²)` as `null_method='clt'`. 
  Typically close to the `eig` (Liu) reference and is recommended when `eig` is too slow.
- **Optional `spatial_key`** when `adj_key` is provided (`SplisosmNP`, `SplisosmGLMM`, `run_hsic_gc` AnnData mode). 
  Non-spatial AnnData (e.g. scRNA-seq with `adata.obsp['connectivities']` from `scanpy.pp.neighbors`) can now be tested end-to-end without coordinates.
  `method='spark-x'` (SV) and `method='hsic-gp'` (DU) raise a targeted `ValueError` at call time when coordinates are absent.
- `IdentityKernel` and `FFTKernel` now document a `centering: bool = False` constructor argument matching `SpatialCovKernel`; 
  HSIC-based SV/DU workflows should always set `centering=True` (no direct impact).

### Documentation

- **`methods.rst`** — mathematical corrections and notation sweep:
  - Liu's chi-squared mixture null: the missing `1/n` factor is now explicit, `Q = tr(Y^T K Y) ≈ (1/n) Σ λ^K_i λ^Y_j Z_{ij}`.
  - Trace/Welch moments: `μ₀ = (1/n) tr(K) tr(Y^T Y)`, `σ₀² = (2/n²) tr(K²) tr((Y^T Y)²)` (previous form had an incorrect `1/(n−1)` scaling).
  - FFT-DU: corrected the claim about the p-value source: covariate and response eigenvalues, not the FFT spatial spectrum.
  - FFT convention: the `1/n` prefactor in the inverse DFT identity `F⁻¹ = (1/n) F*` is now stated explicitly.
  - GLMM model spec rewritten with explicit dimensions (`β ∈ ℝ^{d×q}`, `U ∈ ℝ^{n×q}`), reference-category multinomial-logit
    link, and a Kronecker-form random-effect covariance `Σ = θK + (1−θ)I_n`, `vec(U) ~ N(0, σ² Σ ⊗ I_q)`.
  - Notation unified across the page: `i ∈ {1,…,n}` indexes spots, `j ∈ {1,…,p}` indexes isoforms; `K_{ii'}` for spot-pair kernel entries.
- **`quickstart.rst`** — Inputs-and-outputs section rewritten and consolidated with the "Expected input data format" spec moved in from
  `txquant.rst`. New subsection documenting the non-spatial / single-cell workflow via `adj_key` + targeted errors for operations
  that still require coordinates.
- **README** — platform/feature table, model-class decision tree, non-spatial path, paper + preprint references, and badges.
- All tutorial notebooks: updated to v1.1.1. Add a new `visium_ffpe.ipynb` demo for 10x Visium FFPE (v2, CytAssit) data and for SplisosmNP vs SplisosmFFT comparison.

### Testing

- New: `test_sv_nan_filling_none_uses_per_gene_kernel_moments` (`tests/test_hyptest_np.py`) — regression for the per-gene null fix.
- New: `test_implicit_honors_centering_flag` (`tests/test_kernel.py`) — dense/implicit agreement for `trace`, `square_trace`, and `xtKx_exact` under both
  `centering=True` and `centering=False`.
- Extended `test_null_methods_agreement` to include `welch` alongside `eig` / `clt` / `perm` (Spearman-ρ thresholds at ≥ 0.90 pairwise among the three asymptotic methods).
- New: `test_sv_null_method_trace_alias_deprecated` (`tests/test_hyptest_np.py`) and `test_matrix_mode_null_method_trace_alias_deprecated` (`tests/test_utils.py`) — 
  assert the deprecated `'trace'` alias returns identical results to `'clt'` and emits a `DeprecationWarning`.
- 390 tests passing (up from 371 in v1.1.0), 4 GPU-skipped.

## v1.1.0 (2026-04-08)

### Breaking Changes

**Behavioral changes:**
- `counts_to_ratios` with `nan_filling='mean'` now fills NaNs after ratio transformation instead of before. This does not affect the default no-transformation behavior.
- `SplisosmGLMM` default fitting parameters changed:

  | Parameter | v1.0.4 | v1.1.0 | Rationale |
  |-----------|--------|--------|-----------|
  | `var_fix_sigma` | `False` | `True` | Fix total variance to method-of-moments estimate; faster convergence |
  | `init_ratio` | `"observed"` | `"uniform"` | More robust for sparse data |
  | `fitting_configs["max_epochs"]` | `-1` (10000) | `500` | Efficiency (still recommend large max_epochs if possible) |
  | `var_parameterization_sigma_theta` | `True` (user choice) | always `True` (removed) | Simplify API |

**Renamed attributes (all three classes):**
- `n_isos` → `n_isos_per_gene` in `SplisosmNP` and `SplisosmFFT` (aligns with `SplisosmGLMM`)
- `corr_sp` / `kernel` → `sp_kernel` across all three classes (unified spatial kernel attribute)
- `sv_test_results` / `du_test_results` → now private (`_sv_test_results` / `_du_test_results`); use `get_formatted_test_results()` instead
- `model_type` / `model_configs` → now private in `SplisosmGLMM` (shown in `__str__`)
- `data` / `coordinates` → now private in `SplisosmNP` and `SplisosmGLMM`
- `k_neighbors` / `rho` / `standardize_cov` → now private in `SplisosmNP` (shown in `__str__`)

**Removed methods:**
- `SplisosmGLMM.fitting_results` property — use `get_fitted_models()` or `get_gene_model()` instead
- `SplisosmGLMM.n_isos` property alias — use `n_isos_per_gene` directly

**Changed parameters and arguments:**
- `SplisosmNP` / `run_hsic_gc`:
  - move `approx_rank` from `setup_data` to `test_spatial_variability(null_configs={'approx_rank': ...})` for finer control over kernel construction. Setting `'approx_rank': None` now will override the default low-rank approximation and use the full kernel for large datasets (n > 5000)
  - replace `use_perm_null`/`n_perms_per_gene` in `test_spatial_variability` with `null_method="perm"` and the `null_configs={'n_perms_per_gene': ...}` dict
- `SplisosmGLMM`:
  - `PatienceLogger` now always logs the training loss. Use `store_param_history` instead of `diagnose` to keep track of parameter trajectories.
  - `fit(quiet=True)` now suppresses non-convergence warnings.

**Removed parameters:**
- `setup_data()` from `SplisosmNP` and `SplisosmGLMM` now only accepts AnnData as input; legacy array-based setup removed
- `filter_single_iso_genes` removed from `SplisosmGLMM.setup_data()` — GLMM always requires ≥2 isoforms per gene; the parameter is still available in `SplisosmNP` and `SplisosmFFT`
- `var_parameterization_sigma_theta` retired from `MultinomGLMM` — only the sigma/theta parameterization is supported

**Deprecated functions:**
- `extract_gene_level_statistics()` → use `compute_feature_summaries()` instead (emits `DeprecationWarning`)
- `extract_counts_n_ratios` → use `add_ratio_layer` instead (emits `DeprecationWarning`)

### New Features

**Parallelism and GPU support:**
- `SplisosmNP.test_spatial_variability()` and `test_differential_usage()` now accept `n_jobs` parameter for joblib-based gene-level parallelism (`prefer="threads"`)
- GPU guard: parallelism automatically disabled when `gpr_backend="gpytorch"` with `device != "cpu"`
- GPU support for `SplisosmGLMM` fitting via `SplisosmGLMM(device=...)` with `"cpu"`, `"cuda"`, or `"mps"` backends
- FFT worker auto-coordination: `workers = max(1, cpu_count() // n_jobs)` prevents thread oversubscription

**Performance:**
- Low-rank approximation for GLMM fitting via `SplisosmGLMM(approx_rank=...)`, defaulting to `int(4*sqrt(n))` for n > 5000
- Analytic sigma Hessian (`_get_log_lik_hessian_sigma_expand_analytic`) replaces `torch.autograd.functional.jacobian` — closed-form O(G·(p-1)·rank) computation
- Single-allocation Hessian in `_get_log_lik_hessian_nu` — halves peak memory by filling MVN + multinomial blocks in-place
- Identity covariance fast paths for glmm-null: `_calc_log_prob_joint`, `_inv_cov`, and `_get_log_lik_hessian_nu` bypass eigenvector operations when theta=0
- Lean GLMM storage: per-gene models replaced with lightweight `_FittedGeneState` dataclasses after fitting; `save()`/`load()` no longer duplicates kernel eigenvectors across genes
- Improved `SplisosmGLMM` default configs (see above)

**Kernel module:**
- `SpatialCovKernel` refactored: handles sparse KNN graph construction internally (removing smoother-omics dependency), delays expensive eigen-decomposition until `eigenvalues()` are called
- New `adj_key` parameter in `SplisosmNP.setup_data()` and `run_hsic_gc()`: supports custom adjacency matrices (e.g., expression-based k-NN graphs)
- `skip_spatial_kernel=True` in `SplisosmNP.setup_data()`: uses `IdentityKernel` for DU-only workflows (no CAR kernel construction)
- `min_component_size` parameter in `SplisosmNP.setup_data()` and `run_hsic_gc()`: filters small disconnected tissue fragments from the spatial graph
- `FFTKernel` moved from `hyptest_fft.py` to `kernel.py` and now inherits `Kernel` ABC
- `IdentityKernel` added for identity-covariance and DU-only modes

**Kernel configs exposed in API:**
- `SplisosmNP(k_neighbors=..., rho=..., standardize_cov=...)`
- `SplisosmGLMM(k_neighbors=..., rho=..., approx_rank=...)`
- Optional skipping of kernel construction in `SplisosmNP.setup_data()` for DU-only analyses (`skip_spatial_kernel=True`)

**New SplisosmGLMM features:**
- `SplisosmGLMM(k_neighbors=..., rho=..., approx_rank=...)` — kernel construction configs moved to constructor
- `SplisosmGLMM.load(path)` — static method to load saved models (new; v1.0.4 only had `save()`)
- `get_gene_model(gene_name)` / `model[gene_name]` — retrieve fitted per-gene model
- `get_training_summary()` — per-gene convergence, loss, and timing DataFrame
- `get_fitted_ratios_anndata()` — extract fitted isoform ratios as AnnData layer

**API improvements:**
- `get_formatted_test_results(with_gene_summary=True)` appends gene-level summary statistics to results DataFrame
- `extract_feature_summary(level='gene'/'isoform')` on all three classes — cached gene/isoform statistics
- `filtered_adata` read-only property on `SplisosmNP` and `SplisosmGLMM`
- `compute_feature_summaries()` shared utility function (gene + isoform level)
- `add_ratio_layer()` utility for adding isoform ratio layers to AnnData
- `run_hsic_gc()` now supports AnnData mode with integrated filtering
- Unified `__str__` / `__repr__` across all three classes showing data summary, model config, and test status

**Numerical safety:**
- Woodbury inverse correction clamped to `[-1e6, 1e6]`
- Analytic Hessian eigenvalue derivatives clamped at `min=1e-6`
- Single-isoform gene guards: HSIC-IR returns `(0, 1.0)`, all DU methods return `(zeros, ones)`
- Marginal mode warning when `n_spots > 300`
- `nan_filling="none"` warning for expensive per-gene kernel path

**I/O:**
- `load_visium_sp_meta` moved to `splisosm.io` module (backward-compatible import from `utils`)

**Documentation:**
- New methods.rst sections: FFT-accelerated DU test, SplisosmNP vs SplisosmFFT differences
- quickstart.rst: complete rewrite with full-defaults code blocks, configuration reference tables
- New tutorial notebooks: `sit_mob_demo.ipynb`, `xenium_sc_segmented.ipynb`

### Bug Fixes
- Fix `MultinomGLMM` training:
  - `marginal_newton`: fix epoch counter 
  - `_update_joint_newton`: used beta value instead of gradient for Newton step
  - `get_params_iter()`: fix bug where default_collate was called on a dict
- Fix `ValueError` when `covariate_names` is a numpy array (truthiness check on `or`)
- Fix PyTorch deprecation warning in score test (`linalg.solve` output tensor resize)
- Fix `compute_feature_summaries` assertion when gene display names differ from groupby keys
- Fix `SplisosmGLMM.save()` to strip per-gene kernel buffers and raw adata, reducing file size
- Fix `run_hsic_gc` AnnData mode: correct sparse matrix handling and filtering
- Fix `SpatialCovKernel` eigenvalue caching: recompute when rank changes
- Fix `xtKx` computation for implicit (sparse precision) kernels
- Fix all `warnings.warn` calls to include `stacklevel=2` for correct caller attribution
- Fix AnnData obs index handling (ensure string indices throughout)

### Testing

- 371 tests passing (up from ~250 in v1.0.4), 5 skipped (GPU-only)
- New test classes: `TestParallelNP`, `TestSigmaHessianAnalytic`, `TestMultinomGLMMLowRank`, `TestRunHsicGc`
- Added parallelism determinism tests (n_jobs=1 vs n_jobs=2) for all SV/DU methods in NP and FFT
- Added analytic sigma Hessian tests (7 parameterization × prior combinations)
- Added identity fast path equivalence tests (log-prob and Hessian)
- Added `with_gene_summary`, `filtered_adata`, single-isoform edge case tests
- Added sparse vs dense consistency tests for design matrix and count handling
- Added save/load roundtrip tests with kernel re-linking verification
- Added GPU device tests (CPU/CUDA/MPS) for SplisosmGLMM workflows
