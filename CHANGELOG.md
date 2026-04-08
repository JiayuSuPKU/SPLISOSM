# Changelog

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
  - `fit(quiet=True)` now suppresses not convergence warnings.

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
