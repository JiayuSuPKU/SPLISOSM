Preview release for SPLISOSM v1.2.0. This release candidate is intended for
GitHub-only testing and is not uploaded to PyPI.

Install from the GitHub tag:

```bash
pip install "splisosm[sdata,gp] @ git+https://github.com/JiayuSuPKU/SPLISOSM.git@v1.2.0rc1"
```

Docs: https://splisosm.readthedocs.io/en/latest/

Full changelog: https://github.com/JiayuSuPKU/SPLISOSM/blob/v1.2.0rc1/CHANGELOG.md

## Behavioral change - SV defaults

`SplisosmNP.test_spatial_variability()` and `run_hsic_gc()` still default to
`null_method="liu"/"eig"`, but now estimate cumulants from **full-rank** instead of the v1.1.1 automatic low-rank shortcut.

Use:

```python
model.test_spatial_variability(null_configs={"n_probes": 120})
run_hsic_gc(counts_gene, coordinates, null_configs={"n_probes": 120})
```

to increase the Hutchinson Rademacher trace budget. `n_probes` is not a
low-rank approximation.

P-values and FDR hit counts may change compared with v1.1.1. The old low-rank
path can look more powerful on global spatial patterns because it focuses on
low-frequency structure at the cost of zero sensitivity to local variation. 
For analyses that intentionally emphasize global patterns, prefer a smoother 
full-rank CAR kernel, for example `rho=0.999`, rather than returning to rank truncation.

Permutation p-values now use `(1 + # null >= observed) / (B + 1)`. GLMM SV
permutation nulls are kept per gene instead of being pooled across genes.

## Performance and memory

Expect significant runtime and memory improvements for large datasets due to:

- Memory-aware automatic feature chunking to reduce overhead and improve speed.
- Sparse-preserving algrebra for kernel operations.
- Joblib-based parallelism for the stand-alone `run_hsic_gc()` function.

Below are operation-count estimates relative to v1.1.1; actual gains depend on
sparsity, isoform counts per gene, `n_jobs`, and the spatial kernel.

### `SplisosmNP`

- v1.1.1's default large-data Liu path cached a rank
  `k = ceil(4 * sqrt(n_spots))` eigensummary. Storing both eigenvectors and the
  weighted low-rank factor costs about `2 * n_spots * k * 4` bytes: roughly
  1.0 GB at 100K spots and 32 GB at 1M spots, before eigensolver work arrays.
- v1.2.0's default `n_probes=60` cumulant path uses about
  `3 * n_spots * n_probes * 8` bytes for batched probe/result arrays: roughly
  144 MB at 100K spots and 1.4 GB at 1M spots, plus sparse graph/precision
  storage. This is about 7x lower memory at 100K spots and more than 20x lower
  at 1M spots for the null-calibration state.
- Null setup replaces thousands of Lanczos eigenvectors at million-spot scale
  with `2 * n_probes` kernel applications. The observed statistic is now
  full-rank, so per-gene work can be heavier than the old low-rank shortcut,
  but sparse reductions and 32-column chunks reduce dispatch and solver
  overhead.
- Kernel calls drop from one per gene to about
  `ceil(total_response_columns / 32)`: up to 32x fewer calls for `hsic-gc` and
  usually about 8-16x fewer calls for 2-4 isoforms per gene.

### `SplisosmFFT`

- v1.1.1 formed a per-gene product spectrum
  `lambda_spatial x lambda_response`. This temporary costs about
  `8 * n_grid * rank(response)` bytes per gene: about 24 MB at a 1M-cell grid
  for a 3-dimensional response, or 80 MB for a 10-dimensional response. v1.2.0
  uses cumulants instead, so the null calculation is effectively constant
  memory per gene after the FFT spectrum is cached.
- FFT spatial statistics are packed by response channel. The number of FFT
  kernel calls drops from one per gene to about
  `ceil(total_response_channels / 32)`, with the same 32x (`hsic-gc`) or
  8-16x (typical multi-isoform genes) call-count reduction.
- The automatic chunk cap keeps live FFT work arrays under the 2 GiB per-worker
  budget; at a 1M-cell grid and 32 channels the estimate is about 1.5 GB.

### NUFFT GP residualization

The new `gpr_backend="nufft"` / `"finufft"` path avoids sklearn's dense GP
kernel matrix on irregular 2-D coordinates. Exact sklearn GP fitting uses
`O(n^2)` memory and `O(n^3)` dense linear algebra. NUFFT keeps the RBF kernel
implicit; GP matrix-vector products scale approximately as
`O((n + q) log q)` for `q` Fourier modes, and irregular-grid hyperparameter
fitting stores `O(n * lml_approx_rank)` eigensummary vectors. At 1M spots and
`lml_approx_rank=64`, those vectors are about 512 MB in float64, compared with
about 8 TB for one dense sklearn-size `n x n` kernel.

## Highlights

- New FINUFFT-backed NUFFT GP backend for
  `SplisosmNP.test_differential_usage(method="hsic-gp")`.
- New NUFFT controls: `n_modes`, `max_auto_modes`, and `lml_approx_rank`.
- Sparse-aware, response-column chunked SV tests for `SplisosmNP`,
  `SplisosmFFT`, and `run_hsic_gc`; `chunk_size="auto"` caps both NP and FFT
  at 32 response columns/channels.
- `run_hsic_gc()` now accepts `n_jobs`.
- `hsic-ir` with `nan_filling="none"` uses a masked implicit spatial kernel
  instead of dense per-gene spatial submatrices.
- Deprecated null aliases are routed automatically: `eig` to `liu`; `clt` and
  `trace` to `welch`.

## Fixes

- GLMM SV permutation calibration now uses per-gene permutation nulls.
- NP and GLMM permutation p-values now use the standard finite-permutation
  correction.
- FFT DU t-tests reject constant or all-NaN binary covariates during
  validation.
- FFT `n_jobs=0` now receives the shared input-validation error.
- Sparse linear HSIC keeps null eigenvalues consistent with `centering=False`.

## Docs, API, and packaging

- Documentation and tutorials now describe the full-rank cumulant SV default
  and the NUFFT backend.
- API docs are reorganized into Core API and Advanced Options.
- GP backend docs now include kernel operator classes before residualizer
  classes.
- The source tree now uses `src/splisosm`; advanced helpers are grouped under
  `splisosm.gpr`, `splisosm.utils`, `splisosm.io`, `splisosm.glmm`, and
  `splisosm.hyptest`. Main imports such as
  `from splisosm import SplisosmNP, SplisosmFFT, SplisosmGLMM` are unchanged.

## Validation

- Sphinx docs build cleanly.
- `ruff` and `black --check` pass for touched Python modules.
- Full pytest suite passes in `splisosm_dev` with local runtime-cache settings.
- Tutorial notebooks were refreshed for the preview release.
- Local sdist and wheel build successfully, and `twine check` passes.

**Full diff**: https://github.com/JiayuSuPKU/SPLISOSM/compare/v1.1.1...v1.2.0rc1
