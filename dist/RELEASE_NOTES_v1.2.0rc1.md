Preview release for SPLISOSM v1.2.0. This release candidate is intended for GitHub-only testing and is not uploaded to PyPI.

Install from the GitHub tag:

```bash
pip install "splisosm[sdata,gp] @ git+https://github.com/JiayuSuPKU/SPLISOSM.git@v1.2.0rc1"
```

Docs: https://splisosm.readthedocs.io/en/latest/
Full changelog: https://github.com/JiayuSuPKU/SPLISOSM/blob/v1.2.0rc1/CHANGELOG.md

## Behavior change: SplisosmNP SV defaults

`SplisosmNP.test_spatial_variability()` and `run_hsic_gc()` still default to `null_method="liu"`, but the large-data path changed:

- v1.1.x default: Liu approximation with automatic low-rank spatial truncation.
- v1.2.0 default: Liu approximation from full-rank spatial-kernel cumulants.

Large implicit CAR kernels estimate those cumulants with Hutchinson Rademacher trace probes. Tune that stochastic budget with:

```python
model.test_spatial_variability(null_configs={"n_probes": 120})
run_hsic_gc(counts_gene, coordinates, null_configs={"n_probes": 120})
```

`n_probes` is not a low-rank approximation.

P-values and FDR hit counts may change compared with v1.1.x. The legacy low-rank approach often finds more SVP genes on real datasets because it prioritizes global, low-frequency spatial patterns. That can produce smaller p-values when the biological signal is global, but it sacrifices sensitivity to local, high-frequency variation.

For analyses that intentionally prioritize global patterns, prefer a smoother full-rank CAR kernel, for example `rho=0.999`, rather than returning to rank truncation. This keeps local-pattern support while emphasizing broad spatial structure.

## Speed and memory

### Full-rank Liu cumulants for SV

The v1.2.0 Liu path is full-rank without requiring a full eigendecomposition of the spatial kernel. For an `n`-spot dense kernel, a direct eigendecomposition costs `O(n^3)` time and `O(n^2)` memory. The memory alone is prohibitive at modern spatial scales: one float64 dense `n x n` matrix is about 80 GB at `n=100,000` and about 8 TB at `n=1,000,000`, before storing eigenvectors or temporary work arrays.

For large implicit CAR kernels, SPLISOSM instead estimates trace cumulants with Hutchinson Rademacher probes. With `m = n_probes`, Liu's four cumulants use two kernel applications per probe and scale as roughly `O(m * cost(Kx))` time. The batched probe work arrays scale as `O(nm)` memory; at `n=1,000,000` and `m=60`, three float64 `n x m` arrays are about 1.4 GB, plus the sparse graph/precision storage. Lowering `n_probes` reduces this working memory and runtime linearly, while increasing it improves the stochastic trace accuracy. Spatial cumulants are cached for the default `nan_filling="mean"` path and reused across genes.

This is the key distinction from the older low-rank shortcut: v1.2.0 keeps the full-rank spatial null support, but avoids the `O(n^3)` eigendecomposition that full-rank Liu calibration would otherwise require.

### NUFFT GP residualization

The new `gpr_backend="nufft"` / `"finufft"` path avoids sklearn's dense GP kernel matrix on irregular 2-D coordinates. Exact sklearn GP fitting materializes an `n x n` kernel (`O(n^2)` memory) and uses dense Cholesky/eigendecomposition-style operations (`O(n^3)` time), which is suitable only for small to moderate `n`. SPLISOSM's sklearn large-data fallback fits hyperparameters on `M = n_inducing` sampled observations, reducing the fit to `O(M^3)` time and `O(M^2)` memory, but it learns hyperparameters from only the subset and still predicts back to all observations with `O(nM)` kernel work.

The NUFFT backend keeps the RBF kernel implicit. Each GP matrix-vector product uses type-1/type-2 FINUFFT calls over the effective Fourier grid, approximately `O((n + q) log q)` for `q` Fourier modes, and conjugate gradients solve `(K + eps I)` systems without forming the dense matrix. Hyperparameter fitting on irregular grids uses a leading eigensummary of rank `r = lml_approx_rank` with trace/tail corrections; memory scales as `O(nr)` for those vectors. For example, `n=1,000,000` and `r=64` is about 512 MB in float64 for the eigensummary vectors, compared with about 8 TB for one dense sklearn-size `n x n` matrix.

In practice, sklearn remains a convenient and accurate small-data reference, while NUFFT is the recommended backend when irregular-coordinate GP residualization must use all spots at large scale.

## Highlights

- New FINUFFT-backed NUFFT GP backend for `SplisosmNP.test_differential_usage(method="hsic-gp")` on large irregular 2-D coordinates.
- New NUFFT GP controls, including `n_modes`, `max_auto_modes`, and `lml_approx_rank`.
- Liu p-values now use direct multivariate HSIC cumulants instead of materializing all pairwise spatial/response eigenvalue products.
- Deprecated SV null aliases are routed automatically: `eig` to `liu`; `clt` and `trace` to `welch`.
- Documentation and tutorials now describe the full-rank cumulant default and NUFFT backend.

## Validation

- Sphinx docs build cleanly.
- `ruff` and `black --check` pass for touched Python modules.
- Tutorial notebooks were refreshed for the preview release.
- Full pytest suite passes in `splisosm_dev` with `NUMBA_CACHE_DIR` and
  `LOKY_MAX_CPU_COUNT` set for local dependency/runtime quirks.
- Local sdist and wheel build successfully, and `twine check` passes.
