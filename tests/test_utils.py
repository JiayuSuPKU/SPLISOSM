import unittest
import json
import os
import sys
import tempfile
import types
import warnings
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import torch
from anndata import AnnData
from unittest.mock import patch
from splisosm.utils import (
    counts_to_ratios,
    false_discovery_control,
    load_visium_sp_meta,
    add_ratio_layer,
    extract_gene_level_statistics,
    run_hsic_gc,
    run_sparkx,
    _index_rows_sparse_coo,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        n_spots = 50
        n_genes = 3
        n_iso_per_gene = [2, 3, 1]
        n_total_isos = sum(n_iso_per_gene)

        # Create counts
        counts_dense = np.random.randint(0, 10, size=(n_spots, n_total_isos)).astype(
            np.float32
        )
        # Ensure some zeros for sparsity check
        counts_dense[counts_dense < 3] = 0

        # Create gene info
        gene_symbols = []
        isoform_names = []
        for i, n in enumerate(n_iso_per_gene):
            gene_name = f"Gene_{i}"
            for j in range(n):
                gene_symbols.append(gene_name)
                isoform_names.append(f"{gene_name}_Iso_{j}")

        var_df = pd.DataFrame({"gene_symbol": gene_symbols}, index=isoform_names)

        # Create AnnData
        self.adata = AnnData(X=counts_dense, var=var_df)
        self.adata.layers["counts"] = counts_dense
        # Store for comparison
        self.counts_dense = counts_dense
        self.gene_names = [f"Gene_{i}" for i in range(n_genes)]

    def test_add_ratio_layer_sparse_default(self):
        # Default (fill_nan_with_mean=False) → sparse output, same sparsity as input
        add_ratio_layer(
            self.adata,
            layer="counts",
            group_iso_by="gene_symbol",
            ratio_layer_key="ratios",
        )

        self.assertIn("ratios", self.adata.layers)
        r = self.adata.layers["ratios"]
        self.assertTrue(scipy.sparse.issparse(r))
        # Same nnz as the dense input viewed as sparse
        expected_nnz = scipy.sparse.csr_matrix(self.counts_dense).nnz
        self.assertEqual(r.nnz, expected_nnz)

        # At each non-zero entry, ratios within a gene group must sum to 1 per spot
        r_dense = r.toarray()
        var = self.adata.var
        for gene in var["gene_symbol"].unique():
            iso_idx = np.where(var["gene_symbol"] == gene)[0]
            gene_counts = self.counts_dense[:, iso_idx]
            gene_ratios = r_dense[:, iso_idx]
            expressed = gene_counts.sum(axis=1) > 0
            np.testing.assert_allclose(
                gene_ratios[expressed].sum(axis=1),
                np.ones(expressed.sum()),
                atol=1e-5,
                err_msg=f"Ratios for gene '{gene}' do not sum to 1 at expressed spots",
            )

    def test_add_ratio_layer_dense_fill_mean(self):
        # fill_nan_with_mean=True → dense output, NaN spots filled with column mean
        add_ratio_layer(
            self.adata,
            layer="counts",
            group_iso_by="gene_symbol",
            ratio_layer_key="ratios",
            fill_nan_with_mean=True,
        )

        self.assertIn("ratios", self.adata.layers)
        ratios = self.adata.layers["ratios"]
        self.assertEqual(ratios.shape, self.counts_dense.shape)
        self.assertFalse(scipy.sparse.issparse(ratios))
        self.assertEqual(ratios.dtype, np.float32)
        self.assertFalse(np.isnan(ratios).any(), "Dense output should have no NaNs")

        # At expressed spots, ratios per gene still sum to 1
        var = self.adata.var
        for gene in var["gene_symbol"].unique():
            iso_idx = np.where(var["gene_symbol"] == gene)[0]
            gene_counts = self.counts_dense[:, iso_idx]
            gene_ratios = ratios[:, iso_idx]
            expressed = gene_counts.sum(axis=1) > 0
            np.testing.assert_allclose(
                gene_ratios[expressed].sum(axis=1),
                np.ones(expressed.sum()),
                atol=1e-5,
            )

    def test_add_ratio_layer_sparse_input_sparse_output(self):
        # Sparse count input → sparse ratio output (default)
        counts_sparse = scipy.sparse.csr_matrix(self.counts_dense)
        self.adata.layers["counts_sparse"] = counts_sparse

        add_ratio_layer(
            self.adata,
            layer="counts_sparse",
            group_iso_by="gene_symbol",
            ratio_layer_key="ratios_sp",
        )

        r = self.adata.layers["ratios_sp"]
        self.assertTrue(scipy.sparse.issparse(r))
        self.assertEqual(r.nnz, counts_sparse.nnz)

        # Values should match dense-input sparse-output path
        add_ratio_layer(
            self.adata,
            layer="counts",
            group_iso_by="gene_symbol",
            ratio_layer_key="ratios_ref",
        )
        np.testing.assert_allclose(
            r.toarray(), self.adata.layers["ratios_ref"].toarray(), atol=1e-5
        )

    def test_add_ratio_layer_same_key_raises(self):
        with self.assertRaises(ValueError):
            add_ratio_layer(
                self.adata,
                layer="counts",
                group_iso_by="gene_symbol",
                ratio_layer_key="counts",
            )

    def test_extract_gene_level_statistics_sparse_vs_dense(self):
        # Dense results
        stats_dense = extract_gene_level_statistics(self.adata, layer="counts")

        # Sparse results
        counts_sparse = scipy.sparse.csr_matrix(self.counts_dense)
        self.adata.layers["counts_sparse"] = counts_sparse
        stats_sparse = extract_gene_level_statistics(self.adata, layer="counts_sparse")

        # Compare columns
        cols_to_compare = [
            "n_iso",
            "pct_spot_on",
            "count_avg",
            "count_std",
            "perplexity",
            "major_ratio_avg",
        ]

        # Sort by index to ensure alignment
        stats_dense = stats_dense.sort_index()
        stats_sparse = stats_sparse.sort_index()

        for col in cols_to_compare:
            np.testing.assert_allclose(
                stats_sparse[col].values,
                stats_dense[col].values,
                rtol=1e-5,
                err_msg=f"Mismatch in column {col}",
            )

    def test_run_sparkx_error_on_sparse(self):
        n_spots = 20
        n_genes = 5
        counts_dense = np.random.rand(n_spots, n_genes)
        coords = np.random.rand(n_spots, 2)

        # SciPy sparse
        counts_csr = scipy.sparse.csr_matrix(counts_dense)
        with self.assertRaisesRegex(ValueError, "does not support sparse input"):
            run_sparkx(counts_csr, coords)

        # Torch sparse
        counts_torch = torch.from_numpy(counts_dense).to_sparse()
        with self.assertRaisesRegex(ValueError, "does not support sparse input"):
            run_sparkx(counts_torch, coords)

    def test_run_hsic_gc_formats(self):
        """All input formats produce identical statistics and valid result keys."""
        np.random.seed(0)
        n_spots, n_genes = 50, 4
        counts_np = np.random.randint(0, 5, size=(n_spots, n_genes)).astype(np.float32)
        coords_np = np.random.rand(n_spots, 2).astype(np.float32)

        res_np = run_hsic_gc(counts_np, coords_np)

        # Result structure
        for key in (
            "statistic",
            "pvalue",
            "pvalue_adj",
            "method",
            "null_method",
            "n_spots",
        ):
            self.assertIn(key, res_np)
        self.assertEqual(res_np["method"], "hsic-gc")
        self.assertEqual(res_np["n_spots"], n_spots)
        self.assertEqual(len(res_np["statistic"]), n_genes)
        self.assertTrue(np.all(res_np["pvalue"] >= 0) and np.all(res_np["pvalue"] <= 1))

        # Dense torch
        res_torch = run_hsic_gc(
            torch.from_numpy(counts_np), torch.from_numpy(coords_np)
        )
        np.testing.assert_allclose(
            res_np["statistic"], res_torch["statistic"], rtol=1e-4
        )
        np.testing.assert_allclose(res_np["pvalue"], res_torch["pvalue"], rtol=1e-4)

        # Scipy sparse (CSR)
        res_csr = run_hsic_gc(scipy.sparse.csr_matrix(counts_np), coords_np)
        np.testing.assert_allclose(res_np["statistic"], res_csr["statistic"], rtol=1e-4)

        # Torch sparse
        res_torch_sparse = run_hsic_gc(
            torch.from_numpy(counts_np).to_sparse(), coords_np
        )
        np.testing.assert_allclose(
            res_np["statistic"], res_torch_sparse["statistic"], rtol=1e-4
        )

    def test_run_hsic_gc_anndata_mode(self):
        """AnnData mode loads adata.X / adata.layers[layer] as gene-level counts."""
        np.random.seed(1)
        n_spots, n_genes = 40, 5
        gene_counts = np.random.randint(0, 5, size=(n_spots, n_genes)).astype(
            np.float32
        )
        adata = AnnData(
            X=gene_counts,
            var=pd.DataFrame(index=[f"gene_{k}" for k in range(n_genes)]),
        )
        adata.layers["raw"] = gene_counts
        adata.obsm["spatial"] = np.random.rand(n_spots, 2).astype(np.float32)

        # Default: uses adata.X
        res = run_hsic_gc(adata=adata)
        self.assertEqual(res["method"], "hsic-gc")
        self.assertEqual(res["n_spots"], n_spots)
        self.assertEqual(len(res["statistic"]), n_genes)
        self.assertTrue(np.all(res["pvalue"] >= 0) and np.all(res["pvalue"] <= 1))
        for key in (
            "statistic",
            "pvalue",
            "pvalue_adj",
            "method",
            "null_method",
            "n_spots",
        ):
            self.assertIn(key, res)

        # layer= selects adata.layers[layer]; results should match (same data)
        res_layer = run_hsic_gc(adata=adata, layer="raw")
        np.testing.assert_allclose(res["statistic"], res_layer["statistic"], rtol=1e-5)

        # Sparse adata.X also works
        adata_sp = AnnData(
            X=scipy.sparse.csr_matrix(gene_counts),
            var=adata.var.copy(),
        )
        adata_sp.obsm["spatial"] = adata.obsm["spatial"]
        res_sp = run_hsic_gc(adata=adata_sp)
        np.testing.assert_allclose(res["statistic"], res_sp["statistic"], rtol=1e-4)

        # Mutually exclusive with matrix args
        with self.assertRaises(ValueError):
            run_hsic_gc(adata=adata, counts_gene=gene_counts)

    def test_run_hsic_gc_null_methods(self):
        """Both null methods return valid p-values; p-value rankings agree (high Spearman r)."""
        np.random.seed(2)
        n_spots, n_genes = 50, 30
        counts_np = np.random.randint(0, 5, size=(n_spots, n_genes)).astype(np.float32)
        coords_np = np.random.rand(n_spots, 2).astype(np.float32)

        res_eig = run_hsic_gc(counts_np, coords_np, null_method="eig")
        res_clt = run_hsic_gc(counts_np, coords_np, null_method="clt")

        for res, nm in [(res_eig, "eig"), (res_clt, "clt")]:
            self.assertEqual(res["null_method"], nm)
            self.assertEqual(len(res["pvalue"]), n_genes)
            self.assertTrue(np.all(res["pvalue"] >= 0) and np.all(res["pvalue"] <= 1))
            self.assertIn("pvalue_adj", res)
            self.assertTrue(np.all(res["pvalue_adj"] >= 0))

        # Statistics must be identical (same kernel, same counts; only null differs)
        np.testing.assert_allclose(
            res_eig["statistic"], res_clt["statistic"], rtol=1e-5
        )

        # P-value rankings should agree between methods
        rho, _ = scipy.stats.spearmanr(res_eig["pvalue"], res_clt["pvalue"])
        self.assertGreater(
            rho, 0.9, f"Spearman r of p-values between eig and clt was only {rho:.3f}"
        )

        # Invalid null method
        with self.assertRaises(ValueError):
            run_hsic_gc(counts_np, coords_np, null_method="bad")

    def test_counts_to_ratios_importerror_fallback(self):
        counts = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        orig_import = __import__

        def _import_hook(name, *args, **kwargs):
            if name == "skbio.stats.composition":
                raise ImportError("mocked missing scikit-bio")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_hook):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                y = counts_to_ratios(counts, transformation="clr", nan_filling="mean")

        self.assertEqual(y.shape, (2, 2))
        self.assertTrue(any("Please install scikit-bio" in str(w.message) for w in rec))

    def test_counts_to_ratios_mocked_clr_ilr_alr_and_nan(self):
        comp_mod = types.ModuleType("skbio.stats.composition")

        def _clr(x):
            arr = np.asarray(x, dtype=np.float64)
            return np.log(arr + 1e-8) - np.log(arr + 1e-8).mean(axis=1, keepdims=True)

        def _ilr(x):
            arr = np.asarray(x, dtype=np.float64)
            return np.log(arr[:, :-1] + 1e-8) - np.log(arr[:, -1:] + 1e-8)

        def _alr(x):
            arr = np.asarray(x, dtype=np.float64)
            return np.log(arr[:, :-1] + 1e-8) - np.log(arr[:, -1:] + 1e-8)

        comp_mod.clr = _clr
        comp_mod.ilr = _ilr
        comp_mod.alr = _alr

        counts = np.array([[1.0, 3.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        with patch.dict(
            sys.modules,
            {
                "skbio": types.ModuleType("skbio"),
                "skbio.stats": types.ModuleType("skbio.stats"),
                "skbio.stats.composition": comp_mod,
            },
        ):
            y_clr = counts_to_ratios(counts, transformation="clr", nan_filling="none")
            y_ilr = counts_to_ratios(counts, transformation="ilr", nan_filling="none")
            y_alr = counts_to_ratios(counts, transformation="alr", nan_filling="none")

        self.assertEqual(y_clr.shape, (2, 3))
        self.assertEqual(y_ilr.shape, (2, 2))
        self.assertEqual(y_alr.shape, (2, 2))
        self.assertTrue(torch.isnan(y_clr[1]).all())

    def test_false_discovery_control_validation_branches(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out_nan = false_discovery_control(np.array([0.1, np.nan, 0.5]))
        self.assertEqual(out_nan.shape, (3,))
        self.assertTrue(any("NaNs encountered" in str(w.message) for w in rec))

        with self.assertRaises(ValueError):
            false_discovery_control(np.array([1.2, 0.3]))

        with self.assertRaises(ValueError):
            false_discovery_control(np.array([0.1, 0.2]), method="invalid")

        out_none_axis = false_discovery_control(
            np.array([[0.1, 0.2], [0.3, 0.4]]), axis=None
        )
        self.assertEqual(out_none_axis.shape, (4,))

        with self.assertRaises(ValueError):
            false_discovery_control(np.array([0.1, 0.2]), axis="bad")

        out_single = false_discovery_control(np.array([0.3]))
        self.assertEqual(np.asarray(out_single).size, 1)
        self.assertAlmostEqual(float(np.asarray(out_single).reshape(-1)[0]), 0.3)

        out_by = false_discovery_control(np.array([0.01, 0.02, 0.2]), method="by")
        self.assertEqual(out_by.shape, (3,))

    def test_load_visium_sp_meta_missing_images(self):
        adata = AnnData(
            X=np.zeros((2, 2), dtype=np.float32),
            obs=pd.DataFrame(index=["spot1", "spot2"]),
            var=pd.DataFrame(index=["i1", "i2"]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sp = os.path.join(tmpdir, "spatial")
            os.makedirs(sp, exist_ok=True)

            with open(
                os.path.join(sp, "scalefactors_json.json"), "w", encoding="utf-8"
            ) as f:
                json.dump({"tissue_hires_scalef": 1.0}, f)

            positions = pd.DataFrame(
                {
                    "barcode": ["spot1", "spot2"],
                    "in_tissue": [1, 1],
                    "array_row": [0, 1],
                    "array_col": [0, 1],
                    "pxl_col_in_fullres": [10, 20],
                    "pxl_row_in_fullres": [30, 40],
                }
            )
            positions.to_csv(os.path.join(sp, "tissue_positions.csv"), index=False)

            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                out = load_visium_sp_meta(adata, sp)

        self.assertIn("spatial", out.uns)
        self.assertEqual(out.obsm["spatial"].shape, (2, 2))
        self.assertNotIn("pxl_row_in_fullres", out.obs.columns)
        self.assertTrue(any("Missing 'hires' image" in str(w.message) for w in rec))
        self.assertTrue(any("Missing 'lowres' image" in str(w.message) for w in rec))

    def test_add_ratio_layer_csc_sparse(self):
        # CSC sparse input (non-CSR) → sparse output with same nnz
        counts_csc = scipy.sparse.csc_matrix(self.counts_dense)
        self.adata.layers["counts_csc"] = counts_csc

        add_ratio_layer(
            self.adata,
            layer="counts_csc",
            group_iso_by="gene_symbol",
            ratio_layer_key="ratios_csc",
        )

        self.assertIn("ratios_csc", self.adata.layers)
        ratios = self.adata.layers["ratios_csc"]
        self.assertTrue(scipy.sparse.issparse(ratios))
        self.assertEqual(ratios.nnz, counts_csc.nnz)

        # Should match the dense-input sparse-output path
        add_ratio_layer(
            self.adata,
            layer="counts",
            group_iso_by="gene_symbol",
            ratio_layer_key="ratios_ref",
        )
        np.testing.assert_allclose(
            ratios.toarray(), self.adata.layers["ratios_ref"].toarray(), atol=1e-5
        )

    def test_extract_gene_level_statistics_sparse_lil_and_zero_gene(self):
        # LIL is converted to CSR for AnnData compatibility
        counts_csr = scipy.sparse.lil_matrix(self.counts_dense).tocsr()
        adata_sp = self.adata.copy()
        adata_sp.layers["counts_sp"] = counts_csr
        stats = extract_gene_level_statistics(adata_sp, layer="counts_sp")
        self.assertGreater(stats.shape[0], 0)

        zero_counts = np.zeros((5, 2), dtype=np.float32)
        adata_zero = AnnData(
            X=zero_counts,
            var=pd.DataFrame({"gene_symbol": ["G0", "G0"]}, index=["i1", "i2"]),
        )
        adata_zero.layers["counts"] = zero_counts
        stats_zero = extract_gene_level_statistics(adata_zero, layer="counts")
        self.assertAlmostEqual(float(stats_zero.iloc[0]["major_ratio_avg"]), 0.0)

    def test_run_hsic_gc_branch_coverage_with_mocked_kernel(self):
        class FakeKernel:
            def __init__(self, coords=None, adj_matrix=None, **kwargs):
                self.Q = None

            def eigenvalues(self, k=None):
                return torch.tensor([1.0], dtype=torch.float32)

            def xtKx(self, counts):
                return counts.T @ counts

            def xtKx_exact(self, counts):
                return counts.T @ counts

        with patch("splisosm.kernel.SpatialCovKernel", FakeKernel):
            # n_spots > 5000 branch and centering warning branch
            counts_large = np.ones((5001, 1), dtype=np.float32)
            coords_large = np.random.rand(5001, 2).astype(np.float32)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                res_large = run_hsic_gc(
                    counts_large,
                    coords_large,
                    null_configs={"approx_rank": 100000},
                    centering=False,
                )
            self.assertIn("statistic", res_large)
            self.assertTrue(any("centering" in str(w.message) for w in rec))

            # approx_rank >= n_spots should be reset to None
            counts_small = np.ones((10, 1), dtype=np.float32)
            coords_small = np.random.rand(10, 2).astype(np.float32)
            res_small = run_hsic_gc(
                counts_small, coords_small, null_configs={"approx_rank": 100}
            )
            self.assertIn("pvalue", res_small)

            # scipy sparse branch with conversion to csc/csr
            counts_lil = scipy.sparse.lil_matrix(
                np.random.randint(0, 3, size=(20, 2)).astype(np.float32)
            )
            res_sparse = run_hsic_gc(
                counts_lil, np.random.rand(20, 2).astype(np.float32)
            )
            self.assertEqual(res_sparse["method"], "hsic-gc")

            # torch sparse dtype cast and coalesce branches
            idx = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.long)
            val = torch.tensor([1, 2, 3], dtype=torch.int64)
            counts_t_sparse = torch.sparse_coo_tensor(idx, val, size=(5, 2))
            self.assertFalse(counts_t_sparse.is_coalesced())
            res_t_sparse = run_hsic_gc(
                counts_t_sparse, np.random.rand(5, 2).astype(np.float32)
            )
            self.assertEqual(res_t_sparse["method"], "hsic-gc")

    def test_run_hsic_gc_min_component_size(self):
        """Spots in small components are removed when min_component_size > 1."""
        np.random.seed(42)
        # 8 tightly-packed connected spots on a 0.25 grid
        grid = np.array(
            [[i * 0.25, j * 0.25] for i in range(4) for j in range(2)],
            dtype=np.float32,
        )
        # 2 isolated spots far away (will form their own 2-spot component)
        isolated = np.array([[100.0, 100.0], [100.0, 100.01]], dtype=np.float32)
        coords = np.vstack([grid, isolated])  # (10, 2)
        counts = np.random.randint(1, 5, size=(10, 3)).astype(np.float32)

        # default (min_component_size=1) keeps all 10 spots
        res_default = run_hsic_gc(counts, coords)
        self.assertEqual(res_default["n_spots"], 10)
        self.assertEqual(len(res_default["statistic"]), 3)

        # min_component_size=3 should drop the 2 isolated spots
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_filtered = run_hsic_gc(counts, coords, min_component_size=3)

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(len(user_warns) > 0, "Expected a UserWarning for removed spots")
        warn_msg = str(user_warns[0].message)
        self.assertIn("2", warn_msg)  # removed count
        self.assertIn("8", warn_msg)  # remaining count
        self.assertEqual(res_filtered["n_spots"], 8)
        self.assertEqual(len(res_filtered["statistic"]), 3)  # n_genes unchanged
        self.assertEqual(len(res_filtered["pvalue"]), 3)
        self.assertTrue(np.all(res_filtered["pvalue"] >= 0))
        self.assertTrue(np.all(res_filtered["pvalue"] <= 1))

        # AnnData mode: same filtering behaviour
        gene_counts_fc = np.random.randint(1, 5, size=(10, 3)).astype(np.float32)
        adata_fc = AnnData(
            X=gene_counts_fc,
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(3)]),
        )
        adata_fc.obsm["spatial"] = coords

        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            res_adata = run_hsic_gc(adata=adata_fc, min_component_size=3)

        uw2 = [w for w in caught2 if issubclass(w.category, UserWarning)]
        self.assertTrue(len(uw2) > 0, "Expected UserWarning in AnnData mode")
        self.assertEqual(res_adata["n_spots"], 8)

    def test_run_sparkx_torch_conversion_with_mocked_rpy2(self):
        class DummyCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class DummyConverter:
            def __add__(self, other):
                return self

            def context(self):
                return DummyCtx()

        class DummyRObj:
            def __init__(self, arr):
                self.arr = np.asarray(arr)
                self.colnames = None

        class DummyConvApi:
            def py2rpy(self, x):
                return DummyRObj(x)

            def rpy2py(self, x):
                return x

        class DummyConv:
            def get_conversion(self):
                return DummyConvApi()

        class DummyR:
            def __getitem__(self, key):
                if key == "rownames":
                    return lambda obj: [f"r{i}" for i in range(obj.arr.shape[0])]
                raise KeyError(key)

        class DummySparkRes:
            def rx2(self, key):
                if key == "stats":
                    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                if key == "res_mtest":
                    return {
                        "combinedPval": np.array([0.05, 0.2], dtype=np.float32),
                        "adjustedPval": np.array([0.1, 0.25], dtype=np.float32),
                    }
                raise KeyError(key)

        class DummySpark:
            def sparkx(self, counts_r, coords_r):
                return DummySparkRes()

        ro_mod = types.ModuleType("rpy2.robjects")
        ro_mod.default_converter = DummyConverter()
        ro_mod.conversion = DummyConv()
        ro_mod.r = DummyR()
        ro_mod.vectors = types.SimpleNamespace(StrVector=lambda xs: list(xs))

        numpy2ri_mod = types.ModuleType("rpy2.robjects.numpy2ri")
        numpy2ri_mod.converter = DummyConverter()
        ro_mod.numpy2ri = numpy2ri_mod

        packages_mod = types.ModuleType("rpy2.robjects.packages")
        packages_mod.importr = lambda name: DummySpark()

        with patch.dict(
            sys.modules,
            {
                "rpy2": types.ModuleType("rpy2"),
                "rpy2.robjects": ro_mod,
                "rpy2.robjects.numpy2ri": numpy2ri_mod,
                "rpy2.robjects.packages": packages_mod,
            },
        ):
            counts = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
            out = run_sparkx(counts, coords)

        self.assertIn("method", out)
        self.assertEqual(out["method"], "spark-x")
        self.assertEqual(len(out["pvalue"]), 2)


class TestIndexRowsSparseCoo(unittest.TestCase):
    def test_correctness(self):
        """_index_rows_sparse_coo must match dense row indexing without densifying."""
        torch.manual_seed(0)
        n, m = 20, 5
        mask = torch.rand(n, m) < 0.3
        dense = torch.randn(n, m) * mask.float()
        sparse = dense.to_sparse()
        keep_indices = np.array([0, 3, 7, 11, 15, 19], dtype=np.int64)
        result = _index_rows_sparse_coo(sparse, keep_indices)
        self.assertEqual(result.shape, torch.Size([len(keep_indices), m]))
        torch.testing.assert_close(result.to_dense(), dense[keep_indices])

    def test_uncoalesced(self):
        """Helper must handle uncoalesced (duplicate-index) COO tensors."""
        indices = torch.tensor([[0, 0, 1, 2], [0, 0, 1, 2]], dtype=torch.long)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t = torch.sparse_coo_tensor(indices, values, size=(4, 3))
        self.assertFalse(t.is_coalesced())
        keep_indices = np.array([0, 2], dtype=np.int64)
        result = _index_rows_sparse_coo(t, keep_indices)
        torch.testing.assert_close(
            result.to_dense(), t.coalesce().to_dense()[keep_indices]
        )

    def test_empty_keep(self):
        """Keeping zero rows yields an empty sparse tensor."""
        t = torch.eye(5).to_sparse()
        result = _index_rows_sparse_coo(t, np.array([], dtype=np.int64))
        self.assertEqual(result.shape, torch.Size([0, 5]))
        self.assertEqual(result.to_dense().numel(), 0)


class TestRunHsicGc(unittest.TestCase):
    """Comprehensive tests for run_hsic_gc — matrix mode and AnnData mode."""

    # ── shared fixtures ────────────────────────────────────────────────────────

    @classmethod
    def _make_matrix_inputs(cls, n_spots=50, n_genes=6, seed=0):
        rng = np.random.default_rng(seed)
        counts = rng.integers(0, 8, size=(n_spots, n_genes)).astype(np.float32)
        coords = rng.random((n_spots, 2)).astype(np.float32)
        return counts, coords

    @classmethod
    def _make_adata(cls, n_spots=50, n_genes=6, seed=1, sparse=False):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 8, size=(n_spots, n_genes)).astype(np.float32)
        coords = rng.random((n_spots, 2)).astype(np.float32)
        if sparse:
            X_store = scipy.sparse.csr_matrix(X)
        else:
            X_store = X
        adata = AnnData(
            X=X_store,
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
        )
        adata.layers["raw"] = X_store
        adata.obsm["spatial"] = coords
        return adata, X, coords

    # ── matrix mode ───────────────────────────────────────────────────────────

    def test_matrix_mode_numpy_basic(self):
        """Matrix mode with numpy arrays returns correct result structure."""
        counts, coords = self._make_matrix_inputs()
        res = run_hsic_gc(counts, coords)
        for key in (
            "statistic",
            "pvalue",
            "pvalue_adj",
            "method",
            "null_method",
            "n_spots",
        ):
            self.assertIn(key, res)
        self.assertEqual(res["method"], "hsic-gc")
        self.assertEqual(res["null_method"], "eig")
        self.assertEqual(res["n_spots"], counts.shape[0])
        self.assertEqual(len(res["statistic"]), counts.shape[1])
        self.assertTrue(np.all(res["pvalue"] >= 0))
        self.assertTrue(np.all(res["pvalue"] <= 1))
        self.assertTrue(np.all(res["pvalue_adj"] >= 0))
        self.assertTrue(np.all(res["pvalue_adj"] <= 1))

    def test_matrix_mode_torch_dense(self):
        """Matrix mode with torch dense tensor matches numpy result."""
        counts, coords = self._make_matrix_inputs()
        res_np = run_hsic_gc(counts, coords)
        res_t = run_hsic_gc(torch.from_numpy(counts), torch.from_numpy(coords))
        np.testing.assert_allclose(res_np["statistic"], res_t["statistic"], rtol=1e-4)
        np.testing.assert_allclose(res_np["pvalue"], res_t["pvalue"], rtol=1e-4)

    def test_matrix_mode_scipy_sparse(self):
        """Matrix mode with scipy sparse (CSR and CSC) matches numpy result."""
        counts, coords = self._make_matrix_inputs()
        res_np = run_hsic_gc(counts, coords)
        for fmt in (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix):
            res = run_hsic_gc(fmt(counts), coords)
            np.testing.assert_allclose(res_np["statistic"], res["statistic"], rtol=1e-4)

    def test_matrix_mode_torch_sparse(self):
        """Matrix mode with torch sparse COO matches numpy result."""
        counts, coords = self._make_matrix_inputs()
        res_np = run_hsic_gc(counts, coords)
        res_sp = run_hsic_gc(torch.from_numpy(counts).to_sparse(), coords)
        np.testing.assert_allclose(res_np["statistic"], res_sp["statistic"], rtol=1e-4)

    def test_matrix_mode_null_method_clt(self):
        """'clt' null method returns valid p-values and same statistic as 'eig'."""
        counts, coords = self._make_matrix_inputs()
        res_eig = run_hsic_gc(counts, coords, null_method="eig")
        res_clt = run_hsic_gc(counts, coords, null_method="clt")
        self.assertEqual(res_clt["null_method"], "clt")
        np.testing.assert_allclose(
            res_eig["statistic"], res_clt["statistic"], rtol=1e-5
        )
        self.assertTrue(np.all(res_clt["pvalue"] >= 0))
        self.assertTrue(np.all(res_clt["pvalue"] <= 1))

    def test_matrix_mode_null_method_trace_alias_deprecated(self):
        """'trace' is accepted as a deprecated alias for 'clt'."""
        counts, coords = self._make_matrix_inputs()
        res_clt = run_hsic_gc(counts, coords, null_method="clt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_trace = run_hsic_gc(counts, coords, null_method="trace")
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning) and "trace" in str(w.message)
        ]
        self.assertTrue(len(dep) >= 1, "expected DeprecationWarning for 'trace'")
        self.assertEqual(res_trace["null_method"], "clt")
        np.testing.assert_allclose(res_trace["statistic"], res_clt["statistic"])
        np.testing.assert_allclose(res_trace["pvalue"], res_clt["pvalue"])

    def test_matrix_mode_invalid_null_method(self):
        """Invalid null_method raises ValueError."""
        counts, coords = self._make_matrix_inputs()
        with self.assertRaises(ValueError):
            run_hsic_gc(counts, coords, null_method="bogus")

    def test_matrix_mode_missing_both_args(self):
        """Calling with neither adata nor counts_gene raises ValueError."""
        with self.assertRaises(ValueError):
            run_hsic_gc(counts_gene=None, coordinates=None)

    def test_matrix_mode_missing_coordinates(self):
        """Calling with counts_gene but no coordinates raises ValueError."""
        counts, _ = self._make_matrix_inputs()
        with self.assertRaises(ValueError):
            run_hsic_gc(counts_gene=counts)

    def test_matrix_mode_min_component_size_removes_isolated_spots(self):
        """min_component_size > 1 removes spots in small components (matrix mode)."""
        rng = np.random.default_rng(7)
        # 8 spots on a tight 0.1-spaced grid
        grid = np.array(
            [[i * 0.1, j * 0.1] for i in range(4) for j in range(2)], dtype=np.float32
        )
        # 2 isolated spots far away
        isolated = np.array([[50.0, 50.0], [50.0, 50.01]], dtype=np.float32)
        coords = np.vstack([grid, isolated])
        counts = rng.integers(1, 5, size=(10, 4)).astype(np.float32)

        # Default keeps all spots
        res_default = run_hsic_gc(counts, coords)
        self.assertEqual(res_default["n_spots"], 10)

        # min_component_size=3 drops the 2-spot island
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_filt = run_hsic_gc(counts, coords, min_component_size=3)
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(len(user_warns) > 0)
        self.assertIn("2", str(user_warns[0].message))
        self.assertEqual(res_filt["n_spots"], 8)
        self.assertEqual(len(res_filt["statistic"]), 4)

    def test_matrix_mode_statistic_is_nonneg(self):
        """All HSIC statistics are non-negative."""
        counts, coords = self._make_matrix_inputs(n_genes=10)
        res = run_hsic_gc(counts, coords)
        self.assertTrue(np.all(res["statistic"] >= 0))

    # ── AnnData mode ──────────────────────────────────────────────────────────

    def test_anndata_mode_dense_X(self):
        """AnnData with dense adata.X returns correct gene-level results."""
        adata, X, _ = self._make_adata()
        res = run_hsic_gc(adata=adata)
        for key in (
            "statistic",
            "pvalue",
            "pvalue_adj",
            "method",
            "null_method",
            "n_spots",
        ):
            self.assertIn(key, res)
        self.assertEqual(res["method"], "hsic-gc")
        self.assertEqual(res["n_spots"], X.shape[0])
        self.assertEqual(len(res["statistic"]), X.shape[1])
        self.assertTrue(np.all(res["pvalue"] >= 0))
        self.assertTrue(np.all(res["pvalue"] <= 1))

    def test_anndata_mode_sparse_X(self):
        """AnnData with sparse adata.X matches dense result."""
        adata_dense, _, _ = self._make_adata(sparse=False)
        adata_sparse, _, _ = self._make_adata(sparse=True)
        res_dense = run_hsic_gc(adata=adata_dense)
        res_sparse = run_hsic_gc(adata=adata_sparse)
        np.testing.assert_allclose(
            res_dense["statistic"], res_sparse["statistic"], rtol=1e-4
        )

    def test_anndata_mode_layer_selects_layer(self):
        """layer= selects adata.layers[layer] (same data → same result as adata.X)."""
        adata, _, _ = self._make_adata()
        res_X = run_hsic_gc(adata=adata)
        res_layer = run_hsic_gc(adata=adata, layer="raw")
        np.testing.assert_allclose(
            res_X["statistic"], res_layer["statistic"], rtol=1e-5
        )

    def test_anndata_mode_matches_matrix_mode(self):
        """AnnData mode produces the same statistic as equivalent matrix-mode call."""
        adata, X, coords = self._make_adata()
        res_adata = run_hsic_gc(adata=adata)
        res_matrix = run_hsic_gc(X, coords)
        np.testing.assert_allclose(
            res_adata["statistic"], res_matrix["statistic"], rtol=1e-4
        )

    def test_anndata_mode_min_counts_filter(self):
        """min_counts filters out low-expression genes."""
        rng = np.random.default_rng(3)
        n_spots, n_genes = 40, 8
        X = rng.integers(0, 4, size=(n_spots, n_genes)).astype(np.float32)
        # Force first gene to all zeros → total count = 0
        X[:, 0] = 0.0
        adata = AnnData(X=X, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
        adata.obsm["spatial"] = rng.random((n_spots, 2)).astype(np.float32)
        res = run_hsic_gc(adata=adata, min_counts=1)
        # Gene 0 (all-zero) should be filtered out
        self.assertLess(len(res["statistic"]), n_genes)

    def test_anndata_mode_min_bin_pct_filter(self):
        """min_bin_pct filters out genes expressed in too few spots."""
        rng = np.random.default_rng(4)
        n_spots, n_genes = 40, 8
        X = rng.integers(0, 4, size=(n_spots, n_genes)).astype(np.float32)
        # Force gene 0 to be non-zero in only 1 spot
        X[:, 0] = 0.0
        X[0, 0] = 5.0
        adata = AnnData(X=X, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
        adata.obsm["spatial"] = rng.random((n_spots, 2)).astype(np.float32)
        # gene 0 expressed in 1/40 = 0.025 of spots → filtered when pct > 0.05
        res = run_hsic_gc(adata=adata, min_bin_pct=0.05)
        self.assertLess(len(res["statistic"]), n_genes)

    def test_anndata_mode_min_component_size(self):
        """AnnData mode removes spots in small components when min_component_size > 1."""
        rng = np.random.default_rng(5)
        grid = np.array(
            [[i * 0.1, j * 0.1] for i in range(4) for j in range(2)], dtype=np.float32
        )
        isolated = np.array([[50.0, 50.0], [50.0, 50.01]], dtype=np.float32)
        coords = np.vstack([grid, isolated])
        X = rng.integers(1, 5, size=(10, 3)).astype(np.float32)
        adata = AnnData(X=X, var=pd.DataFrame(index=[f"g{i}" for i in range(3)]))
        adata.obsm["spatial"] = coords

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = run_hsic_gc(adata=adata, min_component_size=3)
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(len(user_warns) > 0)
        self.assertEqual(res["n_spots"], 8)
        self.assertEqual(len(res["statistic"]), 3)

    def test_anndata_mode_mutex_with_matrix_args(self):
        """Providing adata together with counts_gene or coordinates raises ValueError."""
        adata, X, coords = self._make_adata()
        with self.assertRaises(ValueError):
            run_hsic_gc(adata=adata, counts_gene=X)
        with self.assertRaises(ValueError):
            run_hsic_gc(adata=adata, counts_gene=X, coordinates=coords)

    def test_anndata_mode_adj_key(self):
        """adj_key loads a pre-built adjacency from adata.obsp."""
        from splisosm.kernel import _build_adj_from_coords

        adata, X, coords = self._make_adata()
        coords_t = torch.from_numpy(coords)
        adj = _build_adj_from_coords(coords_t, k_neighbors=4, mutual_neighbors=True)
        adata.obsp["spatial_adj"] = adj

        res_adj = run_hsic_gc(adata=adata, adj_key="spatial_adj")
        # Should complete without error and return correct n_genes
        self.assertEqual(len(res_adj["statistic"]), X.shape[1])
        self.assertTrue(np.all(res_adj["pvalue"] >= 0))

    def test_anndata_mode_adj_key_no_spatial_key(self):
        """adj_key alone is sufficient; obsm[spatial_key] is optional."""
        from splisosm.kernel import _build_adj_from_coords

        adata, X, coords = self._make_adata()
        coords_t = torch.from_numpy(coords)
        adj = _build_adj_from_coords(coords_t, k_neighbors=4, mutual_neighbors=True)
        adata.obsp["spatial_adj"] = adj
        # Strip coordinates to simulate a non-spatial AnnData.
        del adata.obsm["spatial"]
        self.assertNotIn("spatial", adata.obsm)

        res_adj = run_hsic_gc(adata=adata, adj_key="spatial_adj")
        self.assertEqual(len(res_adj["statistic"]), X.shape[1])
        self.assertTrue(np.all(res_adj["pvalue"] >= 0))

    def test_anndata_mode_no_spatial_no_adj_raises(self):
        """Missing both spatial_key and adj_key raises a clear error."""
        adata, _, _ = self._make_adata()
        del adata.obsm["spatial"]
        with self.assertRaises(ValueError) as ctx:
            run_hsic_gc(adata=adata)
        msg = str(ctx.exception)
        self.assertIn("spatial", msg)
        self.assertIn("adj_key", msg)


if __name__ == "__main__":
    unittest.main()
