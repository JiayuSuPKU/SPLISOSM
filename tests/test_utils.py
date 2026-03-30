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
import torch
from anndata import AnnData
from unittest.mock import patch
from splisosm.utils import (
    counts_to_ratios,
    false_discovery_control,
    load_visium_sp_meta,
    extract_counts_n_ratios,
    extract_gene_level_statistics,
    run_hsic_gc,
    run_sparkx,
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

    def test_extract_counts_n_ratios_dense(self):
        counts_list, ratios_list, gene_names, ratios_obs = extract_counts_n_ratios(
            self.adata, layer="counts", group_iso_by="gene_symbol", return_sparse=False
        )

        self.assertEqual(len(counts_list), 2)
        self.assertEqual(len(ratios_list), 2)
        self.assertIsInstance(counts_list[0], torch.Tensor)
        self.assertFalse(counts_list[0].is_sparse)
        self.assertIsInstance(ratios_list[0], torch.Tensor)
        self.assertIsNotNone(ratios_obs)

    def test_extract_counts_n_ratios_sparse_scipy(self):
        # Convert layer to sparse
        counts_sparse = scipy.sparse.csr_matrix(self.counts_dense)
        self.adata.layers["counts_sparse"] = counts_sparse

        counts_list, ratios_list, gene_names, ratios_obs = extract_counts_n_ratios(
            self.adata,
            layer="counts_sparse",
            group_iso_by="gene_symbol",
            return_sparse=True,
        )

        self.assertEqual(len(counts_list), 2)
        self.assertEqual(len(ratios_list), 0)
        self.assertIsNone(ratios_obs)

        for i, counts in enumerate(counts_list):
            self.assertTrue(counts.is_sparse)
            # Convert to dense and compare with original
            gene_name = gene_names[i]
            iso_indices = np.where(self.adata.var["gene_symbol"] == gene_name)[0]
            expected = self.counts_dense[:, iso_indices]
            np.testing.assert_allclose(counts.to_dense().numpy(), expected, rtol=1e-5)

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
        n_spots = 50
        n_genes = 4

        counts_np = np.random.randint(0, 5, size=(n_spots, n_genes)).astype(np.float32)
        coords_np = np.random.rand(n_spots, 2).astype(np.float32)

        # 1. Test Dense Numpy
        res_np = run_hsic_gc(counts_np, coords_np)

        # 2. Test Dense Torch
        counts_torch = torch.from_numpy(counts_np)
        coords_torch = torch.from_numpy(coords_np)
        res_torch = run_hsic_gc(counts_torch, coords_torch)

        # 3. Test Scipy Sparse
        counts_csr = scipy.sparse.csr_matrix(counts_np)
        res_csr = run_hsic_gc(counts_csr, coords_np)

        # 4. Test Torch Sparse
        counts_torch_sparse = counts_torch.to_sparse()
        res_torch_sparse = run_hsic_gc(counts_torch_sparse, coords_np)

        # Check consistency

        np.testing.assert_allclose(
            res_np["statistic"], res_torch["statistic"], rtol=1e-5
        )
        np.testing.assert_allclose(res_np["statistic"], res_csr["statistic"], rtol=1e-5)
        np.testing.assert_allclose(
            res_np["statistic"], res_torch_sparse["statistic"], rtol=1e-5
        )

        # Check p-values roughly
        np.testing.assert_allclose(res_np["pvalue"], res_torch["pvalue"], rtol=1e-5)
        np.testing.assert_allclose(res_np["pvalue"], res_csr["pvalue"], rtol=1e-5)
        np.testing.assert_allclose(
            res_np["pvalue"], res_torch_sparse["pvalue"], rtol=1e-5
        )

    def test_run_hsic_gc_null_methods(self):
        """Both null_method='eig' and 'trace' should return valid p-values."""
        n_spots, n_genes = 50, 4
        counts_np = np.random.randint(0, 5, size=(n_spots, n_genes)).astype(np.float32)
        coords_np = np.random.rand(n_spots, 2).astype(np.float32)

        res_eig = run_hsic_gc(counts_np, coords_np, null_method="eig")
        res_trace = run_hsic_gc(counts_np, coords_np, null_method="trace")

        for res, nm in [(res_eig, "eig"), (res_trace, "trace")]:
            self.assertEqual(res["null_method"], nm)
            self.assertEqual(len(res["pvalue"]), n_genes)
            self.assertTrue(np.all(res["pvalue"] >= 0))
            self.assertTrue(np.all(res["pvalue"] <= 1))
            self.assertIn("pvalue_adj", res)

        # statistics should be identical (same kernel, same counts)
        np.testing.assert_allclose(
            res_eig["statistic"], res_trace["statistic"], rtol=1e-5
        )

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

    def test_extract_counts_n_ratios_sparse_lil_dense_return(self):
        counts_lil = scipy.sparse.lil_matrix(self.counts_dense)
        fake_adata = types.SimpleNamespace(
            layers={"counts_lil": counts_lil},
            var=self.adata.var,
        )

        counts_list, ratios_list, gene_names, ratio_obs = extract_counts_n_ratios(
            fake_adata,
            layer="counts_lil",
            group_iso_by="gene_symbol",
            return_sparse=False,
        )

        self.assertGreater(len(counts_list), 0)
        self.assertEqual(len(ratios_list), len(counts_list))
        self.assertIsNotNone(ratio_obs)
        self.assertIsInstance(counts_list[0], torch.Tensor)
        self.assertFalse(counts_list[0].is_sparse)

    def test_extract_gene_level_statistics_sparse_lil_and_zero_gene(self):
        counts_lil = scipy.sparse.lil_matrix(self.counts_dense)
        fake_adata = types.SimpleNamespace(
            layers={"counts_lil": counts_lil},
            var=self.adata.var,
        )
        stats = extract_gene_level_statistics(fake_adata, layer="counts_lil")
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
            def __init__(self, coordinates, **kwargs):
                self.kwargs = kwargs

            def eigenvalues(self, k=None):
                return torch.tensor([1.0], dtype=torch.float32)

            def xtKx(self, counts):
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


if __name__ == "__main__":
    unittest.main()
