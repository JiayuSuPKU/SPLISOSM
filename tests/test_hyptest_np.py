import unittest
import warnings
import torch
import numpy as np
import scipy.sparse
import pandas as pd
from anndata import AnnData
from unittest.mock import patch
from splisosm.utils import run_hsic_gc
from splisosm.hyptest_np import (
    SplisosmNP,
    _calc_ttest_differential_usage,
    _prepare_np_response,
    _quadratic_columns_exact,
    _sparse_counts_to_ratios_centered,
)
from splisosm._hsic_null import _feature_cumulants_from_data
from splisosm.kernel import SpatialCovKernel
from splisosm.utils import counts_to_ratios
from splisosm.simulation import simulate_isoform_counts


def get_simulation_data(n_genes=2, n_isos=3, n_spots_per_dim=20):
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # generate data
    mtc, var = 10, 0.3
    n_spots = n_spots_per_dim**2
    X_spot = torch.concat([torch.randn(n_spots, 2)], dim=1)
    beta_true = torch.ones(2, n_isos - 1)
    data = simulate_isoform_counts(
        n_genes=n_genes,
        grid_size=(n_spots_per_dim, n_spots_per_dim),
        n_isos=n_isos,
        total_counts_expected=mtc,
        var_sp=var,
        var_nsp=var,
        rho=0.99,
        design_mtx=X_spot,
        beta_true=beta_true,
        return_params=False,
    )

    return data


def _make_small_adata(counts_list, coords, design_mtx=None):
    """Build a minimal AnnData from per-gene (n_spots, n_isos) count tensors."""
    coords_np = coords.numpy() if hasattr(coords, "numpy") else np.asarray(coords)
    n_spots = coords_np.shape[0]
    all_c, gene_ids, iso_names = [], [], []
    for i, c in enumerate(counts_list):
        c_np = (
            c.detach().numpy().astype(np.float32)
            if hasattr(c, "numpy")
            else np.asarray(c, dtype=np.float32)
        )
        all_c.append(c_np)
        for j in range(c_np.shape[1]):
            gene_ids.append(f"gene_{i}")
            iso_names.append(f"gene_{i}_iso_{j}")
    X = np.concatenate(all_c, axis=1)
    var = pd.DataFrame({"gene_symbol": gene_ids}, index=iso_names)
    if design_mtx is not None:
        dm = (
            design_mtx.detach().numpy()
            if hasattr(design_mtx, "numpy")
            else np.asarray(design_mtx, dtype=np.float32)
        )
        obs = pd.DataFrame(
            {f"cov_{i+1}": dm[:, i] for i in range(dm.shape[1])},
            index=[str(i) for i in range(n_spots)],
        )
    else:
        obs = pd.DataFrame(index=[str(i) for i in range(n_spots)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X
    adata.obsm["spatial"] = coords_np.astype(np.float32)
    return adata


class TestSplisosmNP(unittest.TestCase):

    def setUp(self):
        # simulate genes with different number of isoforms
        data_3_iso = get_simulation_data(n_genes=10, n_isos=3, n_spots_per_dim=20)
        data_4_iso = get_simulation_data(n_genes=10, n_isos=4, n_spots_per_dim=20)

        design_mtx = data_3_iso["design_mtx"]  # (400, 2)
        coords = data_3_iso["coords"]  # (400, 2)

        # concat counts as list
        counts = [g for g in data_3_iso["counts"]] + [
            g for g in data_4_iso["counts"]
        ]  # len = 20
        gene_names = [f"gene_{i}" for i in range(20)]

        self.counts = counts
        self.gene_names = gene_names
        self.coords = coords
        self.design_mtx = design_mtx

        self.n_spots = coords.shape[0]
        self.n_genes = len(counts)

        iso_name_list = []
        iso_gene_ids = []
        iso_gene_labels = []
        counts_merged = []
        for _gene_name, _counts in zip(self.gene_names, self.counts):
            n_iso = _counts.shape[1]
            counts_merged.append(_counts)
            for _iso_idx in range(n_iso):
                iso_name_list.append(f"{_gene_name}_iso_{_iso_idx}")
                iso_gene_ids.append(_gene_name)
                iso_gene_labels.append(f"{_gene_name}_label")

        adata_counts = torch.concat(counts_merged, dim=1).numpy().astype(np.float32)
        adata_var = pd.DataFrame(
            {"gene_symbol": iso_gene_ids, "gene_label": iso_gene_labels},
            index=iso_name_list,
        )
        design_np = np.asarray(self.design_mtx)
        adata_obs = pd.DataFrame(
            {
                "cov_1": design_np[:, 0],
                "cov_2": design_np[:, 1],
            },
            index=[str(i) for i in range(design_np.shape[0])],
        )
        self.adata = AnnData(X=adata_counts, obs=adata_obs, var=adata_var)
        self.adata.layers["counts"] = adata_counts
        self.adata.obsm["spatial"] = np.asarray(self.coords).astype(np.float32)

        self._is_sparkx_installed = self._test_sparkx_installed()

        # Subsets for tests that only need a few genes
        g5 = [f"gene_{i}" for i in range(5)]
        g10 = [f"gene_{i}" for i in range(10)]
        self.adata_5g = self.adata[:, self.adata.var["gene_symbol"].isin(g5)].copy()
        self.adata_10g = self.adata[:, self.adata.var["gene_symbol"].isin(g10)].copy()

    def _test_sparkx_installed(self):
        try:
            from rpy2.robjects.packages import importr, PackageNotInstalledError

            try:
                _ = importr("SPARK")
                return True
            except PackageNotInstalledError:
                return False
            except Exception:
                return False
        except Exception:
            return False

    def _make_fragmented_data(self):
        """Return (data, coords, design_mtx) where the last 2 spots are isolated.

        Main cluster: 8 spots in a tight 0.25-step grid so k-NN (k=4) connects them.
        Isolated pair: 2 spots placed far away (at coordinate 100, 100 ± 0.01),
        which will form their own tiny component.
        """
        np.random.seed(7)
        # 8 connected spots on a 0–0.75 grid (0.25 spacing → k-NN links them)
        grid = np.array(
            [[i * 0.25, j * 0.25] for i in range(4) for j in range(2)], dtype=np.float32
        )
        # 2 isolated spots far away
        isolated = np.array([[100.0, 100.0], [100.0, 100.01]], dtype=np.float32)
        coords = np.vstack([grid, isolated])  # (10, 2)

        n_spots = coords.shape[0]
        n_genes = 5
        data = [torch.rand(n_spots, 3) for _ in range(n_genes)]

        design_mtx = np.random.randn(n_spots, 2).astype(np.float32)
        return data, coords, design_mtx

    def _make_fragmented_adata(self):
        """Return (adata, n_total) built from _make_fragmented_data."""
        data, coords, design_mtx = self._make_fragmented_data()
        n_total = coords.shape[0]
        adata = _make_small_adata(
            data, torch.from_numpy(coords), torch.from_numpy(design_mtx)
        )
        return adata, n_total

    def test_setup_data(self):
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model_str = str(model)
        self.assertIn("SplisosmNP", model_str)

    def test_calc_ttest_differential_usage(self):
        data = torch.rand(self.n_spots, 2)
        groups = torch.tensor([0] * (self.n_spots // 2) + [1] * (self.n_spots // 2))
        stats, pval = _calc_ttest_differential_usage(data, groups)
        self.assertIsInstance(stats, np.floating)
        self.assertIsInstance(pval, np.floating)

    def test_setup_data_from_anndata(self):
        model = SplisosmNP()
        self.adata.obsm["xy"] = self.adata.obsm["spatial"].copy()

        model.setup_data(
            adata=self.adata,
            spatial_key="xy",
            layer="counts",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
        )

        self.assertEqual(model.n_genes, self.n_genes)
        self.assertEqual(model.n_spots, self.n_spots)
        self.assertEqual(model.covariate_names, ["cov_1", "cov_2"])
        self.assertEqual(model.gene_names[0], f"{self.gene_names[0]}_label")

    def test_setup_data_from_anndata_with_filtering(self):
        counts = np.array(
            [
                [5, 3, 1, 2],
                [4, 2, 0, 2],
                [3, 1, 0, 2],
                [2, 1, 0, 2],
                [3, 1, 0, 1],
                [4, 2, 0, 1],
            ],
            dtype=np.float32,
        )
        var = pd.DataFrame(
            {
                "gene_symbol": ["g1", "g1", "g2", "g2"],
                "gene_label": ["Gene 1", "Gene 1", "Gene 2", "Gene 2"],
            },
            index=["g1_i1", "g1_i2", "g2_i1", "g2_i2"],
        )
        obs = pd.DataFrame(index=[f"spot_{i}" for i in range(counts.shape[0])])
        adata = AnnData(X=counts, obs=obs, var=var)
        adata.layers["counts"] = counts
        adata.obsm["spatial"] = np.stack(
            [np.arange(counts.shape[0]), np.arange(counts.shape[0])], axis=1
        ).astype(np.float32)

        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            gene_names="gene_label",
            min_counts=2,
            min_bin_pct=0.0,
            filter_single_iso_genes=True,
        )

        self.assertEqual(model.n_genes, 1)
        self.assertEqual(model.n_isos_per_gene, [2])
        self.assertEqual(model.gene_names, ["Gene 1"])

    def test_setup_data_from_anndata_with_single_covariate_name(self):
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            spatial_key="spatial",
            layer="counts",
            group_iso_by="gene_symbol",
            design_mtx="cov_1",
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
        )

        self.assertEqual(model._setup_input_mode, "anndata")
        self.assertIs(model.adata, self.adata)
        self.assertEqual(model.n_factors, 1)
        self.assertEqual(model.covariate_names, ["cov_1"])

    def test_docstring_example_spatial_variability_workflow(self):
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_spatial_variability(method="hsic-ir", print_progress=False)
        sv_results = model.get_formatted_test_results("sv")

        self.assertEqual(len(sv_results), self.n_genes)
        self.assertTrue(
            {"statistic", "pvalue", "pvalue_adj"}.issubset(sv_results.columns)
        )

    def test_docstring_example_differential_usage_workflow(self):
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(method="hsic", print_progress=False)
        du_results = model.get_formatted_test_results("du")

        self.assertEqual(len(du_results), self.n_genes * 2)
        self.assertTrue(
            {"statistic", "pvalue", "pvalue_adj"}.issubset(du_results.columns)
        )

    def test_spatial_variability(self):
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        for method in ["hsic-gc", "hsic-ir", "hsic-ic", "spark-x"]:
            if method == "spark-x" and not self._is_sparkx_installed:
                self.skipTest("SPARK-X is not installed. Skipping SPARK-X test.")

            with self.subTest(method=method):
                model.test_spatial_variability(method=method)
                sv_results = model.get_formatted_test_results("sv")
                print(sv_results.head())
                self.assertIn(method, str(model))

    def test_spatial_variability_chunk_size_matches_single_column(self):
        """Column-chunked NP SV matches one-column/singleton execution."""
        adata = self.adata_10g.copy()
        adata.layers["counts"] = scipy.sparse.csr_matrix(adata.layers["counts"])

        ref = None
        last = None
        for chunk_size in (1, 2, "auto"):
            model = SplisosmNP()
            model.setup_data(
                adata=adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                gene_names="gene_label",
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
            )
            res = model.test_spatial_variability(
                method="hsic-ic",
                chunk_size=chunk_size,
                n_jobs=1,
                return_results=True,
                print_progress=False,
            )
            if ref is None:
                ref = res
            else:
                np.testing.assert_allclose(
                    res["statistic"], ref["statistic"], rtol=1e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    res["pvalue"], ref["pvalue"], rtol=1e-5, atol=1e-8
                )
            last = res
        self.assertLessEqual(last["chunk_size"], 32)

    def test_sparse_ratio_response_matches_dense_ratio_statistic(self):
        """Sparse HSIC-IR ratios avoid dense counts while matching dense math."""
        counts_dense = np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0],
                [0.0, 4.0, 4.0],
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
        kernel = SpatialCovKernel.from_coordinates(
            np.column_stack([np.arange(counts_dense.shape[0]), np.zeros(6)]),
            k_neighbors=2,
            rho=0.9,
            centering=True,
        )
        counts_sparse = torch.from_numpy(counts_dense).to_sparse()

        for transformation in ("none", "clr", "ilr", "alr", "radial"):
            with self.subTest(transformation=transformation):
                sparse_centered, nan_mask = _sparse_counts_to_ratios_centered(
                    scipy.sparse.csr_matrix(counts_dense),
                    transformation=transformation,
                    nan_filling="mean",
                )
                self.assertIsNone(nan_mask)
                response, is_centered = _prepare_np_response(
                    counts_sparse,
                    method="hsic-ir",
                    ratio_transformation=transformation,
                )
                dense_ref = counts_to_ratios(
                    torch.from_numpy(counts_dense),
                    transformation=transformation,
                    nan_filling="mean",
                    fill_before_transform=False,
                ).numpy()
                dense_ref = dense_ref - dense_ref.mean(axis=0, keepdims=True)
                np.testing.assert_allclose(
                    sparse_centered.toarray(), dense_ref, atol=1e-6
                )
                self.assertTrue(is_centered)
                self.assertTrue(scipy.sparse.issparse(response))
                np.testing.assert_allclose(response.toarray(), dense_ref, atol=1e-6)
                self.assertEqual(response[1].nnz, 0)
                self.assertEqual(response[4].nnz, 0)

                q_sparse = _quadratic_columns_exact(kernel, response)
                kx = np.asarray(kernel.Kx(dense_ref), dtype=float)
                q_dense = np.sum(dense_ref * kx, axis=0)
                np.testing.assert_allclose(q_sparse, q_dense, rtol=1e-6, atol=1e-8)

    def test_sparse_counts_to_ratios_centered_nan_mask(self):
        """Sparse ratio helper returns centered expressed rows plus NaN mask."""
        counts_dense = np.array(
            [
                [3.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        sparse, nan_mask = _sparse_counts_to_ratios_centered(
            scipy.sparse.csr_matrix(counts_dense),
            transformation="none",
            nan_filling="none",
        )
        np.testing.assert_array_equal(nan_mask, np.array([False, True, False, True]))
        dense_ref = counts_to_ratios(
            torch.from_numpy(counts_dense),
            transformation="none",
            nan_filling="none",
            fill_before_transform=False,
        )
        keep = ~torch.isnan(dense_ref).any(1)
        dense_ref = dense_ref[keep].numpy()
        dense_ref = dense_ref - dense_ref.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(sparse[~nan_mask].toarray(), dense_ref, atol=1e-7)
        self.assertEqual(sparse[nan_mask].nnz, 0)

    def test_sparse_feature_cumulants_match_dense_centering(self):
        """Sparse feature cumulants compute centered Gram without densifying rows."""
        counts_dense = np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0],
                [0.0, 4.0, 4.0],
                [5.0, 0.0, 5.0],
            ],
            dtype=np.float64,
        )
        sparse_counts = scipy.sparse.csc_matrix(counts_dense)
        dense_centered = counts_dense - counts_dense.mean(axis=0, keepdims=True)
        sparse_cumulants = _feature_cumulants_from_data(
            sparse_counts,
            centered=False,
        )
        dense_cumulants = _feature_cumulants_from_data(dense_centered)
        for power in (1, 2, 3, 4):
            self.assertAlmostEqual(
                sparse_cumulants[power],
                dense_cumulants[power],
                places=5,
            )

    def test_sparse_data_handling(self):
        n_spots = self.n_spots
        n_isos_per_gene = [3, 2]
        total_isos = sum(n_isos_per_gene)

        counts_dense = np.random.randint(0, 5, size=(n_spots, total_isos)).astype(
            np.float32
        )
        counts_dense[counts_dense < 2] = 0
        counts_sparse = scipy.sparse.csr_matrix(counts_dense)

        gene_ids = []
        for i, n in enumerate(n_isos_per_gene):
            gene_ids.extend([f"gene_sparse_{i}"] * n)
        var_df = pd.DataFrame(
            {"gene_symbol": gene_ids}, index=[f"iso_{i}" for i in range(total_isos)]
        )

        # Test SplisosmNP with sparse data
        model = SplisosmNP()
        _adata_sparse = AnnData(
            X=counts_sparse,
            obs=self.adata.obs.copy(),
            var=var_df,
        )
        _adata_sparse.layers["counts"] = counts_sparse
        _adata_sparse.obsm["spatial"] = self.adata.obsm["spatial"].copy()
        model.setup_data(
            adata=_adata_sparse,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        # Test spatial variability - HSIC-GC (gene counts)
        model.test_spatial_variability(method="hsic-gc", print_progress=False)
        self.assertTrue(len(model._sv_test_results) > 0)

        # Test spatial variability - HSIC-IR (isoform ratios, requires densification)
        model.test_spatial_variability(method="hsic-ir", print_progress=False)
        self.assertTrue(len(model._sv_test_results) > 0)

        # Test differential usage - HSIC (isoform ratios, requires densification)
        model.test_differential_usage(method="hsic", print_progress=False)
        self.assertTrue(len(model._du_test_results) > 0)

    def test_hsic_gc(self):
        """Make sure the standalone hsic-gc function works as expected."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        # run hsic-gc using the class method
        model.test_spatial_variability(method="hsic-gc")
        sv_results1 = model.get_formatted_test_results("sv")

        # run hsic-gc using the standalone utility function
        counts_g = torch.concat(
            [
                (
                    _counts.to_dense().sum(1, keepdim=True)
                    if _counts.is_sparse
                    else _counts.sum(1, keepdim=True)
                )
                for _counts in model._data
            ],
            axis=1,
        )  # tensor(n_spots, n_genes)
        sv_results2 = run_hsic_gc(counts_gene=counts_g, coordinates=model._coordinates)
        # compare the statistics and p-values
        stats1 = sv_results1["statistic"].values
        stats2 = sv_results2["statistic"]
        self.assertTrue(np.allclose(stats1, stats2, atol=1e-6))

        pvals1 = sv_results1["pvalue"].values
        pvals2 = sv_results2["pvalue"]
        self.assertTrue(np.allclose(pvals1, pvals2, atol=1e-6))

    def test_differential_usage(self):
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        for method in ["hsic", "hsic-gp"]:
            with self.subTest(method=method):
                model.test_differential_usage(method=method, print_progress=False)
                du_results = model.get_formatted_test_results("du")
                self.assertGreater(len(du_results), 0)
                self.assertIn(method, str(model))

    def test_extract_feature_summary_gene(self):
        """extract_feature_summary(level='gene') returns expected shape and columns."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        df = model.extract_feature_summary(level="gene", print_progress=False)
        self.assertEqual(len(df), model.n_genes)
        for col in ("n_isos", "perplexity", "pct_bin_on", "count_avg", "count_std"):
            self.assertIn(col, df.columns)
        self.assertTrue((df["count_avg"] >= 0).all())
        # caching: second call returns same object
        df2 = model.extract_feature_summary(level="gene", print_progress=False)
        self.assertIs(df, df2)

    def test_extract_feature_summary_isoform(self):
        """extract_feature_summary(level='isoform') returns expected shape and columns."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        df = model.extract_feature_summary(level="isoform", print_progress=False)
        self.assertEqual(len(df), sum(model.n_isos_per_gene))
        for col in ("gene_symbol", "ratio_avg", "ratio_std", "ratio_total"):
            self.assertIn(col, df.columns)
        self.assertTrue(np.isfinite(df["ratio_avg"].to_numpy()).all())
        # caching
        df2 = model.extract_feature_summary(level="isoform", print_progress=False)
        self.assertIs(df, df2)

    def test_sv_return_results_true(self):
        """test_spatial_variability(return_results=True) returns the result dict."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        results = model.test_spatial_variability(
            method="hsic-ir", return_results=True, print_progress=False
        )
        self.assertIsNotNone(results)
        self.assertIn("statistic", results)
        self.assertIn("pvalue", results)
        # Should match stored results
        np.testing.assert_array_equal(
            results["statistic"], model._sv_test_results["statistic"]
        )

    def test_with_gene_summary_sv(self):
        """get_formatted_test_results(with_gene_summary=True) appends gene stats."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_spatial_variability(method="hsic-ir", print_progress=False)
        df = model.get_formatted_test_results("sv", with_gene_summary=True)
        self.assertIn("perplexity", df.columns)
        self.assertIn("count_avg", df.columns)
        self.assertEqual(len(df), model.n_genes)

    def test_filtered_adata_property(self):
        """filtered_adata raises before setup_data, returns AnnData after."""
        model = SplisosmNP()
        with self.assertRaises(RuntimeError):
            _ = model.filtered_adata
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        fa = model.filtered_adata
        self.assertIsInstance(fa, AnnData)

    def test_str_reflects_test_status(self):
        """__str__ includes method name after running SV and DU."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        s_before = str(model)
        self.assertIn("SplisosmNP", s_before)

        model.test_spatial_variability(method="hsic-ir", print_progress=False)
        s_sv = str(model)
        self.assertIn("hsic-ir", s_sv)

        model.test_differential_usage(method="hsic", print_progress=False)
        s_du = str(model)
        self.assertIn("hsic", s_du)

    def test_spatial_variability_with_radial_transformation(self):
        """Test spatial variability with radial ratio transformation."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_spatial_variability(
            method="hsic-ir", ratio_transformation="radial", print_progress=False
        )
        sv_results = model.get_formatted_test_results("sv")
        self.assertEqual(len(sv_results), self.n_genes)

    def test_spatial_variability_with_nan_filling_none(self):
        """Test spatial variability with nan_filling='none' and no transformation."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_spatial_variability(
            method="hsic-ir",
            ratio_transformation="none",
            nan_filling="none",
            print_progress=False,
        )
        sv_results = model.get_formatted_test_results("sv")
        self.assertEqual(len(sv_results), self.n_genes)

    def test_sv_nan_filling_none_avoids_kernel_realization(self):
        """The masked-kernel path should not realize a dense parent kernel."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        with patch.object(
            model.sp_kernel,
            "realization",
            side_effect=AssertionError("dense realization should not be called"),
        ):
            res = model.test_spatial_variability(
                method="hsic-ir",
                nan_filling="none",
                null_configs={"n_probes": 8},
                return_results=True,
                print_progress=False,
            )
        self.assertEqual(len(res["pvalue"]), model.n_genes)
        self.assertTrue(np.all(np.isfinite(res["pvalue"])))

    def test_sv_null_methods(self):
        """All three null methods should run and return valid p-values."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        for null_method in ["liu", "welch", "perm"]:
            with self.subTest(null_method=null_method):
                configs = (
                    {"n_perms_per_gene": 50, "perm_batch_size": 10}
                    if null_method == "perm"
                    else {}
                )
                model.test_spatial_variability(
                    method="hsic-ir",
                    null_method=null_method,
                    null_configs=configs,
                    print_progress=False,
                )
                res = model._sv_test_results
                self.assertEqual(res["null_method"], null_method)
                self.assertEqual(len(res["pvalue"]), self.n_genes)
                self.assertTrue(np.all(res["pvalue"] >= 0))
                self.assertTrue(np.all(res["pvalue"] <= 1))

    def test_sv_perm_nan_filling_none(self):
        """Perm null with nan_filling='none' should use the dense per-gene submatrix path."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_spatial_variability(
            method="hsic-ir",
            nan_filling="none",
            null_method="perm",
            null_configs={"n_perms_per_gene": 30, "perm_batch_size": 10},
            print_progress=False,
        )
        res = model._sv_test_results
        self.assertEqual(len(res["pvalue"]), self.n_genes)
        self.assertTrue(np.all(res["pvalue"] >= 0))
        self.assertTrue(np.all(res["pvalue"] <= 1))

    def test_sv_nan_filling_none_uses_per_gene_kernel_moments(self):
        """liu/welch/perm nulls with nan_filling='none' must use per-gene K.

        The `hsic-ir + nan_filling='none'` branch builds a gene-specific
        double-centred kernel submatrix (dropping NaN spots).  The null
        moments / perm kernel must reference that submatrix, not the global
        K_sp; otherwise p-values are mis-calibrated (or shape-mismatch in
        the perm path).  This test injects NaNs into each gene's ratio
        and checks that all three null methods run and return valid p-values.
        """
        # Rebuild adata with a sprinkling of dropout spots so that
        # counts_to_ratios(nan_filling='none') actually emits NaNs.
        rng = np.random.default_rng(0)
        adata = self.adata.copy()
        X = adata.layers["counts"].copy()
        # Zero out ~10% of spots per gene to create NaN ratios after
        # counts_to_ratios when the gene has zero total in those spots.
        for gene in adata.var["gene_symbol"].unique():
            iso_idx = np.where(adata.var["gene_symbol"].values == gene)[0]
            mask = rng.random(self.n_spots) < 0.1
            X[np.ix_(mask, iso_idx)] = 0.0
        adata.layers["counts"] = X

        for null_method in ("liu", "welch", "perm"):
            with self.subTest(null_method=null_method):
                model = SplisosmNP()
                model.setup_data(
                    adata=adata,
                    layer="counts",
                    spatial_key="spatial",
                    group_iso_by="gene_symbol",
                    gene_names="gene_label",
                    min_counts=0,
                    min_bin_pct=0.0,
                    filter_single_iso_genes=False,
                )
                configs = (
                    {"n_perms_per_gene": 30, "perm_batch_size": 10}
                    if null_method == "perm"
                    else {}
                )
                model.test_spatial_variability(
                    method="hsic-ir",
                    nan_filling="none",
                    null_method=null_method,
                    null_configs=configs,
                    print_progress=False,
                )
                res = model._sv_test_results
                self.assertEqual(len(res["pvalue"]), self.n_genes)
                self.assertTrue(np.all(np.isfinite(res["pvalue"])))
                self.assertTrue(np.all(res["pvalue"] >= 0))
                self.assertTrue(np.all(res["pvalue"] <= 1))

    def test_sv_null_method_aliases_deprecated(self):
        """Legacy null names are mapped to canonical Liu/Welch names."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        res_liu = model.test_spatial_variability(
            method="hsic-ir",
            null_method="liu",
            print_progress=False,
            return_results=True,
        )
        res_welch = model.test_spatial_variability(
            method="hsic-ir",
            null_method="welch",
            print_progress=False,
            return_results=True,
        )

        for alias, canonical, expected in (
            ("eig", "liu", res_liu),
            ("clt", "welch", res_welch),
            ("trace", "welch", res_welch),
        ):
            with self.subTest(alias=alias):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res_alias = model.test_spatial_variability(
                        method="hsic-ir",
                        null_method=alias,
                        print_progress=False,
                        return_results=True,
                    )
                dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
                self.assertTrue(len(dep) >= 1, f"expected warning for {alias!r}")
                self.assertEqual(res_alias["null_method"], canonical)
                np.testing.assert_allclose(
                    res_alias["statistic"], expected["statistic"]
                )
                np.testing.assert_allclose(res_alias["pvalue"], expected["pvalue"])

    def test_sv_perm_batch_size_config(self):
        """perm_batch_size in null_configs should be respected."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        for batch_size in [1, 50]:
            with self.subTest(perm_batch_size=batch_size):
                model.test_spatial_variability(
                    method="hsic-gc",
                    null_method="perm",
                    null_configs={
                        "n_perms_per_gene": 50,
                        "perm_batch_size": batch_size,
                    },
                    print_progress=False,
                )
                res = model._sv_test_results
                self.assertEqual(len(res["pvalue"]), self.n_genes)
                self.assertTrue(np.all(res["pvalue"] >= 0))
                self.assertTrue(np.all(res["pvalue"] <= 1))

    def test_differential_usage_t_fisher(self):
        """Test differential usage with t-fisher method using binary factor."""
        model = SplisosmNP()
        # Create a binary design matrix for t-test
        binary_design = torch.zeros(self.design_mtx.shape[0], 1)
        binary_design[::2, 0] = 1  # Alternate between 0 and 1

        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=binary_design,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(method="t-fisher", print_progress=False)
        du_results = model.get_formatted_test_results("du")
        # Should have results for each gene and covariate pair
        self.assertEqual(len(du_results), self.n_genes * 1)

    def test_differential_usage_t_tippett(self):
        """Test differential usage with t-tippett method using binary factor."""
        model = SplisosmNP()
        # Create a binary design matrix for t-test
        binary_design = torch.zeros(self.design_mtx.shape[0], 1)
        binary_design[::2, 0] = 1  # Alternate between 0 and 1

        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=binary_design,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(method="t-tippett", print_progress=False)
        du_results = model.get_formatted_test_results("du")
        self.assertEqual(len(du_results), self.n_genes * 1)

    def test_differential_usage_unconditional_hsic(self):
        """Test unconditional HSIC (method='hsic', no spatial conditioning)."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(method="hsic", print_progress=False)
        du_results = model.get_formatted_test_results("du")
        self.assertEqual(len(du_results), self.n_genes * 2)

    def test_differential_usage_with_no_transformation(self):
        """Test differential usage without transformation."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(
            method="hsic-gp", ratio_transformation="none", print_progress=False
        )
        du_results = model.get_formatted_test_results("du")
        self.assertGreater(len(du_results), 0)

    def test_differential_usage_return_results(self):
        """Test differential usage with return_results=True."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        results = model.test_differential_usage(
            method="hsic", return_results=True, print_progress=False
        )
        self.assertIsNotNone(results)
        self.assertIn("statistic", results)
        self.assertIn("pvalue", results)

    def test_design_mtx_dataframe_name_inference(self):
        """Test that DataFrame column names are inferred for covariate_names."""
        model = SplisosmNP()
        design_df = pd.DataFrame(
            np.asarray(self.design_mtx), columns=["treatment", "batch"]
        )
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=design_df,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        # Verify covariate names were inferred from DataFrame columns
        self.assertEqual(model.covariate_names, ["treatment", "batch"])
        self.assertEqual(model.n_factors, 2)

    def test_design_mtx_explicit_covariate_names_override(self):
        """Test that explicit covariate_names override inferred names."""
        model = SplisosmNP()
        design_df = pd.DataFrame(
            self.design_mtx.numpy(), columns=["default_1", "default_2"]
        )
        custom_names = ["my_treatment", "my_batch"]
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=design_df,
            covariate_names=custom_names,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        # Verify explicit names take priority
        self.assertEqual(model.covariate_names, custom_names)

    def test_design_mtx_dimension_mismatch_error(self):
        """Test that dimension mismatch between design_mtx and covariate_names raises error."""
        model = SplisosmNP()
        design_df = pd.DataFrame(
            self.design_mtx.numpy(), columns=["treatment", "batch"]
        )

        # Number of covariate names doesn't match design matrix columns
        with self.assertRaises(ValueError) as context:
            model.setup_data(
                adata=self.adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                design_mtx=design_df,
                covariate_names=["only_one"],
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
            )

        self.assertIn("must match", str(context.exception))

    def test_design_mtx_sparse_array_conversion(self):
        """Sparse design matrices are stored as scipy CSR (not densified at setup_data time)."""
        import scipy.sparse as sp

        model = SplisosmNP()
        design_sparse = sp.csr_matrix(np.asarray(self.design_mtx))
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=design_sparse,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        # Sparse design matrices are stored as scipy CSR
        self.assertTrue(sp.issparse(model.design_mtx))
        self.assertEqual(model.design_mtx.shape, (self.n_spots, 2))

    def test_design_mtx_numpy_array_conversion(self):
        """Test numpy array conversion and covariate name generation."""
        model = SplisosmNP()

        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=np.asarray(self.design_mtx),
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        # Verify conversion and default name generation
        self.assertIsInstance(model.design_mtx, torch.Tensor)
        self.assertEqual(model.covariate_names, ["factor_1", "factor_2"])
        self.assertEqual(model.n_factors, 2)

    def test_design_mtx_anndata_categorical_encoding(self):
        """Test that categorical columns in AnnData are one-hot encoded."""
        # Create AnnData with a categorical column
        adata = AnnData(X=self.adata.X, obs=self.adata.obs.copy(), var=self.adata.var)
        adata.layers["counts"] = self.adata.layers["counts"].copy()
        adata.obsm["spatial"] = self.adata.obsm["spatial"].copy()

        # Add a categorical column with 3 categories
        adata.obs["treatment"] = pd.Categorical(
            ["A", "B", "C"] * (len(adata.obs) // 3) + ["A"] * (len(adata.obs) % 3)
        )

        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            spatial_key="spatial",
            layer="counts",
            group_iso_by="gene_symbol",
            design_mtx=["treatment"],  # Categorical column should be one-hot encoded
            min_counts=0,
            min_bin_pct=0.0,
        )

        # Should have 3 columns (one-hot encoded) + no extra numerical columns
        self.assertEqual(model.n_factors, 3)
        # Covariate names should include one-hot expanded names
        self.assertTrue(
            all(name.startswith("treatment_") for name in model.covariate_names)
        )

    def test_design_mtx_anndata_mixed_columns(self):
        """Test that numerical and categorical columns are handled correctly."""
        # Create AnnData with both numerical and categorical columns
        adata = AnnData(X=self.adata.X, obs=self.adata.obs.copy(), var=self.adata.var)
        adata.layers["counts"] = self.adata.layers["counts"].copy()
        adata.obsm["spatial"] = self.adata.obsm["spatial"].copy()

        # Add numerical and categorical columns
        adata.obs["age"] = np.linspace(20, 80, len(adata.obs))
        adata.obs["batch"] = pd.Categorical(
            ["batch1", "batch2"] * (len(adata.obs) // 2)
        )

        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            spatial_key="spatial",
            layer="counts",
            group_iso_by="gene_symbol",
            design_mtx=["age", "batch"],
            min_counts=0,
            min_bin_pct=0.0,
        )

        # Should have 1 (age) + 2 (batch categorical) = 3 factors
        self.assertEqual(model.n_factors, 3)
        # First should be age, rest batch-related
        self.assertEqual(model.covariate_names[0], "age")
        self.assertTrue(
            all(name.startswith("batch_") for name in model.covariate_names[1:])
        )

    def test_differential_usage_residualize_both(self):
        """residualize='both' should run without error and return valid p-values."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=np.asarray(self.design_mtx),
            covariate_names=["C1", "C2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(
            method="hsic-gp",
            gpr_backend="sklearn",
            residualize="both",
            print_progress=False,
        )
        df = model.get_formatted_test_results("du")
        self.assertEqual(df.shape[0], 5 * 2)  # 5 genes × 2 covariates
        pv = df["pvalue"].values
        self.assertTrue(np.all(np.isfinite(pv)))
        self.assertTrue(np.all((pv >= 0) & (pv <= 1)))

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("gpytorch") is not None,
        "gpytorch not installed",
    )
    def test_differential_usage_gpytorch_backend(self):
        """gpr_backend='gpytorch' should return valid p-values."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=np.asarray(self.design_mtx),
            covariate_names=["C1", "C2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(
            method="hsic-gp",
            gpr_backend="gpytorch",
            residualize="cov_only",
            print_progress=False,
        )
        df = model.get_formatted_test_results("du")
        self.assertEqual(df.shape[0], 5 * 2)
        pv = df["pvalue"].values
        self.assertTrue(np.all(np.isfinite(pv)))
        self.assertTrue(np.all((pv >= 0) & (pv <= 1)))

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("gpytorch") is not None,
        "gpytorch not installed",
    )
    def test_gpytorch_agrees_with_sklearn(self):
        """GPyTorch and sklearn backends should produce broadly consistent p-value ranks."""
        from scipy.stats import spearmanr

        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_10g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=np.asarray(self.design_mtx),
            covariate_names=["C1", "C2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model.test_differential_usage(
            method="hsic-gp",
            gpr_backend="sklearn",
            residualize="cov_only",
            print_progress=False,
        )
        pv_sk = model.get_formatted_test_results("du")["pvalue"].values

        model.test_differential_usage(
            method="hsic-gp",
            gpr_backend="gpytorch",
            residualize="cov_only",
            print_progress=False,
        )
        pv_pt = model.get_formatted_test_results("du")["pvalue"].values

        rho, _ = spearmanr(
            -np.log10(np.clip(pv_sk, 1e-300, 1)), -np.log10(np.clip(pv_pt, 1e-300, 1))
        )
        self.assertGreater(rho, 0.9, f"Spearman rank correlation={rho:.3f} too low")

    def test_null_methods_agreement(self):
        """Liu, Welch and perm null methods should yield broadly similar p-value ranks.

        We run all methods on the same data and check pairwise Spearman rank
        correlations on –log10(p).  The two asymptotic methods should agree
        tightly (ρ > 0.9); each asymptotic method vs. the permutation
        null should agree moderately (ρ > 0.6), allowing for the discrete, noisy
        nature of a permutation p-value with a finite number of permutations.
        """
        from scipy.stats import spearmanr

        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        pvals = {}
        for null_method in ("liu", "welch", "perm"):
            null_configs = (
                {"n_perms_per_gene": 2000, "perm_batch_size": 100}
                if null_method == "perm"
                else {}
            )
            res = model.test_spatial_variability(
                method="hsic-ir",
                null_method=null_method,
                null_configs=null_configs,
                print_progress=False,
                return_results=True,
            )
            pvals[null_method] = -np.log10(
                np.clip(res["pvalue"].astype(np.float64), 1e-300, 1)
            )

        # Thresholds: asymptotic methods should agree tightly; perm is noisier.
        thresholds = {
            ("liu", "welch"): 0.90,
            ("liu", "perm"): 0.70,
            ("welch", "perm"): 0.70,
        }
        for (m1, m2), thr in thresholds.items():
            rho, _ = spearmanr(pvals[m1], pvals[m2])
            with self.subTest(pair=f"{m1}_vs_{m2}"):
                self.assertGreater(
                    rho,
                    thr,
                    f"Spearman ρ({m1}, {m2}) = {rho:.3f} < {thr} — null methods disagree too much",
                )

    def test_liu_lowrank_vs_fullrank_agreement(self):
        """Low-rank Liu should give p-value ranks consistent with full-rank Liu.

        This is a regression test for the scale-mismatch bug where approx_rank < n_spots
        caused the test stat (full kernel) and the Liu null (rank-k eigenvalues) to be on
        incompatible scales, pushing all p-values to 0.
        """
        from scipy.stats import spearmanr

        model = SplisosmNP()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        res_full = model.test_spatial_variability(
            method="hsic-ir",
            null_method="liu",
            null_configs={},
            print_progress=False,
            return_results=True,
        )
        res_lowrank = model.test_spatial_variability(
            method="hsic-ir",
            null_method="liu",
            null_configs={"approx_rank": 20},
            print_progress=False,
            return_results=True,
        )

        pv_full = res_full["pvalue"]
        pv_lowrank = res_lowrank["pvalue"]

        # Regression: low-rank must NOT produce all-zero p-values
        self.assertFalse(
            np.all(pv_lowrank == 0.0),
            "Low-rank Liu produced all p-values == 0 (scale-mismatch bug regression)",
        )

        # Rank correlation should be high
        rho, _ = spearmanr(
            -np.log10(np.clip(pv_full, 1e-300, 1)),
            -np.log10(np.clip(pv_lowrank, 1e-300, 1)),
        )
        self.assertGreater(
            rho,
            0.85,
            f"Spearman ρ(full-rank, low-rank Liu) = {rho:.3f} — approximation too inaccurate",
        )

    def test_sv_asymptotic_nulls_accept_probe_budget(self):
        """SV asymptotic nulls accept a shared Hutchinson probe budget."""
        model = SplisosmNP()
        model.setup_data(
            adata=self.adata_5g,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        res = model.test_spatial_variability(
            method="hsic-ir",
            null_method="liu",
            null_configs={"n_probes": 8},
            n_jobs=1,
            print_progress=False,
            return_results=True,
        )

        self.assertEqual(res["statistic"].shape, (model.n_genes,))
        self.assertTrue(np.all(np.isfinite(res["pvalue"])))
        self.assertTrue(np.all((res["pvalue"] >= 0.0) & (res["pvalue"] <= 1.0)))

        res = model.test_spatial_variability(
            method="hsic-ir",
            null_method="welch",
            null_configs={
                "n_probes": 8,
            },
            n_jobs=1,
            print_progress=False,
            return_results=True,
        )
        self.assertEqual(res["statistic"].shape, (model.n_genes,))
        self.assertTrue(np.all(np.isfinite(res["pvalue"])))
        self.assertTrue(np.all((res["pvalue"] >= 0.0) & (res["pvalue"] <= 1.0)))

    def test_init_kernel_hyperparams(self):
        """k_neighbors, rho, standardize_cov passed to __init__ should be used in setup_data."""
        model_custom = SplisosmNP(k_neighbors=6, rho=0.5, standardize_cov=False)
        self.assertEqual(model_custom._k_neighbors, 6)
        self.assertAlmostEqual(model_custom._rho, 0.5)
        self.assertFalse(model_custom._standardize_cov)

        model_default = SplisosmNP()
        self.assertEqual(model_default._k_neighbors, 4)
        self.assertAlmostEqual(model_default._rho, 0.99)
        self.assertTrue(model_default._standardize_cov)

        # After setup_data both models work end-to-end
        model_custom.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        model_default.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        # Custom vs default kernel should differ (rho=0.5 vs 0.99 changes eigenvalue spread)
        import torch

        trace_custom = torch.trace(model_custom.sp_kernel.realization()).item()
        trace_default = torch.trace(model_default.sp_kernel.realization()).item()
        self.assertNotAlmostEqual(
            trace_custom,
            trace_default,
            places=3,
            msg="Custom kernel hyperparams had no effect — not wired through",
        )

    # ── min_component_size tests ──────────────────────────────────────────────

    def test_min_component_size_warning_and_filtering(self):
        """Spots in small components are removed and a UserWarning is issued."""
        import warnings as _warnings

        frag_adata, n_total = self._make_fragmented_adata()
        n_isolated = 2

        model = SplisosmNP()
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            model.setup_data(
                adata=frag_adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
                min_component_size=3,
            )

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(len(user_warns) > 0, "Expected at least one UserWarning")
        warning_text = str(user_warns[0].message)
        self.assertIn(str(n_isolated), warning_text)
        self.assertEqual(model.n_spots, n_total - n_isolated)
        self.assertEqual(model._coordinates.shape[0], n_total - n_isolated)
        for g in model._data:
            self.assertEqual(g.shape[0], n_total - n_isolated)

    def test_min_component_size_design_mtx_filtered(self):
        """design_mtx rows are filtered together with spots."""
        import warnings as _warnings

        frag_adata, n_total = self._make_fragmented_adata()
        n_kept = n_total - 2

        model = SplisosmNP()
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            model.setup_data(
                adata=frag_adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                design_mtx=["cov_1", "cov_2"],
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
                min_component_size=3,
            )
        self.assertEqual(model.design_mtx.shape[0], n_kept)

    def test_min_component_size_1_is_noop(self):
        """min_component_size=1 (default) keeps all spots, no warning."""
        import warnings as _warnings

        frag_adata, n_total = self._make_fragmented_adata()

        model = SplisosmNP()
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            try:
                model.setup_data(
                    adata=frag_adata,
                    layer="counts",
                    spatial_key="spatial",
                    group_iso_by="gene_symbol",
                    min_counts=0,
                    min_bin_pct=0.0,
                    filter_single_iso_genes=False,
                    min_component_size=1,
                )
            except UserWarning:
                self.fail("min_component_size=1 should not issue a UserWarning")
        self.assertEqual(model.n_spots, n_total)

    def test_min_component_size_all_removed_raises(self):
        """Raise ValueError when every spot is in a too-small component."""
        coords = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        X = np.ones((2, 2), dtype=np.float32)
        var = pd.DataFrame({"gene_symbol": ["g", "g"]}, index=["g_0", "g_1"])
        obs = pd.DataFrame(index=["s0", "s1"])
        tiny_adata = AnnData(X=X, obs=obs, var=var)
        tiny_adata.layers["counts"] = X
        tiny_adata.obsm["spatial"] = coords
        model = SplisosmNP()
        with self.assertRaises(ValueError):
            model.setup_data(
                adata=tiny_adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
                min_component_size=2,
            )

    def _make_adata_adj_only(self):
        """Return (adata, adj) with adata stripped of obsm['spatial']."""
        from splisosm.kernel import _build_adj_from_coords

        adata = self.adata_5g.copy()
        coords_t = torch.as_tensor(
            np.asarray(adata.obsm["spatial"]), dtype=torch.float32
        )
        adj = _build_adj_from_coords(coords_t, k_neighbors=4, mutual_neighbors=True)
        adata.obsp["adj"] = adj
        del adata.obsm["spatial"]
        return adata, adj

    def test_setup_adj_only_no_spatial_key(self):
        """setup_data accepts adj_key alone; SV HSIC tests run without raw coords."""
        adata, _ = self._make_adata_adj_only()
        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            layer="counts",
            spatial_key="spatial",  # deliberately missing from obsm
            adj_key="adj",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        self.assertIsNone(model._coordinates)
        for m in ("hsic-ir", "hsic-ic", "hsic-gc"):
            with self.subTest(method=m):
                model.test_spatial_variability(method=m, print_progress=False)
                res = model.get_formatted_test_results("sv")
                self.assertEqual(len(res), model.n_genes)
                self.assertTrue(np.all(res["pvalue"].values >= 0))
                self.assertTrue(np.all(res["pvalue"].values <= 1))

    def test_setup_no_spatial_no_adj_raises(self):
        """Missing both spatial_key and adj_key raises a clear error."""
        adata = self.adata_5g.copy()
        del adata.obsm["spatial"]
        model = SplisosmNP()
        with self.assertRaises(ValueError) as ctx:
            model.setup_data(
                adata=adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
            )
        msg = str(ctx.exception)
        self.assertIn("spatial", msg)
        self.assertIn("adj_key", msg)

    def test_sparkx_without_coords_raises(self):
        """spark-x raises a targeted ValueError when coordinates are absent."""
        adata, _ = self._make_adata_adj_only()
        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            layer="counts",
            spatial_key="spatial",
            adj_key="adj",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        with self.assertRaises(ValueError) as ctx:
            model.test_spatial_variability(method="spark-x", print_progress=False)
        self.assertIn("spark-x", str(ctx.exception))
        self.assertIn("spatial", str(ctx.exception))

    def test_hsic_gp_without_coords_raises(self):
        """hsic-gp raises a targeted ValueError when coordinates are absent."""
        adata, _ = self._make_adata_adj_only()
        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            layer="counts",
            spatial_key="spatial",
            adj_key="adj",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        with self.assertRaises(ValueError) as ctx:
            model.test_differential_usage(method="hsic-gp", print_progress=False)
        self.assertIn("hsic-gp", str(ctx.exception))
        self.assertIn("spatial", str(ctx.exception))

    def test_calc_ttest_sparse_groups_matches_dense(self):
        """Sparse groups column produces identical results to the dense path."""
        np.random.seed(7)
        n_spots = 80
        n_isos = 3
        data = torch.from_numpy(np.random.rand(n_spots, n_isos).astype(np.float32))
        # Binary 0/1 group vector: first half = 0, second half = 1
        groups_dense = torch.tensor([0] * (n_spots // 2) + [1] * (n_spots // 2))
        # Sparse equivalent: (n, 1) CSR with nonzeros at group-1 positions
        g1_indices = np.where(groups_dense.numpy() == 1)[0]
        groups_sparse = scipy.sparse.csr_matrix(
            (
                np.ones(len(g1_indices), dtype=np.float32),
                (g1_indices, np.zeros(len(g1_indices), dtype=int)),
            ),
            shape=(n_spots, 1),
        )

        stats_dense, pval_dense = _calc_ttest_differential_usage(data, groups_dense)
        stats_sparse, pval_sparse = _calc_ttest_differential_usage(data, groups_sparse)

        self.assertAlmostEqual(float(stats_dense), float(stats_sparse), places=5)
        self.assertAlmostEqual(float(pval_dense), float(pval_sparse), places=5)

    def test_linear_hsic_sparse_X_matches_dense(self):
        """linear_hsic_test with sparse X gives same result as dense X."""
        from splisosm.kernel_gpr import linear_hsic_test

        np.random.seed(42)
        n, p, q = 100, 2, 4
        # Dense continuous X
        X_np = np.random.randn(n, p).astype(np.float32)
        Y = torch.from_numpy(np.random.randn(n, q).astype(np.float32))
        X_dense = torch.from_numpy(X_np)
        # Sparse version of the same X
        X_sparse = scipy.sparse.csr_matrix(X_np)

        hsic_dense, pval_dense = linear_hsic_test(X_dense, Y, centering=True)
        hsic_sparse, pval_sparse = linear_hsic_test(X_sparse, Y, centering=True)

        self.assertAlmostEqual(hsic_dense, hsic_sparse, places=4)
        self.assertAlmostEqual(pval_dense, pval_sparse, places=4)

    def test_differential_usage_sparse_design_matrix(self):
        """All four DU methods work with a sparse design matrix.

        Uses a binary 0/1 numeric covariate (n_factors=1) so the t-test
        methods have exactly two groups and the result shapes are predictable.

        For the three deterministic methods (t-fisher, t-tippett, hsic) sparse
        and dense design matrices must produce identical p-values.  hsic-gp
        always densifies the design column internally via _get_design_col
        (GPR residuals are always dense), so only a validity check is done
        for that method.
        """
        adata = AnnData(
            X=self.adata.X,
            obs=self.adata.obs.copy(),
            var=self.adata.var,
        )
        adata.layers["counts"] = self.adata.layers["counts"].copy()
        adata.obsm["spatial"] = self.adata.obsm["spatial"].copy()
        # Numeric binary covariate (not categorical) → single design column → n_factors=1.
        # Alternating 0/1 gives balanced groups for the t-test.
        n_obs = len(adata.obs)
        adata.obs["group"] = np.tile([0.0, 1.0], n_obs // 2 + 1)[:n_obs].astype(
            np.float32
        )

        def _run(method, sparse_mtx):
            np.random.seed(0)
            torch.manual_seed(0)
            model = SplisosmNP()
            model.setup_data(
                adata=adata,
                spatial_key="spatial",
                layer="counts",
                group_iso_by="gene_symbol",
                design_mtx=["group"],
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
            )
            if sparse_mtx:
                # Force design_mtx to scipy sparse CSR to exercise the sparse path
                model.design_mtx = scipy.sparse.csr_matrix(model.design_mtx.numpy())
            model.test_differential_usage(method=method, print_progress=False)
            return model._du_test_results

        n_genes = self.n_genes
        n_factors = 1  # single binary covariate

        # ── validity: all four methods run without error and return legal p-values ──
        for method in ["t-fisher", "t-tippett", "hsic", "hsic-gp"]:
            with self.subTest(method=method, check="validity"):
                res = _run(method, sparse_mtx=True)
                self.assertIn("pvalue", res)
                pvals = res["pvalue"]
                self.assertEqual(
                    pvals.shape,
                    (n_genes, n_factors),
                    f"{method} sparse: unexpected p-value shape",
                )
                self.assertTrue(
                    np.all((pvals >= 0) & (pvals <= 1)),
                    f"{method} sparse: p-values outside [0, 1]",
                )

        # ── consistency: deterministic methods must match their dense counterpart ──
        # hsic-gp is excluded: _get_design_col densifies the design column before
        # GPR fitting, so the sparse vs dense comparison is trivially identical;
        # checking it would only add expensive GPR fitting time.
        for method in ["t-fisher", "t-tippett", "hsic"]:
            with self.subTest(method=method, check="sparse==dense"):
                res_dense = _run(method, sparse_mtx=False)
                res_sparse = _run(method, sparse_mtx=True)
                np.testing.assert_allclose(
                    res_dense["pvalue"],
                    res_sparse["pvalue"],
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"{method}: dense vs sparse p-values differ",
                )


class TestParallelNP(unittest.TestCase):
    """Verify that joblib parallelism in SplisosmNP produces identical results to sequential."""

    @classmethod
    def setUpClass(cls):
        """Build a shared small dataset (5 genes, 10×10 grid, 2 covariates)."""
        torch.manual_seed(123)
        np.random.seed(123)
        n_spots_per_dim = 10
        n_genes, n_isos = 5, 3
        mtc, var = 10, 0.3
        n_spots = n_spots_per_dim**2
        X_spot = torch.randn(n_spots, 2)
        data = simulate_isoform_counts(
            n_genes=n_genes,
            grid_size=(n_spots_per_dim, n_spots_per_dim),
            n_isos=n_isos,
            total_counts_expected=mtc,
            var_sp=var,
            var_nsp=var,
            rho=0.99,
            design_mtx=X_spot,
            beta_true=torch.ones(2, n_isos - 1),
            return_params=False,
        )
        counts_list = [data["counts"][i] for i in range(n_genes)]
        cls.adata = _make_small_adata(counts_list, data["coords"], X_spot)
        cls.n_genes = n_genes

        # Binary covariate for t-test methods
        binary_design = torch.zeros(n_spots, 1)
        binary_design[::2, 0] = 1
        cls.adata_binary = _make_small_adata(counts_list, data["coords"], binary_design)

    def _setup_model(self, adata, design_cols=None):
        """Create and setup a fresh SplisosmNP model."""
        model = SplisosmNP()
        model.setup_data(
            adata=adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=design_cols,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        return model

    # ------------------------------------------------------------------
    # Spatial variability: n_jobs=1 vs n_jobs=2
    # ------------------------------------------------------------------

    def test_sv_parallel_liu_hsic_ir(self):
        """SV with hsic-ir + Liu null: n_jobs=1 and n_jobs=2 give identical results."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata)
            model.test_spatial_variability(
                method="hsic-ir", null_method="liu", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="SV statistic differs between n_jobs=1 and n_jobs=2",
                )
                np.testing.assert_allclose(
                    model._sv_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="SV pvalue differs between n_jobs=1 and n_jobs=2",
                )

    def test_sv_parallel_welch_null(self):
        """SV with hsic-ir + Welch null: parallel results are identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata)
            model.test_spatial_variability(
                method="hsic-ir",
                null_method="welch",
                n_jobs=n_jobs,
                print_progress=False,
            )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    model._sv_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                )

    def test_sv_parallel_perm_null(self):
        """SV with hsic-ir + perm null: parallel results are identical (fixed seed)."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata)
            model.test_spatial_variability(
                method="hsic-ir",
                null_method="perm",
                null_configs={"n_perms_per_gene": 30, "perm_batch_size": 10},
                n_jobs=n_jobs,
                print_progress=False,
            )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                # Perm stats should still match (deterministic per-gene body)
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                )

    def test_sv_parallel_hsic_ic(self):
        """SV with hsic-ic method: parallel results are identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata)
            model.test_spatial_variability(
                method="hsic-ic", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    model._sv_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                )

    def test_sv_parallel_hsic_gc(self):
        """SV with hsic-gc method: parallel results are identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata)
            model.test_spatial_variability(
                method="hsic-gc", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    model._sv_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                )

    def test_sv_parallel_nan_filling_none(self):
        """SV with nan_filling='none' (per-gene kernel path): parallel identical."""
        import warnings

        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.test_spatial_variability(
                    method="hsic-ir",
                    nan_filling="none",
                    n_jobs=n_jobs,
                    print_progress=False,
                )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    model._sv_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                )

    # ------------------------------------------------------------------
    # Differential usage: n_jobs=1 vs n_jobs=2
    # ------------------------------------------------------------------

    def test_du_parallel_hsic(self):
        """DU with method='hsic': parallel results are identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata, design_cols=["cov_1", "cov_2"])
            model.test_differential_usage(
                method="hsic", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._du_test_results
            else:
                np.testing.assert_allclose(
                    model._du_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="DU hsic statistic differs",
                )
                np.testing.assert_allclose(
                    model._du_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="DU hsic pvalue differs",
                )

    def test_du_parallel_hsic_gp(self):
        """DU with method='hsic-gp' (sklearn backend): parallel results are identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata, design_cols=["cov_1", "cov_2"])
            model.test_differential_usage(
                method="hsic-gp",
                gpr_backend="sklearn",
                n_jobs=n_jobs,
                print_progress=False,
            )
            if n_jobs == 1:
                ref = model._du_test_results
            else:
                np.testing.assert_allclose(
                    model._du_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="DU hsic-gp statistic differs",
                )
                np.testing.assert_allclose(
                    model._du_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="DU hsic-gp pvalue differs",
                )

    def test_du_parallel_t_fisher(self):
        """DU with method='t-fisher' (binary covariate): parallel identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata_binary, design_cols=["cov_1"])
            model.test_differential_usage(
                method="t-fisher", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._du_test_results
            else:
                np.testing.assert_allclose(
                    model._du_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="DU t-fisher statistic differs",
                )
                np.testing.assert_allclose(
                    model._du_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="DU t-fisher pvalue differs",
                )

    def test_du_parallel_t_tippett(self):
        """DU with method='t-tippett' (binary covariate): parallel identical."""
        for n_jobs in [1, 2]:
            model = self._setup_model(self.adata_binary, design_cols=["cov_1"])
            model.test_differential_usage(
                method="t-tippett", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._du_test_results
            else:
                np.testing.assert_allclose(
                    model._du_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="DU t-tippett statistic differs",
                )
                np.testing.assert_allclose(
                    model._du_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="DU t-tippett pvalue differs",
                )


if __name__ == "__main__":
    unittest.main()
