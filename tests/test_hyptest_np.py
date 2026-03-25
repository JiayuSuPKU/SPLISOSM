import unittest
import torch
import numpy as np
import scipy.sparse
import pandas as pd
from anndata import AnnData
from splisosm.utils import run_hsic_gc, extract_counts_n_ratios
from splisosm.hyptest_np import (
    SplisosmNP,
    _calc_ttest_differential_usage,
)
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
            }
        )
        self.adata = AnnData(X=adata_counts, obs=adata_obs, var=adata_var)
        self.adata.layers["counts"] = adata_counts
        self.adata.obsm["spatial"] = np.asarray(self.coords).astype(np.float32)

        self._is_sparkx_installed = self._test_sparkx_installed()

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

    def test_setup_data(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        model_str = str(model)
        self.assertIn("Non-parametric SPLISOSM", model_str)

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
        self.assertEqual(model.n_isos, [2])
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

        self.assertEqual(model.setup_input_mode, "anndata")
        self.assertIs(model.adata, self.adata)
        self.assertEqual(model.n_factors, 1)
        self.assertEqual(model.covariate_names, ["cov_1"])

    def test_docstring_example_spatial_variability_workflow(self):
        model = SplisosmNP()
        model.setup_data(data=self.counts, coordinates=self.coords)
        model.test_spatial_variability(method="hsic-ir", print_progress=False)
        sv_results = model.get_formatted_test_results("sv")

        self.assertEqual(len(sv_results), self.n_genes)
        self.assertTrue(
            {"statistic", "pvalue", "pvalue_adj"}.issubset(sv_results.columns)
        )

    def test_docstring_example_differential_usage_workflow(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
        )
        model.test_differential_usage(method="hsic", print_progress=False)
        du_results = model.get_formatted_test_results("du")

        self.assertEqual(len(du_results), self.n_genes * self.design_mtx.shape[1])
        self.assertTrue(
            {"statistic", "pvalue", "pvalue_adj"}.issubset(du_results.columns)
        )

    def test_spatial_variability(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        for method in ["hsic-gc", "hsic-ir", "hsic-ic", "spark-x"]:
            if method == "spark-x" and not self._is_sparkx_installed:
                self.skipTest("SPARK-X is not installed. Skipping SPARK-X test.")

            with self.subTest(method=method):
                model.test_spatial_variability(method=method)
                sv_results = model.get_formatted_test_results("sv")
                print(sv_results.head())
                self.assertIn(method, str(model))

    def test_sparse_data_handling(self):
        # 1. Prepare sparse data using AnnData and extract_counts_n_ratios
        n_spots = self.n_spots
        n_isos_per_gene = [3, 2]
        total_isos = sum(n_isos_per_gene)

        # Create random counts with some zeros
        counts_dense = np.random.randint(0, 5, size=(n_spots, total_isos)).astype(
            np.float32
        )
        counts_dense[counts_dense < 2] = 0
        # Make it sparse
        counts_sparse = scipy.sparse.csr_matrix(counts_dense)

        # Create var dataframe
        gene_ids = []
        for i, n in enumerate(n_isos_per_gene):
            gene_ids.extend([f"gene_sparse_{i}"] * n)
        var = pd.DataFrame(
            {"gene_symbol": gene_ids}, index=[f"iso_{i}" for i in range(total_isos)]
        )

        adata = AnnData(X=counts_sparse, var=var)
        adata.layers["counts"] = counts_sparse

        # Extract sparse tensors (should return list of sparse torch tensors)
        data_sparse, _, gene_names, _ = extract_counts_n_ratios(
            adata, "counts", "gene_symbol", return_sparse=True
        )

        # Verify inputs are sparse
        for tensor in data_sparse:
            self.assertTrue(tensor.is_sparse)

        # 2. Test SplisosmNP with sparse data
        model = SplisosmNP()
        model.setup_data(
            data=data_sparse,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=gene_names,
        )

        # Test spatial variability - HSIC-GC (gene counts)
        model.test_spatial_variability(method="hsic-gc", print_progress=False)
        self.assertTrue(len(model.sv_test_results) > 0)

        # Test spatial variability - HSIC-IR (isoform ratios, requires densification)
        model.test_spatial_variability(method="hsic-ir", print_progress=False)
        self.assertTrue(len(model.sv_test_results) > 0)

        # Test differential usage - HSIC (isoform ratios, requires densification)
        model.test_differential_usage(method="hsic", print_progress=False)
        self.assertTrue(len(model.du_test_results) > 0)

    def test_hsic_gc(self):
        """Make sure the standalone hsic-gc function works as expected."""
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        # run hsic-gc using the class method
        model.test_spatial_variability(method="hsic-gc")
        sv_results1 = model.get_formatted_test_results("sv")

        # run hsic-gc using the standalone utility function
        counts_g = torch.concat(
            [_counts.sum(1, keepdim=True) for _counts in self.counts], axis=1
        )  # tensor(n_spots, n_genes)
        sv_results2 = run_hsic_gc(counts_gene=counts_g, coordinates=model.coordinates)
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
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        for method in ["hsic", "hsic-gp"]:
            with self.subTest(method=method):
                model.test_differential_usage(method=method, print_progress=False)
                du_results = model.get_formatted_test_results("du")
                self.assertGreater(len(du_results), 0)
                self.assertIn(method, str(model))

    def test_spatial_variability_with_none_transformation(self):
        """Test spatial variability with no transformation."""
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        model.test_spatial_variability(
            method="hsic-ir", ratio_transformation="none", print_progress=False
        )
        sv_results = model.get_formatted_test_results("sv")
        self.assertEqual(len(sv_results), self.n_genes)
        self.assertIn("pvalue", sv_results.columns)

    def test_spatial_variability_with_radial_transformation(self):
        """Test spatial variability with radial ratio transformation."""
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
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
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        model.test_spatial_variability(
            method="hsic-ir",
            ratio_transformation="none",
            nan_filling="none",
            print_progress=False,
        )
        sv_results = model.get_formatted_test_results("sv")
        self.assertEqual(len(sv_results), self.n_genes)

    def test_differential_usage_t_fisher(self):
        """Test differential usage with t-fisher method using binary factor."""
        model = SplisosmNP()
        # Create a binary design matrix for t-test
        binary_design = torch.zeros(self.design_mtx.shape[0], 1)
        binary_design[::2, 0] = 1  # Alternate between 0 and 1

        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=binary_design,
            gene_names=self.gene_names,
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
            data=self.counts,
            coordinates=self.coords,
            design_mtx=binary_design,
            gene_names=self.gene_names,
        )
        model.test_differential_usage(method="t-tippett", print_progress=False)
        du_results = model.get_formatted_test_results("du")
        self.assertEqual(len(du_results), self.n_genes * 1)

    def test_differential_usage_unconditional_hsic(self):
        """Test unconditional HSIC (method='hsic', no spatial conditioning)."""
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
        )
        model.test_differential_usage(method="hsic", print_progress=False)
        du_results = model.get_formatted_test_results("du")
        self.assertEqual(len(du_results), self.n_genes * self.design_mtx.shape[1])

    def test_differential_usage_with_no_transformation(self):
        """Test differential usage without transformation."""
        model = SplisosmNP()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
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
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
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
            self.design_mtx.numpy(), columns=["treatment", "batch"]
        )
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=design_df,
            # covariate_names=None (not provided, should be inferred)
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
            data=self.counts,
            coordinates=self.coords,
            design_mtx=design_df,
            covariate_names=custom_names,
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
                data=self.counts,
                coordinates=self.coords,
                design_mtx=design_df,
                covariate_names=["only_one"],
            )

        self.assertIn("must match", str(context.exception))

    def test_design_mtx_sparse_array_conversion(self):
        """Test that sparse matrices are properly converted to dense."""
        import scipy.sparse as sp

        model = SplisosmNP()
        # Convert design matrix to sparse scipy format
        design_sparse = sp.csr_matrix(self.design_mtx.numpy())

        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=design_sparse,
        )

        # Verify sparse matrix was converted to torch tensor
        self.assertIsInstance(model.design_mtx, torch.Tensor)
        self.assertEqual(model.design_mtx.shape, (self.n_spots, 2))

    def test_design_mtx_numpy_array_conversion(self):
        """Test numpy array conversion and covariate name generation."""
        model = SplisosmNP()
        design_np = self.design_mtx.numpy()

        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=design_np,
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
            data=self.counts[:5],
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            covariate_names=["C1", "C2"],
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
            data=self.counts[:5],
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            covariate_names=["C1", "C2"],
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
            data=self.counts[:10],
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            covariate_names=["C1", "C2"],
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


if __name__ == "__main__":
    unittest.main()
