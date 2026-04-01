import unittest
import os
import tempfile
import warnings
from unittest.mock import patch
import torch
import numpy as np
import scipy.sparse
import pandas as pd
from anndata import AnnData
from splisosm.model import MultinomGLMM

from splisosm.simulation import simulate_isoform_counts

from splisosm.hyptest_glmm import (
    IsoFullModel,
    IsoNullNoSpVar,
    _fit_model_one_gene,
    _fit_null_full_sv_one_gene,
    _fit_perm_one_gene,
    _calc_llr_spatial_variability,
    _calc_wald_differential_usage,
    _calc_score_differential_usage,
    SplisosmGLMM,
)


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
        obs = pd.DataFrame({f"cov_{i+1}": dm[:, i] for i in range(dm.shape[1])})
    else:
        obs = pd.DataFrame(index=[str(i) for i in range(n_spots)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X
    adata.obsm["spatial"] = coords_np.astype(np.float32)
    return adata


class TestHypTestGLMM(unittest.TestCase):

    def setUp(self):
        # Set up mock data for testing
        data = get_simulation_data()
        self.counts = data["counts"]
        self.design_mtx = data["design_mtx"]
        self.cov_sp = data["cov_sp"]
        self.model_configs = {"fitting_configs": {"max_epochs": 5}}
        corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.cov_sp)
        self.corr_sp_eigvals = corr_sp_eigvals
        self.corr_sp_eigvecs = corr_sp_eigvecs

    def test_fit_model_one_gene_glm(self):
        pars = _fit_model_one_gene(
            self.model_configs,
            "glm",
            self.counts,
            None,
            None,
            self.design_mtx,
            quiet=True,
            random_seed=42,
        )
        self.assertIn("beta", pars)
        self.assertIn("bias_eta", pars)

    def test_fit_model_one_gene_glmm_full(self):
        pars = _fit_model_one_gene(
            self.model_configs,
            "glmm-full",
            self.counts,
            self.corr_sp_eigvals,
            self.corr_sp_eigvecs,
            self.design_mtx,
            quiet=True,
            random_seed=42,
        )
        self.assertIn("nu", pars)
        self.assertIn("beta", pars)
        self.assertIn("bias_eta", pars)

    def test_fit_null_full_sv_one_gene(self):
        null_pars, full_pars = _fit_null_full_sv_one_gene(
            self.model_configs,
            self.counts,
            self.corr_sp_eigvals,
            self.corr_sp_eigvecs,
            None,
            quiet=True,
            random_seed=42,
        )
        self.assertIn("nu", null_pars)
        self.assertIn("nu", full_pars)

    def test_calc_llr_spatial_variability(self):
        null_model = IsoNullNoSpVar(**self.model_configs)
        full_model = IsoFullModel(**self.model_configs)
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_model.fit()
        full_model.fit()
        sv_llr, df = _calc_llr_spatial_variability(null_model, full_model)
        self.assertIsNotNone(sv_llr)
        self.assertIsInstance(df, int)

    def test_calc_wald_differential_usage(self):
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.fit()
        wald_stat, df = _calc_wald_differential_usage(full_model)
        self.assertIsNotNone(wald_stat)
        self.assertIsInstance(df, int)

    def test_sparse_data_handling(self):
        n_spots = self.counts[0].shape[
            0
        ]  # roughly 400 from get_simulation_data defaults
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
        var = pd.DataFrame(
            {"gene_symbol": gene_ids}, index=[f"iso_{i}" for i in range(total_isos)]
        )

        coords = torch.rand(n_spots, 2)

        # 2. Test SplisosmGLMM with sparse data
        # Use a very low max_epochs to speed up test
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 1})

        design_mtx_np = np.random.randn(n_spots, 2).astype(np.float32)
        obs = pd.DataFrame({"cov_1": design_mtx_np[:, 0], "cov_2": design_mtx_np[:, 1]})
        sparse_adata = AnnData(X=counts_sparse, obs=obs, var=var)
        sparse_adata.layers["counts"] = counts_sparse
        sparse_adata.obsm["spatial"] = coords.numpy().astype(np.float32)

        model.setup_data(
            adata=sparse_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )

        # Fit the model
        model.fit(quiet=True, print_progress=False)
        self.assertTrue(model._is_trained)

    def test_calc_score_differential_usage(self):
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts,
            design_mtx=None,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.fit()
        score_stat, df = _calc_score_differential_usage(full_model, self.design_mtx)
        self.assertIsNotNone(score_stat)
        self.assertIsInstance(df, int)


class TestSplisosmGLMM(unittest.TestCase):
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

    def test_splisosm_glmm_setup_data(self):
        model = SplisosmGLMM()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        self.assertIsNotNone(model.design_mtx)
        self.assertIsNotNone(model.coordinates)

    def test_splisosm_glmm_setup_data_from_anndata(self):
        model = SplisosmGLMM()
        model.setup_data(
            adata=self.adata,
            spatial_key="spatial",
            layer="counts",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            gene_names="gene_label",
            min_counts=0,
            min_bin_pct=0.0,
            group_gene_by_n_iso=True,
        )

        self.assertEqual(model._setup_input_mode, "anndata")
        self.assertIs(model.adata, self.adata)
        self.assertEqual(model.n_genes, len(self.gene_names))
        self.assertEqual(model.n_spots, self.n_spots)
        self.assertEqual(model.covariate_names, ["cov_1", "cov_2"])

    def test_docstring_example_workflow(self):
        data_3_iso = get_simulation_data(n_genes=3, n_isos=3, n_spots_per_dim=8)
        data_4_iso = get_simulation_data(n_genes=3, n_isos=4, n_spots_per_dim=8)
        counts = [g for g in data_3_iso["counts"]] + [g for g in data_4_iso["counts"]]
        coordinates = data_3_iso["coords"]
        design_mtx = data_3_iso["design_mtx"]
        local_adata = _make_small_adata(counts, coordinates, design_mtx)

        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(
            n_jobs=1,
            batch_size=3,
            with_design_mtx=False,
            quiet=True,
            print_progress=False,
        )

        fitted_models = model.get_fitted_models()
        self.assertEqual(len(fitted_models), 6)
        self.assertIsInstance(fitted_models[0], MultinomGLMM)

        model.test_differential_usage(method="score")
        du_results = model.get_formatted_test_results("du")
        self.assertEqual(len(du_results), 6 * 2)  # 6 genes × 2 covariates

    def test_splisosm_glm_fit(self):
        model = SplisosmGLMM(model_type="glm")
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
            group_gene_by_n_iso=True,
        )
        model.fit(quiet=True, batch_size=5)
        fitted_models = model.get_fitted_models()
        self.assertTrue(len(fitted_models) == 20)

    def test_splisosm_glmm_fit(self):
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(quiet=True, batch_size=5)
        fitted_models = model.get_fitted_models()
        self.assertTrue(len(fitted_models) == 20)

    def test_splisosm_glmm_test_spatial_variability(self):
        model = SplisosmGLMM(
            model_type="glmm-full", fitting_configs={"max_epochs": 100}
        )
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=False, from_null=True, quiet=True)
        model.test_spatial_variability(method="llr", use_perm_null=False)
        sv_results = model.get_formatted_test_results("sv")

        print(str(model))
        print(sv_results.head())
        self.assertIsNotNone(sv_results)

    def test_splisosm_glmm_test_differential_usage(self):
        model = SplisosmGLMM(
            model_type="glmm-full", fitting_configs={"max_epochs": 100}
        )
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=False, quiet=True)
        model.test_differential_usage(method="score")
        du_results = model.get_formatted_test_results("du")

        print(str(model))
        print(du_results.head())
        self.assertIsNotNone(du_results)

    def test_splisosm_glmm_test_differential_usage_wald(self):
        """Test differential usage with Wald method."""
        model = SplisosmGLMM(
            model_type="glmm-full", fitting_configs={"max_epochs": 100}
        )
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=True, quiet=True)
        model.test_differential_usage(method="wald")
        du_results = model.get_formatted_test_results("du")
        self.assertIsNotNone(du_results)

    def test_splisosm_glmm_test_spatial_variability_permutation(self):
        """Test spatial variability with permutation-based null."""
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=False, from_null=True, quiet=True)
        model.test_spatial_variability(
            method="llr", use_perm_null=True, n_perms_per_gene=10
        )
        sv_results = model.get_formatted_test_results("sv")
        self.assertIsNotNone(sv_results)

    def test_splisosm_glmm_test_spatial_variability_return_results(self):
        """Test spatial variability with return_results=True."""
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=False, from_null=True, quiet=True)
        results = model.test_spatial_variability(method="llr", return_results=True)
        self.assertIsNotNone(results)
        self.assertIn("statistic", results)

    def test_splisosm_glmm_test_differential_usage_return_results(self):
        """Test differential usage with return_results=True."""
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=False, quiet=True)
        results = model.test_differential_usage(method="score", return_results=True)
        self.assertIsNotNone(results)
        self.assertIn("statistic", results)

    def test_splisosm_glmm_setup_data_without_design_mtx(self):
        """Test setup_data without design matrix."""
        model = SplisosmGLMM()
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=None,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        self.assertIsNone(model.design_mtx)

    def test_splisosm_glmm_fit_with_design_matrix(self):
        """Test fitting with design matrix."""
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(with_design_mtx=True, quiet=True)
        self.assertTrue(model._is_trained)

    def test_splisosm_glmm_different_batch_size(self):
        """Test fitting with different batch_size."""
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 5})
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(quiet=True, batch_size=10)
        fitted_models = model.get_fitted_models()
        self.assertTrue(len(fitted_models) > 0)


class TestIsoModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in IsoFullModel and IsoNullNoSpVar."""

    def setUp(self):
        data = get_simulation_data(n_genes=2, n_isos=3, n_spots_per_dim=10)
        self.counts = data["counts"]
        self.design_mtx = data["design_mtx"]
        self.cov_sp = data["cov_sp"]
        self.model_configs = {"fitting_configs": {"max_epochs": 2}}
        corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.cov_sp)
        self.corr_sp_eigvals = corr_sp_eigvals
        self.corr_sp_eigvecs = corr_sp_eigvecs

    def test_iso_null_no_sp_var_unsupported_method_raises_error(self):
        """Test that IsoNullNoSpVar raises error with unsupported method."""
        # joint_newton is not supported because Newton's method requires sigma to be optimized
        # This raises AssertionError from the parent class, not ValueError
        with self.assertRaises(AssertionError):
            _ = IsoNullNoSpVar(
                fitting_method="joint_newton", fitting_configs=self.model_configs
            )

    def test_iso_full_model_from_null_no_sp_var(self):
        """Test IsoFullModel creation from null model."""
        null_model = IsoNullNoSpVar(**self.model_configs)
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_model.fit(quiet=True)

        # Create full model from null
        full_model = IsoFullModel.from_trained_null_no_sp_var_model(null_model)
        self.assertEqual(full_model.n_genes, null_model.n_genes)
        self.assertEqual(
            full_model.fitting_method, "joint_gd"
        )  # Should change from joint_newton

    def test_iso_full_model_with_marginal_newton_to_gd(self):
        """Test IsoFullModel conversion from marginal_newton to marginal_gd."""
        null_model = IsoNullNoSpVar(
            fitting_method="marginal_newton", fitting_configs=self.model_configs
        )
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_model.fit(quiet=True)

        full_model = IsoFullModel.from_trained_null_no_sp_var_model(null_model)
        self.assertEqual(full_model.fitting_method, "marginal_gd")

    def test_iso_null_from_full_model(self):
        """Test IsoNullNoSpVar creation from full model."""
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.fit(quiet=True)

        # Create null model from full
        null_model = IsoNullNoSpVar.from_trained_full_model(full_model)
        self.assertEqual(null_model.n_genes, full_model.n_genes)

    def test_iso_null_spatial_variance_turned_off(self):
        """Test that spatial variance is turned off in null model."""
        null_model = IsoNullNoSpVar(**self.model_configs)
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        # Check that either theta_logit or sigma_sp has no gradient
        if hasattr(null_model, "theta_logit"):
            self.assertFalse(null_model.theta_logit.requires_grad)
        elif hasattr(null_model, "sigma_sp"):
            self.assertFalse(null_model.sigma_sp.requires_grad)

    def test_fit_model_one_gene_glm_different_types(self):
        """Test _fit_model_one_gene with different model types."""
        for model_type in ["glm", "glmm-full", "glmm-null"]:
            with self.subTest(model_type=model_type):
                if model_type == "glm":
                    pars = _fit_model_one_gene(
                        self.model_configs,
                        model_type,
                        self.counts,
                        None,
                        None,
                        self.design_mtx,
                        quiet=True,
                        random_seed=42,
                    )
                    self.assertIn("beta", pars)
                else:
                    pars = _fit_model_one_gene(
                        self.model_configs,
                        model_type,
                        self.counts,
                        self.corr_sp_eigvals,
                        self.corr_sp_eigvecs,
                        self.design_mtx,
                        quiet=True,
                        random_seed=42,
                    )
                    self.assertIn("nu", pars) if model_type == "glmm-full" else True

    def test_fit_null_full_sv_with_refit(self):
        """Test _fit_null_full_sv_one_gene with refit_null=True."""
        null_pars, full_pars = _fit_null_full_sv_one_gene(
            self.model_configs,
            self.counts,
            self.corr_sp_eigvals,
            self.corr_sp_eigvecs,
            self.design_mtx,
            refit_null=True,
            quiet=True,
            random_seed=42,
        )
        self.assertIn("nu", null_pars)
        self.assertIn("nu", full_pars)

    def test_fit_null_full_sv_without_refit(self):
        """Test _fit_null_full_sv_one_gene with refit_null=False."""
        null_pars, full_pars = _fit_null_full_sv_one_gene(
            self.model_configs,
            self.counts,
            self.corr_sp_eigvals,
            self.corr_sp_eigvecs,
            self.design_mtx,
            refit_null=False,
            quiet=True,
            random_seed=42,
        )
        self.assertIn("nu", null_pars)
        self.assertIn("nu", full_pars)

    def test_iso_full_model_str_representation(self):
        """Test string representation of IsoFullModel."""
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        str_repr = str(full_model)
        self.assertIn("Multinomial Generalized Linear Mixed Model", str_repr)

    def test_iso_null_str_representation(self):
        """Test string representation of IsoNullNoSpVar."""
        null_model = IsoNullNoSpVar(**self.model_configs)
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        str_repr = str(null_model)
        self.assertIn("Multinomial Generalized Linear Mixed Model", str_repr)

    def test_iso_full_model_clone(self):
        """Test cloning IsoFullModel."""
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.fit(quiet=True)
        cloned = full_model.clone()
        self.assertEqual(cloned.n_genes, full_model.n_genes)
        self.assertEqual(cloned.n_spots, full_model.n_spots)

    def test_iso_null_clone(self):
        """Test cloning IsoNullNoSpVar."""
        null_model = IsoNullNoSpVar(**self.model_configs)
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_model.fit(quiet=True)
        cloned = null_model.clone()
        self.assertEqual(cloned.n_genes, null_model.n_genes)
        self.assertEqual(cloned.n_spots, null_model.n_spots)

    def test_iso_null_from_full_model_with_joint_gd(self):
        """Test from_trained_full_model where full model uses joint_gd method."""
        full_model = IsoFullModel(
            fitting_method="joint_gd", fitting_configs=self.model_configs
        )
        full_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.fit(quiet=True)
        null_model = IsoNullNoSpVar.from_trained_full_model(full_model)
        # joint_gd should stay as joint_gd
        self.assertEqual(null_model.fitting_method, "joint_gd")

    def test_iso_full_model_from_null_with_joint_gd(self):
        """Test IsoFullModel.from_trained_null_no_sp_var_model with joint_gd method."""
        null_model = IsoNullNoSpVar(
            fitting_method="joint_gd", fitting_configs=self.model_configs
        )
        null_model.setup_data(
            counts=self.counts,
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_model.fit(quiet=True)
        full_model = IsoFullModel.from_trained_null_no_sp_var_model(null_model)
        # joint_gd should stay the same
        self.assertEqual(full_model.fitting_method, "joint_gd")
        self.assertEqual(null_model.n_genes, full_model.n_genes)


class TestSplisosmGLMMCoverageBranches(unittest.TestCase):
    """Target uncovered branches in splisosm.hyptest_glmm."""

    def setUp(self):
        data = get_simulation_data(n_genes=2, n_isos=3, n_spots_per_dim=8)
        self.counts_tensor = data["counts"]
        self.counts_list = [g for g in data["counts"]]
        self.coords = data["coords"]
        self.design_mtx = data["design_mtx"]
        self.cov_sp = data["cov_sp"]
        self.corr_sp_eigvals, self.corr_sp_eigvecs = torch.linalg.eigh(self.cov_sp)

    def test_setup_data_numpy_1d_design_and_constant_warning(self):
        model = SplisosmGLMM()
        design_1d_np = np.ones(self.coords.shape[0], dtype=np.float32)
        local_adata = _make_small_adata(self.counts_list, self.coords)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            model.setup_data(
                adata=local_adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                design_mtx=design_1d_np,
                covariate_names=["const"],
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
            )

        self.assertEqual(model.design_mtx.shape[1], 1)
        self.assertTrue(
            any("zero variance" in str(w.message) for w in records),
            "Expected zero-variance covariate warning.",
        )

    def test_setup_data_eigh_fallback_to_eig(self):
        model = SplisosmGLMM()
        n_spots = self.coords.shape[0]
        eigvals = torch.ones(n_spots, dtype=torch.complex64)
        eigvecs = torch.eye(n_spots, dtype=torch.complex64)
        local_adata = _make_small_adata(self.counts_list, self.coords, self.design_mtx)

        with (
            patch(
                "splisosm.hyptest_glmm.torch.linalg.eigh",
                side_effect=RuntimeError("forced eigh failure"),
            ),
            patch(
                "splisosm.hyptest_glmm.torch.linalg.eig",
                return_value=(eigvals, eigvecs),
            ),
        ):
            model.setup_data(
                adata=local_adata,
                layer="counts",
                spatial_key="spatial",
                group_iso_by="gene_symbol",
                design_mtx=["cov_1", "cov_2"],
                min_counts=0,
                min_bin_pct=0.0,
                filter_single_iso_genes=False,
            )

        self.assertTrue(torch.is_floating_point(model._corr_sp_eigvals))
        self.assertTrue(torch.is_floating_point(model._corr_sp_eigvecs))

    def test_fit_warns_when_batch_size_without_grouping(self):
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 1})
        local_adata = _make_small_adata(self.counts_list, self.coords, self.design_mtx)
        model.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=False,
        )

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            model.fit(
                batch_size=2, quiet=True, print_progress=False, with_design_mtx=False
            )

        self.assertTrue(
            any("Ignoring batch size argument" in str(w.message) for w in records)
        )

    def test_get_fitted_models_glmm_null_and_save(self):
        model = SplisosmGLMM(model_type="glmm-null", fitting_configs={"max_epochs": 1})
        local_adata = _make_small_adata(self.counts_list, self.coords, self.design_mtx)
        model.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(quiet=True, print_progress=False, with_design_mtx=False, batch_size=1)

        fitted = model.get_fitted_models()
        self.assertEqual(len(fitted), len(self.counts_list))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            save_path = tmp.name
        try:
            model.save(save_path)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_fit_helpers_with_sparse_counts(self):
        counts_sparse = self.counts_tensor.to_sparse()

        pars_glm = _fit_model_one_gene(
            {"fitting_configs": {"max_epochs": 1}},
            "glm",
            counts_sparse,
            None,
            None,
            self.design_mtx,
            quiet=True,
            random_seed=0,
        )
        self.assertIn("beta", pars_glm)

        null_pars, full_pars = _fit_null_full_sv_one_gene(
            {"fitting_configs": {"max_epochs": 1}},
            counts_sparse,
            self.corr_sp_eigvals,
            self.corr_sp_eigvecs,
            self.design_mtx,
            refit_null=False,
            quiet=True,
            random_seed=0,
        )
        self.assertIn("nu", null_pars)
        self.assertIn("nu", full_pars)

    def test_fit_perm_one_gene_path(self):
        perm_idx = torch.randperm(self.counts_tensor.shape[1])
        sv_llr = _fit_perm_one_gene(
            perm_idx,
            {"fitting_configs": {"max_epochs": 1}},
            self.counts_tensor.to_sparse(),
            self.corr_sp_eigvals,
            self.corr_sp_eigvecs,
            self.design_mtx,
            refit_null=False,
            random_seed=0,
        )
        self.assertGreaterEqual(sv_llr.numel(), 1)

    def test_score_helper_1d_and_invalid_3d_covariate(self):
        full_model = IsoFullModel(fitting_configs={"max_epochs": 1})
        full_model.setup_data(
            counts=self.counts_tensor[:1],
            design_mtx=None,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_model.fit(quiet=True, verbose=False, diagnose=False, random_seed=0)

        stat, df = _calc_score_differential_usage(full_model, self.design_mtx[:, 0])
        self.assertIsNotNone(stat)
        self.assertEqual(df, full_model.n_isos - 1)

        with self.assertRaises(ValueError):
            _calc_score_differential_usage(
                full_model, torch.randn(self.design_mtx.shape[0], 1, 1)
            )

    def test_spatial_variability_requires_null_models(self):
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 1})
        local_adata = _make_small_adata(self.counts_list, self.coords, self.design_mtx)
        model.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(
            quiet=True, print_progress=False, from_null=False, with_design_mtx=False
        )

        with self.assertRaises(ValueError):
            model.test_spatial_variability(method="llr", print_progress=False)

    def test_spatial_variability_uses_cached_permutation_results(self):
        model = SplisosmGLMM(model_type="glmm-full", fitting_configs={"max_epochs": 1})
        local_adata = _make_small_adata(self.counts_list, self.coords, self.design_mtx)
        model.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        model.fit(
            quiet=True,
            print_progress=False,
            from_null=True,
            with_design_mtx=False,
            refit_null=False,
        )
        model.fitting_results["sv_llr_perm_stats"] = torch.zeros(model.n_genes)

        with patch("builtins.print") as print_mock:
            model.test_spatial_variability(
                method="llr",
                use_perm_null=True,
                print_progress=False,
            )

        print_mock.assert_any_call("Using cached permutation results...")

    def test_differential_usage_error_paths(self):
        local_adata = _make_small_adata(self.counts_list, self.coords, self.design_mtx)

        model_no_design = SplisosmGLMM(fitting_configs={"max_epochs": 1})
        model_no_design.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=None,
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        with self.assertRaises(ValueError):
            model_no_design.test_differential_usage(
                method="score", print_progress=False
            )

        model_score = SplisosmGLMM(fitting_configs={"max_epochs": 1})
        model_score.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        with self.assertRaises(ValueError):
            model_score.test_differential_usage(method="score", print_progress=False)

        model_score.fit(
            quiet=True,
            print_progress=False,
            with_design_mtx=True,
            from_null=False,
        )
        with self.assertRaises(ValueError):
            model_score.test_differential_usage(method="score", print_progress=False)

        model_wald = SplisosmGLMM(fitting_configs={"max_epochs": 1})
        model_wald.setup_data(
            adata=local_adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            design_mtx=["cov_1", "cov_2"],
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
            group_gene_by_n_iso=True,
        )
        with self.assertRaises(ValueError):
            model_wald.test_differential_usage(method="wald", print_progress=False)

        model_wald.fit(
            quiet=True,
            print_progress=False,
            with_design_mtx=False,
            from_null=False,
        )
        with self.assertRaises(ValueError):
            model_wald.test_differential_usage(method="wald", print_progress=False)

    def test_iso_conversion_branches_with_sigma_sp_parameterization(self):
        base_joint = MultinomGLMM(
            fitting_method="joint_newton",
            var_parameterization_sigma_theta=False,
            var_fix_sigma=False,
            fitting_configs={"max_epochs": 1},
        )
        base_joint.setup_data(
            self.counts_tensor[:1],
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        full_from_joint = IsoFullModel.from_trained_null_no_sp_var_model(base_joint)
        self.assertEqual(full_from_joint.fitting_method, "joint_gd")
        self.assertTrue(full_from_joint.sigma_sp.requires_grad)

        base_marg = MultinomGLMM(
            fitting_method="marginal_newton",
            var_parameterization_sigma_theta=False,
            fitting_configs={"max_epochs": 1},
        )
        base_marg.setup_data(
            self.counts_tensor[:1],
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_from_marg = IsoNullNoSpVar.from_trained_full_model(base_marg)
        self.assertEqual(null_from_marg.fitting_method, "marginal_gd")
        self.assertFalse(null_from_marg.sigma_sp.requires_grad)
        self.assertTrue(
            torch.allclose(
                null_from_marg.sigma_sp, torch.zeros_like(null_from_marg.sigma_sp)
            )
        )

        base_joint_for_null = MultinomGLMM(
            fitting_method="joint_newton",
            var_parameterization_sigma_theta=False,
            var_fix_sigma=False,
            fitting_configs={"max_epochs": 1},
        )
        base_joint_for_null.setup_data(
            self.counts_tensor[:1],
            design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        null_from_joint = IsoNullNoSpVar.from_trained_full_model(base_joint_for_null)
        self.assertEqual(null_from_joint.fitting_method, "joint_gd")

    def test_iso_null_unsupported_method_raises_value_error(self):
        with self.assertRaises(ValueError):
            IsoNullNoSpVar(
                fitting_method="joint_newton",
                var_fix_sigma=False,
                fitting_configs={"max_epochs": 1},
            )


class TestSplisosmGLMMNewFeatures(unittest.TestCase):
    """Tests for the new SplisosmGLMM features: kernel configs and feature summaries."""

    def setUp(self):
        data = get_simulation_data(n_genes=5, n_isos=3, n_spots_per_dim=10)
        gene_names = [f"gene_{i}" for i in range(5)]
        iso_name_list = []
        iso_gene_ids = []
        counts_merged = []
        for _gene_name, _counts in zip(gene_names, data["counts"]):
            counts_merged.append(_counts)
            for _iso_idx in range(3):
                iso_name_list.append(f"{_gene_name}_iso_{_iso_idx}")
                iso_gene_ids.append(_gene_name)

        adata_counts = torch.concat(counts_merged, dim=1).numpy().astype(np.float32)
        adata_var = pd.DataFrame({"gene_symbol": iso_gene_ids}, index=iso_name_list)
        self.adata = AnnData(X=adata_counts, var=adata_var)
        self.adata.layers["counts"] = adata_counts
        self.adata.obsm["spatial"] = np.asarray(data["coords"]).astype(np.float32)
        self.gene_names = gene_names

    def _make_model(self, **kwargs):
        """Helper: build and setup a default model."""
        model = SplisosmGLMM(**kwargs)
        model.setup_data(
            adata=self.adata,
            layer="counts",
            spatial_key="spatial",
            group_iso_by="gene_symbol",
            min_counts=0,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        return model

    # ------------------------------------------------------------------
    # Kernel config tests
    # ------------------------------------------------------------------

    def test_default_kernel_configs_stored(self):
        """Default k_neighbors/rho are stored; standardize_cov is always True internally."""
        model = SplisosmGLMM()
        self.assertEqual(model._kernel_k_neighbors, 4)
        self.assertAlmostEqual(model._kernel_rho, 0.99)
        self.assertTrue(
            model._kernel_standardize_cov
        )  # always True — not user-configurable

    def test_custom_kernel_configs_stored(self):
        """Custom k_neighbors/rho passed to __init__ are stored."""
        model = SplisosmGLMM(k_neighbors=6, rho=0.95)
        self.assertEqual(model._kernel_k_neighbors, 6)
        self.assertAlmostEqual(model._kernel_rho, 0.95)
        self.assertTrue(
            model._kernel_standardize_cov
        )  # always True regardless of params

    def test_custom_k_neighbors_affects_corr_sp(self):
        """Different k_neighbors produce different spatial covariance matrices."""
        model_k4 = self._make_model(k_neighbors=4)
        model_k6 = self._make_model(k_neighbors=6)
        # Matrices should differ (k=4 vs k=6 adjacency)
        self.assertFalse(torch.allclose(model_k4.corr_sp, model_k6.corr_sp))

    def test_eigvals_eigvecs_populated_after_setup(self):
        """setup_data populates _corr_sp_eigvals and _corr_sp_eigvecs."""
        model = self._make_model()
        n = model.n_spots
        self.assertEqual(model._corr_sp_eigvals.shape, (n,))
        self.assertEqual(model._corr_sp_eigvecs.shape, (n, n))

    def test_eigvals_consistent_with_corr_sp(self):
        """Eigendecomposition reconstructs corr_sp correctly."""
        model = self._make_model()
        V = model._corr_sp_eigvecs
        L = model._corr_sp_eigvals
        recon = V @ torch.diag(L) @ V.t()
        self.assertTrue(
            torch.allclose(recon, model.corr_sp, atol=1e-4),
            "Reconstructed corr_sp does not match stored corr_sp.",
        )

    # ------------------------------------------------------------------
    # approx_rank tests
    # ------------------------------------------------------------------

    def test_approx_rank_default_small_n_is_full_rank(self):
        """Default approx_rank uses full rank for small n_spots (<= 5000)."""
        model = self._make_model()
        n = model.n_spots
        # Auto-selection: full rank for n <= 5000
        self.assertEqual(model._corr_sp_eigvals.shape[0], n)
        self.assertEqual(model._corr_sp_eigvecs.shape, (n, n))
        self.assertIsNotNone(model.corr_sp)

    def test_approx_rank_explicit_int(self):
        """Passing approx_rank=k truncates eigenvectors to k columns."""
        model = self._make_model(approx_rank=5)
        n = model.n_spots
        self.assertEqual(model._corr_sp_eigvals.shape[0], 5)
        self.assertEqual(model._corr_sp_eigvecs.shape, (n, 5))
        # corr_sp is None for truncated decomposition
        self.assertIsNone(model.corr_sp)

    def test_approx_rank_none_is_full_rank(self):
        """Explicit approx_rank=None forces full-rank decomposition."""
        model = self._make_model(approx_rank=None)
        n = model.n_spots
        self.assertEqual(model._corr_sp_eigvals.shape[0], n)
        self.assertEqual(model._corr_sp_eigvecs.shape, (n, n))

    def test_approx_rank_stored_on_init(self):
        """approx_rank is stored as _approx_rank attribute."""
        import splisosm.hyptest_glmm as hg

        model_auto = SplisosmGLMM()
        self.assertIs(model_auto._approx_rank, hg._APPROX_RANK_AUTO)

        model_none = SplisosmGLMM(approx_rank=None)
        self.assertIsNone(model_none._approx_rank)

        model_int = SplisosmGLMM(approx_rank=10)
        self.assertEqual(model_int._approx_rank, 10)

    def test_approx_rank_model_fits_low_rank(self):
        """Low-rank model (approx_rank=k) can be fit and produces results."""
        model = self._make_model(approx_rank=5)
        model.fit(quiet=True)
        self.assertTrue(model._is_trained)

    # ------------------------------------------------------------------
    # Feature summary tests
    # ------------------------------------------------------------------

    def test_extract_feature_summary_errors_before_setup(self):
        """extract_feature_summary raises RuntimeError if called before setup_data."""
        model = SplisosmGLMM()
        with self.assertRaises(RuntimeError):
            model.extract_feature_summary(level="gene")

    def test_extract_feature_summary_invalid_level(self):
        """extract_feature_summary raises ValueError for unknown level."""
        model = self._make_model()
        with self.assertRaises(ValueError):
            model.extract_feature_summary(level="invalid")

    def test_extract_gene_summary_shape_and_columns(self):
        """Gene-level summary has one row per gene and expected columns."""
        model = self._make_model()
        df = model.extract_feature_summary(level="gene", print_progress=False)
        self.assertEqual(len(df), len(self.gene_names))
        for col in ("n_isos", "perplexity", "pct_bin_on", "count_avg", "count_std"):
            self.assertIn(col, df.columns)

    def test_extract_isoform_summary_shape_and_columns(self):
        """Isoform-level summary has one row per isoform and expected columns."""
        model = self._make_model()
        df = model.extract_feature_summary(level="isoform", print_progress=False)
        n_isoforms = sum(3 for _ in self.gene_names)  # 3 isos per gene
        self.assertEqual(len(df), n_isoforms)
        for col in (
            "pct_bin_on",
            "count_total",
            "count_avg",
            "count_std",
            "ratio_total",
            "ratio_avg",
            "ratio_std",
        ):
            self.assertIn(col, df.columns)

    def test_gene_summary_values_are_non_negative(self):
        """Gene-level summary statistics are non-negative."""
        model = self._make_model()
        df = model.extract_feature_summary(level="gene", print_progress=False)
        for col in ("n_isos", "perplexity", "pct_bin_on", "count_avg", "count_std"):
            self.assertTrue(
                (df[col] >= 0).all(), f"Column {col!r} has negative values."
            )

    def test_feature_summary_is_cached(self):
        """Repeated calls return the same object (caching works)."""
        model = self._make_model()
        df1 = model.extract_feature_summary(level="gene", print_progress=False)
        df2 = model.extract_feature_summary(level="gene", print_progress=False)
        self.assertIs(df1, df2)

    def test_feature_summary_ratio_sums_to_one(self):
        """Per-gene ratio_total should sum to 1 (or 0 for all-zero genes)."""
        model = self._make_model()
        iso_df = model.extract_feature_summary(level="isoform", print_progress=False)
        gene_df = model.extract_feature_summary(level="gene", print_progress=False)
        for gene in gene_df.index:
            iso_subset = iso_df[iso_df["gene"] == gene]
            ratio_sum = iso_subset["ratio_total"].sum()
            # Either sums to ~1 (expressed gene) or ~0 (unexpressed gene)
            self.assertTrue(
                abs(ratio_sum - 1.0) < 1e-5 or abs(ratio_sum) < 1e-5,
                f"Gene {gene!r}: ratio_total sums to {ratio_sum} (expected ~0 or ~1).",
            )


if __name__ == "__main__":
    unittest.main()
