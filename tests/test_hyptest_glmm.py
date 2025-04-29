import unittest
import torch
import numpy as np

from splisosm.simulation import simulate_isoform_counts

from splisosm.hyptest_glmm import (
    IsoFullModel,
    IsoNullNoSpVar,
    _fit_model_one_gene,
    _fit_null_full_sv_one_gene,
    _calc_llr_spatial_variability,
    _calc_wald_differential_usage,
    _calc_score_differential_usage,
    SplisosmGLMM
)

def get_simulation_data(n_genes=2, n_isos=3, n_spots_per_dim=20):
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # generate data
    mtc, var = 10, 0.3
    n_spots = n_spots_per_dim ** 2
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

class TestHypTestGLMM(unittest.TestCase):

    def setUp(self):
        # Set up mock data for testing
        data = get_simulation_data()
        self.counts = data["counts"]
        self.design_mtx = data["design_mtx"]
        self.cov_sp = data["cov_sp"]
        self.model_configs = {'fitting_configs': {"max_epochs": 5}}
        corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.cov_sp)
        self.corr_sp_eigvals = corr_sp_eigvals
        self.corr_sp_eigvecs = corr_sp_eigvecs

    def test_fit_model_one_gene_glm(self):
        pars = _fit_model_one_gene(
            self.model_configs, 'glm', self.counts,
            None, None, self.design_mtx,
            quiet=True, random_seed=42
        )
        self.assertIn('beta', pars)
        self.assertIn('bias_eta', pars)

    def test_fit_model_one_gene_glmm_full(self):
        pars = _fit_model_one_gene(
            self.model_configs, 'glmm-full', self.counts,
            self.corr_sp_eigvals, self.corr_sp_eigvecs, self.design_mtx,
            quiet=True, random_seed=42
        )
        self.assertIn('nu', pars)
        self.assertIn('beta', pars)
        self.assertIn('bias_eta', pars)

    def test_fit_null_full_sv_one_gene(self):
        null_pars, full_pars = _fit_null_full_sv_one_gene(
            self.model_configs, self.counts, self.corr_sp_eigvals, self.corr_sp_eigvecs, None,
            quiet=True, random_seed=42
        )
        self.assertIn('nu', null_pars)
        self.assertIn('nu', full_pars)

    def test_calc_llr_spatial_variability(self):
        null_model = IsoNullNoSpVar(**self.model_configs)
        full_model = IsoFullModel(**self.model_configs)
        null_model.setup_data(
            counts=self.counts, design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals, corr_sp_eigvecs=self.corr_sp_eigvecs
        )
        full_model.setup_data(
            counts=self.counts, design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals, corr_sp_eigvecs=self.corr_sp_eigvecs
        )
        null_model.fit()
        full_model.fit()
        sv_llr, df = _calc_llr_spatial_variability(null_model, full_model)
        self.assertIsNotNone(sv_llr)
        self.assertIsInstance(df, int)

    def test_calc_wald_differential_usage(self):
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts, design_mtx=self.design_mtx,
            corr_sp_eigvals=self.corr_sp_eigvals, corr_sp_eigvecs=self.corr_sp_eigvecs
        )
        full_model.fit()
        wald_stat, df = _calc_wald_differential_usage(full_model)
        self.assertIsNotNone(wald_stat)
        self.assertIsInstance(df, int)

    def test_calc_score_differential_usage(self):
        full_model = IsoFullModel(**self.model_configs)
        full_model.setup_data(
            counts=self.counts, design_mtx=None,
            corr_sp_eigvals=self.corr_sp_eigvals, corr_sp_eigvecs=self.corr_sp_eigvecs
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

        design_mtx = data_3_iso["design_mtx"] # (400, 2)
        coords = data_3_iso["coords"] # (400, 2)

        # concat counts as list
        counts = [g for g in data_3_iso["counts"]] + [g for g in data_4_iso["counts"]] # len = 20
        gene_names = [f"gene_{i}" for i in range(20)]

        self.counts = counts
        self.gene_names = gene_names
        self.coords = coords
        self.design_mtx = design_mtx

    def test_splisosm_glmm_setup_data(self):
        model = SplisosmGLMM()
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
            group_gene_by_n_iso=True
        )
        self.assertIsNotNone(model.design_mtx)
        self.assertIsNotNone(model.coordinates)

    def test_splisosm_glm_fit(self):
        model = SplisosmGLMM(
            model_type='glm'
        )
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
            group_gene_by_n_iso=True
        )
        model.fit(quiet=True, batch_size=5)
        fitted_models = model.get_fitted_models()
        self.assertTrue(len(fitted_models) == 20)

    def test_splisosm_glmm_fit(self):
        model = SplisosmGLMM(
            model_type='glmm-full',
            fitting_configs={"max_epochs": 5}
        )
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
            group_gene_by_n_iso=True
        )
        model.fit(quiet=True, batch_size=5)
        fitted_models = model.get_fitted_models()
        self.assertTrue(len(fitted_models) == 20)

    def test_splisosm_glmm_test_spatial_variability(self):
        model = SplisosmGLMM(
            model_type='glmm-full',
            fitting_configs={"max_epochs": 100}
        )
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
            group_gene_by_n_iso=True
        )
        model.fit(with_design_mtx=False, from_null=True, quiet=True)
        model.test_spatial_variability(method="llr", use_perm_null=False)
        sv_results = model.get_formatted_test_results('sv')

        print(str(model))
        print(sv_results.head())
        self.assertIsNotNone(sv_results)

    def test_splisosm_glmm_test_differential_usage(self):
        model = SplisosmGLMM(
            model_type='glmm-full',
            fitting_configs={"max_epochs": 100}
        )
        model.setup_data(
            data=self.counts,
            coordinates=self.coords,
            design_mtx=self.design_mtx,
            gene_names=self.gene_names,
            group_gene_by_n_iso=True
        )
        model.fit(with_design_mtx=False, quiet=True)
        model.test_differential_usage(method="score")
        du_results = model.get_formatted_test_results('du')

        print(str(model))
        print(du_results.head())
        self.assertIsNotNone(du_results)

if __name__ == '__main__':
    unittest.main()