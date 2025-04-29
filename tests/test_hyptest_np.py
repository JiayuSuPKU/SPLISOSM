import unittest
import torch
import numpy as np
from splisosm.hyptest_np import SplisosmNP, _calc_ttest_differential_usage, linear_hsic_test
from splisosm.simulation import simulate_isoform_counts

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

class TestSplisosmNP(unittest.TestCase):

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

        self.n_spots = coords.shape[0]
        self.n_genes = len(counts)

    def test_setup_data(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts, coordinates=self.coords,
            design_mtx=self.design_mtx, gene_names=self.gene_names
        )
        model_str = str(model)
        self.assertIn("Non-parametric SPLISOSM", model_str)

    def test_calc_ttest_differential_usage(self):
        data = torch.rand(self.n_spots, 2)
        groups = torch.tensor([0] * (self.n_spots // 2) + [1] * (self.n_spots // 2))
        stats, pval = _calc_ttest_differential_usage(data, groups)
        self.assertIsInstance(stats, float)
        self.assertIsInstance(pval, float)

    def test_spatial_variability(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts, coordinates=self.coords,
            design_mtx=self.design_mtx, gene_names=self.gene_names
        )
        for method in ['hsic-gc', 'hsic-ir', 'hsic-ic', 'spark-x']:
            with self.subTest(method=method):
                model.test_spatial_variability(method=method)
                sv_results = model.get_formatted_test_results('sv')
                print(sv_results.head())
                self.assertIn(method, str(model))

    def test_differential_usage(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts, coordinates=self.coords,
            design_mtx=self.design_mtx, gene_names=self.gene_names
        )
        for method in ['hsic', 'hsic-knn', 'hsic-gp']:
            with self.subTest(method=method):
                model.test_differential_usage(method=method, hsic_eps=1e-3)
                du_results = model.get_formatted_test_results('du')
                print(du_results.head())
                self.assertIn(method, str(model))

if __name__ == "__main__":
    unittest.main()