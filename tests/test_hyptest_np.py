import unittest
import torch
import numpy as np
from splisosm.utils import run_hsic_gc
from splisosm.hyptest_np import SplisosmNP, _calc_ttest_differential_usage, linear_hsic_test
from splisosm.simulation import simulate_isoform_counts
try:
    import rpy2
    from rpy2.robjects.packages import importr, PackageNotInstalledError
except (ImportError , ModuleNotFoundError):
    PackageNotInstalledError = None

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

        self._is_sparkx_installed = self._test_sparkx_installed()

    def _test_sparkx_installed(self):
        try:
            import rpy2
            from rpy2.robjects.packages import importr
            spark = importr('SPARK')
            return True
        except (ImportError, PackageNotInstalledError):
            return False

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
        self.assertIsInstance(stats, np.floating)
        self.assertIsInstance(pval, np.floating)

    def test_spatial_variability(self):
        model = SplisosmNP()
        model.setup_data(
            data=self.counts, coordinates=self.coords,
            design_mtx=self.design_mtx, gene_names=self.gene_names
        )
        for method in ['hsic-gc', 'hsic-ir', 'hsic-ic', 'spark-x']:
            if method == 'spark-x' and not self._is_sparkx_installed:
                self.skipTest("SPARK-X is not installed. Skipping SPARK-X test.")

            with self.subTest(method=method):
                model.test_spatial_variability(method=method)
                sv_results = model.get_formatted_test_results('sv')
                print(sv_results.head())
                self.assertIn(method, str(model))

    def test_hsic_gc(self):
        """Make sure the standalone hsic-gc function works as expected."""
        model = SplisosmNP()
        model.setup_data(
            data=self.counts, coordinates=self.coords,
            design_mtx=self.design_mtx, gene_names=self.gene_names
        )
        # run hsic-gc using the class method
        model.test_spatial_variability(method='hsic-gc')
        sv_results1 = model.get_formatted_test_results('sv')

        # run hsic-gc using the standalone utility function
        counts_g = torch.concat([_counts.sum(1, keepdim=True) for _counts in self.counts], axis = 1) # tensor(n_spots, n_genes)
        sv_results2 = run_hsic_gc(
            counts_gene=counts_g,
            coordinates=model.coordinates
        )
        # compare the statistics and p-values
        stats1 = sv_results1['statistic'].values
        stats2 = sv_results2['statistic']
        self.assertTrue(np.allclose(stats1, stats2, atol=1e-6))

        pvals1 = sv_results1['pvalue'].values
        pvals2 = sv_results2['pvalue']
        self.assertTrue(np.allclose(pvals1, pvals2, atol=1e-6))

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