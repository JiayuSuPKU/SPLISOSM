import unittest
import torch
import numpy as np

from splisosm.simulation import (
    _sample_multinom_sp_single_gene,
    simulate_isoform_counts_single_gene,
    simulate_isoform_counts,
)

class TestSimulation(unittest.TestCase):
    def test_sample_multinom_sp_single_gene(self):
        # simulate a gene with 3 isoforms in 2 spots
        iso_ratio_expected = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        total_counts_expected = 100
        counts = _sample_multinom_sp_single_gene(iso_ratio_expected, total_counts_expected)

        self.assertEqual(counts.shape, (2, 3))

    def test_simulate_isoform_counts_single_gene(self):
        grid_size = (10, 10)
        n_isos = 3
        total_counts_expected = 100
        var_sp = 0.5
        var_nsp = 0.5
        rho = 0.9

        data, params = simulate_isoform_counts_single_gene(
            grid_size=grid_size,
            n_isos=n_isos,
            total_counts_expected=total_counts_expected,
            var_sp=var_sp,
            var_nsp=var_nsp,
            rho=rho,
            return_params=True,
        )

        self.assertEqual(data["counts"].shape, (grid_size[0] * grid_size[1], n_isos))
        self.assertEqual(data["coords"].shape, (grid_size[0] * grid_size[1], 2))
        self.assertEqual(params["grid_size"], grid_size)
        self.assertEqual(params["n_isos"], n_isos)

    def test_simulate_isoform_counts(self):
        n_genes = 5
        grid_size = (10, 10)
        n_isos = 3
        total_counts_expected = 100
        var_sp = 0.5
        var_nsp = 0.5
        rho = 0.9

        data, params = simulate_isoform_counts(
            n_genes=n_genes,
            grid_size=grid_size,
            n_isos=n_isos,
            total_counts_expected=total_counts_expected,
            var_sp=var_sp,
            var_nsp=var_nsp,
            rho=rho,
            return_params=True,
        )

        self.assertEqual(data["counts"].shape, (n_genes, grid_size[0] * grid_size[1], n_isos))
        self.assertEqual(data["coords"].shape, (grid_size[0] * grid_size[1], 2))
        self.assertEqual(params["grid_size"], grid_size)
        self.assertEqual(params["n_isos"], n_isos)
        self.assertEqual(params["n_spots"], grid_size[0] * grid_size[1])

if __name__ == "__main__":
    unittest.main()