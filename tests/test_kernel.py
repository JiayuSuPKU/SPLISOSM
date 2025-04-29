import unittest
import numpy as np
import torch
import itertools
import scipy
from splisosm.kernel import SpatialCovKernel
from splisosm.utils import get_cov_sp


class TestSpatialCovKernel(unittest.TestCase):

    def setUp(self):
        # Simulate 80x80 grid coordinates
        x = np.linspace(0, 1, 30)
        y = np.linspace(0, 1, 30)
        self.n_spots = 900
        self.coords = np.array(list(itertools.product(x, y)))
        self.H = torch.eye(900) - 1 / 900

    def test_full_rank_kernel(self):
        # spatial kernel from the icar prior directly
        cov_sp1 = get_cov_sp(self.coords)
        K1 = self.H @ cov_sp1 @ self.H
        L1 = torch.linalg.eigvalsh(K1) # sorted in ascending order
        self.assertTrue(L1.numpy()[0] > 0)

        # spatial kernel from the icar prior using the kernel class
        cov_sp2 = SpatialCovKernel(self.coords, approx_rank=None, centering=True)
        self.assertTrue(cov_sp2.rank() == 900)
        self.assertTrue(cov_sp2.shape() == (900, 900))

        # check eigenvalues
        K2 = cov_sp2.realization()
        L2 = cov_sp2.eigenvalues()
        L2 = torch.flip(L2, dims=(0,)) # sort in descending order
        self.assertTrue(torch.allclose(K1, K2, atol=1e-4))
        self.assertTrue(torch.allclose(L1, L2, atol=1e-4))

    def test_low_rank_approximation(self):
        cov_sp1 = SpatialCovKernel(self.coords, approx_rank=800, centering=True, standardize_cov=False)
        cov_sp2 = SpatialCovKernel(self.coords, approx_rank=100, centering=True, standardize_cov=False)
        self.assertTrue(cov_sp1.rank() == 800)
        self.assertTrue(cov_sp2.rank() == 100)

        L1 = cov_sp1.eigenvalues()
        L2 = cov_sp2.eigenvalues()
        self.assertEqual(len(L2), 100)
        self.assertTrue((L1[:100] - L2).abs().mean() < 0.1)

    def test_kernel_realization(self):
        cov_sp1 = SpatialCovKernel(self.coords, approx_rank=None, centering=True, standardize_cov=False)
        cov_sp2 = SpatialCovKernel(self.coords, approx_rank=100, centering=True, standardize_cov=False)

        self.assertTrue((cov_sp1.realization() - cov_sp2.realization()).abs().mean() < 0.1)

if __name__ == "__main__":
    unittest.main()
