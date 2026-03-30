import unittest
import numpy as np
import torch
import itertools
from splisosm.kernel import SpatialCovKernel, _build_adj_from_coords
from splisosm.utils import get_cov_sp


class TestSpatialCovKernel(unittest.TestCase):

    def setUp(self):
        # Small 30x30 grid for dense tests
        x = np.linspace(0, 1, 30)
        y = np.linspace(0, 1, 30)
        self.n_spots = 900
        self.coords = np.array(list(itertools.product(x, y)))

        # Very small grid for fast exact checks
        x_small = np.linspace(0, 1, 5)
        y_small = np.linspace(0, 1, 5)
        self.n_small = 25
        self.coords_small = np.array(list(itertools.product(x_small, y_small)))

    def test_from_coordinates(self):
        # Factory method
        K = SpatialCovKernel.from_coordinates(
            self.coords_small, centering=True, standardize_cov=True
        )
        self.assertEqual(K.shape(), (self.n_small, self.n_small))
        self.assertEqual(K.rank(), self.n_small)

        # Positional constructor (coords path)
        K2 = SpatialCovKernel(self.coords_small, centering=True, standardize_cov=True)
        self.assertEqual(K2.shape(), (self.n_small, self.n_small))

        # realization() is symmetric
        R = K.realization()
        self.assertTrue(torch.allclose(R, R.T, atol=1e-5))

        # eigenvalues are positive (skip near-zero)
        eigvals = K.eigenvalues()
        self.assertTrue((eigvals[eigvals > 1e-5] > 0).all())

        # eigenvalues are sorted in descending order
        self.assertTrue((eigvals[:-1] >= eigvals[1:]).all())

    def test_from_adjacency(self):
        # Build adjacency matrix directly from coords
        W = _build_adj_from_coords(self.coords_small, k_neighbors=4)
        K = SpatialCovKernel.from_adjacency(
            W, rho=0.99, standardize_cov=True, centering=False
        )
        self.assertEqual(K.shape(), (self.n_small, self.n_small))
        R = K.realization()
        self.assertTrue(torch.allclose(R, R.T, atol=1e-5))

        # adj_matrix keyword path via constructor should produce same result
        K2 = SpatialCovKernel(
            adj_matrix=W, rho=0.99, standardize_cov=True, centering=False
        )
        self.assertTrue(torch.allclose(R, K2.realization(), atol=1e-5))

        # Should match from_coordinates with the same params
        K3 = SpatialCovKernel.from_coordinates(
            self.coords_small,
            k_neighbors=4,
            rho=0.99,
            standardize_cov=True,
            centering=False,
        )
        self.assertTrue(torch.allclose(R, K3.realization(), atol=1e-5))

    def test_highd_coords(self):
        # coords can be PC embeddings with > 2 dimensions
        coords_10d = np.random.rand(self.n_small, 10)
        K = SpatialCovKernel.from_coordinates(coords_10d, centering=True)
        self.assertEqual(K.shape(), (self.n_small, self.n_small))
        R = K.realization()
        self.assertTrue(torch.allclose(R, R.T, atol=1e-5))

    def test_trace_and_square_trace(self):
        K = SpatialCovKernel.from_coordinates(
            self.coords_small, centering=True, standardize_cov=True
        )
        R = K.realization()

        expected_trace = torch.trace(R)
        expected_sq_trace = R.pow(2).sum()

        self.assertTrue(torch.allclose(K.trace(), expected_trace, atol=1e-4))
        self.assertTrue(torch.allclose(K.square_trace(), expected_sq_trace, atol=1e-3))

    def test_xtKx(self):
        K = SpatialCovKernel.from_coordinates(
            self.coords_small, centering=True, standardize_cov=True
        )
        R = K.realization()

        torch.manual_seed(0)
        y = torch.rand(self.n_small, 3)

        via_xtKx = torch.trace(K.xtKx(y))
        manual = torch.trace(y.T @ R @ y)
        self.assertTrue(torch.allclose(via_xtKx, manual, atol=1e-4))

    def test_eigenvalues_partial(self):
        K = SpatialCovKernel.from_coordinates(
            self.coords, centering=True, standardize_cov=True
        )
        k = 10
        eigvals_partial = K.eigenvalues(k=k)
        self.assertEqual(len(eigvals_partial), k)

        eigvals_all = K.eigenvalues()
        self.assertTrue(torch.allclose(eigvals_partial, eigvals_all[:k], atol=1e-5))

    def test_lazy_low_rank(self):
        K = SpatialCovKernel.from_coordinates(
            self.coords_small, centering=True, standardize_cov=True
        )
        # Q should not be set before eigendecomposition
        self.assertIsNone(K.Q)

        # Trigger eigendecomposition
        K.eigenvalues(k=10)

        # Q should now be cached
        self.assertIsNotNone(K.Q)

        # xtKx should still give correct result using cached Q
        torch.manual_seed(0)
        y = torch.rand(self.n_small, 3)
        R = K.realization()
        via_xtKx = torch.trace(K.xtKx(y))
        manual = torch.trace(y.T @ R @ y)
        self.assertTrue(torch.allclose(via_xtKx, manual, atol=1e-3))

    def test_get_cov_sp(self):
        # utils.get_cov_sp should return a dense symmetric matrix
        cov = get_cov_sp(self.coords_small)
        self.assertEqual(cov.shape, (self.n_small, self.n_small))
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
