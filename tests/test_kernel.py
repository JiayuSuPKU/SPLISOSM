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
        # --- Dense mode: full-rank Q → xtKx == x^T K x exactly ---
        K = SpatialCovKernel.from_coordinates(
            self.coords_small, centering=True, standardize_cov=True
        )
        self.assertIsNone(K.Q)

        K.eigenvalues(k=10)  # triggers eigendecomposition → full-rank Q (n×n)
        self.assertIsNotNone(K.Q)
        self.assertEqual(K.Q.shape[1], self.n_small)  # full rank

        torch.manual_seed(0)
        y = torch.rand(self.n_small, 3)
        R = K.realization()
        # xtKx uses full-rank Q → equivalent to x^T K x exactly
        via_xtKx = torch.trace(K.xtKx(y))
        manual = torch.trace(y.T @ R @ y)
        self.assertTrue(torch.allclose(via_xtKx, manual, atol=1e-3))

        # --- Consistency: xtKx(y) == trace((y^T Q)(y^T Q)^T) always ---
        xtQ = y.t() @ K.Q
        via_Q = torch.trace(xtQ @ xtQ.t())
        self.assertTrue(torch.allclose(via_xtKx, via_Q, atol=1e-5))

    def test_xtKx_consistency_with_Q_implicit(self):
        """xtKx and Q @ Q.T must agree in implicit mode after eigenvalues() is called.

        This is the regression test for the discrepancy reported by users:
            torch.trace(K.xtKx(x)) != torch.trace((x.T @ K.Q) @ (x.T @ K.Q).T)
        The fix: xtKx uses Q (rank-k) when Q is available, for both modes.
        """
        rng = np.random.default_rng(1)
        n_impl = 5002
        coords_impl = rng.uniform(0, 10, size=(n_impl, 2))
        K = SpatialCovKernel.from_coordinates(
            coords_impl, k_neighbors=4, centering=True, standardize_cov=True
        )
        self.assertIsNone(K.K_sp, "Expected implicit mode for n > DENSE_THRESHOLD")

        # Before eigenvalues(): Q is None, xtKx uses exact LU solve.
        self.assertIsNone(K.Q)
        torch.manual_seed(0)
        d = 5
        x = torch.randn(n_impl, d)
        exact = torch.trace(K.xtKx(x))  # exact via LU solve

        # After eigenvalues(k=20): Q is rank-20; xtKx must use Q.
        K.eigenvalues(k=20)
        self.assertIsNotNone(K.Q)
        self.assertEqual(K.Q.shape[1], 20)

        via_xtKx = torch.trace(K.xtKx(x))
        xtQ = x.t() @ K.Q
        via_Q = torch.trace(xtQ @ xtQ.t())

        # xtKx and Q @ Q.T must now agree (core consistency guarantee)
        self.assertAlmostEqual(
            via_xtKx.item(),
            via_Q.item(),
            places=3,
            msg="xtKx and Q@Q.T disagree in implicit mode",
        )

        # Rank-20 approximation is necessarily ≤ the exact (full-K) result
        # (top-k eigenvalues capture less than the full variance)
        self.assertLess(
            via_xtKx.item(),
            exact.item() * 1.01,
            "Rank-k approx should not exceed full-K result",
        )

    def test_get_cov_sp(self):
        # utils.get_cov_sp should return a dense symmetric matrix
        cov = get_cov_sp(self.coords_small)
        self.assertEqual(cov.shape, (self.n_small, self.n_small))
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))

    def test_eigenvalues_cache_reuse_dense(self):
        """Dense mode: cached full decomp is always sufficient; k > n is clipped at n."""
        K = SpatialCovKernel.from_coordinates(
            self.coords_small, centering=True, standardize_cov=True
        )
        # Trigger full decomp with a small k — dense path always stores all n eigenvalues.
        K.eigenvalues(k=5)
        self.assertEqual(len(K.K_eigvals), self.n_small)

        # Requesting k > n should return all n (clipped), not raise or recompute.
        eigvals_over = K.eigenvalues(k=self.n_small + 10)
        self.assertEqual(len(eigvals_over), self.n_small)
        self.assertTrue(torch.all(eigvals_over[:-1] >= eigvals_over[1:]))  # descending

    def test_eigenvalues_cache_recompute_implicit(self):
        """Implicit mode: calling eigenvalues(k_new) with k_new > cached k reruns eigsh.

        Uses n > DENSE_THRESHOLD to force the implicit (LU-solve) code path.
        """
        # Build a minimal implicit-mode kernel (n > 5000)
        rng = np.random.default_rng(0)
        n_impl = 5002
        coords_impl = rng.uniform(0, 10, size=(n_impl, 2))
        K = SpatialCovKernel.from_coordinates(
            coords_impl, k_neighbors=4, centering=True, standardize_cov=True
        )
        self.assertIsNone(K.K_sp, "Expected implicit (LU) mode for n > DENSE_THRESHOLD")

        # First call: cache k=5 eigenvalues
        ev5 = K.eigenvalues(k=5)
        self.assertEqual(len(ev5), 5)
        self.assertEqual(len(K.K_eigvals), 5)
        self.assertEqual(K.Q.shape[1], 5)

        # Second call with k=15: must recompute, not silently return only 5
        ev15 = K.eigenvalues(k=15)
        self.assertEqual(len(ev15), 15, "Should return 15 eigenvalues after recompute")
        self.assertEqual(
            len(K.K_eigvals), 15, "K_eigvals cache should be updated to 15"
        )
        self.assertEqual(K.Q.shape[1], 15, "Q factor should be updated to rank 15")

        # The leading 5 eigenvalues should be consistent before/after recompute
        self.assertTrue(
            torch.allclose(ev5, ev15[:5], atol=1e-2),
            f"Leading eigenvalues changed unexpectedly after recompute:\n{ev5}\nvs\n{ev15[:5]}",
        )

        # All eigenvalues should be positive (real symmetric PSD kernel)
        self.assertTrue(torch.all(ev15 > 0))

        # eigenvalues must remain in descending order
        self.assertTrue(torch.all(ev15[:-1] >= ev15[1:]))


if __name__ == "__main__":
    unittest.main()
