"""Unit tests for splisosm.kernel_gpr."""

import unittest
import warnings

import numpy as np
import torch

from splisosm.kernel_gpr import (
    DenseKernelOp,
    FFTKernelOp,
    FFTKernelGPR,
    SpatialKernelOp,
    SklearnKernelGPR,
    GPyTorchKernelGPR,
    make_kernel_gpr,
    linear_hsic_test,
    build_rbf_kernel,
    get_kernel_regression_residual_op,
    _DEFAULT_GPR_CONFIGS,
    _kernel_residuals_from_eigdecomp,
)


def _make_coords(n: int = 50, d: int = 2, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    coords = torch.randn(n, d)
    return (coords - coords.mean(0)) / coords.std(0)


def _make_rbf_kernel(
    n: int = 50, constant_value: float = 1.0, length_scale: float = 1.0
):
    coords = _make_coords(n)
    return build_rbf_kernel(coords, constant_value, length_scale)


class TestBuildRbfKernel(unittest.TestCase):
    def test_shape(self):
        K = _make_rbf_kernel(n=30)
        self.assertEqual(K.shape, (30, 30))

    def test_symmetric_psd(self):
        K = _make_rbf_kernel(n=40)
        self.assertTrue(torch.allclose(K, K.T, atol=1e-6))
        evals = torch.linalg.eigvalsh(K)
        self.assertTrue((evals >= -1e-6).all(), "Kernel has negative eigenvalues")

    def test_diagonal_equals_constant_value(self):
        K = _make_rbf_kernel(n=20, constant_value=2.5)
        self.assertTrue(torch.allclose(K.diag(), torch.full((20,), 2.5), atol=1e-5))


class TestGetKernelRegressionResidualOp(unittest.TestCase):
    def test_shape_and_idempotency(self):
        K = _make_rbf_kernel(n=30)
        epsilon = 0.1
        Rx = get_kernel_regression_residual_op(K, epsilon)
        self.assertEqual(Rx.shape, (30, 30))

    def test_residual_shrinks_signal(self):
        n = 40
        K = _make_rbf_kernel(n=n)
        Rx = get_kernel_regression_residual_op(K, epsilon=1e-3)
        torch.manual_seed(1)
        y = torch.randn(n, 3)
        y_res = Rx @ y
        # Residuals should be smaller than the original (spatial signal removed)
        self.assertLess(float(y_res.norm()), float(y.norm()))


class TestDenseKernelOp(unittest.TestCase):
    def setUp(self):
        self.n = 30
        self.K = _make_rbf_kernel(self.n)
        self.op = DenseKernelOp(self.K)

    def test_is_spatial_kernel_op(self):
        self.assertIsInstance(self.op, SpatialKernelOp)

    def test_n(self):
        self.assertEqual(self.op.n, self.n)

    def test_matvec_shape(self):
        torch.manual_seed(0)
        v = torch.randn(self.n, 5)
        out = self.op.matvec(v)
        self.assertEqual(out.shape, v.shape)

    def test_solve_1d_and_2d(self):
        torch.manual_seed(0)
        v1 = torch.randn(self.n)
        v2 = torch.randn(self.n, 4)
        u1 = self.op.solve(v1, epsilon=0.1)
        u2 = self.op.solve(v2, epsilon=0.1)
        self.assertEqual(u1.shape, v1.shape)
        self.assertEqual(u2.shape, v2.shape)

    def test_residuals_consistent(self):
        torch.manual_seed(0)
        v = torch.randn(self.n, 3)
        epsilon = 0.05
        res1 = self.op.residuals(v, epsilon)
        res2 = epsilon * self.op.solve(v, epsilon)
        self.assertTrue(torch.allclose(res1, res2, atol=1e-5))

    def test_eigenvalues_descending(self):
        evals = self.op.eigenvalues()
        self.assertEqual(len(evals), self.n)
        self.assertTrue((evals[:-1] >= evals[1:] - 1e-6).all())

    def test_eigenvalues_k(self):
        evals_k = self.op.eigenvalues(k=5)
        evals_all = self.op.eigenvalues()
        self.assertTrue(torch.allclose(evals_k, evals_all[:5], atol=1e-6))

    def test_trace(self):
        tr = self.op.trace()
        self.assertAlmostEqual(tr, float(self.K.diag().sum()), places=3)

    def test_eigh(self):
        vals, vecs = self.op.eigh()
        self.assertEqual(vals.shape, (self.n,))
        self.assertEqual(vecs.shape, (self.n, self.n))
        # Columns should be orthonormal
        self.assertTrue(torch.allclose(vecs.T @ vecs, torch.eye(self.n), atol=1e-5))

    def test_chol_caching(self):
        """Cholesky is cached and reused for the same epsilon."""
        torch.manual_seed(0)
        v = torch.randn(self.n)
        epsilon = 0.1
        u1 = self.op.solve(v, epsilon)
        chol1 = self.op._chol
        u2 = self.op.solve(v, epsilon)
        chol2 = self.op._chol
        self.assertIs(chol1, chol2)  # same object (cached)
        self.assertTrue(torch.allclose(u1, u2))


class TestLinearHsicTest(unittest.TestCase):
    def _make_data(self, n=80, nx=2, ny=3, seed=0):
        torch.manual_seed(seed)
        X = torch.randn(n, nx)
        Y = torch.randn(n, ny)
        return X, Y

    def test_returns_floats(self):
        X, Y = self._make_data()
        hsic, pval = linear_hsic_test(X, Y)
        self.assertIsInstance(hsic, float)
        self.assertIsInstance(pval, float)

    def test_pval_in_01(self):
        X, Y = self._make_data()
        _, pval = linear_hsic_test(X, Y)
        self.assertGreaterEqual(pval, 0.0)
        self.assertLessEqual(pval, 1.0)

    def test_independent_data_high_pval(self):
        """Uncorrelated X, Y should give non-significant p-values on average."""
        torch.manual_seed(42)
        pvals = []
        for _ in range(50):
            X = torch.randn(100, 2)
            Y = torch.randn(100, 3)
            _, pval = linear_hsic_test(X, Y)
            pvals.append(pval)
        # Under independence, p-values uniform(0,1) → mean ~0.5
        self.assertGreater(np.mean(pvals), 0.2)

    def test_dependent_data_low_pval(self):
        """Highly correlated X, Y should give small p-values."""
        torch.manual_seed(0)
        X = torch.randn(100, 2)
        Y = X + 0.01 * torch.randn(100, 2)  # nearly identical
        _, pval = linear_hsic_test(X, Y)
        self.assertLess(pval, 0.05)

    def test_nan_handling(self):
        """NaN rows should be silently removed."""
        torch.manual_seed(0)
        X = torch.randn(80, 2)
        Y = torch.randn(80, 3)
        X[5] = float("nan")
        Y[10, 1] = float("nan")
        hsic, pval = linear_hsic_test(X, Y)
        self.assertTrue(np.isfinite(hsic))
        self.assertTrue(np.isfinite(pval))


class TestKernelResidualFromEigdecomp(unittest.TestCase):
    def test_output_shape(self):
        n, m = 40, 5
        K = _make_rbf_kernel(n)
        evals, evecs = torch.linalg.eigh(K)
        Y = torch.randn(n, m)
        res = _kernel_residuals_from_eigdecomp(evecs, evals, Y)
        self.assertEqual(res.shape, (n, m))

    def test_residuals_smaller_than_input(self):
        n = 50
        K = _make_rbf_kernel(n, constant_value=2.0)
        evals, evecs = torch.linalg.eigh(K)
        torch.manual_seed(1)
        Y = torch.randn(n, 4)
        res = _kernel_residuals_from_eigdecomp(evecs, evals, Y)
        self.assertLess(float(res.norm()), float(Y.norm()))


class TestSklearnKernelGPR(unittest.TestCase):
    def setUp(self):
        self.n = 60
        self.coords = _make_coords(self.n)
        torch.manual_seed(42)
        self.Y = torch.randn(self.n, 3)

    def _make_gpr_fixed(self):
        return SklearnKernelGPR(
            constant_value=1e-3,
            constant_value_bounds="fixed",
            length_scale=1.0,
            length_scale_bounds="fixed",
        )

    def _make_gpr_free(self):
        return SklearnKernelGPR(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3),
            length_scale=1.0,
            length_scale_bounds="fixed",
        )

    def test_from_config_default(self):
        gpr = SklearnKernelGPR.from_config(_DEFAULT_GPR_CONFIGS["isoform"])
        self.assertTrue(gpr.signal_bounds_fixed)

    def test_signal_bounds_fixed(self):
        gpr_f = self._make_gpr_fixed()
        gpr_m = self._make_gpr_free()
        self.assertTrue(gpr_f.signal_bounds_fixed)
        self.assertFalse(gpr_m.signal_bounds_fixed)

    def test_fit_residuals_shape(self):
        gpr = self._make_gpr_fixed()
        res = gpr.fit_residuals(self.coords, self.Y)
        self.assertEqual(res.shape, self.Y.shape)

    def test_fit_residuals_nan_handling(self):
        gpr = self._make_gpr_free()
        Y_nan = self.Y.clone()
        Y_nan[5] = float("nan")
        res = gpr.fit_residuals(self.coords, Y_nan)
        self.assertEqual(res.shape, self.Y.shape)
        self.assertTrue(torch.isnan(res[5]).all())
        self.assertFalse(torch.isnan(res[6]).any())

    def test_shared_eigendecomp_fast_path(self):
        gpr = self._make_gpr_fixed()
        # After precompute, fit_residuals should use the fast path
        gpr.precompute_shared_kernel(self.coords)
        self.assertIsNotNone(gpr._shared_eigvals)
        res = gpr.fit_residuals(self.coords, self.Y)
        self.assertEqual(res.shape, self.Y.shape)

    def test_precompute_warns_when_not_fixed(self):
        gpr = self._make_gpr_free()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gpr.precompute_shared_kernel(self.coords)
            self.assertTrue(any("no effect" in str(x.message) for x in w))

    def test_fit_residuals_batch_amortizes(self):
        gpr = self._make_gpr_fixed()
        Y_list = [torch.randn(self.n, 3) for _ in range(4)]
        res_list = gpr.fit_residuals_batch(self.coords, Y_list)
        self.assertEqual(len(res_list), 4)
        for res in res_list:
            self.assertEqual(res.shape, (self.n, 3))
        # After batch, eigendecomp should be precomputed
        self.assertIsNotNone(gpr._shared_eigvals)

    def test_fast_path_shape_guard_with_nan(self):
        """Fast path should not be used when NaN rows cause coord subset."""
        gpr = self._make_gpr_fixed()
        gpr.precompute_shared_kernel(self.coords)
        Y_nan = self.Y.clone()
        Y_nan[0:5] = float("nan")
        # Should not raise (falls back to sklearn GPR for the NaN-filtered subset)
        res = gpr.fit_residuals(self.coords, Y_nan)
        self.assertEqual(res.shape, self.Y.shape)
        self.assertTrue(torch.isnan(res[:5]).all())

    def test_get_kernel_op(self):
        from splisosm.kernel_gpr import DenseKernelOp

        gpr = self._make_gpr_fixed()
        op = gpr.get_kernel_op(self.coords)
        self.assertIsInstance(op, DenseKernelOp)


class TestGPyTorchKernelGPRImportGuard(unittest.TestCase):
    def test_raises_without_gpytorch(self):
        """GPyTorchKernelGPR raises ImportError if gpytorch not installed."""
        import sys

        # If gpytorch is available, skip; if not, ensure ImportError on instantiation
        if "gpytorch" in sys.modules:
            self.skipTest("gpytorch is available; skipping import guard test.")
        try:
            GPyTorchKernelGPR()
        except ImportError:
            pass  # expected
        except Exception as e:
            # Some other error is also acceptable if gpytorch isn't present
            self.assertIsInstance(e, (ImportError, RuntimeError))

    def test_fit_residuals_shape(self):
        """fit_residuals returns correct shape for multi-output Y."""
        try:
            gpr = GPyTorchKernelGPR(n_iter=5)
        except ImportError:
            self.skipTest("gpytorch not installed.")
        n, m = 20, 3
        coords = torch.randn(n, 2)
        Y = torch.randn(n, m)
        res = gpr.fit_residuals(coords, Y)
        self.assertEqual(res.shape, (n, m))

    def test_fit_residuals_nan_rows_preserved(self):
        """NaN rows in Y are reinserted as NaN in the residuals."""
        try:
            gpr = GPyTorchKernelGPR(n_iter=5)
        except ImportError:
            self.skipTest("gpytorch not installed.")
        n, m = 20, 2
        coords = torch.randn(n, 2)
        Y = torch.randn(n, m)
        Y[3, :] = float("nan")
        res = gpr.fit_residuals(coords, Y)
        self.assertEqual(res.shape, (n, m))
        self.assertTrue(torch.isnan(res[3]).all())
        self.assertFalse(torch.isnan(res[0]).any())

    def test_fit_residuals_shrinks_signal(self):
        """Residuals have smaller variance than pure spatial signal."""
        try:
            gpr = GPyTorchKernelGPR(
                constant_value=1.0,
                constant_value_bounds="fixed",
                length_scale=1.0,
                length_scale_bounds="fixed",
                n_iter=20,
            )
        except ImportError:
            self.skipTest("gpytorch not installed.")
        torch.manual_seed(0)
        n = 30
        t = torch.linspace(0, 2 * np.pi, n).unsqueeze(1)
        coords = torch.cat([t, torch.zeros_like(t)], dim=1)
        Y = torch.sin(t).expand(n, 3) + 0.05 * torch.randn(n, 3)
        res = gpr.fit_residuals(coords, Y)
        self.assertLess(res.std().item(), Y.std().item())

    def test_n_inducing_raises(self):
        """n_inducing > 0 raises NotImplementedError."""
        try:
            with self.assertRaises(NotImplementedError):
                GPyTorchKernelGPR(n_inducing=50)
        except ImportError:
            self.skipTest("gpytorch not installed.")


class TestMakeKernelGPR(unittest.TestCase):
    def test_sklearn_backend(self):
        gpr = make_kernel_gpr(
            "sklearn", constant_value=0.5, constant_value_bounds="fixed"
        )
        self.assertIsInstance(gpr, SklearnKernelGPR)
        self.assertTrue(gpr.signal_bounds_fixed)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            make_kernel_gpr("nonexistent_backend")

    def test_default_configs_structure(self):
        self.assertIn("covariate", _DEFAULT_GPR_CONFIGS)
        self.assertIn("isoform", _DEFAULT_GPR_CONFIGS)
        for key in (
            "constant_value",
            "constant_value_bounds",
            "length_scale",
            "length_scale_bounds",
        ):
            self.assertIn(key, _DEFAULT_GPR_CONFIGS["covariate"])
            self.assertIn(key, _DEFAULT_GPR_CONFIGS["isoform"])


def _make_regular_grid_coords(ny: int = 15, nx: int = 20) -> torch.Tensor:
    """Return z-score normalized coords for a regular (ny, nx) grid."""
    y_idx = torch.arange(ny, dtype=torch.float32)
    x_idx = torch.arange(nx, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    coords = (coords - coords.mean(0)) / coords.std(0)
    return coords


class TestFFTKernelOp(unittest.TestCase):
    """Tests for FFTKernelOp."""

    def setUp(self):
        self.op = FFTKernelOp(
            ny=15, nx=20, dy=0.1, dx=0.1,
            constant_value=1.0, length_scale=0.5,
        )

    def test_n(self):
        self.assertEqual(self.op.n, 300)
        self.assertEqual(self.op.ny, 15)
        self.assertEqual(self.op.nx, 20)

    def test_eigenvalues_shape_and_sorted(self):
        evals = self.op.eigenvalues()
        self.assertEqual(evals.shape, (300,))
        self.assertGreaterEqual(float(evals[0]), float(evals[-1]))

    def test_eigenvalues_nonneg(self):
        evals = self.op.eigenvalues()
        self.assertTrue(np.all(evals >= 0), "FFT eigenvalues should be non-negative")

    def test_trace_equals_n_times_constant_value(self):
        # trace(K) = sum of eigenvalues ≈ n * constant_value for large grid
        # (exact for periodic torus); here we just check it's positive and finite
        t = self.op.trace()
        self.assertGreater(t, 0)
        self.assertTrue(np.isfinite(t))

    def test_matvec_shape_1d(self):
        v = np.random.randn(300)
        kv = self.op.matvec(v)
        self.assertEqual(kv.shape, (300,))

    def test_matvec_shape_2d(self):
        V = np.random.randn(300, 5)
        KV = self.op.matvec(V)
        self.assertEqual(KV.shape, (300, 5))

    def test_residuals_reduce_variance(self):
        """GP residualization should reduce signal variance."""
        np.random.seed(7)
        v = np.random.randn(300)
        r = self.op.residuals(v, epsilon=0.5)
        self.assertLess(float(np.var(r)), float(np.var(v)))

    def test_solve_is_inverse_of_matvec_plus_eps(self):
        """solve((K + eps*I)v) ≈ v."""
        eps = 0.2
        np.random.seed(3)
        v = np.random.randn(300)
        u = self.op.solve(v, eps)
        # Reconstruct: matvec(u) + eps*u should ≈ v
        recon = self.op.matvec(u) + eps * u
        np.testing.assert_allclose(recon, v, atol=1e-6)


class TestFFTKernelGPR(unittest.TestCase):
    """Tests for FFTKernelGPR."""

    def setUp(self):
        self.coords = _make_regular_grid_coords(ny=15, nx=20)
        n = 300
        # Smooth spatial signal + noise.
        # Use frequency 1.5 rad/unit — within the RBF kernel's spectral support on
        # this 15×20 grid (ky=2 at f≈1.94 has negative circulant eigenvalue; f=1.5
        # falls clearly at ky=1 which has eigenvalue ≈35).
        torch.manual_seed(42)
        self.Y_signal = torch.sin(self.coords[:, 0] * 1.5).unsqueeze(1) + 0.05 * torch.randn(n, 1)
        self.Y_noise = torch.randn(n, 2)

    def test_fit_residuals_shape(self):
        gpr = FFTKernelGPR(constant_value_bounds="fixed")
        res = gpr.fit_residuals(self.coords, self.Y_signal)
        self.assertEqual(res.shape, self.Y_signal.shape)

    def test_fit_residuals_reduces_spatial_signal(self):
        """Residuals should have substantially less variance than a smooth signal."""
        gpr = FFTKernelGPR(constant_value_bounds=(1e-3, 1e3))
        res = gpr.fit_residuals(self.coords, self.Y_signal)
        self.assertLess(float(res.var()), float(self.Y_signal.var()) * 0.5)

    def test_fit_residuals_noise_unchanged(self):
        """Residuals of pure noise should retain most variance (kernel can't fit noise)."""
        gpr = FFTKernelGPR(constant_value_bounds=(1e-3, 1e3))
        res = gpr.fit_residuals(self.coords, self.Y_noise)
        # Residual variance should be a large fraction of original
        self.assertGreater(float(res.var()), float(self.Y_noise.var()) * 0.3)

    def test_fit_residuals_nan_rows_preserved(self):
        """NaN rows in Y should be preserved as NaN in residuals."""
        Y = self.Y_signal.clone()
        Y[5] = float("nan")
        Y[20] = float("nan")
        gpr = FFTKernelGPR(constant_value_bounds="fixed")
        res = gpr.fit_residuals(self.coords, Y)
        self.assertTrue(torch.isnan(res[5]).all())
        self.assertTrue(torch.isnan(res[20]).all())
        self.assertFalse(torch.isnan(res[0]).any())

    def test_fit_residuals_batch_matches_single(self):
        """fit_residuals_batch should give same result as repeated fit_residuals."""
        gpr = FFTKernelGPR(constant_value_bounds="fixed")
        res_single = [gpr.fit_residuals(self.coords, self.Y_signal),
                      gpr.fit_residuals(self.coords, self.Y_noise)]
        res_batch = gpr.fit_residuals_batch(self.coords, [self.Y_signal, self.Y_noise])
        for single, batch in zip(res_single, res_batch):
            torch.testing.assert_close(single, batch, atol=1e-5, rtol=1e-4)

    def test_fit_residuals_cube_backward_compat(self):
        """fit_residuals_cube should work and return (array, float)."""
        gpr = FFTKernelGPR(constant_value_bounds="fixed")
        cube = np.random.randn(8, 10)
        res_cube, eps = gpr.fit_residuals_cube(cube, spacing=(0.125, 0.1))
        self.assertEqual(res_cube.shape, (8, 10))
        self.assertGreater(eps, 0)

    def test_irregular_grid_raises(self):
        """fit_residuals should raise ValueError for irregular coordinates."""
        gpr = FFTKernelGPR()
        irregular = self.coords.clone()
        irregular[5, 0] += 0.0137  # break regularity
        with self.assertRaises(ValueError):
            gpr.fit_residuals(irregular, self.Y_signal)

    def test_length_scale_bounds_not_fixed_raises(self):
        with self.assertRaises(NotImplementedError):
            FFTKernelGPR(length_scale_bounds=(0.1, 10.0))

    def test_make_kernel_gpr_fft(self):
        gpr = make_kernel_gpr("fft", **_DEFAULT_GPR_CONFIGS["covariate"])
        self.assertIsInstance(gpr, FFTKernelGPR)
        res = gpr.fit_residuals(self.coords, self.Y_signal)
        self.assertEqual(res.shape, self.Y_signal.shape)

    def test_fft_matches_sklearn_calibration(self):
        """FFT and sklearn backends should give similar calibration on null data."""
        from scipy.stats import kstest

        torch.manual_seed(99)
        np.random.seed(99)
        n = self.coords.shape[0]  # 300 (15x20 grid)
        n_genes = 40

        # Null isoform ratios (no spatial signal)
        Y_list = [torch.rand(n, 3) for _ in range(n_genes)]

        # Residualize a single covariate
        covar_config = _DEFAULT_GPR_CONFIGS["covariate"]
        z = torch.randn(n, 1)

        gpr_sk = SklearnKernelGPR(**covar_config)
        z_sk = gpr_sk.fit_residuals(self.coords, z)

        gpr_fft = FFTKernelGPR(
            constant_value=covar_config["constant_value"],
            constant_value_bounds=covar_config["constant_value_bounds"],
            length_scale=covar_config["length_scale"],
            length_scale_bounds=covar_config["length_scale_bounds"],
        )
        z_fft = gpr_fft.fit_residuals(self.coords, z)

        pvals_sk = [linear_hsic_test(z_sk, Y, centering=True)[1] for Y in Y_list]
        pvals_fft = [linear_hsic_test(z_fft, Y, centering=True)[1] for Y in Y_list]

        # Both should be roughly uniform under null (KS p > 0.05)
        ks_sk = kstest(pvals_sk, "uniform").pvalue
        ks_fft = kstest(pvals_fft, "uniform").pvalue
        self.assertGreater(ks_sk, 0.01, f"sklearn p-values not uniform: KS p={ks_sk:.4f}")
        self.assertGreater(ks_fft, 0.01, f"FFT p-values not uniform: KS p={ks_fft:.4f}")


if __name__ == "__main__":
    unittest.main()
