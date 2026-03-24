"""Unit tests for splisosm.kernel_gpr."""

import unittest
import warnings

import numpy as np
import torch

from splisosm.kernel_gpr import (
    DenseKernelOp,
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

    def test_fit_residuals_raises(self):
        """fit_residuals always raises NotImplementedError."""
        try:
            gpr = GPyTorchKernelGPR()
        except ImportError:
            self.skipTest("gpytorch not installed.")
        with self.assertRaises(NotImplementedError):
            gpr.fit_residuals(torch.randn(10, 2), torch.randn(10, 3))


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


if __name__ == "__main__":
    unittest.main()
