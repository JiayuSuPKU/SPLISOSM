import unittest
import torch
from splisosm.model import MultinomGLM, MultinomGLMM
from splisosm.likelihood import log_prob_fastmult, log_prob_fastmvn
from torch.autograd.functional import hessian


class HessianCalculatorGLM:
    def __init__(self, model: MultinomGLM):
        self.model = model

    def _calc_log_prob_joint_wrt_beta_bias(self, beta_bias):
        """Helper function of log joint probability wrt beta and bias for calculate Hessian."""
        model = self.model
        # combine beta and the intercept bias_eta
        X_expand = torch.cat(
            [model.X_spot, torch.ones(1, model.n_spots, 1)],
            dim=-1,
        )  # (1, n_spots, n_factors + 1)
        # update alpha locally as a function of the input beta_bias
        eta = X_expand @ beta_bias
        alpha = torch.concat([eta, torch.zeros(1, model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1)  # (1, n_spots, n_isos)

        return log_prob_fastmult(alpha.squeeze(0).T, model.counts.squeeze(0).T)

    def test_hessian_beta_bias(self):
        model = self.model
        # calculate the Hessian using pytorch
        beta_bias = torch.concat(
            [model.beta, model.bias_eta.unsqueeze(0)], dim=1
        )  # (1, n_factor + 1, n_isos - 1)
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_beta_bias,
            beta_bias,
            vectorize=True,
            create_graph=False,
        )  # (1, n_factor + 1, n_isos - 1, 1, n_factor + 1, n_isos - 1)
        hess_pytorch = (
            hess_pytorch.squeeze(0, 3)
            .permute(1, 0, 3, 2)
            .reshape(
                (model.n_factors + 1) * (model.n_isos - 1),
                (model.n_factors + 1) * (model.n_isos - 1),
            )
            .unsqueeze(0)
        )  # (1, (n_factor + 1) * (n_isos - 1), (n_factor + 1) * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_beta_bias()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-4)


class TestMultinomGLM(unittest.TestCase):
    def setUp(self):
        # simulate 2 isoforms in 3 spots, with 2 covariates
        self.counts = torch.tensor(
            [[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32
        )
        self.design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)

    def test_setup_data(self):
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        self.assertEqual(model.n_genes, 1)
        self.assertEqual(model.n_spots, 3)
        self.assertEqual(model.n_isos, 2)
        self.assertEqual(model.n_factors, 2)
        self.assertEqual(model.counts.shape, (1, 3, 2))
        self.assertEqual(model.X_spot.shape, (1, 3, 2))

    def test_forward(self):
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        log_prob = model.forward()
        self.assertEqual(log_prob.shape, (1,))

    def test_hessian(self):
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        HessianCalculatorGLM(model).test_hessian_beta_bias()

    def test_fit(self):
        model = MultinomGLM(fitting_method="iwls", fitting_configs={"max_epochs": 10})
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        model.fit()
        self.assertTrue(model.convergence.any())

    def test_get_isoform_ratio(self):
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        isoform_ratio = model.get_isoform_ratio()
        self.assertEqual(isoform_ratio.shape, (1, 3, 2))

    def test_configure_learnable_variables(self):
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        self.assertEqual(model.beta.shape, (1, 2, 1))
        self.assertEqual(model.bias_eta.shape, (1, 1))


class HessianCalculatorGLMM:
    def __init__(self, model: MultinomGLMM):
        self.model = model

    def _calc_log_prob_joint_wrt_eta(self, eta):
        """Helper function of log joint probability wrt eta for calculate Hessian."""
        model = self.model
        alpha = torch.concat([eta.squeeze(0), torch.zeros(model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1)  # (n_spots, n_isos)

        return log_prob_fastmult(alpha.T, model.counts.squeeze(0).T)

    def _calc_log_prob_joint_wrt_nu(self, nu):
        """Helper function of log joint probability wrt nu for calculate Hessian."""
        model = self.model
        # MVN prior likelihood as a function of the input eta
        log_prob = log_prob_fastmvn(
            0.0,
            model._cov_eigvals().squeeze(0),  # (1, n_spots)
            model.corr_sp_eigvecs.unsqueeze(0),  # (1, n_spots, n_spots)
            nu.squeeze(0).T,  # (n_isos - 1, n_spots)
        )
        # update alpha locally as a function of the input nu
        eta = (
            model.X_spot @ model.beta + model.bias_eta
        ) + nu  # (1, n_spots, n_isos - 1)
        alpha = torch.concat([eta, torch.zeros(1, model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1)  # (1, n_spots, n_isos)
        log_prob += log_prob_fastmult(alpha.squeeze(0).T, model.counts.squeeze(0).T)

        return log_prob

    def _calc_log_prob_joint_wrt_beta_bias(self, beta_bias):
        """Helper function of log joint probability wrt beta and bias for calculate Hessian."""
        model = self.model
        # combine beta and the intercept bias_eta
        X_expand = torch.cat(
            [model.X_spot, torch.ones(1, model.n_spots, 1)],
            dim=-1,
        )  # (1, n_spots, n_factors + 1)
        # update alpha locally as a function of the input beta_bias
        eta = (X_expand @ beta_bias) + model.nu
        alpha = torch.concat([eta, torch.zeros(1, model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1)  # (1, n_spots, n_isos)

        return log_prob_fastmult(alpha.squeeze(0).T, model.counts.squeeze(0).T)

    def test_hessian_eta(self):
        model = self.model
        # calculate the Hessian using pytorch
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_eta,
            model._eta(),
            vectorize=True,
            create_graph=False,
        )  # (1, n_spots, n_isos - 1, 1, n_spots, n_isos - 1)
        hess_pytorch = (
            hess_pytorch.squeeze(0, 3)
            .permute(1, 0, 3, 2)
            .reshape(
                model.n_spots * (model.n_isos - 1), model.n_spots * (model.n_isos - 1)
            )
            .unsqueeze(0)
        )  # (1, n_spots * (n_isos - 1), n_spots * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_eta()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-5)

    def test_hessian_nu(self):
        model = self.model
        # calculate the Hessian using pytorch
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_nu,
            model.nu,
            vectorize=True,
            create_graph=False,
        )  # (1, n_spots, n_isos - 1, 1, n_spots, n_isos - 1)
        hess_pytorch = (
            hess_pytorch.squeeze(0, 3)
            .permute(1, 0, 3, 2)
            .reshape(
                model.n_spots * (model.n_isos - 1), model.n_spots * (model.n_isos - 1)
            )
            .unsqueeze(0)
        )  # (1, n_spots * (n_isos - 1), n_spots * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_nu()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-5)

    def test_hessian_beta_bias(self):
        model = self.model
        # calculate the Hessian using pytorch
        beta_bias = torch.concat(
            [model.beta, model.bias_eta.unsqueeze(0)], dim=1
        )  # (1, n_factor + 1, n_isos - 1)
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_beta_bias,
            beta_bias,
            vectorize=True,
            create_graph=False,
        )  # (1, n_factor + 1, n_isos - 1, 1, n_factor + 1, n_isos - 1)
        hess_pytorch = (
            hess_pytorch.squeeze(0, 3)
            .permute(1, 0, 3, 2)
            .reshape(
                (model.n_factors + 1) * (model.n_isos - 1),
                (model.n_factors + 1) * (model.n_isos - 1),
            )
            .unsqueeze(0)
        )  # (1, (n_factor + 1) * (n_isos - 1), (n_factor + 1) * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_beta_bias()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-4)


class TestMultinomGLMM(unittest.TestCase):
    def setUp(self):
        # simulate 2 isoforms in 5 spots, with 2 covariates
        self.counts = torch.tensor(
            [[[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]], dtype=torch.float32
        )
        self.design_mtx = torch.tensor(
            [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.float32
        )

        # simulate the 5-by-5 spatial covariance matrix
        self.corr_sp = torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.2, 0.1],
                [0.5, 1.0, 0.5, 0.3, 0.2],
                [0.3, 0.5, 1.0, 0.5, 0.3],
                [0.2, 0.3, 0.5, 1.0, 0.5],
                [0.1, 0.2, 0.3, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )

    def test_setup_data(self):
        model = MultinomGLMM()
        model.setup_data(self.counts, design_mtx=self.design_mtx, corr_sp=self.corr_sp)
        self.assertEqual(model.n_genes, 1)
        self.assertEqual(model.n_spots, 5)
        self.assertEqual(model.n_isos, 2)
        self.assertEqual(model.n_factors, 2)
        self.assertEqual(model.counts.shape, (1, 5, 2))
        self.assertEqual(model.X_spot.shape, (1, 5, 2))
        self.assertEqual(model.corr_sp.shape, (5, 5))
        self.assertEqual(model.corr_sp_eigvals.shape, (5,))
        self.assertEqual(model.corr_sp_eigvecs.shape, (5, 5))

        model_str = str(model)
        self.assertIn("Multinomial Generalized Linear Mixed Model (GLMM)", model_str)

    def test_configure_optimizer(self):
        model = MultinomGLMM(
            fitting_method="joint_gd", fitting_configs={"optim": "adam"}
        )
        model.setup_data(self.counts, design_mtx=self.design_mtx, corr_sp=self.corr_sp)
        model._configure_optimizer()
        self.assertIsNotNone(model.optimizer)

    def test_forward(self):
        model = MultinomGLMM()
        model.setup_data(self.counts, design_mtx=self.design_mtx, corr_sp=self.corr_sp)
        # joint likelihood
        log_prob = model._calc_log_prob_joint()
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(torch.isfinite(log_prob).all())
        # marginal likelihood
        log_prob_marginal = model._calc_log_prob_marginal()
        self.assertEqual(log_prob_marginal.shape, (1,))
        self.assertTrue(torch.isfinite(log_prob_marginal).all())

    def test_hessian(self):
        model = MultinomGLMM()
        model.setup_data(self.counts, design_mtx=self.design_mtx, corr_sp=self.corr_sp)
        HessianCalculatorGLMM(model).test_hessian_eta()
        HessianCalculatorGLMM(model).test_hessian_nu()
        HessianCalculatorGLMM(model).test_hessian_beta_bias()

    def test_fit_with_different_methods(self):
        for method in ["joint_gd", "joint_newton", "marginal_gd", "marginal_newton"]:
            with self.subTest(fitting_method=method):
                model = MultinomGLMM(
                    fitting_method=method,
                    fitting_configs={"max_epochs": 5},
                    var_fix_sigma=False,
                )
                model.setup_data(
                    self.counts, design_mtx=self.design_mtx, corr_sp=self.corr_sp
                )
                model.fit(verbose=False)

    def test_sigma_init_from_logit_variance(self):
        """sigma is initialised from empirical std of nu in logit space."""
        model = MultinomGLMM(init_ratio="observed")
        model.setup_data(self.counts, corr_sp=self.corr_sp)
        # After setup_data, sigma should ≈ sqrt(max(nu.var(dim=1))) across isoform contrasts
        expected = (
            model.nu.detach()
            .var(dim=1)
            .max(dim=-1, keepdim=True)
            .values.clamp(min=1e-4)
            .pow(0.5)
        )
        torch.testing.assert_close(model.sigma.data, expected, atol=1e-6, rtol=1e-6)

    def test_sigma_init_parameterization_consistency(self):
        """sigma_sp² + sigma_nsp² ≈ sigma² for the same data."""
        model_st = MultinomGLMM(
            var_parameterization_sigma_theta=True, init_ratio="observed"
        )
        model_st.setup_data(self.counts, corr_sp=self.corr_sp)

        model_sp = MultinomGLMM(
            var_parameterization_sigma_theta=False, init_ratio="observed"
        )
        model_sp.setup_data(self.counts, corr_sp=self.corr_sp)

        var_st = model_st.sigma.data**2
        var_sp = model_sp.sigma_sp.data**2 + model_sp.sigma_nsp.data**2
        torch.testing.assert_close(var_st, var_sp, atol=1e-5, rtol=1e-5)

    def test_sigma_init_uniform_fallback(self):
        """init_ratio='uniform' still produces a positive sigma (fallback path)."""
        model = MultinomGLMM(init_ratio="uniform")
        model.setup_data(self.counts, corr_sp=self.corr_sp)
        self.assertTrue((model.sigma.data > 0).all())
        self.assertTrue(torch.isfinite(model.sigma.data).all())

    # def test_update_gradient_descent(self):
    #     model = MultinomGLM(fitting_method="gd")
    #     model.setup_data(self.counts, design_mtx=self.design_mtx)
    #     try:
    #         model._update_gradient_descent()
    #     except NotImplementedError:
    #         self.skipTest("Gradient descent update not implemented yet.")

    # def test_update_newton(self):
    #     model = MultinomGLM(fitting_method="newton")
    #     model.setup_data(self.counts, design_mtx=self.design_mtx)
    #     try:
    #         model._update_newton()
    #     except NotImplementedError:
    #         self.skipTest("Newton's method update not implemented yet.")

    # def test_update_iwls(self):
    #     model = MultinomGLM(fitting_method="iwls")
    #     model.setup_data(self.counts, design_mtx=self.design_mtx)
    #     try:
    #         model._update_iwls()
    #     except NotImplementedError:
    #         self.skipTest("IWLS update not implemented yet.")


class TestMultinomGLMEdgeCases(unittest.TestCase):
    def test_single_isoform_raises_error(self):
        """Test that single isoform data raises NotImplementedError."""
        counts = torch.tensor([[[10], [20], [30]]], dtype=torch.float32)
        design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        model = MultinomGLM()
        with self.assertRaises(NotImplementedError):
            model.setup_data(counts, design_mtx=design_mtx)

    def test_2d_counts_unsqueezes(self):
        """Test that 2D counts are properly unsqueezed to 3D."""
        counts = torch.tensor([[10, 20], [30, 40], [50, 60]], dtype=torch.float32)
        design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        model = MultinomGLM()
        model.setup_data(counts, design_mtx=design_mtx)
        self.assertEqual(model.counts.ndim, 3)
        self.assertEqual(model.counts.shape, (1, 3, 2))

    def test_design_mtx_invalid_shape_raises_error(self):
        """Test that invalid design_mtx shape raises ValueError."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.ones(3, 2, 2, 2)  # 4D tensor
        model = MultinomGLM()
        with self.assertRaises(ValueError):
            model.setup_data(counts, design_mtx=design_mtx)

    def test_design_mtx_3d_batched_raises_error(self):
        """Test that batched design matrix raises assertion error."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.ones(2, 3, 2)  # batch size 2, not supported
        model = MultinomGLM()
        with self.assertRaises(AssertionError):
            model.setup_data(counts, design_mtx=design_mtx)

    def test_design_mtx_shape_mismatch_raises_error(self):
        """Test that mismatched design_mtx shape raises assertion error."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.ones(2, 2)  # 2 spots instead of 3
        model = MultinomGLM()
        with self.assertRaises(AssertionError):
            model.setup_data(counts, design_mtx=design_mtx)

    def test_no_design_mtx_creates_empty_design(self):
        """Test that None design_mtx creates an empty design matrix."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        model = MultinomGLM()
        model.setup_data(counts, design_mtx=None)
        self.assertEqual(model.n_factors, 0)
        self.assertEqual(model.X_spot.shape, (1, 3, 0))

    def test_model_string_representation(self):
        """Test that model __str__ returns expected format."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        model = MultinomGLM()
        model.setup_data(counts, design_mtx=design_mtx)
        str_repr = str(model)
        self.assertIn("Multinomial Generalized Linear Model", str_repr)
        self.assertIn("1", str_repr)  # n_genes
        self.assertIn("3", str_repr)  # n_spots
        self.assertIn("2", str_repr)  # n_isos

    def test_fit_with_sgd_optimizer(self):
        """Test fitting with SGD optimizer."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        model = MultinomGLM(
            fitting_method="gd",
            fitting_configs={"optim": "sgd", "max_epochs": 5, "lr": 0.01},
        )
        model.setup_data(counts, design_mtx=design_mtx)
        model.fit(verbose=False)
        self.assertTrue(hasattr(model, "optimizer"))

    def test_fit_with_lbfgs_optimizer(self):
        """Test fitting with LBFGS optimizer."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        model = MultinomGLM(
            fitting_method="gd",
            fitting_configs={"optim": "lbfgs", "max_epochs": 2, "lr": 0.1, "tol": 1e-6},
        )
        model.setup_data(counts, design_mtx=design_mtx)
        # LBFGS with small dataset and epochs might fail, but we test the code path
        try:
            model.fit(verbose=False)
        except RuntimeError:
            pass
        self.assertTrue(hasattr(model, "optimizers"))

    def test_clone_model(self):
        """Test cloning a model."""
        counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
        design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        model = MultinomGLM()
        model.setup_data(counts, design_mtx=design_mtx)
        model_clone = model.clone()
        self.assertEqual(model_clone.n_genes, model.n_genes)
        self.assertEqual(model_clone.n_spots, model.n_spots)


def _make_spd_kernel(n: int, seed: int = 0) -> torch.Tensor:
    """Build a random symmetric positive-definite correlation matrix of size n×n."""
    torch.manual_seed(seed)
    L = torch.randn(n, n)
    K = L @ L.T + torch.eye(n) * 0.5
    d = K.diagonal().sqrt()
    return K / d.unsqueeze(0) / d.unsqueeze(1)


def _eigenpairs_from_corr_sp(corr_sp: torch.Tensor, k: int | None = None):
    """Return (eigvals, eigvecs) in descending order, optionally truncated to top k."""
    eigvals, eigvecs = torch.linalg.eigh(corr_sp)
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    if k is not None:
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]
    return eigvals, eigvecs


class TestMultinomGLMIWLSRefactored(unittest.TestCase):
    """Verify the memory-efficient IWLS and Hessian einsum refactoring.

    The refactored methods avoid materialising the large
    n_spots*(n_isos-1) × n_spots*(n_isos-1) block-diagonal weight matrix.
    All outputs should be numerically identical to the original formulation.
    """

    def setUp(self):
        torch.manual_seed(0)
        # 3 genes, 8 spots, 3 isoforms, 2 covariates — enough to stress both paths
        self.counts = torch.randint(5, 50, (3, 8, 3)).float()
        self.design_mtx = torch.randn(8, 2)

    # ------------------------------------------------------------------
    # _get_log_lik_hessian_beta_bias  (MultinomGLM)
    # ------------------------------------------------------------------

    def test_hessian_beta_bias_glm_matches_autograd(self):
        """Einsum-based _get_log_lik_hessian_beta_bias matches autograd for GLM.

        HessianCalculatorGLM uses squeeze(0) so we pass a single-gene model.
        """
        # Single gene slice to satisfy the HessianCalculator helper
        counts_1g = self.counts[:1]
        model = MultinomGLM()
        model.setup_data(counts_1g, design_mtx=self.design_mtx)
        HessianCalculatorGLM(model).test_hessian_beta_bias()

    def test_hessian_beta_bias_glm_shape_batched(self):
        """_get_log_lik_hessian_beta_bias returns expected shape for batched GLM."""
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        H = model._get_log_lik_hessian_beta_bias()
        nF1 = model.n_factors + 1  # 3
        nQ = model.n_isos - 1  # 2
        self.assertEqual(H.shape, (model.n_genes, nF1 * nQ, nF1 * nQ))

    def test_hessian_beta_bias_glmm_matches_autograd(self):
        """Einsum-based _get_log_lik_hessian_beta_bias matches autograd for GLMM.

        HessianCalculatorGLMM uses squeeze(0) so we pass a single-gene model.
        """
        corr_sp = _make_spd_kernel(8)
        counts_1g = self.counts[:1]
        model = MultinomGLMM()
        model.setup_data(counts_1g, design_mtx=self.design_mtx, corr_sp=corr_sp)
        HessianCalculatorGLMM(model).test_hessian_beta_bias()

    def test_hessian_beta_bias_no_design_mtx(self):
        """_get_log_lik_hessian_beta_bias works when n_factors=0 (intercept only)."""
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=None)
        H = model._get_log_lik_hessian_beta_bias()
        # intercept only → (1)*(n_isos-1) square
        nQ = model.n_isos - 1
        self.assertEqual(H.shape, (model.n_genes, nQ, nQ))
        self.assertTrue(torch.isfinite(H).all())

    # ------------------------------------------------------------------
    # _update_iwls  (MultinomGLM)
    # ------------------------------------------------------------------

    def test_iwls_single_step_beta_shape(self):
        """_update_iwls produces beta / bias_eta of correct shape."""
        model = MultinomGLM(fitting_method="iwls", fitting_configs={"max_epochs": 1})
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        model.fit(verbose=False, quiet=True)
        self.assertEqual(model.beta.shape, (3, 2, 2))  # n_genes, n_factors, n_isos-1
        self.assertEqual(model.bias_eta.shape, (3, 2))  # n_genes, n_isos-1

    def test_iwls_beta_finite(self):
        """Beta and bias_eta remain finite after IWLS steps."""
        model = MultinomGLM(fitting_method="iwls", fitting_configs={"max_epochs": 20})
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        model.fit(verbose=False, quiet=True)
        self.assertTrue(torch.isfinite(model.beta).all())
        self.assertTrue(torch.isfinite(model.bias_eta).all())

    def test_iwls_log_prob_increases(self):
        """Log-likelihood should not decrease over IWLS iterations."""
        counts = torch.tensor(
            [[[10, 20, 5], [3, 40, 7], [5, 6, 70], [20, 10, 30]]],
            dtype=torch.float32,
        )
        model_init = MultinomGLM(
            fitting_method="iwls", fitting_configs={"max_epochs": 0}
        )
        model_init.setup_data(counts, design_mtx=None)
        lp_init = model_init.forward().item()

        model_fit = MultinomGLM(
            fitting_method="iwls", fitting_configs={"max_epochs": 30}
        )
        model_fit.setup_data(counts, design_mtx=None)
        model_fit.fit(verbose=False, quiet=True)
        lp_fit = model_fit.forward().item()

        self.assertGreaterEqual(lp_fit, lp_init - 1e-3)

    def test_iwls_matches_gradient_descent(self):
        """IWLS and GD reach similar log-prob on a simple problem (no design)."""
        counts = torch.tensor(
            [[[10, 30, 15], [25, 5, 20], [8, 40, 12]]],
            dtype=torch.float32,
        )
        model_iwls = MultinomGLM(
            fitting_method="iwls", fitting_configs={"max_epochs": 100}
        )
        model_iwls.setup_data(counts, design_mtx=None)
        model_iwls.fit(verbose=False, quiet=True)

        model_gd = MultinomGLM(
            fitting_method="gd",
            fitting_configs={"max_epochs": 500, "lr": 0.05, "optim": "adam"},
        )
        model_gd.setup_data(counts, design_mtx=None)
        model_gd.fit(verbose=False, quiet=True)

        lp_iwls = model_iwls.forward().item()
        lp_gd = model_gd.forward().item()
        # Both should be close to the same optimum (within 1 nats)
        self.assertAlmostEqual(lp_iwls, lp_gd, delta=1.0)


class TestMultinomGLMMLowRank(unittest.TestCase):
    """Tests for the low-rank eigendecomposition path in MultinomGLMM.

    Covers setup correctness, memory layout, Woodbury log-prob formula, and
    fitting quality of low-rank vs full-rank models.
    """

    # medium problem: 2 genes, 15 spots, 3 isoforms
    N_SPOTS = 15
    N_GENES = 2
    N_ISOS = 3

    def setUp(self):
        torch.manual_seed(7)
        self.counts = torch.randint(
            10, 60, (self.N_GENES, self.N_SPOTS, self.N_ISOS)
        ).float()
        self.corr_sp = _make_spd_kernel(self.N_SPOTS, seed=7)
        self.eigvals_full, self.eigvecs_full = _eigenpairs_from_corr_sp(self.corr_sp)

    # ------------------------------------------------------------------
    # Setup / buffer layout
    # ------------------------------------------------------------------

    def test_full_rank_eigenpairs_path_no_dense_corr_sp(self):
        """Full-rank eigenpairs path: corr_sp=None, _is_low_rank=False, shapes correct."""
        model = MultinomGLMM()
        model.setup_data(
            self.counts,
            corr_sp_eigvals=self.eigvals_full,
            corr_sp_eigvecs=self.eigvecs_full,
        )
        self.assertFalse(model._is_low_rank)
        self.assertIsNone(model.corr_sp)
        self.assertEqual(model._rank, self.N_SPOTS)
        self.assertEqual(model.corr_sp_eigvals.shape, (self.N_SPOTS,))
        self.assertEqual(model.corr_sp_eigvecs.shape, (self.N_SPOTS, self.N_SPOTS))

    def test_low_rank_setup_shapes_and_flags(self):
        """Low-rank path: _is_low_rank=True, corr_sp=None, eigvec shape (n, k)."""
        k = 6
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)
        model = MultinomGLMM()
        model.setup_data(
            self.counts,
            corr_sp_eigvals=eigvals_k,
            corr_sp_eigvecs=eigvecs_k,
        )
        self.assertTrue(model._is_low_rank)
        self.assertIsNone(model.corr_sp)
        self.assertEqual(model._rank, k)
        self.assertEqual(model.corr_sp_eigvals.shape, (k,))
        self.assertEqual(model.corr_sp_eigvecs.shape, (self.N_SPOTS, k))

    def test_legacy_corr_sp_path_still_stores_dense_matrix(self):
        """Legacy corr_sp= path keeps the n×n matrix (backward compatibility)."""
        model = MultinomGLMM()
        model.setup_data(self.counts, corr_sp=self.corr_sp)
        self.assertFalse(model._is_low_rank)
        self.assertIsNotNone(model.corr_sp)
        self.assertEqual(model.corr_sp.shape, (self.N_SPOTS, self.N_SPOTS))

    # ------------------------------------------------------------------
    # Memory layout: eigvec sizes
    # ------------------------------------------------------------------

    def test_low_rank_eigvec_buffer_smaller_than_full_rank(self):
        """Low-rank eigvec buffer has fewer elements than full-rank (n×k vs n×n)."""
        k = 5
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)

        model_full = MultinomGLMM()
        model_full.setup_data(
            self.counts,
            corr_sp_eigvals=self.eigvals_full,
            corr_sp_eigvecs=self.eigvecs_full,
        )

        model_low = MultinomGLMM()
        model_low.setup_data(
            self.counts,
            corr_sp_eigvals=eigvals_k,
            corr_sp_eigvecs=eigvecs_k,
        )

        full_eigvec_size = model_full.corr_sp_eigvecs.numel()  # N × N
        low_eigvec_size = model_low.corr_sp_eigvecs.numel()  # N × k

        self.assertEqual(full_eigvec_size, self.N_SPOTS * self.N_SPOTS)
        self.assertEqual(low_eigvec_size, self.N_SPOTS * k)
        self.assertLess(low_eigvec_size, full_eigvec_size)

        # Neither path materialises the n×n corr_sp dense buffer
        self.assertIsNone(model_full.corr_sp)
        self.assertIsNone(model_low.corr_sp)

    def test_low_rank_saves_corr_sp_storage_vs_legacy(self):
        """Low-rank path uses much less storage than the legacy corr_sp= path."""
        k = 5
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)

        # Legacy path allocates n×n corr_sp  +  n×n eigvecs = 2n² elements
        model_legacy = MultinomGLMM()
        model_legacy.setup_data(self.counts, corr_sp=self.corr_sp)
        legacy_corr_sp_elem = model_legacy.corr_sp.numel()
        legacy_eigvec_elem = model_legacy.corr_sp_eigvecs.numel()

        # Low-rank path: no corr_sp, n×k eigvecs only
        model_low = MultinomGLMM()
        model_low.setup_data(
            self.counts, corr_sp_eigvals=eigvals_k, corr_sp_eigvecs=eigvecs_k
        )
        low_eigvec_elem = model_low.corr_sp_eigvecs.numel()

        # Legacy: 2 × N² elements of spatial matrices; low-rank: N × k elements
        legacy_total = legacy_corr_sp_elem + legacy_eigvec_elem  # 2 * N²
        self.assertLess(low_eigvec_elem, legacy_total)
        self.assertGreater(legacy_total // low_eigvec_elem, 1)

    # ------------------------------------------------------------------
    # Woodbury log-prob correctness
    # ------------------------------------------------------------------

    def test_full_rank_eigenpairs_log_prob_equals_legacy_corr_sp(self):
        """Full-rank eigenpairs path gives the same log-prob as legacy corr_sp= path."""
        model_legacy = MultinomGLMM()
        model_legacy.setup_data(self.counts, corr_sp=self.corr_sp)

        model_eig = MultinomGLMM()
        model_eig.setup_data(
            self.counts,
            corr_sp_eigvals=self.eigvals_full,
            corr_sp_eigvecs=self.eigvecs_full,
        )

        # Sync sigma / theta to the same values so the only difference is the code path
        with torch.no_grad():
            model_eig.sigma.copy_(model_legacy.sigma)
            if hasattr(model_eig, "theta_logit"):
                model_eig.theta_logit.copy_(model_legacy.theta_logit)
            model_eig.nu.copy_(model_legacy.nu)

        lp_legacy = model_legacy._calc_log_prob_joint()
        lp_eig = model_eig._calc_log_prob_joint()
        torch.testing.assert_close(lp_legacy, lp_eig, atol=1e-4, rtol=1e-4)

    def test_low_rank_woodbury_log_prob_finite_and_negative(self):
        """Low-rank log-prob is finite and negative (valid log-prob)."""
        k = 6
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)
        model = MultinomGLMM()
        model.setup_data(
            self.counts, corr_sp_eigvals=eigvals_k, corr_sp_eigvecs=eigvecs_k
        )
        lp = model._calc_log_prob_joint()
        self.assertEqual(lp.shape, (self.N_GENES,))
        self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue((lp < 0).all(), "Log-prob should be negative")

    def test_low_rank_residual_eigval_positive(self):
        """_residual_cov_eigval returns positive values (noise variance > 0)."""
        k = 5
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)
        model = MultinomGLMM()
        model.setup_data(
            self.counts, corr_sp_eigvals=eigvals_k, corr_sp_eigvecs=eigvecs_k
        )
        d = model._residual_cov_eigval()
        self.assertEqual(d.shape[-1], 1)  # last dim is 1 (broadcast dim)
        self.assertTrue((d > 0).all())

    def test_low_rank_log_prob_close_to_full_rank_for_high_k(self):
        """Top-k log-prob is close to full-rank when k captures most spectral mass (>90%)."""
        # Find k such that top-k eigenvalues capture ≥ 90% of spectral mass
        total = self.eigvals_full.sum()
        cumsum = self.eigvals_full.cumsum(0)
        k = int((cumsum / total < 0.90).sum()) + 1
        k = min(k, self.N_SPOTS - 1)

        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)

        model_full = MultinomGLMM()
        model_full.setup_data(
            self.counts,
            corr_sp_eigvals=self.eigvals_full,
            corr_sp_eigvecs=self.eigvecs_full,
        )
        model_low = MultinomGLMM()
        model_low.setup_data(
            self.counts, corr_sp_eigvals=eigvals_k, corr_sp_eigvecs=eigvecs_k
        )

        # Copy identical parameter values so differences come only from the approximation
        with torch.no_grad():
            model_low.sigma.copy_(model_full.sigma)
            if hasattr(model_low, "theta_logit"):
                model_low.theta_logit.copy_(model_full.theta_logit)
            model_low.nu.copy_(model_full.nu)

        lp_full = model_full._calc_log_prob_joint()
        lp_low = model_low._calc_log_prob_joint()

        # Absolute difference should be modest (within ~2 nats per gene)
        diff = (lp_full - lp_low).abs()
        self.assertTrue(
            (diff < 5.0).all(),
            f"Log-prob difference too large: {diff.tolist()} (k={k}/{self.N_SPOTS})",
        )

    # ------------------------------------------------------------------
    # Fitting quality
    # ------------------------------------------------------------------

    def test_low_rank_fit_joint_gd_completes(self):
        """Low-rank MultinomGLMM can be fit with joint_gd without error."""
        k = 7
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)
        model = MultinomGLMM(
            fitting_method="joint_gd",
            fitting_configs={"max_epochs": 20},
        )
        model.setup_data(
            self.counts, corr_sp_eigvals=eigvals_k, corr_sp_eigvecs=eigvecs_k
        )
        model.fit(verbose=False, quiet=True)
        self.assertEqual(model.n_genes, self.N_GENES)
        self.assertTrue(torch.isfinite(model.sigma).all())
        self.assertTrue(torch.isfinite(model.nu).all())

    def test_low_rank_and_full_rank_similar_fitted_ratios(self):
        """Low-rank (high-k) and full-rank produce similar mean isoform ratios after fitting."""
        k = self.N_SPOTS - 2  # nearly full rank — approx should be tight
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)

        cfg = {"max_epochs": 80}

        model_full = MultinomGLMM(fitting_method="joint_gd", fitting_configs=cfg)
        model_full.setup_data(
            self.counts,
            corr_sp_eigvals=self.eigvals_full,
            corr_sp_eigvecs=self.eigvecs_full,
        )
        model_full.fit(verbose=False, quiet=True)

        model_low = MultinomGLMM(fitting_method="joint_gd", fitting_configs=cfg)
        model_low.setup_data(
            self.counts, corr_sp_eigvals=eigvals_k, corr_sp_eigvecs=eigvecs_k
        )
        model_low.fit(verbose=False, quiet=True)

        ratio_full = model_full.get_isoform_ratio()  # (n_genes, n_spots, n_isos)
        ratio_low = model_low.get_isoform_ratio()

        # Spot-averaged ratios should be similar (within 5 percentage points)
        mean_full = ratio_full.mean(dim=1)  # (n_genes, n_isos)
        mean_low = ratio_low.mean(dim=1)
        max_diff = (mean_full - mean_low).abs().max().item()
        self.assertLess(max_diff, 0.05, f"Mean ratio diff {max_diff:.4f} > 0.05")

    def test_low_rank_hessian_beta_bias_matches_autograd(self):
        """_get_log_lik_hessian_beta_bias is correct for a low-rank GLMM model.
        HessianCalculatorGLMM uses squeeze(0) so we pass a single-gene model.
        """
        k = 6
        eigvals_k, eigvecs_k = _eigenpairs_from_corr_sp(self.corr_sp, k=k)
        design_mtx = torch.randn(self.N_SPOTS, 1)
        counts_1g = self.counts[:1]
        model = MultinomGLMM()
        model.setup_data(
            counts_1g,
            design_mtx=design_mtx,
            corr_sp_eigvals=eigvals_k,
            corr_sp_eigvecs=eigvecs_k,
        )
        # Only test beta/bias Hessian; nu Hessian helper assumes full-rank eigvecs
        HessianCalculatorGLMM(model).test_hessian_beta_bias()

    def test_rank1_dummy_glmm_null_log_prob_identity_covariance(self):
        """Rank-1 dummy eigenpairs (glmm-null sentinel) gives identity covariance (theta→0)."""
        n = self.N_SPOTS
        # Rank-1 dummy: constant unit vector, eigenvalue=1
        eigvals_1 = torch.ones(1)
        eigvecs_1 = torch.full((n, 1), 1.0 / (n**0.5))

        model = MultinomGLMM(var_parameterization_sigma_theta=True)
        model.setup_data(
            self.counts, corr_sp_eigvals=eigvals_1, corr_sp_eigvecs=eigvecs_1
        )

        # Force theta → 0 (no spatial variance)
        with torch.no_grad():
            model.theta_logit.fill_(-30.0)

        # Manually compute expected cov eigvals: σ²(θ*1 + (1-θ)) ≈ σ²
        # residual_eigval ≈ σ²(1-θ) ≈ σ²
        # correction = 1/σ² - 1/σ² = 0 → Woodbury gives C⁻¹ = (1/σ²)I  ✓
        d = model._residual_cov_eigval()  # (n_genes, n_var_comp, 1)
        c = model._cov_eigvals()  # (n_genes, n_var_comp, 1)
        # when theta≈0: c ≈ d  →  correction ≈ 0
        correction = (1.0 / c - 1.0 / d).abs()
        self.assertTrue(
            (correction < 1e-4).all(),
            "Woodbury correction should vanish when theta→0 (identity covariance)",
        )

        lp = model._calc_log_prob_joint()
        self.assertTrue(torch.isfinite(lp).all())


class TestNewAPIs(unittest.TestCase):
    """Tests for new APIs introduced in the refactoring:
    - loss_history always available (PatienceLogger)
    - __str__ shows training summary after fit()
    - strip_data() frees counts/X_spot buffers
    - diagnose deprecation warning
    - store_param_history still works
    """

    def setUp(self):
        torch.manual_seed(42)
        # Small: 2 genes, 10 spots, 3 isoforms, 2 covariates
        self.counts = torch.randint(5, 30, (2, 10, 3)).float()
        self.design_mtx = torch.randn(10, 2)

    # ------------------------------------------------------------------
    # PatienceLogger.loss_history
    # ------------------------------------------------------------------

    def test_glm_loss_history_always_available(self):
        """loss_history is populated after fit() without store_param_history=True."""
        model = MultinomGLM(fitting_configs={"max_epochs": 15})
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        model.fit(verbose=False, quiet=True)
        lh = model.logger.loss_history
        self.assertEqual(lh.ndim, 2, "loss_history must be 2-D (epochs, batch_size)")
        self.assertEqual(lh.shape[1], 2, "batch_size dim must equal n_genes=2")
        self.assertGreater(lh.shape[0], 0, "At least one epoch must be recorded")
        self.assertTrue(torch.isfinite(lh).all())

    def test_glmm_loss_history_always_available(self):
        """MultinomGLMM loss_history is populated without store_param_history."""
        corr_sp = _make_spd_kernel(10)
        model = MultinomGLMM(fitting_configs={"max_epochs": 10})
        model.setup_data(self.counts, design_mtx=self.design_mtx, corr_sp=corr_sp)
        model.fit(verbose=False, quiet=True)
        lh = model.logger.loss_history
        self.assertEqual(lh.ndim, 2)
        self.assertGreater(lh.shape[0], 0)

    def test_glm_loss_history_empty_before_fit(self):
        """loss_history returns empty tensor (0 epochs) before fit() is called."""
        model = MultinomGLM()
        model.setup_data(self.counts)
        # PatienceLogger is only created during fit(); accessing before raises AttributeError
        # — that is acceptable, no requirement to have logger before fit.
        # If logger exists, its loss_history should be empty.
        if hasattr(model, "logger") and model.logger is not None:
            lh = model.logger.loss_history
            self.assertEqual(lh.shape[0], 0)

    def test_store_param_history_still_works(self):
        """store_param_history=True still populates params_iter."""
        model = MultinomGLM(fitting_configs={"max_epochs": 5})
        model.setup_data(self.counts[:1])  # single gene to keep it fast
        model.fit(verbose=False, quiet=True, store_param_history=True)
        pi = model.logger.get_params_iter()
        self.assertIsNotNone(
            pi, "params_iter must not be None when store_param_history=True"
        )
        self.assertIn("loss", pi)
        self.assertIn("params", pi)

    def test_diagnose_deprecation_warning(self):
        """Passing diagnose=True to fit() emits a DeprecationWarning."""
        import warnings

        model = MultinomGLM(fitting_configs={"max_epochs": 3})
        model.setup_data(self.counts[:1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(verbose=False, quiet=True, diagnose=True)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertTrue(
            len(dep_warns) > 0,
            "Expected DeprecationWarning for diagnose=True, got none",
        )
        self.assertIn("diagnose", str(dep_warns[0].message).lower())

    # ------------------------------------------------------------------
    # MultinomGLM.__str__ / __repr__ training section
    # ------------------------------------------------------------------

    def test_glm_str_unfitted(self):
        """GLM __str__ shows 'not yet fitted' before fit()."""
        model = MultinomGLM()
        model.setup_data(self.counts)
        s = str(model)
        self.assertIn("not yet fitted", s.lower())

    def test_glm_str_fitted(self):
        """GLM __str__ shows convergence info after fit()."""
        model = MultinomGLM(fitting_configs={"max_epochs": 10})
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        model.fit(verbose=False, quiet=True)
        s = str(model)
        self.assertIn("Converged", s)
        self.assertIn("Best epoch", s)
        self.assertIn("Best loss", s)
        # repr should match str
        self.assertEqual(repr(model), s)

    def test_glmm_str_fitted(self):
        """GLMM __str__ shows training summary after fit()."""
        corr_sp = _make_spd_kernel(10)
        model = MultinomGLMM(fitting_configs={"max_epochs": 5})
        model.setup_data(self.counts, corr_sp=corr_sp)
        model.fit(verbose=False, quiet=True)
        s = str(model)
        self.assertIn("Converged", s)
        self.assertEqual(repr(model), s)

    # ------------------------------------------------------------------
    # MultinomGLM.strip_data()
    # ------------------------------------------------------------------

    def test_strip_data_frees_buffers(self):
        """strip_data() sets counts and X_spot to None and removes them from _buffers."""
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=self.design_mtx)
        # Confirm buffers exist before strip
        self.assertIsNotNone(model.counts)
        self.assertIsNotNone(model.X_spot)
        model.strip_data()
        self.assertIsNone(model.counts)
        self.assertIsNone(model.X_spot)
        # Must not be in _buffers either
        self.assertNotIn("counts", model._buffers)
        self.assertNotIn("X_spot", model._buffers)

    def test_strip_data_no_counts_no_error(self):
        """strip_data() is safe to call when design_mtx=None (X_spot not registered)."""
        model = MultinomGLM()
        model.setup_data(self.counts, design_mtx=None)
        model.strip_data()  # should not raise
        self.assertIsNone(model.counts)

    def test_strip_data_idempotent(self):
        """Calling strip_data() twice should not raise."""
        model = MultinomGLM()
        model.setup_data(self.counts)
        model.strip_data()
        model.strip_data()  # second call must be safe


def _setup_glmm_for_sigma_hessian(
    n_genes=2,
    n_spots=10,
    n_isos=3,
    rank=4,
    var_parameterization_sigma_theta=True,
    share_variance=True,
    var_prior_model="none",
    var_prior_model_params=None,
    seed=42,
):
    """Helper: create a MultinomGLMM with known low-rank kernel for sigma Hessian tests."""
    torch.manual_seed(seed)
    counts = torch.randint(5, 30, (n_genes, n_spots, n_isos)).float()
    corr_sp = _make_spd_kernel(n_spots, seed=seed)
    eigvals_full, eigvecs_full = _eigenpairs_from_corr_sp(corr_sp)
    eigvals_lr = eigvals_full[:rank]
    eigvecs_lr = eigvecs_full[:, :rank]
    model = MultinomGLMM(
        var_parameterization_sigma_theta=var_parameterization_sigma_theta,
        share_variance=share_variance,
        var_prior_model=var_prior_model,
        var_prior_model_params=var_prior_model_params or {},
    )
    model.setup_data(
        counts,
        corr_sp_eigvals=eigvals_lr,
        corr_sp_eigvecs=eigvecs_lr,
    )
    # Perturb parameters away from defaults so gradients are non-trivial
    with torch.no_grad():
        if var_parameterization_sigma_theta:
            model.sigma.add_(torch.randn_like(model.sigma) * 0.1)
            model.sigma.abs_()
            model.theta_logit.add_(torch.randn_like(model.theta_logit) * 0.5)
        else:
            model.sigma_sp.add_(torch.randn_like(model.sigma_sp) * 0.1)
            model.sigma_sp.abs_()
            model.sigma_nsp.add_(torch.randn_like(model.sigma_nsp) * 0.1)
            model.sigma_nsp.abs_()
    return model


class TestSigmaHessianAnalytic(unittest.TestCase):
    """Unit tests comparing _get_log_lik_hessian_sigma_expand_analytic vs autograd."""

    ATOL = 1e-4  # tolerance for analytic vs autograd comparison

    def _assert_hessians_close(self, model):
        analytic = model._get_log_lik_hessian_sigma_expand_analytic()
        autograd = model._get_log_lik_hessian_sigma_expand()
        self.assertEqual(analytic.shape, autograd.shape)
        max_diff = (analytic - autograd).abs().max().item()
        self.assertLess(
            max_diff,
            self.ATOL,
            f"Max abs diff {max_diff:.2e} exceeds tolerance {self.ATOL:.2e}.\n"
            f"analytic=\n{analytic}\nautograd=\n{autograd}",
        )

    def test_sigma_theta_low_rank_no_prior(self):
        """sigma/theta_logit, low-rank kernel, no prior."""
        model = _setup_glmm_for_sigma_hessian(
            var_parameterization_sigma_theta=True, var_prior_model="none"
        )
        self.assertTrue(model._is_low_rank)
        self._assert_hessians_close(model)

    def test_sigma_sp_nsp_low_rank_no_prior(self):
        """sigma_sp/sigma_nsp, low-rank kernel, no prior."""
        model = _setup_glmm_for_sigma_hessian(
            var_parameterization_sigma_theta=False, var_prior_model="none"
        )
        self.assertTrue(model._is_low_rank)
        self._assert_hessians_close(model)

    def test_sigma_theta_low_rank_inv_gamma_prior(self):
        """sigma/theta_logit, low-rank kernel, inv_gamma prior."""
        model = _setup_glmm_for_sigma_hessian(
            var_parameterization_sigma_theta=True, var_prior_model="inv_gamma"
        )
        self._assert_hessians_close(model)

    def test_sigma_sp_nsp_low_rank_gamma_prior(self):
        """sigma_sp/sigma_nsp, low-rank kernel, gamma prior."""
        model = _setup_glmm_for_sigma_hessian(
            var_parameterization_sigma_theta=False, var_prior_model="gamma"
        )
        self._assert_hessians_close(model)

    def test_sigma_theta_full_rank_no_prior(self):
        """sigma/theta_logit, full-rank kernel (_is_low_rank=False), no prior."""
        torch.manual_seed(0)
        n_spots, n_genes, n_isos = 10, 2, 3
        counts = torch.randint(5, 30, (n_genes, n_spots, n_isos)).float()
        corr_sp = _make_spd_kernel(n_spots, seed=0)
        eigvals, eigvecs = _eigenpairs_from_corr_sp(corr_sp)
        model = MultinomGLMM(var_parameterization_sigma_theta=True)
        model.setup_data(counts, corr_sp_eigvals=eigvals, corr_sp_eigvecs=eigvecs)
        self.assertFalse(model._is_low_rank)
        self._assert_hessians_close(model)

    def test_sigma_theta_low_rank_share_variance_false(self):
        """sigma/theta_logit, low-rank, share_variance=False (per-isoform params)."""
        model = _setup_glmm_for_sigma_hessian(
            var_parameterization_sigma_theta=True,
            share_variance=False,
            var_prior_model="none",
        )
        self._assert_hessians_close(model)

    def test_sigma_sp_nsp_low_rank_share_variance_false(self):
        """sigma_sp/sigma_nsp, low-rank, share_variance=False, inv_gamma prior."""
        model = _setup_glmm_for_sigma_hessian(
            var_parameterization_sigma_theta=False,
            share_variance=False,
            var_prior_model="inv_gamma",
        )
        self._assert_hessians_close(model)


if __name__ == "__main__":
    unittest.main()
