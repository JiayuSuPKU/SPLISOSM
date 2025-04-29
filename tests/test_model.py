import unittest
import torch
from splisosm.model import MultinomGLM, MultinomGLMM
from splisosm.likelihood import log_prob_fastmult, log_prob_fastmvn
from torch.autograd.functional import hessian


class TestHessianGLM():
    def __init__(self, model: MultinomGLM):
        self.model = model

    def _calc_log_prob_joint_wrt_beta_bias(self, beta_bias):
        """Helper function of log joint probability wrt beta and bias for calculate Hessian."""
        model = self.model
        # combine beta and the intercept bias_eta
        X_expand = torch.cat(
            [model.X_spot, torch.ones(1, model.n_spots, 1)], dim=-1,
        ) # (1, n_spots, n_factors + 1)
        # update alpha locally as a function of the input beta_bias
        eta = (X_expand @ beta_bias)
        alpha = torch.concat([eta, torch.zeros(1, model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1) # (1, n_spots, n_isos)

        return log_prob_fastmult(alpha.squeeze(0).T, model.counts.squeeze(0).T)

    def test_hessian_beta_bias(self):
        model = self.model
        # calculate the Hessian using pytorch
        beta_bias = torch.concat([model.beta, model.bias_eta.unsqueeze(0)], dim=1) # (1, n_factor + 1, n_isos - 1)
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_beta_bias, beta_bias, vectorize=True, create_graph = False
        ) # (1, n_factor + 1, n_isos - 1, 1, n_factor + 1, n_isos - 1)
        hess_pytorch = hess_pytorch.squeeze(0, 3).permute(1,0,3,2).reshape(
            (model.n_factors + 1) * (model.n_isos - 1), (model.n_factors + 1) * (model.n_isos - 1)
        ).unsqueeze(0) # (1, (n_factor + 1) * (n_isos - 1), (n_factor + 1) * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_beta_bias()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-4)

class TestMultinomGLM(unittest.TestCase):
    def setUp(self):
        # simulate 2 isoforms in 3 spots, with 2 covariates
        self.counts = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.float32)
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
        TestHessianGLM(model).test_hessian_beta_bias()

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


class TestHessianGLMM():
    def __init__(self, model: MultinomGLMM):
        self.model = model

    def _calc_log_prob_joint_wrt_eta(self, eta):
        """Helper function of log joint probability wrt eta for calculate Hessian."""
        model = self.model
        alpha = torch.concat([eta.squeeze(0), torch.zeros(model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1) # (n_spots, n_isos)

        return log_prob_fastmult(alpha.T, model.counts.squeeze(0).T)

    def _calc_log_prob_joint_wrt_nu(self, nu):
        """Helper function of log joint probability wrt nu for calculate Hessian."""
        model = self.model
        # MVN prior likelihood as a function of the input eta
        log_prob = log_prob_fastmvn(
            0.0,
            model._cov_eigvals().squeeze(0), # (1, n_spots)
            model.corr_sp_eigvecs.unsqueeze(0), # (1, n_spots, n_spots)
            nu.squeeze(0).T # (n_isos - 1, n_spots)
        )
        # update alpha locally as a function of the input nu
        eta = (model.X_spot @ model.beta + model.bias_eta) + nu # (1, n_spots, n_isos - 1)
        alpha = torch.concat([eta, torch.zeros(1, model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1) # (1, n_spots, n_isos)
        log_prob += log_prob_fastmult(alpha.squeeze(0).T, model.counts.squeeze(0).T)

        return log_prob

    def _calc_log_prob_joint_wrt_beta_bias(self, beta_bias):
        """Helper function of log joint probability wrt beta and bias for calculate Hessian."""
        model = self.model
        # combine beta and the intercept bias_eta
        X_expand = torch.cat(
            [model.X_spot, torch.ones(1, model.n_spots, 1)], dim=-1,
        ) # (1, n_spots, n_factors + 1)
        # update alpha locally as a function of the input beta_bias
        eta = (X_expand @ beta_bias) + model.nu
        alpha = torch.concat([eta, torch.zeros(1, model.n_spots, 1)], dim=-1)
        alpha = torch.softmax(alpha, dim=-1) # (1, n_spots, n_isos)

        return log_prob_fastmult(alpha.squeeze(0).T, model.counts.squeeze(0).T)

    def test_hessian_eta(self):
        model = self.model
        # calculate the Hessian using pytorch
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_eta, model._eta(), vectorize=True, create_graph = False
        ) # (1, n_spots, n_isos - 1, 1, n_spots, n_isos - 1)
        hess_pytorch = hess_pytorch.squeeze(0, 3).permute(1,0,3,2).reshape(
            model.n_spots * (model.n_isos - 1), model.n_spots * (model.n_isos - 1)
        ).unsqueeze(0) # (1, n_spots * (n_isos - 1), n_spots * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_eta()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-5)

    def test_hessian_nu(self):
        model = self.model
        # calculate the Hessian using pytorch
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_nu, model.nu, vectorize=True, create_graph = False
        ) # (1, n_spots, n_isos - 1, 1, n_spots, n_isos - 1)
        hess_pytorch = hess_pytorch.squeeze(0,3).permute(1,0,3,2).reshape(
            model.n_spots * (model.n_isos - 1), model.n_spots * (model.n_isos - 1)
        ).unsqueeze(0) # (1, n_spots * (n_isos - 1), n_spots * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_nu()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-5)

    def test_hessian_beta_bias(self):
        model = self.model
        # calculate the Hessian using pytorch
        beta_bias = torch.concat([model.beta, model.bias_eta.unsqueeze(0)], dim=1) # (1, n_factor + 1, n_isos - 1)
        hess_pytorch = hessian(
            self._calc_log_prob_joint_wrt_beta_bias, beta_bias, vectorize=True, create_graph = False
        ) # (1, n_factor + 1, n_isos - 1, 1, n_factor + 1, n_isos - 1)
        hess_pytorch = hess_pytorch.squeeze(0, 3).permute(1,0,3,2).reshape(
            (model.n_factors + 1) * (model.n_isos - 1), (model.n_factors + 1) * (model.n_isos - 1)
        ).unsqueeze(0) # (1, (n_factor + 1) * (n_isos - 1), (n_factor + 1) * (n_isos - 1))

        # calculate the Hessian using the analytic formula
        hess_analytic = model._get_log_lik_hessian_beta_bias()

        # print(hess_pytorch, hess_analytic)
        assert torch.allclose(hess_pytorch, hess_analytic, atol=1e-4)

class TestMultinomGLMM(unittest.TestCase):
    def setUp(self):
        # simulate 2 isoforms in 5 spots, with 2 covariates
        self.counts = torch.tensor([[[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]], dtype=torch.float32)
        self.design_mtx = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.float32)
        
        # simulate the 5-by-5 spatial covariance matrix
        self.corr_sp = torch.tensor([[1.0, 0.5, 0.3, 0.2, 0.1],
                                      [0.5, 1.0, 0.5, 0.3, 0.2],
                                      [0.3, 0.5, 1.0, 0.5, 0.3],
                                      [0.2, 0.3, 0.5, 1.0, 0.5],
                                      [0.1, 0.2, 0.3, 0.5, 1.0]], dtype=torch.float32)
        
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
        model = MultinomGLMM(fitting_method="joint_gd", fitting_configs={"optim": "adam"})
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
        TestHessianGLMM(model).test_hessian_eta()
        TestHessianGLMM(model).test_hessian_nu()
        TestHessianGLMM(model).test_hessian_beta_bias()

    def test_fit_with_different_methods(self):
        for method in ['joint_gd', 'joint_newton', 'marginal_gd', 'marginal_newton']:
            with self.subTest(fitting_method=method):
                model = MultinomGLMM(
                    fitting_method=method, 
                    fitting_configs={"max_epochs": 5},
                    var_fix_sigma=False
                )
                model.setup_data(self.counts, design_mtx=self.design_mtx, corr_sp=self.corr_sp)
                model.fit(verbose=False, diagnose=False)

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



if __name__ == "__main__":
    unittest.main()