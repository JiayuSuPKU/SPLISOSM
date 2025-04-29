import warnings
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import hessian, jacobian
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import default_collate
from torch.distributions import Gamma, InverseGamma

from splisosm.likelihood import (
    log_prob_fastmult,
    log_prob_fastmvn,
    log_prob_fastmult_batched,
    log_prob_fastmvn_batched,
)
from splisosm.logger import PatienceLogger


class BaseModel(ABC):
    """API for the GLM and GLMM model."""

    @abstractmethod
    def setup_data(self, counts, corr_sp, design_mtx=None):
        """Set up the data for the model.

        Args:
                counts: tensor(n_genes, n_spots, n_isoforms), genes with the same number of isoforms
                corr_sp: tensor(n_spots, n_spots)
                design_mtx: tensor(n_spots, n_covariates)
        """
        pass

    def forward(self):
        """Calculate the log-likelihood or log-marginal-likelihood of the model."""
        pass

    @abstractmethod
    def fit(self):
        """Fit the model using all data."""
        pass

    @abstractmethod
    def get_isoform_ratio(self):
        """Extract the fitted isoform ratio across space."""
        pass

    @abstractmethod
    def clone(self):
        """Clone the model with data and model parameters."""
        pass


def _melt_tensor_along_first_dim(tensor_in):
    """Melt a 4D tensor into 3D and reorder entries by spots.

    tensor_in[:, i, j, k] -> matrix_out[:, i + j * n, i + k * n] where n = tensor_in.shape[1]

    Args:
            tensor_in: tensor(n_genes, n_spots, n_isos - 1, n_isos - 1)

    Returns:
            matrix_out: tensor(n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
    """
    b, n, m = tensor_in.shape[:3]
    assert tensor_in.shape == (b, n, m, m)

    # example: at spot i, isoform j and k are connected via tensor[i, j, k]
    # tensor[i, j, k] -> out[i + j * n_spots, i + k * n_spots]
    matrix_out = torch.zeros(b, n * m, n * m, device=tensor_in.device)
    i, j, k = torch.meshgrid(
        torch.arange(n), torch.arange(m), torch.arange(m), indexing="ij"
    )
    row_indices = i + j * n
    col_indices = i + k * n
    matrix_out[:, row_indices.view(-1), col_indices.view(-1)] += tensor_in.flatten(1)

    return matrix_out


@torch.no_grad
def update_at_idx(
    params: torch.Tensor, new_params: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    idx = idx.view(-1, *([1] * (params.ndim - 1))).float()
    params = params * idx + new_params * (1 - idx)
    return params


class MultinomGLM(BaseModel, nn.Module):
    """The Multinomial Generalized Linear Model for spatial isoform expression.

    Compared to MultinomGLMM, this model does not have a random effect term.
            Y ~ Multinomial(alpha, Y.sum(1))
            eta = multinomial-logit(alpha) = X @ beta + bias_eta

    Given isoform counts of a gene Y (n_spots, n_isos) and design matrix X (n_spots, n_factors),
    MultinomGLM.fit will find the MAP estimates of the following learnable parameters:
    - beta: (n_factors, n_isos - 1) covariate coefficients of the fixed effect term.
    - bias_eta: (n_isos - 1) intercepts of the fixed effect term.

    Inference is performed by maximizing the log likelihood using one of the following methods:
    - 'iwls': (default) Iteratively reweighted least squares.
    - 'newton': Newton's method.
    - 'gd': Gradient descent.
    """

    def __init__(self, fitting_method: str = "iwls", fitting_configs: dict = {}):
        super().__init__()
        # will be set up later by calling self.setup_data()
        self.n_spots = None  # number of spots
        self.n_isos = None  # number of isoforms
        self.n_factors = None  # number of covariates

        # specify the fitting method
        assert fitting_method in ["gd", "newton", "iwls"]
        self.fitting_method = fitting_method

        # specify the fitting configurations
        self.fitting_configs = {  # default configurations
            "lr": 1e-2,
            "optim": "adam",
            "tol": 1e-5,
            "max_epochs": 1000,
            "patience": 5 if fitting_method == "gd" else 2,
        }
        self.fitting_configs.update(fitting_configs)

        # for now, restricting the optimization method to Adam, SGD and lbfgs
        assert self.fitting_configs["optim"] in ["adam", "sgd", "lbfgs"]

        # specify the fitting outcomes
        self.fitting_time = 0

    def __str__(self):
        return (
            f"A Multinomial Generalized Linear Model (GLM)\n"
            + f"- Number of genes in the batch: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of isoforms per gene: {self.n_isos}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Fitting method: {self.fitting_method}"
        )

    def setup_data(self, counts, design_mtx=None, device="cpu") -> None:
        """Set up the data for the model.

        Args:
                counts: tensor(n_genes, n_spots, n_isoforms)
                design_mtx: tensor(n_spots, n_covariates)
                device: 'cpu', or 'cuda'
                        'mps' currently not supported (torch.lgamma not supported on mps)
        """
        # need to switch to a different count model (e.g., Poisson) when only one isoform is provided
        if counts.shape[-1] == 1:
            raise NotImplementedError(
                "Only one isoform provided. Please use a different count model."
            )

        assert device in ["cpu", "cuda"]
        self.device = torch.device(device)

        if counts.ndim == 2:
            counts = counts.unsqueeze(0) # (1, n_spots, n_isos)
            print(
                "Batched calculation has been implemented. Provide a batch of counts to speed up calculation."
            )
        # set model dimensions based on the input shape
        self.n_genes, self.n_spots, self.n_isos = counts.shape

        if design_mtx is None:
            # initialize an empty design matrix of shape (1, n_spots, 0)
            design_mtx = torch.ones(1, self.n_spots, 0)
        elif design_mtx.ndim == 2:
            design_mtx = design_mtx.unsqueeze(0) # (1, n_spots, n_factors)
        elif design_mtx.ndim == 3:
            # all genes to test should share the same design matrix
            assert design_mtx.shape[0] == 1, "Batched design matrix is currently not supported."
        else:
            raise ValueError(
                f"design_mtx must be a 2D or 3D tensor. Got shape: {design_mtx.shape}"
            )
        self.n_factors = design_mtx.shape[-1]
        assert design_mtx.shape == (1, self.n_spots, self.n_factors), f"Invalide design matrix shape: {design_mtx.shape}"

        # (n_genes, n_spots, n_isos), int, the observed counts
        self.register_buffer("counts", counts)
        # (1, n_spots, n_factors), the input design matrix of n_factors covariates
        self.register_buffer("X_spot", design_mtx)
        self.register_buffer("convergence", torch.zeros(self.n_genes, dtype=bool))

        # set up learnable parameters according to the model architecture
        self._configure_learnable_variables()

        # send to device
        self.to(self.device)

    def _configure_learnable_variables(self, val=None):
        """Set up learnable parameters according to the model architecture."""
        # the fixed effect terms
        # covariate coefficients
        beta = torch.zeros(self.n_genes, self.n_factors, self.n_isos - 1)
        # intercepts
        bias_eta = torch.zeros(self.n_genes, self.n_isos - 1)

        # optimize using gradient descent / newton's method
        self.register_parameter("beta", nn.Parameter(beta))
        self.register_parameter("bias_eta", nn.Parameter(bias_eta))

    def _configure_optimizer(self, verbose=False):
        """Configure the optimizer and learning rate scheduler."""
        # initialize optimizer
        if self.fitting_configs["optim"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.fitting_configs["lr"]
            )
            self._closure = None
        elif self.fitting_configs["optim"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.fitting_configs["lr"]
            )
            self._closure = None
        else:
            # with torch.no_grad():
            #     def _fake_parameters():
            #         for p in self.parameters():
            #             yield torch.tensor(
            #                 p.flatten(end_dim=1), requires_grad=True, device=p.device
            #             )
            #     self._fake_parameters = _fake_parameters
            if hasattr(self, "_fake_params"):
                del self._fake_params
            self.register_module(
                "_fake_params",
                nn.ParameterList(
                    [
                        nn.ParameterList(
                            [nn.Parameter(p[i].flatten()) for p in self.parameters()]
                        )
                        for i in range(self.n_genes)
                    ]
                ),
            )
            # from here on, no new parameters should be added
            self.optimizers = [
                torch.optim.LBFGS(
                    self._fake_params[i],
                    lr=self.fitting_configs["lr"],
                    max_iter=10,
                    tolerance_change=self.fitting_configs["tol"],
                    line_search_fn="strong_wolfe",
                )
                for i in range(self.n_genes)
            ]

            def closure():
                i = self._temp_mask
                self.optimizers[i].zero_grad()
                with torch.no_grad():
                    for p, fp in zip(self.parameters(), self._fake_params[i]):
                        # print(p.shape, fp.shape)
                        if p.grad is not None:
                            p.grad[i].zero_()
                        else:
                            p.grad = torch.zeros_like(p)
                        # copy value from fake to real
                        p[i].data.copy_(fp.data.reshape_as(p[i]))
                neg_log_prob = -self()[i].sum()
                neg_log_prob.backward()
                # print(self.beta.grad)
                with torch.no_grad():  # copy gradient from real to fake
                    for p, fp in zip(self.parameters(), self._fake_params[i]):
                        if fp.grad is None:
                            fp.grad = torch.zeros_like(fp)
                        fp.grad.copy_(p.grad[i].reshape_as(fp))
                return neg_log_prob

            self._closure = closure

        # learning rate scheduler
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer,
        #     patience=int(self.fitting_configs["patience"] / 4) + 1,
        #     factor=0.1,
        #     verbose=verbose,
        # )

    def _eta(self):
        """Output the eta based on the linear model."""
        # (1, n_spots, n_factors) @ (n_genes, n_factors, n_isos - 1) + (n_genes, 1, n_isos - 1)
        # -> (n_genes, n_spots, n_isos - 1)
        return self.X_spot.matmul(self.beta) + self.bias_eta.unsqueeze(-2)

    def _alpha(self):
        """Convert eta (n_isos - 1) to alpha (n_isos)."""
        # alpha is the expected proportion of isoforms
        # alpha.shape = (n_genes, n_spots, n_isos)
        # the last isoform will have constant zero eta across space and
        # its proportion is given by 1/(1 + sum(exp(eta)))
        eta = self._eta()  # (n_genes, n_spots, n_isos - 1)
        alpha = torch.cat(
            [eta, torch.zeros(self.n_genes, self.n_spots, 1, device=self.device)],
            dim=-1,
        ) # (n_genes, n_spots, n_isos)
        alpha = torch.softmax(alpha, dim=-1)  # (n_genes, n_spots, n_isos)
        return alpha

    def get_isoform_ratio(self):
        """Extract the fitted isoform ratio across space."""
        return self._alpha().detach()  # (n_genes, n_spots, n_isos)

    def forward(self):
        """Calculate log probability given data."""
        return log_prob_fastmult_batched(
            self._alpha().transpose(1, 2), self.counts.transpose(1, 2)
        )

    """Functions to calculate Hessian.
    """

    def _get_log_lik_gradient_beta_bias(self):
        """Get the gradient of the log joint probability wrt beta and bias."""
        # calculate the gradient wrt eta
        # (n_genes, n_spots, n_isos)
        d_l_d_eta = self.counts - self._alpha() * self.counts.sum(axis=-1, keepdim=True)

        # calculate the gradient wrt beta and bias
        X_expand = torch.cat(
            [
                self.X_spot,
                torch.ones(1, self.n_spots, 1, device=self.device),
            ],
            dim=-1,
        ) # (1, n_spots, n_factors + 1)
        # score of shape (n_genes, n_factors + 1, n_isos - 1)
        score = X_expand.transpose(1, 2).matmul(d_l_d_eta.detach()[..., :-1])

        return score

    def _get_log_lik_hessian_eta(self):
        """Get the Hessian matrix of the log joint probability wrt eta."""
        n_isos = self.n_isos
        props = self._alpha()  # (n_genes, n_spots, n_isos)

        # multinom_hessian[s] = - sum(counts[s,:]) * {diag(p) - p@p.T}
        # (n_genes, n_spots, n_isos - 1, n_isos - 1)
        multinom_hessian = -self.counts.sum(-1).view(self.n_genes, -1, 1, 1) * (
            props[..., :-1].unsqueeze(-1).expand(-1, -1, -1, n_isos - 1)
            * torch.eye(n_isos - 1, device=self.device)
            - torch.einsum("bsi,bsj->bsij", (props[..., :-1], props[..., :-1]))
        )

        # reshape the hessian into (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        # (1) zeros for spots i != j
        # (2) at spot i, isoform j and k are connected via multinom_hessian[i, j, k]
        return _melt_tensor_along_first_dim(multinom_hessian)

    def _get_log_lik_hessian_beta_bias(self):
        """Get the Hessian matrix of the log joint probability wrt the fixed effects."""
        # combine beta and the intercept bias_eta
        X_expand = torch.cat(
            [
                self.X_spot,
                torch.ones(1, self.n_spots, 1, device=self.device),
            ],
            dim=-1,
        ) # (1, n_spots, n_factors + 1)
        # (n_genes, n_spots * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        X_expand_full = torch.stack(
            [torch.block_diag(*[x] * (self.n_isos - 1)) for x in X_expand], dim=0
        )

        # calculate the hessian of the log likelihood wrt [beta, bias]
        # W is the hessian of the log multinomial likelihood wrt eta := X@beta + nu
        # W[s] = - sum(counts[s,:]) * {diag(p) - p@p.T}
        # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        W = self._get_log_lik_hessian_eta()
        # hessian_beta = X.T @ W @ X, ((n_factors + 1)(n_isos - 1), (n_factors + 1)(n_isos - 1))
        # (n_genes, (n_factors + 1) * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        hessian_beta_expand = (
            X_expand_full.transpose(1, 2).matmul(W).matmul(X_expand_full)
        )

        return hessian_beta_expand

    """Functions for model fitting.
    """

    def _update_gradient_descent(self):
        """Update the model parameters using gradient descent."""
        if self.fitting_configs["optim"] == "lbfgs":
            [optimizer.zero_grad() for optimizer in self.optimizers]
            with torch.no_grad():
                for p in self.parameters():
                    if not p.grad is None:
                        p.grad.zero_()
        else:
            self.optimizer.zero_grad()

        # minimize the negative log-likelihood or the negative log-marginal-likelihood
        neg_log_prob = -self()[~self.convergence].sum()
        neg_log_prob.backward()

        # gradient-based updates
        if self.fitting_configs["optim"] == "lbfgs":
            # update all parameters using L-BFGS
            [optimizer.zero_grad() for optimizer in self.optimizers]
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            for idx, optimizer in enumerate(self.optimizers):
                self._temp_mask = idx
                optimizer.step(self._closure)
            with torch.no_grad():
                for idx, fps in enumerate(self._fake_params):
                    for p, fp in zip(self.parameters(), fps):
                        p[idx].data.copy_(fp.data.reshape_as(p[idx]))
            # print(self.beta[0], self.bias_eta[0])
        elif self.fitting_configs["optim"] in ["adam", "sgd"]:
            # update the remaining parameters with non-zero gradients using gradient descent
            self.optimizer.step()
        else:
            raise NotImplementedError(
                f"Optimization method {self.fitting_configs['optim']} is not supported."
            )

    def _update_newton(self, step=0.9):
        """Update the model parameters using Newton's method."""
        n_genes, n_isos, n_factors = self.n_genes, self.n_isos, self.n_factors

        # combine beta and bias_eta
        # (n_genes, n_factors, n_isos - 1), (n_genes, 1, n_isos - 1) -> (n_genes, n_factors + 1, n_isos - 1)
        beta_expand = torch.cat([self.beta, self.bias_eta.unsqueeze(1)], dim=1)

        # calculate gradient and hessian
        # gradient_beta_expand = torch.cat([self.beta.grad, self.bias_eta.grad.reshape(1,-1)], dim=0)
        # (n_genes, n_factors + 1, n_isos - 1)
        gradient_beta_expand = self._get_log_lik_gradient_beta_bias()
        # (n_genes, (n_factors + 1) * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        hessian_beta_expand = self._get_log_lik_hessian_beta_bias()
        # for numerical stability
        # (n_genes, (n_factors + 1) * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        hessian_beta_expand += 1e-5 * torch.eye(
            (n_factors + 1) * (n_isos - 1), device=self.device
        )

        # find the new beta and bias_eta using the Newton's method
        # (n_genes, (n_factors + 1) * (n_isos - 1))
        right = torch.einsum(
            "bij,bj->bi", hessian_beta_expand, beta_expand.transpose(1, 2).flatten(1)
        ) - step * gradient_beta_expand.transpose(1, 2).flatten(1)
        beta_expand_new = (
            torch.linalg.solve(hessian_beta_expand, right)
            .reshape(n_genes, n_isos - 1, n_factors + 1)
            .transpose(1, 2)
        )

        # extract beta and bias_eta
        beta_new = beta_expand_new[:, :-1, :]  # (n_genes, n_factors, n_isos - 1)
        bias_eta_new = beta_expand_new[:, -1, :]  # (n_genes, n_isos - 1)

        # update the parameters and clear the gradients
        self.beta.data.copy_(update_at_idx(self.beta, beta_new, self.convergence))
        self.bias_eta.data.copy_(
            update_at_idx(self.bias_eta, bias_eta_new, self.convergence)
        )

    def _update_iwls(self):
        """Update the model parameters using the iteratively reweighted least squares (IWLS)."""
        n_genes, n_spots, n_isos, n_factors = (
            self.n_genes,
            self.n_spots,
            self.n_isos,
            self.n_factors,
        )
        props = self._alpha()  # (n_genes, n_spots, n_isos)

        # define the working variable (pseudo data)
        # working_y = eta + [d_eta/d_mu][counts - mu]
        # [d_eta/d_mu] = [d_mu/d_eta]^-1 = W^-1
        # W = N_i * (diag(alpha[i,:-1]) - alpha[i,:-1].T @ alpha[i,:-1]) at given location i

        # W = sum(counts[s,:]) * {diag(p) - p@p.T}
        # (n_genes, n_spots, n_isos - 1, n_isos - 1)
        W = self.counts.sum(-1).view(n_genes, n_spots, 1, 1) * (
            props[..., :-1].unsqueeze(-1).expand(-1, -1, -1, n_isos - 1)
            * torch.eye(n_isos - 1, device=self.device)
            - torch.einsum("bsi,bsj->bsij", (props[..., :-1], props[..., :-1]))
        )
        # for stability
        W_inv = torch.linalg.inv(W + 1e-5 * torch.eye(n_isos - 1, device=self.device))
        residuals = self.counts - props * self.counts.sum(-1, keepdim=True)
        working_y = residuals[..., :-1].unsqueeze(-1)  # (n_genes, n_spots, n_isos - 1, 1)
        working_y = (W_inv.matmul(working_y)).squeeze(-1)  # (n_genes, n_spots, n_isos - 1)
        working_y += self._eta()  # (n_genes, n_spots, n_isos - 1)

        # calculate the IWLS update for beta and bias_eta
        # beta_new = (X.T @ W @ X)^-1 @ X.T @ W @ y
        # reshape W
        # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        W = _melt_tensor_along_first_dim(W)
        # reshape X to include the intercept term
        # (1, n_spots, n_factors + 1)
        X_expand = torch.cat(
            [self.X_spot, torch.ones(1, n_spots, 1, device=self.device)], dim=-1
        )
        # (1, n_spots * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        X_expand_full = torch.stack(
            [torch.block_diag(*[x] * (n_isos - 1)) for x in X_expand], dim=0
        )
        # calculate the analytical solution
        # (n_genes, (n_factors + 1) * (n_isos - 1), n_spots * (n_isos - 1))
        Xt_W = X_expand_full.transpose(1, 2).matmul(W)
        # (n_genes, (n_factors + 1) * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        Xt_W_X = Xt_W.matmul(X_expand_full)
        # add a small value to the diagonal for stability
        Xt_W_X += 1e-5 * torch.eye((n_factors + 1) * (n_isos - 1))

        # (n_genes, (n_factors + 1) x (n_isos - 1))
        Xt_W_y = torch.einsum("bij,bj->bi", Xt_W, working_y.transpose(1, 2).flatten(1))
        res = torch.linalg.solve(Xt_W_X, Xt_W_y).reshape(
            n_genes, n_isos - 1, n_factors + 1
        )

        # extract beta and bias_eta
        beta_new = res[..., :-1].transpose(1, 2)  # (n_genes, n_factors, n_isos - 1)
        bias_eta_new = res[..., -1]  # (n_genes, n_isos - 1)

        # update the parameters and clear the gradients
        # with torch.no_grad():
        #     self.beta[~self.convergence].copy_(beta_new[~self.convergence])
        #     self.bias_eta[~self.convergence].copy_(bias_eta_new[~self.convergence])
        self.beta.data.copy_(update_at_idx(self.beta, beta_new, self.convergence))
        self.bias_eta.data.copy_(
            update_at_idx(self.bias_eta, bias_eta_new, self.convergence)
        )

    def fit(self, diagnose: bool = False, verbose: bool = False, quiet: bool = False, random_seed = None):
        """Fit the model using all data"""
        if random_seed is not None: # set random seed for reproducibility
            torch.manual_seed(random_seed)

        if self.fitting_method == "gd":  # use gradient descent
            # configure the optimizer and start training
            self._configure_optimizer(verbose=verbose)
            self.train()

        max_epochs = self.fitting_configs["max_epochs"]
        patience = self.fitting_configs["patience"]
        tol = self.fitting_configs["tol"]
        max_epochs = 10000 if max_epochs == -1 else max_epochs
        patience = patience if patience > 0 else 1

        # set iteration limits
        batch_size = self.n_genes
        t_start = timer()
        logger = PatienceLogger(batch_size, patience, min_delta=tol, diagnose=diagnose)

        while logger.epoch < max_epochs and not logger.convergence.all():
            # update the model parameters
            if self.fitting_method == "gd":
                self._update_gradient_descent()
            elif self.fitting_method == "newton":
                self._update_newton()
            elif self.fitting_method == "iwls":
                self._update_iwls()

            # check convergence
            with torch.no_grad():
                # calculate the negative log-likelihood
                neg_log_prob = -self().detach().cpu()

                # d_loss = prev_loss - neg_log_prob.detach().item()

                # # update the epoch and patience
                # epoch += 1
                # if d_loss < 0:  # if loss increases
                #     patience -= 1

                # # update the loss
                # prev_loss = neg_log_prob.detach().item()

            # if self.fitting_method == "gd":
            # update learning rate
            # self.scheduler.step(neg_log_prob[~self.convergence].mean())

            logger.log(
                neg_log_prob,
                {
                    k: v.detach().cpu()
                    for k, v in self.named_parameters()
                    if "_fake" not in k
                },
            )
            self.convergence.copy_(logger.convergence)

            if (verbose and not quiet) and logger.epoch % 10 == 0:
                print(
                    f"Epoch {logger.epoch}. Loss (neg_log_prob): {logger.best_loss.mean():.4f}. "
                )

        # check model convergence
        num_not_converge = (~logger.convergence).sum()
        if num_not_converge:
            warnings.warn(
                f"{num_not_converge} Genes did not converge after epoch {max_epochs}. "
                "Try larger max_epochs."
            )

        # save runtime
        t_end = timer()
        self.fitting_time = t_end - t_start

        if not quiet:  # print final message
            print(
                f"Time {self.fitting_time:.2f}s. Total epoch {logger.epoch}. Final loss "
                f"(neg_log_prob): {neg_log_prob.mean():.3f}."
            )

        # collect parameters corresponding to the best epoch for each sample in batch
        if max_epochs > 0:
            for k, v in self.named_parameters():
                if "_fake" not in k:
                    v.data.copy_(logger.best_params[k])

        self.logger = logger

        return logger.params_iter

    def clone(self):
        """Clone a model with the same set of parameters."""
        new_model = type(self)(
            fitting_method=self.fitting_method, fitting_configs=self.fitting_configs
        )
        new_model.setup_data(counts=self.counts, design_mtx=self.X_spot)
        new_model.load_state_dict(self.state_dict())

        return new_model

    def update_params_from_dict(self, params):
        """Update a subset of model parameters with a dictionary of parameters.

        Args:
                params: a dictionary of parameters to be updated. The keys must be
                        existing parameter names in the model.
        """
        new_params = self.state_dict()
        new_params.update(params)
        self.load_state_dict(new_params)


class MultinomGLMM(MultinomGLM, BaseModel, nn.Module):
    """The Multinomial Generalized Linear Mixed Model for spatial isoform expression.

    Y ~ Multinomial(alpha, Y.sum(1))
    eta = multinomial-logit(alpha) = X @ beta + bias_eta + nu
    nu ~ MVN(0, sigma^2 * (theta * V_sp + (1-theta) * I) =
            MVN(0, sigma_sp^2 * V_sp + sigma_nsp^2 * I)

    Given isoform counts of a gene Y (n_spots, n_isos) and design matrix X (n_spots, n_factors),
    MultinomGLMM.fit will find the MAP estimates of the following learnable parameters:
    - beta: (n_factors, n_isos - 1) covariate coefficients of the fixed effect term.
    - bias_eta: (n_isos - 1) intercepts of the fixed effect term.
    - nu: (n_spots, n_isos - 1) the random effect term.
    - variance components: each of length n_isos - 1 or 1 if self.share_variance
            if self.var_parameterization_sigma_theta: # by default, False
                    - (sigma, theta):
                            the total variance and the proportion of spatial variability of the random effect term.
            else:
                    - (sigma_sp, sigma_nsp):
                            the spatial and non-spatial variance components of the random effect term.

    Inference algorithms can be categorized into two types based on the optimization objective:
    - Joint: Maximize the joint likelihood (with the random effect nu).
            This is equivalent to the first-order Laplace approximation of the marginal likelihood.
    - Marginal: Maximize the marginal likelihood (with the random effect nu integrated out).
            The integral is approximated by a second-order Laplace approximation.

    Methods implemented:
    - 'joint_gd': Maximize the joint likelihood using gradient descent.
    - 'joint_newton': Maximize the joint likelihood using Newton's method.
    - 'marginal_gd': Maximize the marginal likelihood using gradient descent.
    - 'marginal_newton': Maximize the marginal likelihood using Newton's method.
            nu is first updated using Newton's method every 'update_nu_every_k' iterations, and
            beta, bias_eta, and variance components are updated using gradient descent.

    See README for complete details of the model.

    TODO:
    - Use a parameter to toggle between using `n_isos - 1` and `n_isos`. For `n_isos`,
            other normalization methods are needed for Multinomial.
    - Implement held-out likelihood for model selection.
    """

    def __init__(
        self,
        share_variance: bool = True,
        var_parameterization_sigma_theta: bool = True,
        var_fix_sigma: bool = True,
        var_prior_model: str = "none",
        var_prior_model_params: dict = {},
        init_ratio: str = "observed",
        fitting_method: str = "joint_gd",
        fitting_configs: dict = {},
    ):
        super().__init__()

        # specify the parameterization of variance components
        # if True:
        # 	var(sigma, theta) = sigma^2 (theta * V_sp + (1-theta) * I)
        # else:
        # 	var(sigma_sp, sigma_nsp) = sigma_sp^2 * V_sp + sigma_nsp^2 * I
        self.var_parameterization_sigma_theta = var_parameterization_sigma_theta
        self.share_variance = (
            share_variance  # whether to share variance across isoforms
        )
        self.var_fix_sigma = var_fix_sigma  # whether to fix sigma

        # specify the prior model on sigma^2 (or sigma_sp^2 + sigma_nsp^2))
        assert var_prior_model in ["none", "gamma", "inv_gamma"]
        self.var_prior_model = var_prior_model  # prior on the variance size sigma
        if self.var_prior_model == "gamma":
            # Chung, Yeojin, et al. Psychometrika 78.4 (2013): 685-709.
            # this prior is applied on sigma (or sqrt(sigma_sp^2 + sigma_nsp^2))
            # Gamma(2, 0.3): prior mode of sigma ~= 3
            self.var_prior_model_params = {
                "alpha": 2.0,
                "beta": 0.3,
            }
            self.var_prior_model_params.update(var_prior_model_params)
            self.var_prior_model_dist = Gamma(
                self.var_prior_model_params["alpha"],
                self.var_prior_model_params["beta"],
            )
        elif self.var_prior_model == "inv_gamma":
            # conjugacy prior
            # this prior is applied on sigma^2 (or sigma_sp^2 + sigma_nsp^2)
            # InverseGamma(3, 0.5): prior mode of sigma^2 ~= 0.25
            self.var_prior_model_params = {
                "alpha": 3,
                "beta": 0.5,
            }
            self.var_prior_model_params.update(var_prior_model_params)
            self.var_prior_model_dist = InverseGamma(
                self.var_prior_model_params["alpha"],
                self.var_prior_model_params["beta"],
            )
        else:  # 	no/flat prior on sigma
            self.var_prior_model_params = {}
            self.var_prior_model_dist = None

        # specify the initialization method for the logit isoform usage ratio gamma
        assert init_ratio in ["observed", "uniform"]
        self.init_ratio = init_ratio

        # specify the fitting method
        assert fitting_method in [
            "joint_gd",
            "joint_newton",  # joint likelihood
            "marginal_gd",
            "marginal_newton",  # marginal likelihood
        ]
        self.fitting_method = fitting_method

        # specify the fitting configurations
        self.fitting_configs = {  # default configurations
            "lr": 1e-2,
            "optim": "adam",
            "tol": 1e-5,
            "max_epochs": 1000,
            "patience": 5,
        }
        self.fitting_configs.update(fitting_configs)
        if self.fitting_method == "joint_newton":
            # Newton's method is fast but can't very well handel saddle points
            # use small patience to avoid loss increase in the final iterations
            self.fitting_configs["patience"] = 2
            assert (
                self.var_fix_sigma is False
            ), "Newton's method requires sigma to be optimized."

        elif self.fitting_method == "marginal_gd":
            self.fitting_configs["lr"] = 1e-1

        elif self.fitting_method == "marginal_newton":
            # update nu using Newton's method every 'update_nu_every_k' iterations
            # update beta, bias_eta, and variance components using gradient descent
            self.fitting_configs["lr"] = 1e-1
            self.fitting_configs["patience"] = 10
            self.fitting_configs["update_nu_every_k"] = 3

        # override the default if user provides a different configuration
        self.fitting_configs.update(fitting_configs)

        # for now, restricting the optimization method to Adam, SGD and lbfgs
        assert self.fitting_configs["optim"] in ["adam", "sgd", "lbfgs"]

        # specify the fitting outcomes
        self.fitting_time = 0

    def __str__(self):
        return (
            f"A Multinomial Generalized Linear Mixed Model (GLMM)\n"
            + f"- Number of genes in the batch: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of isoforms per gene: {self.n_isos}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + "- Variance formulation:\n"
            + f"\t* Parameterized using sigma and theta: {self.var_parameterization_sigma_theta}\n"
            + f"\t* Learnable variance: {not self.var_fix_sigma}\n"
            + f"\t* Same variance across classes: {self.share_variance}\n"
            + f"\t* Prior on total variance: {self.var_prior_model}\n"
            + f"- Initialization method: {self.init_ratio}\n"
            + f"- Fitting method: {self.fitting_method}"
        )

    def setup_data(
        self,
        counts,
        corr_sp=None,
        design_mtx=None,
        device="cpu",
        corr_sp_eigvals=None,
        corr_sp_eigvecs=None,
    ):
        """Set up the data for the model.

        Args:
                counts: tensor(n_genes, n_spots, n_isoforms)
                corr_sp: tensor(n_spots, n_spots), spatial covariance matrix
                design_mtx: tensor(n_spots, n_covariates)
                device: 'cpu', or 'cuda'
                        'mps' currently not supported (torch.lgamma not supported on mps)
                corr_sp_eigvals: tensor(n_spots,)
                corr_sp_eigvecs: tensor(n_spots, n_spots)
        """
        # need to switch to a different count model (e.g., Poisson) when only one isoform is provided

        assert device in ["cpu", "cuda"]
        self.device = torch.device(device)

        if counts.ndim == 2:
            counts = counts.unsqueeze(0) # (1, n_spots, n_isos)
            print(
                "Batched calculation has been implemented. Provide a batch of counts to speed up calculation."
            )
        else:
            if not counts.ndim == 3:
                raise ValueError(
                    f"counts must be a 2D or 3D tensor. Got shape: {counts.shape}"
                )
        if counts.shape[2] == 1:
            raise NotImplementedError(
                "Only one isoform provided. Please use a different count model."
            )
        # set model dimensions based on the input shape
        self.n_genes, self.n_spots, self.n_isos = counts.shape

        if design_mtx is None:
            # initialize an empty design matrix of shape (1, n_spots, 0)
            design_mtx = torch.ones(1, self.n_spots, 0)
        elif design_mtx.ndim == 2:
            design_mtx = design_mtx.unsqueeze(0) # (1, n_spots, n_factors)
        elif design_mtx.ndim == 3:
            # all genes to test should share the same design matrix
            assert design_mtx.shape[0] == 1, "Batched design matrix is currently not supported."
        else:
            raise ValueError(
                f"design_mtx must be a 2D or 3D tensor. Got shape: {design_mtx.shape}"
            )
        self.n_factors = design_mtx.shape[-1]
        assert design_mtx.shape == (1, self.n_spots, self.n_factors), f"Invalide design matrix shape: {design_mtx.shape}"

        # (n_genes, n_spots, n_isos), int, the observed counts
        self.register_buffer("counts", counts)
        # (1, n_spots, n_factors), the input design matrix of n_factors covariates
        self.register_buffer("X_spot", design_mtx)

        # either the corr_sp or the corr_sp_eigvals and corr_sp_eigvecs must be provided
        assert (corr_sp is not None) or (
            corr_sp_eigvals is not None and corr_sp_eigvecs is not None
        )

        if corr_sp is not None:
            # ignore eigendecomposition if corr_sp is provided
            if corr_sp_eigvals is not None or corr_sp_eigvecs is not None:
                warnings.warn(
                    "Both the correlation matrix and its eigendecomposition are provided. "
                    "The latter will be ignored."
                )

            assert corr_sp.shape == (self.n_spots, self.n_spots)
            # (n_spots, n_spots), the spatial covariance matrix
            self.register_buffer("corr_sp", corr_sp)

            # precompute the eigendecomposition of corr_sp to speed up matrix inverse
            # corr_sp = corr_sp_eigvecs @ diag(corr_sp_eigvals) @ corr_sp_eigvecs.T
            try:
                corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.corr_sp)
            except RuntimeError:
                # fall back to eig if eigh fails
                # related to a pytorch bug on M1 macs, see https://github.com/pytorch/pytorch/issues/83818
                corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eig(self.corr_sp)
                corr_sp_eigvals = torch.real(corr_sp_eigvals)
                corr_sp_eigvecs = torch.real(corr_sp_eigvecs)

            self.register_buffer("corr_sp_eigvals", corr_sp_eigvals)
            self.register_buffer("corr_sp_eigvecs", corr_sp_eigvecs)

        else:
            assert corr_sp_eigvals.shape == (self.n_spots,)
            assert corr_sp_eigvecs.shape == (self.n_spots, self.n_spots)

            corr_sp = corr_sp_eigvecs.matmul(torch.diag(corr_sp_eigvals)).matmul(
                corr_sp_eigvecs.t()
            )
            # (n_spots, n_spots), the spatial covariance matrix
            self.register_buffer("corr_sp", corr_sp)
            # (n_spots,), the eigenvalues of the spatial covariance matrix
            self.register_buffer("corr_sp_eigvals", corr_sp_eigvals)
            # (n_spots, n_spots), the eigenvectors of the spatial covariance matrix
            self.register_buffer("corr_sp_eigvecs", corr_sp_eigvecs)

        self.register_buffer("convergence", torch.zeros(self.n_genes, dtype=bool))

        # set up learnable parameters according to the model architecture
        self._configure_learnable_variables()
        self._initialize_params()

        # send to device
        self.to(self.device)

    def _configure_learnable_variables(self):
        """Set up learnable parameters according to the model architecture."""
        # the random effect term
        nu = torch.zeros(self.n_genes, self.n_spots, self.n_isos - 1)

        # the fixed effect terms
        beta = torch.zeros(
            self.n_genes, self.n_factors, self.n_isos - 1
        )  # covariate coefficients
        bias_eta = torch.zeros(self.n_genes, self.n_isos - 1)  # intercepts

        # optimize using gradient descent / newton's method
        self.register_parameter("nu", nn.Parameter(nu))
        self.register_parameter("beta", nn.Parameter(beta))
        self.register_parameter("bias_eta", nn.Parameter(bias_eta))

        # the variance components
        n_var_components = 1 if self.share_variance else self.n_isos - 1
        if self.var_parameterization_sigma_theta:
            # cov = sigma^2 (theta * V_sp + (1-theta) * I)
            sigma = torch.ones(self.n_genes, n_var_components)
            theta_logit = torch.zeros(self.n_genes, n_var_components)
            self.register_parameter("sigma", nn.Parameter(sigma))
            self.register_parameter("theta_logit", nn.Parameter(theta_logit))

            if self.var_fix_sigma:
                self.sigma.requires_grad = False
        else:
            # cov = sigma_sp^2 * V_sp + sigma_nsp^2 * I
            sigma_sp = torch.ones(self.n_genes, n_var_components)
            sigma_nsp = torch.ones(self.n_genes, n_var_components)
            self.register_parameter("sigma_sp", nn.Parameter(sigma_sp))
            self.register_parameter("sigma_nsp", nn.Parameter(sigma_nsp))

            if self.var_fix_sigma:
                self.sigma_nsp.requires_grad = False

    def _initialize_params(self):
        """Initialize model parameters."""
        # initialize the random effect term
        if self.init_ratio == "observed":
            # initialize isoform ratios from observed probabilities
            # (n_genes, n_spots, n_isos)
            counts_props = self.counts / (self.counts.sum(-1, keepdim=True) + 1e-5)
            with torch.no_grad():
                self.nu.copy_(
                    (
                        (counts_props[..., :-1] + 1e-5)
                        / (counts_props[..., -1:] + 1e-5)
                    ).log()
                )
        elif self.init_ratio == "uniform":
            # initialize isoforms uniformly across space
            with torch.no_grad():
                self.nu.copy_(torch.zeros(self.n_genes, self.n_spots, self.n_isos - 1))

        # initial estimate of the variance components
        sigma_init = (
            (self.counts.var(1).mean(1) / self.counts.sum(2).mean(1))
            .clamp(max=0.9)
            .pow(0.5)
        ).unsqueeze(-1)

        # initialize proportion of spatial variance to be ~0.05
        if self.var_parameterization_sigma_theta:
            with torch.no_grad():
                self.sigma.copy_(torch.ones_like(self.sigma) * sigma_init)
                self.theta_logit.copy_(torch.ones_like(self.theta_logit) * -3.0)
        else:
            with torch.no_grad():
                self.sigma_nsp.copy_(torch.ones_like(self.sigma_nsp) * sigma_init)
                self.sigma_sp.copy_(torch.ones_like(self.sigma_sp) * 0.2 * sigma_init)

    """Below are a bunch of helper functions to update intermediate variables after each optimization step.
    """

    def var_total(self):
        """Output the total variance."""
        if self.var_parameterization_sigma_theta:
            var_total = self.sigma**2
        else:
            var_total = self.sigma_sp**2 + self.sigma_nsp**2

        if var_total.min() < 1e-2:
            warnings.warn("Total variance is close to zero.")
        var_total = torch.clip(var_total, min=1e-2) # (n_genes, n_var_components)
        return var_total

    def var_sp_prop(self):
        """Output the proporptions of the spatial variance."""
        # return: (n_genes, n_var_components)
        if self.var_parameterization_sigma_theta:
            return torch.sigmoid(self.theta_logit)
        else:
            return self.sigma_sp**2 / self.var_total()

    def _corr_eigvals(self):
        """Output the eigenvalues of the correlation matrix."""
        var_sp_prop = self.var_sp_prop().unsqueeze(-1)  # (n_genes, n_var_components, 1)
        # return: (n_genes, n_var_components, n_spots)
        return var_sp_prop * self.corr_sp_eigvals.unsqueeze(0) + (1 - var_sp_prop)
        # return torch.stack(
        #     [(p * self.corr_sp_eigvals + (1 - p)) for p in self.var_sp_prop()], dim=0
        # )

    def _cov_eigvals(self):
        """Output the eigenvalues of the covariance matrix."""
        # (n_genes, n_var_components, n_spots) * (n_genes, n_var_components, 1)
        # return: (n_genes, n_var_components, n_spots)
        return self._corr_eigvals() * self.var_total().unsqueeze(-1)

    def _cov(self):
        """Reconstruct the covariance of the random effect."""
        # return (
        #     self.corr_sp_eigvecs.unsqueeze(0)
        #     @ torch.diag_embed(self._cov_eigvals())
        #     @ self.corr_sp_eigvecs.T.unsqueeze(0)
        # )  # (n_isos - 1) or 1 x n_spots x n_spots
        # return: (n_genes, n_var_components, n_spots, n_spots)
        return self.corr_sp_eigvecs[None, None, ...].matmul(
            torch.diag_embed(self._cov_eigvals()).matmul(
                self.corr_sp_eigvecs.transpose(0, 1)[None, None, ...]
            )
        )

    def _inv_cov(self):
        """Reconstruct the inverse covariance of the random effect."""
        # return (
        #     self.corr_sp_eigvecs.unsqueeze(0)
        #     @ torch.diag_embed(1 / self._cov_eigvals())
        #     @ self.corr_sp_eigvecs.T.unsqueeze(0)
        # )  # (n_isos - 1) or 1 x n_spots x n_spots
        # return: (n_genes, n_var_components, n_spots, n_spots)
        return self.corr_sp_eigvecs[None, None, ...].matmul(
            torch.diag_embed(1 / self._cov_eigvals()).matmul(
                self.corr_sp_eigvecs.transpose(0, 1)[None, None, ...]
            )
        )

    def _eta(self):
        """Output the eta based on the linear model."""
        # eta = X @ beta + bias_eta + nu
        # eta.shape = (n_genes, n_spots, n_isos - 1)
        return (
            self.nu
            + self.X_spot.matmul(self.beta)
            + self.bias_eta.unsqueeze(1)
        )

    """Functions to calculate log likelihoods.
    """

    def _calc_log_prob_prior_sigma(self):
        """Calculate log prob of the prior on sigma."""
        if self.var_prior_model == "inv_gamma":  # prior on sigma^2
            return self.var_prior_model_dist.log_prob(self.var_total()).sum(-1)
        elif self.var_prior_model == "gamma":  # prior on sigma
            return self.var_prior_model_dist.log_prob(self.var_total().pow(0.5)).sum(-1)
        else:
            return torch.zeros(self.n_genes, device=self.device)

    def _calc_log_prob_joint(self):
        """Calculate log joint probability given data."""
        # add prior prob of sigma_total
        log_prob = self._calc_log_prob_prior_sigma() # (n_genes,)

        # add mvn prob of nu ~ MVN(0, S)
        data = self.nu.transpose(1, 2) # (n_genes, n_isos - 1, n_spots)
        # use the same variance components for all genes if cov_eigvals.shape[1] == 1
        cov_eigvals = self._cov_eigvals().expand(
            self.n_genes, data.shape[1], self.n_spots
        )
        cov_eigvecs = self.corr_sp_eigvecs.expand(
            1, data.shape[1], self.n_spots, self.n_spots
        )
        log_prob += log_prob_fastmvn_batched(
            locs=torch.zeros_like(data),
            cov_eigvals=cov_eigvals,
            cov_eigvecs=cov_eigvecs,
            data=data,
        )

        # add Multinomial likelihood of the counts
        log_prob += log_prob_fastmult_batched(
            self._alpha().transpose(1, 2), self.counts.transpose(1, 2)
        )

        return log_prob # (n_genes,)

    def _calc_log_prob_marginal(self):
        """Calculate the log marginal probability (integrating out random effect nu)."""
        # by Laplace approximation, the log marginal probability is given by the following:
        # log_marginal ~= log_joint - 1/2 logdet_neg_hessian
        log_prob = self._calc_log_prob_joint() # (n_genes,)
        full_hessian = (
            self._get_log_lik_hessian_nu()
        )  # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))

        # save the cholesky for fast matrix inverse in Newton updates
        # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        self._chol_hessian_nu = torch.linalg.cholesky(-full_hessian)
        logdet_neg_hessian = 2 * torch.diagonal(
            self._chol_hessian_nu, dim1 = -2, dim2 = -1
        ).log().sum(-1) # (n_genes,)

        return log_prob - 0.5 * logdet_neg_hessian # (n_genes,)

    def forward(self):
        """Calculate the log-likelihood or log-marginal-likelihood of the model."""
        if self.fitting_method in ["marginal_gd", "marginal_newton"]:
            # calculate log marginal prob
            return self._calc_log_prob_marginal() # (n_genes,)
        else:
            # calculate log joint prob given data
            return self._calc_log_prob_joint() # (n_genes,)

    """Functions to calculate Hessian.
    """

    def _get_log_lik_hessian_nu(self):
        """Get the Hessian matrix of the log joint probability wrt the random effect nu."""
        n_genes, n_isos = self.n_genes, self.n_isos
        # full_hessian = multinom_hessian + mvn_hessian # n_spots * (n_isos - 1) x n_spots * (n_isos - 1)
        # mvn_hessian: (n_isos - 1) x n_spots x n_spots
        # mvn_hessian[q] = - cov[q]^-1 # n_spots * n_spots
        mvn_hessian = -self._inv_cov()  # (n_genes, n_var_components, n_spots, n_spots)
        if self.share_variance:  # the same variance across isoforms
            # (n_genes, n_isos - 1, n_spots, n_spots)
            mvn_hessian = mvn_hessian.expand(n_genes, n_isos - 1, -1, -1)
        # reshape the hessian into (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        full_hessian = torch.stack([
            torch.block_diag(*[_m for _m in m_gene])
            for m_gene in mvn_hessian
        ], dim = 0)

        # multinom_hessian[s] = - sum(counts[s,:]) * {diag(p) - p@p.T}
        # self._get_log_lik_hessian_eta() already reshapes the hessian into
        # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        multinom_hessian = self._get_log_lik_hessian_eta()
        full_hessian += multinom_hessian

        return full_hessian

    def _calc_log_prob_mvn_wrt_sigma(self, sigma_expand):
        """Helper function of log joint probability wrt sigmas for calculate Hessian.

        Args:
                sigma_expand: tensor(n_genes, n_var_components, 2). The two variance parameters,
                    either (sigma, theta) or (sigma_sp, sigma_nsp), are stacked along the last dimension.
        """
        if self.share_variance:  # the same variance components across isoforms
            sigma_expand = sigma_expand.expand(self.n_genes, self.n_isos - 1, -1)

        if self.var_parameterization_sigma_theta:
            sigma, theta_logit = sigma_expand[..., 0], sigma_expand[..., 1]
            var_sp_prop = torch.sigmoid(theta_logit).unsqueeze(-1) # (n_genes, n_iso - 1, -1)
            sigma_total = sigma # (n_genes, n_iso - 1)
            # (n_genes, n_iso - 1, n_spots) * (n_genes, n_iso - 1, 1) -> (n_genes, n_iso - 1, n_spots)
            cov_eigvals = (var_sp_prop * self.corr_sp_eigvals.unsqueeze(0) + (1 - var_sp_prop)) * \
                sigma_total.unsqueeze(-1)
        else:
            sigma_sp, sigma_nsp = sigma_expand[..., 0], sigma_expand[..., 1]
            sigma_total = torch.sqrt(sigma_sp**2 + sigma_nsp**2) # (n_genes, n_iso - 1)
            # (n_genes, n_iso - 1, 1) * (1, n_spots) -> (n_genes, n_iso - 1, n_spots)
            cov_eigvals = sigma_sp.unsqueeze(-1).pow(2) * self.corr_sp_eigvals.unsqueeze(0) + \
                sigma_nsp.unsqueeze(-1).pow(2)

        # MVN prior likelihood as a function of the input eta
        data = self.nu.transpose(1, 2) # (n_genes, n_isos - 1, n_spots)
        # use the same variance components for all genes if cov_eigvals.shape[1] == 1
        cov_eigvecs = self.corr_sp_eigvecs.expand(
            1, data.shape[1], self.n_spots, self.n_spots
        )
        log_prob = log_prob_fastmvn_batched(
            locs=torch.zeros_like(data),
            cov_eigvals=cov_eigvals,
            cov_eigvecs=cov_eigvecs,
            data=data,
        )

        # add prior prob of sigma_total
        if self.var_prior_model == "inv_gamma":  # prior on sigma^2
            log_prob += self.var_prior_model_dist.log_prob(sigma_total**2).sum(-1)
        elif self.var_prior_model == "gamma":  # prior on sigma
            log_prob += self.var_prior_model_dist.log_prob(sigma_total.abs()).sum(-1)

        return log_prob.mean()

    def _get_sum_of_grad_log_prob_mvn_wrt_sigma(self, sigma_expand):
        """Get the sum of gradients of the log joint probability wrt the variance components."""
        log_prob = self._calc_log_prob_mvn_wrt_sigma(sigma_expand)
        # sum over the batch dim to get shape of (n_var_components, 2)
        return torch.autograd.grad(log_prob, sigma_expand, create_graph=True)[0].sum(0)

    def _get_log_lik_hessian_sigma_expand(self):
        """Get the Hessian matrix of the log mvn wrt the variance components."""
        if self.var_parameterization_sigma_theta:
            sigma_expand = torch.stack(
                [self.sigma, self.theta_logit], dim=-1
            )  # (n_genes, n_var_components, 2)
        else:
            sigma_expand = torch.stack(
                [self.sigma_sp, self.sigma_nsp], dim=-1
            )  # (n_genes, n_var_components, 2)

        # calculate the hessian using pytorch's functional
        # x = sigma_expand.clone().detach().requires_grad_(True)
        hessian_sigma_expand = jacobian(
            self._get_sum_of_grad_log_prob_mvn_wrt_sigma, sigma_expand, vectorize=True
        ).permute(2, 0, 1, 3, 4) # (n_genes, n_var_components, 2, n_genes, n_var_components, 2)

        # hessian_sigma_expand = hessian(
        #     self._calc_log_prob_mvn_wrt_sigma, sigma_expand
        # )  # (n_genes, n_var_components, 2, n_genes, n_var_components, 2)

        n_var_comps = hessian_sigma_expand.shape[-2]
        hessian_sigma_expand = hessian_sigma_expand.reshape(
            self.n_genes, n_var_comps * 2, n_var_comps * 2
        ) # (n_genes, n_var_components * 2, n_var_components * 2)

        return hessian_sigma_expand

    """Optimization functions.
    """

    def _update_joint_sigma_expand_newton(self, return_variables=False):
        """Update the variance components by Newton's method.

        Need to first backpropagate the gradients of the log likelihood w.r.t the variance components.
        For concave functions like f(x) = - log(1/x), Newton's update will give worse results.
        """
        n_genes = self.n_genes

        # calculate the updates for variance parameters
        if self.var_parameterization_sigma_theta:
            sigma_expand = torch.stack(
                [self.sigma, self.theta_logit], dim=-1
            )  # (n_genes, n_var_components, 2)
            sigma_expand_grad = torch.stack(
                [self.sigma.grad, self.theta_logit.grad], dim=-1
            )  # (n_genes, n_var_components, 2)
        else:
            sigma_expand = torch.stack(
                [self.sigma_sp, self.sigma_nsp], dim=-1
            )  # (n_genes, n_var_components, 2)
            sigma_expand_grad = torch.stack(
                [self.sigma_sp.grad, self.sigma_nsp.grad], dim=-1
            )  # (n_genes, n_var_components, 2)

        # TODO: analytical solutions when var_prior_model in ['none', 'inv_gamma']?
        hessian_sigma_expand = (
            -self._get_log_lik_hessian_sigma_expand()
        )  # (n_genes, n_var_components * 2, n_var_components * 2)
        hessian_sigma_expand += 1e-5 * torch.eye(hessian_sigma_expand.shape[-1]).unsqueeze(0).to(
            self.device
        )  # for stability
        right = hessian_sigma_expand.matmul(
            sigma_expand.reshape(n_genes, -1, 1)
        ) - sigma_expand_grad.reshape(n_genes, -1, 1)  # (n_genes, n_var_components * 2, 1)
        sigma_expand_new = torch.linalg.solve(
            hessian_sigma_expand, right
        )  # (n_genes, n_var_components * 2, 1)
        sigma_expand_new = sigma_expand_new.reshape(
            sigma_expand.shape
        )  # (n_genes, n_var_components, 2)

        # update the parameters and clear the gradients
        with torch.no_grad():
            if self.var_parameterization_sigma_theta:
                self.sigma.copy_(sigma_expand_new[..., 0])
                self.theta_logit.copy_(sigma_expand_new[..., 1])
                self.sigma.grad.zero_()
                self.theta_logit.grad.zero_()
            else:
                self.sigma_sp.copy_(sigma_expand_new[..., 0])
                self.sigma_nsp.copy_(sigma_expand_new[..., 1])
                self.sigma_sp.grad.zero_()
                self.sigma_nsp.grad.zero_()

        if return_variables:
            return sigma_expand_new

    def _update_joint_newton(self, return_variables=False):
        """Calculate the Newton update of the fixed and random effects.

        For a given parameter p, the Newton update is given by:
                p_new = p - step * hessian^(-1) * gradient

        Returns: (if return_variables)
                nu_new: tensor(n_genes, n_spots, n_isos - 1)
                beta_new: tensor(n_genes, n_factors, n_isos - 1)
                bias_eta_new: tensor(n_genes, n_isos - 1)
                sigma_expand_new: tensor(n_genes, n_var_components, 2)
        """
        n_genes, n_spots, n_isos, n_factors = (
            self.n_genes, self.n_spots, self.n_isos, self.n_factors
        )
        step = 1  # step size of each update

        # update the random effect term nu
        # self._get_log_lik_hessian_nu() returns the hessian of the log joint likelihood
        # to minize loss (neg likelihood), need to take the negative
        hessian_nu = (
            -self._get_log_lik_hessian_nu()
        )  # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        hessian_nu += 1e-5 * torch.eye(hessian_nu.shape[-1]).unsqueeze(0).to(
            self.device
        )  # for stability
        gradient_nu = self.nu.grad.transpose(1, 2).reshape(n_genes, -1, 1)  # (n_genes, n_spots * (n_isos - 1), 1)
        right = (
            hessian_nu.matmul(self.nu.transpose(1, 2).reshape(n_genes, -1, 1)) - step * gradient_nu
        )  # (n_genes, n_spots * (n_isos - 1), -1)
        nu_new = torch.linalg.solve(hessian_nu, right).reshape(n_genes, n_isos - 1, n_spots).transpose(1, 2)
        # update the parameters and clear the gradients
        with torch.no_grad():
            self.nu.copy_(nu_new)
            self.nu.grad.zero_()

        # update the fixed effect term beta and bias
        # self._get_log_lik_hessian_beta_bias() returns the hessian of the log joint likelihood
        # to minize loss (neg likelihood), need to take the negative
        hessian_beta_expand = (
            -self._get_log_lik_hessian_beta_bias()
        )  # (n_genes, (n_factors + 1) * (n_isos - 1), (n_factors + 1) * (n_isos - 1))
        hessian_beta_expand += 1e-5 * torch.eye(hessian_beta_expand.shape[-1]).unsqueeze(0).to(
            self.device
        )

        # combine beta and bias_eta
        # (n_genes, n_factors + 1, n_isos - 1)
        gradient_beta_expand = torch.cat(
            [self.beta.grad, self.bias_eta.grad.unsqueeze(1)], dim=1
        )
        # (n_genes, n_factors + 1, n_isos - 1)
        beta_expand = torch.cat([self.beta, self.bias_eta.unsqueeze(1)], dim=1)
        right = hessian_beta_expand.matmul(
            beta_expand.transpose(1, 2).reshape(n_genes, -1, 1)
        ) - step * beta_expand.transpose(1, 2).reshape(n_genes, -1, 1)  # (n_genes, (n_factors + 1) * (n_isos - 1), 1)
        beta_expand_new = (
            torch.linalg.solve(hessian_beta_expand, right)
            .reshape(n_genes, n_isos - 1, n_factors + 1).transpose(1, 2)
        ) # (n_genes, n_factors + 1, n_isos - 1)

        # extract beta and bias_eta
        beta_new = beta_expand_new[:, :-1, :]  # (n_genes, n_factors, n_isos - 1)
        bias_eta_new = beta_expand_new[:, -1, :]  # (n_genes, n_isos - 1)
        # update the parameters and clear the gradients
        with torch.no_grad():
            self.beta.copy_(beta_new)
            self.bias_eta.copy_(bias_eta_new)
            self.beta.grad.zero_()
            self.bias_eta.grad.zero_()

        # update variance components
        sigma_expand_new = self._update_joint_sigma_expand_newton(
            return_variables=return_variables
        )

        if return_variables:
            return nu_new, beta_new, bias_eta_new, sigma_expand_new

    def _update_marginal_nu_newton(self, return_variables=False):
        """Calculate the Newton update of the random effects given the hessian."""
        try:
            # use the stored hessian cholesky computed by self._calc_log_prob_margin()
            chol = self._chol_hessian_nu # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        except AttributeError:
            # if cholesky is not available, compute it on the fly
            chol = torch.linalg.cholesky(-self._get_log_lik_hessian_nu())

        n_genes, n_spots, n_isos = self.n_genes, self.n_spots, self.n_isos
        step = 1  # step size

        hessian_nu_inv = torch.cholesky_inverse(chol) # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        gradient_nu = self.nu.grad.transpose(1, 2).reshape(n_genes, -1, 1)  # (n_genes, n_spots * (n_isos - 1), 1)
        nu_new = self.nu.transpose(1, 2).reshape(n_genes, -1, 1) - (step * hessian_nu_inv).matmul(
            gradient_nu
        )  # (n_genes, n_spots * (n_isos - 1), 1)
        nu_new = nu_new.reshape(n_genes, n_isos - 1, n_spots).transpose(1, 2)
        # update the parameters and clear the gradients
        with torch.no_grad():
            self.nu.copy_(nu_new)
            self.nu.grad.zero_()

        if return_variables:
            return nu_new

    def _fit(self, diagnose: bool = False, verbose: bool = False, quiet: bool = False, random_seed = None):
        """Main fitting function to find the MAP estimates using specified fitting method."""
        # extract configs
        fitting_method = self.fitting_method
        optim = self.fitting_configs["optim"]
        max_epochs = self.fitting_configs["max_epochs"]
        patience = self.fitting_configs["patience"]
        tol = self.fitting_configs["tol"]
        max_epochs = 10000 if max_epochs == -1 else max_epochs
        patience = patience if patience > 0 else 1

        if random_seed is not None: # set random seed for reproducibility
            torch.manual_seed(random_seed)

        # set iteration limits
        epoch = 0
        batch_size = self.n_genes

        # set iteration limits
        batch_size = self.n_genes
        t_start = timer()
        logger = PatienceLogger(batch_size, patience, min_delta=tol, diagnose=diagnose)

        # start training
        self.train()

        while logger.epoch < max_epochs and not logger.convergence.all():
            self.optimizer.zero_grad()

            # minimize the negative log-likelihood or the negative log-marginal-likelihood
            neg_log_prob = -self() # (n_genes,)
            neg_log_prob.mean().backward()  # backpropate gradients

            if fitting_method == "joint_newton":
                # update nu, beta, bias, and sigmas using Newton's method
                self._update_joint_newton()

            elif fitting_method == "marginal_newton":
                # update nu every k epochs
                if epoch % self.fitting_configs["update_nu_every_k"] == 0:
                    self._update_marginal_nu_newton()
                # skip gradient descent update
                self.nu.grad.zero_()

            # gradient-based updates
            if optim == "lbfgs":
                # update all parameters using L-BFGS
                self.optimizer.zero_grad()
                neg_log_prob = self.optimizer.step(self._closure)
            else:
                # update the remaining parameters with non-zero gradients using gradient descent
                self.optimizer.step()

            # check convergence
            with torch.no_grad():
                # calculate the negative log-likelihood
                neg_log_prob = -self().detach().cpu()

            logger.log(
                neg_log_prob,
                {
                    k: v.detach().cpu()
                    for k, v in self.named_parameters()
                    if "_fake" not in k
                },
            )
            self.convergence.copy_(logger.convergence)

            if (verbose and not quiet) and logger.epoch % 10 == 0:
                print(
                    f"Epoch {logger.epoch}. Loss (neg_log_prob): {logger.best_loss.mean():.4f}. "
                )

        # make sure constraints are satisfied
        self._final_sanity_check()

        # check model convergence
        num_not_converge = (~logger.convergence).sum()
        if num_not_converge:
            warnings.warn(
                f"{num_not_converge} Genes did not converge after epoch {max_epochs}. "
                "Try larger max_epochs."
            )

        # save runtime
        t_end = timer()
        self.fitting_time = t_end - t_start

        if not quiet:  # print final message
            print(
                f"Time {self.fitting_time:.2f}s. Total epoch {logger.epoch}. Final loss "
                f"(neg_log_prob): {neg_log_prob.mean():.3f}."
            )

        # collect parameters corresponding to the best epoch for each sample in batch
        if max_epochs > 0:
            for k, v in self.named_parameters():
                if "_fake" not in k:
                    v.data.copy_(logger.best_params[k])

        self.logger = logger

        return logger.params_iter


    @torch.no_grad()
    def _final_sanity_check(self):
        """Make sure constraints are satisfied."""
        # ensure positive parameters
        if self.var_parameterization_sigma_theta:
            self.sigma.abs_()
        else:
            self.sigma_sp.abs_()
            self.sigma_nsp.abs_()

    def fit(self, diagnose: bool = False, verbose: bool = False, quiet: bool = False, random_seed = None):
        """Fit the model using all data"""
        self._configure_optimizer(verbose=verbose)
        diag_outputs = self._fit(diagnose=diagnose, verbose=verbose, quiet=quiet, random_seed=random_seed)
        if diagnose:
            return diag_outputs

    def clone(self):
        """Clone a Multinomial GLMM model with the same set of parameters."""
        new_model = type(self)(
            share_variance=self.share_variance,
            var_parameterization_sigma_theta=self.var_parameterization_sigma_theta,
            var_fix_sigma=self.var_fix_sigma,
            var_prior_model=self.var_prior_model,
            var_prior_model_params=self.var_prior_model_params,
            init_ratio=self.init_ratio,
            fitting_method=self.fitting_method,
            fitting_configs=self.fitting_configs,
        )
        new_model.setup_data(
            counts=self.counts,
            corr_sp=None,
            design_mtx=self.X_spot,
            corr_sp_eigvals=self.corr_sp_eigvals,
            corr_sp_eigvecs=self.corr_sp_eigvecs,
        )
        new_model.load_state_dict(self.state_dict())

        return new_model
