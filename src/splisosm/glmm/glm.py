"""Multinomial GLM implementation for isoform usage."""

from __future__ import annotations

import warnings
from timeit import default_timer as timer
from typing import Literal, Optional

import torch
import torch.nn as nn

from splisosm.glmm.base import (
    BaseModel,
    _melt_tensor_along_first_dim,
    update_at_idx,
)
from splisosm.glmm.likelihood import log_prob_fastmult_batched
from splisosm.glmm.logger import PatienceLogger

__all__ = ["MultinomGLM"]


class MultinomGLM(BaseModel, nn.Module):
    """The Multinomial Generalized Linear Model for spatial isoform expression.

    Compared to MultinomGLMM, this model does not have a random effect term::

        Y ~ Multinomial(alpha, Y.sum(1))
        eta = multinomial-logit(alpha) = X @ beta + bias_eta

    Given isoform counts of a gene ``Y`` (n_spots, n_isos) and design matrix ``X`` (n_spots, n_factors),
    MultinomGLM.fit will find the MAP estimates of the following learnable parameters:

    - ``beta``: (n_factors, n_isos - 1) covariate coefficients of the fixed effect term.
    - ``bias_eta``: (n_isos - 1) intercepts of the fixed effect term.

    Inference is performed by maximizing the log likelihood using `fitting_method` (default: ``'iwls'``)

    Example
    -------
    >>> from splisosm.glmm import MultinomGLM
    >>> import torch
    >>> # Generate synthetic data
    >>> counts = torch.randint(0, 10, (5, 100, 3))  # 5 genes, 100 spots, each 3 isoforms
    >>> # Fit the GLM model
    >>> model = MultinomGLM(fitting_method='iwls')
    >>> model.setup_data(counts, design_mtx=None)
    >>> model.fit()
    >>> print(model)
    >>> # Extract the fitted isoform ratios
    >>> isoform_ratios = model.get_isoform_ratio()  # shape (5, 100, 3)
    >>> # Fitted parameters
    >>> print(model.beta.shape)  # shape (5, 0, 2)
    >>> print(model.bias_eta.shape)  # shape (5, 2)
    """

    n_spots: int
    """Number of samples/spots"""

    n_genes: int
    """Number of genes in the batch."""

    n_isos: int
    """Number of isoforms per gene in the batch."""

    n_factors: int | None
    """Number of covariates in the design matrix"""

    fitting_method: Literal["iwls", "newton", "gd"]
    """Method for fitting the model."""

    fitting_configs: dict
    """Dictionary of fitting configurations."""

    fitting_time: float
    """Time taken for fitting the model."""

    def __init__(
        self,
        fitting_method: Literal["iwls", "newton", "gd"] = "iwls",
        fitting_configs: dict | None = None,
    ):
        """
        Parameters
        ----------
        fitting_method
            Method for fitting the model.
            ``'iwls'``: Iteratively reweighted least squares.
            ``'newton'``: Newton's method.
            ``'gd'``: Gradient descent.
        fitting_configs
            Dictionary of fitting configurations. Keys include

            - ``'lr'``: float, Learning rate for gradient descent or Newton's method.
            - ``'optim'``: str, Optimizer type, one of ``'adam'``, ``'sgd'``, or ``'lbfgs'``.
            - ``'tol'``: float, Tolerance for convergence.
            - ``'tol_relative'``: bool, Whether to use relative tolerance for improvement.
            - ``'max_epochs'``: int, Maximum number of epochs for fitting.
            - ``'patience'``: int, Number of epochs to wait for improvement before stopping.
        """
        super().__init__()
        # will be set up later by calling self.setup_data()
        self.n_spots = None  # number of spots
        self.n_isos = None  # number of isoforms
        self.n_factors = None  # number of covariates

        # specify the fitting method
        valid_fitting_methods = ["gd", "newton", "iwls"]
        if fitting_method not in valid_fitting_methods:
            raise ValueError(
                f"Invalid fitting method. Must be one of {valid_fitting_methods}."
            )
        self.fitting_method = fitting_method

        # specify the fitting configurations
        self.fitting_configs = {  # default configurations
            "lr": 1e-2,
            "optim": "adam",
            "tol": 1e-5,
            "tol_relative": False,
            "max_epochs": 1000,
            "patience": 5 if fitting_method == "gd" else 2,
        }
        if fitting_configs is not None:
            self.fitting_configs.update(fitting_configs)

        # for now, restricting the optimization method to Adam, SGD and lbfgs
        valid_optimizers = ["adam", "sgd", "lbfgs"]
        if self.fitting_configs["optim"] not in valid_optimizers:
            raise ValueError(f"Invalid optimizer. Must be one of {valid_optimizers}.")

        # specify the fitting outcomes
        self.fitting_time = 0

    def __str__(self):
        base = (
            "A Multinomial Generalized Linear Model (GLM)\n"
            + f"- Number of genes in the batch: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of isoforms per gene: {self.n_isos}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Fitting method: {self.fitting_method}"
        )
        lg = getattr(self, "logger", None)
        if lg is not None:
            n_conv = int(lg.convergence.sum().item())
            med_epoch = int(lg.best_epoch.median().item())
            med_loss = float(lg.best_loss.median().item())
            training = (
                "\n=== Training summary (fitted)"
                + f"\n- Fitting time: {self.fitting_time:.2f} s"
                + f"\n- Converged: {n_conv} / {self.n_genes} genes"
                + f"\n- Best epoch (median): {med_epoch}"
                + f"\n- Best loss (median): {med_loss:.4f}"
            )
        else:
            training = "\n=== Training summary (not yet fitted)"
        return base + training

    __repr__ = __str__

    def setup_data(
        self,
        counts: torch.Tensor,
        design_mtx: Optional[torch.Tensor] = None,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ) -> None:
        """Set up the data for the model.

        Parameters
        ----------
        counts
            Shape (n_genes, n_spots, n_isos) or (n_spots, n_isos).
            For batched calculations, all genes in the batch must have the same number of isoforms.
        design_mtx
            Shape (n_spots, n_factors). Design matrix of spatial covariates.
            If None, an intercept-only design matrix will be used.
        device
            'cpu', 'cuda', or 'mps' (torch v2.11.0+).
        """
        # need to switch to a different count model (e.g., Poisson) when only one isoform is provided
        if counts.shape[-1] == 1:
            raise NotImplementedError(
                "Only one isoform provided. Please use a different count model."
            )

        valid_devices = ["cpu", "cuda", "mps"]
        if device not in valid_devices:
            raise ValueError(f"Invalid device. Must be one of {valid_devices}.")
        self.device = torch.device(device)

        if counts.ndim == 2:
            counts = counts.unsqueeze(0)  # (1, n_spots, n_isos)

        # convert sparse tensors to dense tensors
        if counts.is_sparse:
            counts = counts.to_dense()

        # set model dimensions based on the input shape
        self.n_genes, self.n_spots, self.n_isos = counts.shape

        if design_mtx is None:
            # initialize an empty design matrix of shape (1, n_spots, 0)
            design_mtx = torch.ones(1, self.n_spots, 0)
        elif design_mtx.ndim == 2:
            design_mtx = design_mtx.unsqueeze(0)  # (1, n_spots, n_factors)
        elif design_mtx.ndim == 3:
            # all genes to test should share the same design matrix
            if design_mtx.shape[0] != 1:
                raise ValueError("Batched design matrix is currently not supported.")
        else:
            raise ValueError(
                f"design_mtx must be a 2D or 3D tensor. Got shape: {design_mtx.shape}"
            )
        self.n_factors = design_mtx.shape[-1]
        expected_shape = (1, self.n_spots, self.n_factors)
        if design_mtx.shape != expected_shape:
            raise ValueError(
                f"Invalid design matrix shape. Expected {expected_shape}; "
                f"got {tuple(design_mtx.shape)}."
            )

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
        )  # (n_genes, n_spots, n_isos)
        alpha = torch.softmax(alpha, dim=-1)  # (n_genes, n_spots, n_isos)
        return alpha

    def get_isoform_ratio(self) -> torch.Tensor:
        """Extract the fitted isoform ratio across space.

        Returns
        -------
        ratio : torch.Tensor
            Shape (n_genes, n_spots, n_isos), the fitted isoform ratio across space.
        """
        return self._alpha().detach()  # (n_genes, n_spots, n_isos)

    def forward(self) -> torch.Tensor:
        """Calculate log probability given data.

        Returns
        -------
        log_prob : torch.Tensor
            Shape (n_genes,), the log probability for each gene.
        """
        return log_prob_fastmult_batched(
            self._alpha().transpose(1, 2), self.counts.transpose(1, 2)
        )

    # Functions to calculate Hessian.

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
        )  # (1, n_spots, n_factors + 1)
        # score of shape (n_genes, n_factors + 1, n_isos - 1)
        score = X_expand.transpose(1, 2).matmul(d_l_d_eta.detach()[..., :-1])

        return score

    def _get_multinom_hessian_raw(self):
        """Get the per-spot multinomial Hessian blocks (compact form, before melting).

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_spots, n_isos - 1, n_isos - 1)``.
            Entry ``[b, s, j, k]`` is the (j,k) element of the Hessian block at spot s
            for gene b.
        """
        n_isos = self.n_isos
        props = self._alpha()  # (n_genes, n_spots, n_isos)
        # multinom_hessian[b,s,j,k] = -counts_total[b,s] * (δ_jk * p_j - p_j * p_k)
        return -self.counts.sum(-1).view(self.n_genes, -1, 1, 1) * (
            props[..., :-1].unsqueeze(-1).expand(-1, -1, -1, n_isos - 1)
            * torch.eye(n_isos - 1, device=self.device)
            - torch.einsum("bsi,bsj->bsij", (props[..., :-1], props[..., :-1]))
        )

    def _get_log_lik_hessian_eta(self):
        """Get the Hessian matrix of the log joint probability wrt eta."""
        # reshape the hessian into (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        # (1) zeros for spots i != j
        # (2) at spot i, isoform j and k are connected via multinom_hessian[i, j, k]
        return _melt_tensor_along_first_dim(self._get_multinom_hessian_raw())

    def _get_log_lik_hessian_beta_bias(self):
        """Get the Hessian matrix of the log joint probability wrt the fixed effects.

        Computes ``X^T H_eta X`` directly from the per-spot Hessian blocks using
        einsum, avoiding the large ``n_spots*(n_isos-1) × n_spots*(n_isos-1)``
        intermediate matrix produced by ``_get_log_lik_hessian_eta``.

        Note: uses ``self.X_spot.shape[-1]`` for the factor dimension rather than
        ``self.n_factors``, because some callers (e.g. score test) extend ``X_spot``
        without updating ``n_factors``.
        """
        n_isos = self.n_isos
        props = self._alpha()  # (n_genes, n_spots, n_isos)

        # Per-spot log-lik Hessian blocks w.r.t. eta:
        # H_block[b, s, r, k] = -counts[b,s] * (diag(p) - p*p^T)[r,k]
        # Shape (n_genes, n_spots, n_isos-1, n_isos-1)
        H_block = -self.counts.sum(-1).view(self.n_genes, -1, 1, 1) * (
            props[..., :-1].unsqueeze(-1).expand(-1, -1, -1, n_isos - 1)
            * torch.eye(n_isos - 1, device=self.device)
            - torch.einsum("bsi,bsj->bsij", props[..., :-1], props[..., :-1])
        )

        # X_s: (n_spots, n_factors_actual+1) — design matrix augmented with intercept column.
        # Use X_spot.shape[-1] (not self.n_factors) to handle cases where X_spot is
        # temporarily extended by callers (e.g., score test).
        X_s = torch.cat(
            [self.X_spot.squeeze(0), torch.ones(self.n_spots, 1, device=self.device)],
            dim=-1,
        )  # (n_spots, n_factors_actual+1)
        nF1 = X_s.shape[-1]  # n_factors_actual + 1

        # X^T H X directly via einsum (no large W matrix needed):
        # XtHX[b, r*nF1+f, k*nF1+g] = Σ_s X_s[s,f] * H_block[b,s,r,k] * X_s[s,g]
        XtHX_5d = torch.einsum("sf, bsrk, sg -> brfkg", X_s, H_block, X_s)
        # (n_genes, n_isos-1, nF1, n_isos-1, nF1)
        hessian_beta_expand = XtHX_5d.reshape(
            self.n_genes,
            (n_isos - 1) * nF1,
            (n_isos - 1) * nF1,
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
                    if p.grad is not None:
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
        """Update the model parameters using the iteratively reweighted least squares (IWLS).

        Avoids materialising the large ``n_spots*(n_isos-1) × n_spots*(n_isos-1)``
        block-diagonal weight matrix by computing ``X^T W X`` and ``X^T W y``
        directly via einsum on the per-spot weight blocks.
        """
        n_genes, n_spots, n_isos, n_factors = (
            self.n_genes,
            self.n_spots,
            self.n_isos,
            self.n_factors,
        )
        props = self._alpha()  # (n_genes, n_spots, n_isos)

        # Per-spot IWLS weight blocks:
        # W[b, s, r, k] = counts[b,s] * (diag(p) - p*p^T)[r, k]
        # Shape (n_genes, n_spots, n_isos-1, n_isos-1)
        W = self.counts.sum(-1).view(n_genes, n_spots, 1, 1) * (
            props[..., :-1].unsqueeze(-1).expand(-1, -1, -1, n_isos - 1)
            * torch.eye(n_isos - 1, device=self.device)
            - torch.einsum("bsi,bsj->bsij", props[..., :-1], props[..., :-1])
        )
        # Invert per-spot weight (small (n_isos-1)×(n_isos-1) blocks, not n×n)
        W_inv = torch.linalg.inv(W + 1e-5 * torch.eye(n_isos - 1, device=self.device))

        # Pseudo-response: y_tilde = eta + W^{-1}(counts - mu)
        residuals = self.counts - props * self.counts.sum(-1, keepdim=True)
        working_y = W_inv.matmul(residuals[..., :-1].unsqueeze(-1)).squeeze(-1)
        working_y += self._eta()  # (n_genes, n_spots, n_isos-1)

        # X_s: (n_spots, n_factors+1) — design with intercept column (no n_spots² allocation)
        X_s = torch.cat(
            [self.X_spot.squeeze(0), torch.ones(n_spots, 1, device=self.device)], dim=-1
        )  # (n_spots, n_factors+1)

        # Efficient X^T W X via einsum (avoids the n_spots*(n_isos-1) × n_spots*(n_isos-1) W):
        # XtWX[b, r*(F+1)+f, k*(F+1)+g] = Σ_s X_s[s,f] * W[b,s,r,k] * X_s[s,g]
        XtWX_5d = torch.einsum("sf, bsrk, sg -> brfkg", X_s, W, X_s)
        # (n_genes, n_isos-1, n_factors+1, n_isos-1, n_factors+1)
        Xt_W_X = XtWX_5d.reshape(
            n_genes, (n_isos - 1) * (n_factors + 1), (n_isos - 1) * (n_factors + 1)
        )
        Xt_W_X += 1e-5 * torch.eye((n_isos - 1) * (n_factors + 1), device=self.device)

        # Efficient X^T W y via einsum:
        # XtWy[b, r*(F+1)+f] = Σ_s X_s[s,f] * Σ_k W[b,s,r,k] * y[b,s,k]
        Wy = torch.einsum(
            "bsrk, bsk -> bsr", W, working_y
        )  # (n_genes, n_spots, n_isos-1)
        XtWy = torch.einsum(
            "sf, bsr -> brf", X_s, Wy
        )  # (n_genes, n_isos-1, n_factors+1)
        Xt_W_y = XtWy.reshape(n_genes, -1)  # (n_genes, (n_isos-1)*(n_factors+1))

        res = torch.linalg.solve(Xt_W_X, Xt_W_y).reshape(
            n_genes, n_isos - 1, n_factors + 1
        )

        # extract beta and bias_eta
        beta_new = res[..., :-1].transpose(1, 2)  # (n_genes, n_factors, n_isos-1)
        bias_eta_new = res[..., -1]  # (n_genes, n_isos-1)

        self.beta.data.copy_(update_at_idx(self.beta, beta_new, self.convergence))
        self.bias_eta.data.copy_(
            update_at_idx(self.bias_eta, bias_eta_new, self.convergence)
        )

    def fit(
        self,
        store_param_history: bool = False,
        verbose: bool = False,
        quiet: bool = False,
        random_seed: Optional[int] = None,
        diagnose: Optional[bool] = None,
    ) -> dict:
        """Fit the model using all data.

        Parameters
        ----------
        store_param_history
            Whether to store per-epoch parameter snapshots during training.
            Loss history is always recorded via ``model.logger.loss_history``
            regardless of this flag.
        verbose
            Whether to print verbose information during fitting.
        quiet
            Whether to suppress output during fitting.
        random_seed
            Random seed for reproducibility.
        diagnose
            Deprecated alias for ``store_param_history``.

        Returns
        -------
        params_iter : dict or None
            If ``store_param_history=True``, returns a dictionary of parameter
            snapshots during training.  Otherwise returns ``None``.
        """
        if diagnose is not None:
            warnings.warn(
                "The `diagnose` parameter is deprecated; use `store_param_history` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            store_param_history = diagnose

        if random_seed is not None:  # set random seed for reproducibility
            torch.manual_seed(random_seed)

        if self.fitting_method == "gd":  # use gradient descent
            # configure the optimizer and start training
            self._configure_optimizer(verbose=verbose)
            self.train()

        max_epochs = self.fitting_configs["max_epochs"]
        patience = self.fitting_configs["patience"]
        tol = self.fitting_configs["tol"]
        tol_relative = self.fitting_configs.get("tol_relative", False)
        max_epochs = 10000 if max_epochs == -1 else max_epochs
        patience = patience if patience > 0 else 1

        batch_size = self.n_genes
        t_start = timer()
        logger = PatienceLogger(
            batch_size,
            patience,
            min_delta=tol,
            tol_relative=tol_relative,
            store_param_history=store_param_history,
        )

        while logger.epoch < max_epochs and not logger.convergence.all():
            # update the model parameters
            if self.fitting_method == "gd":
                self._update_gradient_descent()
            elif self.fitting_method == "newton":
                self._update_newton()
            elif self.fitting_method == "iwls":
                self._update_iwls()

            # evaluate post-step loss for convergence checking
            with torch.no_grad():
                neg_log_prob = -self().detach().cpu()

            # NaN/Inf guard — mark affected genes converged to avoid wasted iterations
            if not torch.isfinite(neg_log_prob).all():
                nan_genes = int((~torch.isfinite(neg_log_prob)).sum().item())
                warnings.warn(
                    f"{nan_genes} gene(s) produced non-finite loss at epoch "
                    f"{logger.epoch}. Check for numerical issues (extreme counts, "
                    "degenerate design matrix).",
                    UserWarning,
                    stacklevel=2,
                )
                self.convergence |= ~torch.isfinite(neg_log_prob)
                neg_log_prob = torch.nan_to_num(neg_log_prob, nan=float("inf"))

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
                    f"Epoch {logger.epoch}. Loss (neg_log_prob): {logger.best_loss.mean():.4f}."
                )

        # check model convergence
        num_not_converge = (~logger.convergence).sum()
        if num_not_converge and not quiet:
            warnings.warn(
                f"{num_not_converge} gene(s) did not converge after epoch {max_epochs}. "
                "Try larger max_epochs.",
                UserWarning,
                stacklevel=2,
            )

        # save runtime
        t_end = timer()
        self.fitting_time = t_end - t_start

        if not quiet:  # print final message
            print(
                f"Time {self.fitting_time:.2f}s. Total epoch {logger.epoch}. Final loss "
                f"(neg_log_prob): {neg_log_prob.mean():.3f}."
            )

        # restore parameters from the best epoch for each sample in the batch
        if max_epochs > 0:
            for k, v in self.named_parameters():
                if "_fake" not in k:
                    v.data.copy_(logger.best_params[k])

        self.logger = logger

        return logger.params_iter

    def clone(self) -> "MultinomGLM":
        """Clone a model with the same set of parameters."""
        new_model = type(self)(
            fitting_method=self.fitting_method, fitting_configs=self.fitting_configs
        )
        new_model.setup_data(
            counts=self.counts, design_mtx=self.X_spot, device=self.device.type
        )
        new_model.load_state_dict(self.state_dict())

        return new_model

    def update_params_from_dict(self, params: dict) -> None:
        """Update a subset of model parameters with a dictionary of parameters.

        Parameters
        ----------
        params
            A dictionary of parameters to be updated. The keys must be
            existing parameter names in the model.
        """
        new_params = self.state_dict()
        new_params.update(params)
        self.load_state_dict(new_params)
