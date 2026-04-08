"""Multinomial GLM/GLMM model implementations for SPLISOSM."""

import warnings
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.distributions import Gamma, InverseGamma

from splisosm.likelihood import (
    log_prob_fastmult_batched,
    log_prob_fastmvn_batched,
)
from splisosm.logger import PatienceLogger

__all__ = ["MultinomGLM", "MultinomGLMM"]


class BaseModel(ABC):
    """API for the GLM and GLMM model."""

    @abstractmethod
    def setup_data(
        self,
        counts: torch.Tensor,
        corr_sp: Optional[torch.Tensor],
        design_mtx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set up the data for the model.

        Parameters
        ----------
        counts : torch.Tensor
            Shape (n_genes, n_spots, n_isoforms), genes with the same number of isoforms.
        corr_sp : torch.Tensor
            Shape (n_spots, n_spots).
        design_mtx : torch.Tensor, optional
            Shape (n_spots, n_factors).
        """
        pass

    def forward(self) -> torch.Tensor:
        """Calculate the log-likelihood or log-marginal-likelihood of the model."""
        pass

    @abstractmethod
    def fit(self) -> None:
        """Fit the model using all data."""
        pass

    @abstractmethod
    def get_isoform_ratio(self) -> torch.Tensor:
        """Extract the fitted isoform ratio across space."""
        pass

    @abstractmethod
    def clone(self) -> "BaseModel":
        """Clone the model with data and model parameters."""
        pass


def _melt_tensor_along_first_dim(tensor_in):
    """Melt a 4D tensor into 3D and reorder entries by spots.

    tensor_in[:, i, j, k] -> matrix_out[:, i + j * n, i + k * n] where n = tensor_in.shape[1]

    Parameters
    ----------
    tensor_in : torch.Tensor
        Shape (n_genes, n_spots, n_isos - 1, n_isos - 1).

    Returns
    -------
    matrix_out : torch.Tensor
        Shape (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1)).
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
    >>> from splisosm.model import MultinomGLM
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
        fitting_configs: dict = {},
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
        assert fitting_method in ["gd", "newton", "iwls"]
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
        self.fitting_configs.update(fitting_configs)

        # for now, restricting the optimization method to Adam, SGD and lbfgs
        assert self.fitting_configs["optim"] in ["adam", "sgd", "lbfgs"]

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

        assert device in [
            "cpu",
            "cuda",
            "mps",
        ], f"device must be 'cpu', 'cuda', or 'mps'; got {device!r}"
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
            assert (
                design_mtx.shape[0] == 1
            ), "Batched design matrix is currently not supported."
        else:
            raise ValueError(
                f"design_mtx must be a 2D or 3D tensor. Got shape: {design_mtx.shape}"
            )
        self.n_factors = design_mtx.shape[-1]
        assert design_mtx.shape == (
            1,
            self.n_spots,
            self.n_factors,
        ), f"Invalide design matrix shape: {design_mtx.shape}"

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


class MultinomGLMM(MultinomGLM, BaseModel, nn.Module):
    """The Multinomial Generalized Linear Mixed Model for spatial isoform expression.

    The model is defined as follows::

        Y ~ Multinomial(alpha, Y.sum(1))
        eta = multinomial-logit(alpha) = X @ beta + bias_eta + nu
        nu ~ MVN(0, sigma^2 * (theta * V_sp + (1-theta) * I))

    Given isoform counts of a gene ``Y`` (n_spots, n_isos), design matrix ``X`` (n_spots, n_factors),
    and spatial covariance matrix ``V_sp`` (n_spots, n_spots), the model estimates the isoform usage
    ratio ``alpha`` (n_spots, n_isos) across space.
    Specifically, `MultinomGLMM.fit` will find the MAP estimates of the following learnable parameters:

    - ``beta``: (n_factors, n_isos - 1) covariate coefficients of the fixed effect term.
    - ``bias_eta``: (n_isos - 1) intercepts of the fixed effect term.
    - ``nu``: (n_spots, n_isos - 1) the random effect term.
    - variance components: (``sigma``, ``theta_logit``), each of length n_isos - 1
      (or 1 if `share_variance` is True), representing total variance and logit of
      spatial variance proportion ``theta``.

    Inference algorithms can be categorized into two types based on the optimization objective:

    - Joint: Maximize the joint likelihood (with the random effect ``nu``).
      This is equivalent to the first-order Laplace approximation of the marginal likelihood.
    - Marginal: Maximize the marginal likelihood (with the random effect ``nu`` integrated out).
      The integral is approximated by a second-order Laplace approximation.

    Methods implemented:

    - ``'joint_gd'``: Maximize the joint likelihood using gradient descent.
    - ``'joint_newton'``: Maximize the joint likelihood using Newton's method.
    - ``'marginal_gd'``: Maximize the marginal likelihood using gradient descent.
    - ``'marginal_newton'``: Maximize the marginal likelihood using Newton's method.
      In this method, ``nu`` is first updated using Newton's method every
      ``'update_nu_every_k'`` iterations, and ``beta``, ``bias_eta``, and variance components
      are updated using gradient descent.

    Notes
    -----
    It is also possible to implement held-out likelihood for model selection.

    Example
    -------
    >>> from splisosm.model import MultinomGLMM
    >>> from splisosm.utils import get_cov_sp
    >>> import torch
    >>> # Generate synthetic data
    >>> counts = torch.randint(0, 10, (5, 100, 3))  # 5 genes, 100 spots, each 3 isoforms
    >>> coords = torch.rand(100, 2)  # 100 spots with 2D coordinates
    >>> K_sp = get_cov_sp(coords, k=4, rho=0.9) # spatial covariance matrix of shape (100, 100)
    >>> # Fit the GLMM model
    >>> model = MultinomGLMM(fitting_method='joint_gd')
    >>> model.setup_data(counts, corr_sp=K_sp, design_mtx=None)
    >>> model.fit()
    >>> print(model)
    >>> # Extract the fitted isoform ratios
    >>> isoform_ratios = model.get_isoform_ratio()  # shape (5, 100, 3)
    >>> # Fitted parameters
    >>> print(model.beta.shape)  # shape (5, 0, 2)
    >>> print(model.bias_eta.shape)  # shape (5, 2)
    >>> print(model.nu.shape)  # shape (5, 100, 2)
    >>> print(model.sigma.shape)  # shape (5, 1)
    >>> print(model.theta_logit.shape)  # shape (5, 1)
    """

    share_variance: bool
    """Whether to use the same variance across isoforms."""

    var_fix_sigma: bool
    """Whether to fix the total variance (``sigma``) to its initial value.

    When ``True`` (default), sigma is frozen at the Fano-factor estimate
    and only ``theta_logit`` (spatial variance proportion) is learned.
    This yields conservative but well-calibrated SV and DU tests:
    near-zero false positive rates with strong power on true signals.
    Set to ``False`` to learn sigma jointly, which may increase power
    at the cost of inflated false positive rates."""

    var_prior_model: Literal["none", "gamma", "inv_gamma"]
    """The prior model on the total variance ``sigma``."""

    var_prior_model_params: dict
    """The parameters for the prior model on the total variance ``sigma``."""

    init_ratio: Literal["observed", "uniform"]
    """The initialization method for the logit isoform usage ratio ``gamma``."""

    fitting_method: Literal[
        "joint_gd", "joint_newton", "marginal_gd", "marginal_newton"
    ]
    """The fitting method to use."""

    fitting_configs: dict
    """A dictionary of fitting configurations."""

    fitting_time: float
    """The time taken to fit the model."""

    def __init__(
        self,
        share_variance: bool = True,
        var_fix_sigma: bool = True,
        var_prior_model: Literal["none", "gamma", "inv_gamma"] = "none",
        var_prior_model_params: dict = {},
        init_ratio: Literal["observed", "uniform"] = "uniform",
        fitting_method: Literal[
            "joint_gd", "joint_newton", "marginal_gd", "marginal_newton"
        ] = "joint_gd",
        fitting_configs: dict = {},
    ):
        """
        Parameters
        ----------
        share_variance
            Whether to use the same variance across isoforms. If True, the variance components
            will be of length 1. If False, the variance components will be of length n_isos - 1.
        var_fix_sigma
            Whether to fix the total variance (``sigma``) to the Fano-factor
            initial estimate.  When ``True`` (default), only ``theta_logit``
            is learned, producing conservative but well-calibrated hypothesis
            tests.  Set to ``False`` to learn sigma jointly with other
            parameters; this may yield higher power for the SV test but can
            inflate false positive rates for both SV and DU tests.
        var_prior_model
            The prior model on the total variance ``sigma``. Default is ``'none'`` with no prior.
            Other options are ``'gamma'`` (Gamma prior) and ``'inv_gamma'`` (Inverse Gamma prior).
        var_prior_model_params
            The parameters for the prior model on the total variance ``sigma``.
            For ``'gamma'``, the default parameters are ``{'alpha': 2.0, 'beta': 0.3}``.
            For ``'inv_gamma'``, the default parameters are ``{'alpha': 3, 'beta': 0.5}``.
        init_ratio
            The initialization method for the logit isoform usage ratio. Options are ``'observed'`` (initialize using observed counts)
            and ``'uniform'`` (equal isoform usage across space).
        fitting_method
            The fitting method to use. Options are ``'joint_gd'`` (joint likelihood with gradient descent),
            ``'joint_newton'`` (joint likelihood with Newton's method),
            ``'marginal_gd'`` (marginal likelihood with gradient descent),
            and ``'marginal_newton'`` (marginal likelihood with Newton's method).
        fitting_configs
            A dictionary of fitting configurations with the following keys:

            - ``'lr'``: float, learning rate
            - ``'optim'``: str, optimization method (one of ``'adam'``, ``'sgd'``, or ``'lbfgs'``)
            - ``'tol'``: float, tolerance for convergence
            - ``'max_epochs'``: int, maximum number of epochs
            - ``'patience'``: int, number of epochs to wait for improvement before stopping
            - ``'update_nu_every_k'``: int, number of iterations to update ``nu`` when using ``fitting_method='marginal_newton'``
        """
        super().__init__()

        # variance parameterization:
        # 	var(sigma, theta) = sigma^2 (theta * V_sp + (1-theta) * I)
        self.share_variance = (
            share_variance  # whether to share variance across isoforms
        )
        self.var_fix_sigma = var_fix_sigma  # whether to fix sigma

        # specify the prior model on sigma^2
        assert var_prior_model in ["none", "gamma", "inv_gamma"]
        self.var_prior_model = var_prior_model  # prior on the variance size sigma
        if self.var_prior_model == "gamma":
            # Chung, Yeojin, et al. Psychometrika 78.4 (2013): 685-709.
            # this prior is applied on sigma
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
            # this prior is applied on sigma^2
            # InverseGamma(2, 0.1): weakly informative, mode at beta/(alpha+1) = 0.033
            self.var_prior_model_params = {
                "alpha": 2,
                "beta": 0.1,
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
        base = (
            "A Multinomial Generalized Linear Mixed Model (GLMM)\n"
            + f"- Number of genes in the batch: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of isoforms per gene: {self.n_isos}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + "- Variance formulation:\n"
            + f"\t* Learnable variance: {not self.var_fix_sigma}\n"
            + f"\t* Same variance across isoforms: {self.share_variance}\n"
            + f"\t* Prior on total variance: {self.var_prior_model}\n"
            + f"- Initialization method: {self.init_ratio}\n"
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
        corr_sp: Optional[torch.Tensor] = None,
        design_mtx: Optional[torch.Tensor] = None,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        corr_sp_eigvals: Optional[torch.Tensor] = None,
        corr_sp_eigvecs: Optional[torch.Tensor] = None,
    ) -> None:
        """Set up the data for the model.

        Parameters
        ----------
        counts
            Shape (n_genes, n_spots, n_isoforms) or (n_spots, n_isoforms).
            For batched calculations, all genes in the batch must have the same number of isoforms.
        corr_sp
            Shape (n_spots, n_spots), spatial covariance matrix.
            If None, the eigendecomposition of the spatial covariance matrix must be provided.
        design_mtx
            Shape (n_spots, n_factors). Design matrix of spatial covariates.
            If None, an intercept-only design matrix will be used.
        device
            'cpu', 'cuda', or 'mps' (torch v2.11.0+).
        corr_sp_eigvals
            Shape (n_spots,), eigenvalues of spatial covariance.
            If None, the spatial covariance matrix `corr_sp` must be provided.
        corr_sp_eigvecs
            Shape (n_spots, n_spots), eigenvectors of spatial covariance.
            If None, the spatial covariance matrix `corr_sp` must be provided.
        """
        # need to switch to a different count model (e.g., Poisson) when only one isoform is provided

        assert device in [
            "cpu",
            "cuda",
            "mps",
        ], f"device must be 'cpu', 'cuda', or 'mps'; got {device!r}"
        self.device = torch.device(device)

        if counts.ndim == 2:
            counts = counts.unsqueeze(0)  # (1, n_spots, n_isos)
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

        # Warn early when marginal mode is used with a large spot count.
        # The Hessian has shape (n_spots × (n_isos-1))², leading to an O(n³)
        # Cholesky decomposition per epoch.  For n_spots=500, n_isos=3 this is
        # a 1000×1000 Cholesky — already ~1 Gflop per gene-epoch.
        if (
            self.fitting_method in ("marginal_gd", "marginal_newton")
            and self.n_spots > 300
        ):
            warnings.warn(
                f"fitting_method='{self.fitting_method}' with n_spots={self.n_spots} > 300. "
                "Marginal mode requires a Cholesky decomposition of the full "
                f"({self.n_spots} × (n_isos-1))² Hessian per epoch (O(n³) cost), "
                "which becomes prohibitively slow for large datasets. "
                "Consider 'joint_gd' or 'joint_newton' instead.",
                UserWarning,
                stacklevel=2,
            )

        # switch to float type
        if not counts.dtype.is_floating_point:
            counts = counts.float()

        # convert sparse tensors to dense tensors
        if counts.is_sparse:
            counts = counts.to_dense()

        if design_mtx is None:
            # initialize an empty design matrix of shape (1, n_spots, 0)
            design_mtx = torch.ones(1, self.n_spots, 0)
        elif design_mtx.ndim == 2:
            design_mtx = design_mtx.unsqueeze(0)  # (1, n_spots, n_factors)
        elif design_mtx.ndim == 3:
            # all genes to test should share the same design matrix
            assert (
                design_mtx.shape[0] == 1
            ), "Batched design matrix is currently not supported."
        else:
            raise ValueError(
                f"design_mtx must be a 2D or 3D tensor. Got shape: {design_mtx.shape}"
            )
        self.n_factors = design_mtx.shape[-1]
        assert design_mtx.shape == (
            1,
            self.n_spots,
            self.n_factors,
        ), f"Invalide design matrix shape: {design_mtx.shape}"

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
                    "The latter will be ignored.",
                    UserWarning,
                    stacklevel=2,
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
            # Full eigendecomposition — always full-rank
            self._rank = self.n_spots
            self._is_low_rank = False

        else:
            rank = corr_sp_eigvals.shape[0]
            assert corr_sp_eigvecs.shape[0] == self.n_spots, (
                f"corr_sp_eigvecs must have {self.n_spots} rows, "
                f"got {corr_sp_eigvecs.shape[0]}"
            )
            assert corr_sp_eigvecs.shape[1] == rank, (
                "corr_sp_eigvecs columns must match corr_sp_eigvals length "
                f"({rank}), got {corr_sp_eigvecs.shape[1]}"
            )

            # Store the rank as a plain Python int (not a buffer, not a param).
            # This is used by _cov/_inv_cov and the log-prob methods.
            self._rank = rank
            self._is_low_rank = rank < self.n_spots

            # Never reconstruct the n×n corr_sp from eigenpairs — it is not used
            # in any computation (only the eigenpairs are needed).
            self.register_buffer("corr_sp", None)

            # (rank,), leading eigenvalues of the spatial correlation matrix
            self.register_buffer("corr_sp_eigvals", corr_sp_eigvals)
            # (n_spots, rank), corresponding orthonormal eigenvectors
            self.register_buffer("corr_sp_eigvecs", corr_sp_eigvecs)

        self.register_buffer("convergence", torch.zeros(self.n_genes, dtype=bool))

        # set up learnable parameters according to the model architecture
        self._configure_learnable_variables()

        # send to device before initialising params so that parameters and
        # counts are on the same device (avoids cross-device ops for non-CPU)
        self.to(self.device)
        self._move_prior_to_device()
        self._initialize_params()

    def _move_prior_to_device(self):
        """Recreate the variance prior distribution on the model's device."""
        if self.var_prior_model_dist is not None:
            alpha = torch.tensor(
                self.var_prior_model_params["alpha"],
                device=self.device,
                dtype=torch.float32,
            )
            beta = torch.tensor(
                self.var_prior_model_params["beta"],
                device=self.device,
                dtype=torch.float32,
            )
            if self.var_prior_model == "gamma":
                self.var_prior_model_dist = Gamma(alpha, beta)
            elif self.var_prior_model == "inv_gamma":
                self.var_prior_model_dist = InverseGamma(alpha, beta)

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
        # cov = sigma^2 (theta * V_sp + (1-theta) * I)
        sigma = torch.ones(self.n_genes, n_var_components)
        theta_logit = torch.zeros(self.n_genes, n_var_components)
        self.register_parameter("sigma", nn.Parameter(sigma))
        self.register_parameter("theta_logit", nn.Parameter(theta_logit))

        if self.var_fix_sigma:
            self.sigma.requires_grad = False

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

        # initial estimate of the variance parameter sigma using the Fano-factor heuristic
        sigma_init = (
            (self.counts.var(1).mean(1) / self.counts.sum(2).mean(1).clamp(min=1))
            .clamp(min=1e-4, max=0.9)
            .pow(0.5)
        ).unsqueeze(-1)

        # initialise spatial variance proportion to ~5% (theta ≈ sigmoid(-3) ≈ 0.047).
        with torch.no_grad():
            self.sigma.copy_(torch.ones_like(self.sigma) * sigma_init)
            self.theta_logit.copy_(torch.ones_like(self.theta_logit) * -3.0)

    """Below are a bunch of helper functions to update intermediate variables after each optimization step.
    """

    def var_total(self) -> torch.Tensor:
        """Output the total variance.

        Returns
        -------
        var_total : torch.Tensor
            The total variance ``sigma`` of shape (n_genes, n_var_components).
        """
        var_total = self.sigma**2

        if var_total.min() < 1e-2:
            warnings.warn(
                "Total variance is close to zero.",
                UserWarning,
                stacklevel=2,
            )
        var_total = torch.clip(var_total, min=1e-2)  # (n_genes, n_var_components)
        return var_total

    def var_sp_prop(self) -> torch.Tensor:
        """Output the proporptions of the spatial variance.

        Returns
        -------
        var_sp_prop : torch.Tensor
            The proportion ``theta`` of spatial variance of shape (n_genes, n_var_components).
        """
        # return: (n_genes, n_var_components)
        return torch.sigmoid(self.theta_logit)

    def _corr_eigvals(self):
        """Output the leading eigenvalues of the correlation matrix.

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_var_components, rank)`` where ``rank`` is the
            number of stored eigenpairs (``n_spots`` for full-rank, ``k`` for
            low-rank approximations).
        """
        var_sp_prop = self.var_sp_prop().unsqueeze(-1)  # (n_genes, n_var_components, 1)
        return var_sp_prop * self.corr_sp_eigvals.unsqueeze(0) + (1 - var_sp_prop)

    def _cov_eigvals(self):
        """Output the leading eigenvalues of the covariance matrix.

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_var_components, rank)``.
        """
        return self._corr_eigvals() * self.var_total().unsqueeze(-1)

    def _residual_cov_eigval(self):
        """Residual (noise-only) covariance eigenvalue for uncaptured modes.

        In the low-rank approximation the ``n_spots - rank`` uncaptured spatial
        modes are treated as pure white noise with eigenvalue
        ``σ²(1 − θ)`` (equivalently ``σ_nsp²``).

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_var_components, 1)``.  Each entry equals the
            residual covariance eigenvalue ``d`` for the corresponding gene and
            variance component.
        """
        # (1 - θ) * σ²:  var_total = σ², var_sp_prop = θ   → d = σ²(1-θ)
        var_sp_prop = self.var_sp_prop().unsqueeze(-1)  # (n_genes, n_var_comp, 1)
        return (1.0 - var_sp_prop) * self.var_total().unsqueeze(-1)

    def _cov(self):
        """Reconstruct the covariance matrix of the random effect.

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_var_components, n_spots, n_spots)``.

        Notes
        -----
        **Full-rank**: ``C = V diag(c) V^T``

        **Low-rank** (when ``_is_low_rank`` is True):
        ``C = d I + V_k diag(c_k − d) V_k^T``
        where ``d = _residual_cov_eigval()`` and ``c_k = _cov_eigvals()``.
        """
        V = self.corr_sp_eigvecs[None, None, ...]  # (1, 1, n_spots, rank)
        c = self._cov_eigvals()  # (n_genes, n_var_comp, rank)
        if not self._is_low_rank:
            return V.matmul(torch.diag_embed(c).matmul(V.transpose(-1, -2)))

        # Low-rank: C = d*I + V_k diag(c_k - d) V_k^T
        d = self._residual_cov_eigval().unsqueeze(-1)  # (n_genes, n_var_comp, 1, 1)
        delta = c - d.squeeze(-1)  # (n_genes, n_var_comp, rank)
        eye = torch.eye(
            self.n_spots, device=V.device, dtype=V.dtype
        )  # (n_spots, n_spots)
        return d * eye + V.matmul(torch.diag_embed(delta).matmul(V.transpose(-1, -2)))

    @property
    def _is_identity_cov(self) -> bool:
        """True when the spatial variance proportion is zero (C = σ²I).

        This holds for null models where ``theta_logit = -inf``,
        making the spatial kernel contribute nothing.
        """
        sp = self.var_sp_prop()  # (n_genes, n_var_comp)
        return bool((sp.detach().abs() < 1e-10).all())

    def _inv_cov(self):
        """Reconstruct the inverse covariance of the random effect.

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_var_components, n_spots, n_spots)``.

        Notes
        -----
        **Identity fast path** (when ``theta = 0``):
        ``C = σ²I`` so ``C^{-1} = (1/σ²) I``.

        **Full-rank**: ``C^{-1} = V diag(1/c) V^T``

        **Low-rank** (Woodbury identity):
        ``C^{-1} = (1/d) I + V_k diag(1/c_k − 1/d) V_k^T``
        """
        # Fast path: when theta = 0, covariance is sigma^2 * I
        if self._is_identity_cov:
            inv_var = (1.0 / self.var_total()).unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(self.n_spots, device=self.device, dtype=self.nu.dtype)
            return inv_var * eye  # (n_genes, n_var_comp, n_spots, n_spots)

        V = self.corr_sp_eigvecs[None, None, ...]  # (1, 1, n_spots, rank)
        c = self._cov_eigvals()  # (n_genes, n_var_comp, rank)
        if not self._is_low_rank:
            return V.matmul(torch.diag_embed(1.0 / c).matmul(V.transpose(-1, -2)))

        # Low-rank Woodbury: C^{-1} = (1/d)I + V_k diag(1/c_k - 1/d) V_k^T
        d = self._residual_cov_eigval().unsqueeze(-1)  # (n_genes, n_var_comp, 1, 1)
        # Clamp to prevent catastrophic cancellation when c ≈ d (theta near 0)
        inv_correction = (1.0 / c - 1.0 / d.squeeze(-1)).clamp(-1e6, 1e6)
        eye = torch.eye(
            self.n_spots, device=V.device, dtype=V.dtype
        )  # (n_spots, n_spots)
        return (1.0 / d) * eye + V.matmul(
            torch.diag_embed(inv_correction).matmul(V.transpose(-1, -2))
        )

    def _eta(self):
        """Output the eta based on the linear model."""
        # eta = X @ beta + bias_eta + nu
        # eta.shape = (n_genes, n_spots, n_isos - 1)
        return self.nu + self.X_spot.matmul(self.beta) + self.bias_eta.unsqueeze(1)

    # Functions to calculate log likelihoods.

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
        log_prob = self._calc_log_prob_prior_sigma()  # (n_genes,)

        # add mvn prob of nu ~ MVN(0, S)
        data = self.nu.transpose(1, 2)  # (n_genes, n_isos - 1, n_spots)

        if self._is_identity_cov:
            # Fast path: C = σ²I.  MVN log-prob =
            #   -0.5 * (n*p*log(2π) + n*p*log(σ²) + ||ν||² / σ²)
            n_p = data.shape[1]  # n_isos - 1
            var_total = self.var_total()  # (n_genes, n_var_comp)
            if self.share_variance:
                var_total = var_total.expand(self.n_genes, n_p)
            import math

            log_2pi = torch.log(
                torch.tensor(2.0 * math.pi, device=self.device, dtype=data.dtype)
            )
            # sum over isoform classes and spots
            nu_sq = (data**2).sum(-1)  # (n_genes, n_isos-1)
            log_prob += (
                -0.5
                * (
                    self.n_spots * log_2pi
                    + self.n_spots * var_total.log()
                    + nu_sq / var_total
                )
            ).sum(-1)
        else:
            # General eigenspace path
            cov_eigvals = self._cov_eigvals().expand(
                self.n_genes, data.shape[1], -1
            )  # (n_genes, n_isos-1, rank)
            cov_eigvecs = self.corr_sp_eigvecs.expand(
                1, data.shape[1], self.n_spots, self._rank
            )  # (1, n_isos-1, n_spots, rank)

            residual_eigval = None
            if self._is_low_rank:
                residual_eigval = self._residual_cov_eigval().expand(
                    self.n_genes, data.shape[1], 1
                )

            log_prob += log_prob_fastmvn_batched(
                locs=torch.zeros_like(data),
                cov_eigvals=cov_eigvals,
                cov_eigvecs=cov_eigvecs,
                data=data,
                residual_eigval=residual_eigval,
            )

        # add Multinomial likelihood of the counts
        log_prob += log_prob_fastmult_batched(
            self._alpha().transpose(1, 2), self.counts.transpose(1, 2)
        )

        return log_prob  # (n_genes,)

    def _calc_log_prob_marginal(self):
        """Calculate the log marginal probability (integrating out random effect nu)."""
        # by Laplace approximation, the log marginal probability is given by the following:
        # log_marginal ~= log_joint - 1/2 logdet_neg_hessian
        log_prob = self._calc_log_prob_joint()  # (n_genes,)
        full_hessian = (
            self._get_log_lik_hessian_nu()
        )  # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))

        # save the cholesky for fast matrix inverse in Newton updates
        # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        self._chol_hessian_nu = torch.linalg.cholesky(-full_hessian)
        logdet_neg_hessian = 2 * torch.diagonal(
            self._chol_hessian_nu, dim1=-2, dim2=-1
        ).log().sum(
            -1
        )  # (n_genes,)

        return log_prob - 0.5 * logdet_neg_hessian  # (n_genes,)

    def forward(self) -> torch.Tensor:
        """Calculate the log-likelihood or log-marginal-likelihood of the model.

        Returns
        -------
        log_prob : torch.Tensor
            Shape (n_genes,), the log probability for each gene.
        """
        if self.fitting_method in ["marginal_gd", "marginal_newton"]:
            # calculate log marginal prob
            return self._calc_log_prob_marginal()  # (n_genes,)
        else:
            # calculate log joint prob given data
            return self._calc_log_prob_joint()  # (n_genes,)

    # Functions to calculate Hessian.

    def _get_log_lik_hessian_nu(self):
        """Get the Hessian matrix of the log joint probability wrt the random effect nu."""
        n_genes, n_isos, n_spots = self.n_genes, self.n_isos, self.n_spots
        n_p = n_spots * (n_isos - 1)

        # Single allocation: pre-allocate the output tensor once to avoid a second
        # (n_genes, n_p, n_p) allocation from _get_log_lik_hessian_eta / _melt_tensor.
        full_hessian = torch.zeros(
            n_genes, n_p, n_p, device=self.device, dtype=self.nu.dtype
        )

        # Fill MVN blocks in-place.
        # mvn_hessian[q] = -cov[q]^{-1}, shape (n_genes, n_spots, n_spots)
        # In the isoform-major index ordering, isoform q occupies
        # rows/cols [q*n_spots : (q+1)*n_spots].
        if self._is_identity_cov:
            # Fast path: C = σ²I → C^{-1} = (1/σ²)I, so MVN Hessian = -(1/σ²)I
            # Fill the diagonal directly — O(n_spots) per isoform, no dense matrix.
            inv_var = 1.0 / self.var_total()  # (n_genes, n_var_comp)
            if self.share_variance:
                inv_var = inv_var.expand(n_genes, n_isos - 1)
            diag_idx = torch.arange(n_spots, device=self.device)
            for q in range(n_isos - 1):
                full_hessian[
                    :, q * n_spots + diag_idx, q * n_spots + diag_idx
                ] -= inv_var[:, q].unsqueeze(-1)
        else:
            mvn_hessian = -self._inv_cov()  # (n_genes, n_var_comp, n_spots, n_spots)
            if self.share_variance:
                mvn_hessian = mvn_hessian.expand(n_genes, n_isos - 1, -1, -1)
            for q in range(n_isos - 1):
                full_hessian[
                    :, q * n_spots : (q + 1) * n_spots, q * n_spots : (q + 1) * n_spots
                ].add_(mvn_hessian[:, q])

        # Fill multinomial blocks in-place via scatter (compact form avoids second
        # large allocation; uses the same index arithmetic as _melt_tensor_along_first_dim).
        # raw: (n_genes, n_spots, n_isos-1, n_isos-1)
        raw = self._get_multinom_hessian_raw()
        i_idx, j_idx, k_idx = torch.meshgrid(
            torch.arange(n_spots, device=self.device),
            torch.arange(n_isos - 1, device=self.device),
            torch.arange(n_isos - 1, device=self.device),
            indexing="ij",
        )
        row_idx = (i_idx + j_idx * n_spots).view(-1)
        col_idx = (i_idx + k_idx * n_spots).view(-1)
        full_hessian[:, row_idx, col_idx] += raw.flatten(1)

        return full_hessian

    def _calc_log_prob_mvn_wrt_sigma(self, sigma_expand):
        """Helper function of log joint probability wrt sigmas for calculate Hessian.

        Parameters
        ----------
        sigma_expand : torch.Tensor
            Shape (n_genes, n_var_components, 2). The two variance parameters
            (sigma, theta_logit) are stacked along the last dimension.
        """
        if self.share_variance:  # the same variance components across isoforms
            sigma_expand = sigma_expand.expand(self.n_genes, self.n_isos - 1, -1)

        sigma, theta_logit = sigma_expand[..., 0], sigma_expand[..., 1]
        var_sp_prop = torch.sigmoid(theta_logit).unsqueeze(
            -1
        )  # (n_genes, n_iso - 1, 1)
        sigma_total = sigma  # (n_genes, n_iso - 1)
        # (n_genes, n_iso - 1, rank)
        cov_eigvals = (
            var_sp_prop * self.corr_sp_eigvals.unsqueeze(0) + (1 - var_sp_prop)
        ) * sigma_total.unsqueeze(-1)
        # residual = σ²(1-θ), shape (n_genes, n_iso-1, 1)
        residual_eigval = (1.0 - var_sp_prop) * sigma_total.unsqueeze(-1)

        # MVN prior likelihood as a function of the input eta
        data = self.nu.transpose(1, 2)  # (n_genes, n_isos - 1, n_spots)
        cov_eigvecs = self.corr_sp_eigvecs.expand(
            1, data.shape[1], self.n_spots, self._rank
        )
        log_prob = log_prob_fastmvn_batched(
            locs=torch.zeros_like(data),
            cov_eigvals=cov_eigvals,
            cov_eigvecs=cov_eigvecs,
            data=data,
            residual_eigval=residual_eigval if self._is_low_rank else None,
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

    def _get_log_lik_hessian_sigma_expand_analytic(self):
        """Analytic Hessian of log(MVN prior + sigma prior) wrt variance parameters.

        Drop-in replacement for :meth:`_get_log_lik_hessian_sigma_expand` that avoids
        the expensive ``torch.autograd.functional.jacobian`` call by computing all
        second derivatives in closed form.

        Returns
        -------
        torch.Tensor
            Shape ``(n_genes, n_var_comps * 2, n_var_comps * 2)``.  Each gene-slice
            is ``(1/n_genes) * d²log_prob_g / d(sigma_params)²``, matching the scale
            of the autograd version.
        """
        n_genes, n_isos, n_spots = self.n_genes, self.n_isos, self.n_spots
        rank = self._rank
        n_var_comps = self.sigma.shape[1]
        dtype = self.nu.dtype

        V_k = self.corr_sp_eigvecs  # (n_spots, rank)
        lam = self.corr_sp_eigvals  # (rank,)

        # Data projections: z[b,q,k] = (V_k^T nu_q)[b,q,k]
        data = self.nu.transpose(1, 2)  # (n_genes, n_isos-1, n_spots)
        z = torch.einsum("bqs,sk->bqk", data, V_k)  # (n_genes, n_isos-1, rank)
        if self._is_low_rank:
            # ||x_perp_q||^2 = ||nu_q||^2 - ||z_q||^2  (n_genes, n_isos-1)
            x_perp_sq = (data**2).sum(-1) - (z**2).sum(-1)

        # Variance params (n_genes, n_var_comps)
        sigma = self.sigma
        theta = torch.sigmoid(self.theta_logit)
        D = theta * (1 - theta)  # sigmoid derivative, (n_genes, n_var_comps)

        # Accumulate per-class MVN Hessian contributions into (n_genes, n_var_comps, 2, 2)
        H_diag = torch.zeros(
            n_genes, n_var_comps, 2, 2, device=self.device, dtype=dtype
        )

        for q in range(n_isos - 1):
            vc = 0 if self.share_variance else q
            zq = z[:, q, :]  # (n_genes, rank)

            # Covariance eigenvalues c_k and residual d for this class
            # NOTE: in _calc_log_prob_mvn_wrt_sigma, σ enters LINEARLY:
            #   c_k = σ · (θ·λ_k + (1-θ)),  d = σ · (1-θ)
            s = sigma[:, vc]  # (n_genes,)
            th = theta[:, vc]
            Dq = D[:, vc]
            c_q = s.unsqueeze(-1) * (
                th.unsqueeze(-1) * (lam - 1) + 1
            )  # (n_genes, rank)
            d_q = (1 - th) * s  # (n_genes,)

            # First/second derivs of l_{g,q} wrt c_k (n_genes, rank) and d (n_genes,)
            # Clamp c_q and d_q to prevent overflow in reciprocal powers
            c_q_safe = c_q.clamp(min=1e-6)
            dl_dc = -0.5 / c_q_safe + 0.5 * zq**2 / c_q_safe**2
            d2l_dc2 = 0.5 / c_q_safe**2 - zq**2 / c_q_safe**3

            if self._is_low_rank:
                xp = x_perp_sq[:, q]  # (n_genes,)
                d_q_safe = d_q.clamp(min=1e-6)
                dl_dd = -0.5 * (n_spots - rank) / d_q_safe + 0.5 * xp / d_q_safe**2
                d2l_dd2 = 0.5 * (n_spots - rank) / d_q_safe**2 - xp / d_q_safe**3

            # Jacobians: ∂c_k/∂p1, ∂c_k/∂p2, ∂d/∂p1, ∂d/∂p2, and their second derivatives
            # p1=sigma, p2=theta_logit,  θ=sigmoid(p2),  D=θ(1-θ)
            # c_k = σ·(θ(λ_k-1)+1),  d = σ·(1-θ)  (LINEAR in σ)
            dc_dp1 = c_q / s.unsqueeze(-1)  # = θ(λ_k-1)+1
            dc_dp2 = s.unsqueeze(-1) * (lam - 1) * Dq.unsqueeze(-1)
            d2c_dp11 = torch.zeros_like(c_q)
            d2c_dp22 = (
                s.unsqueeze(-1)
                * (lam - 1)
                * Dq.unsqueeze(-1)
                * (1 - 2 * th).unsqueeze(-1)
            )
            d2c_dp12 = (lam - 1) * Dq.unsqueeze(-1).expand_as(c_q)
            dd_dp1 = d_q / s  # = (1-θ)
            dd_dp2 = -s * Dq
            d2d_dp11 = torch.zeros_like(s)
            d2d_dp22 = -s * Dq * (1 - 2 * th)
            d2d_dp12 = -Dq

            def _hc(dc_pi, dc_pj, d2c_ij):
                return (d2l_dc2 * dc_pi * dc_pj + dl_dc * d2c_ij).sum(
                    -1
                )  # sum over rank

            def _hd(dd_pi, dd_pj, d2d_ij):
                if self._is_low_rank:
                    return d2l_dd2 * dd_pi * dd_pj + dl_dd * d2d_ij
                return torch.zeros(n_genes, device=self.device, dtype=dtype)

            H_diag[:, vc, 0, 0] += _hc(dc_dp1, dc_dp1, d2c_dp11) + _hd(
                dd_dp1, dd_dp1, d2d_dp11
            )
            H_diag[:, vc, 0, 1] += _hc(dc_dp1, dc_dp2, d2c_dp12) + _hd(
                dd_dp1, dd_dp2, d2d_dp12
            )
            H_diag[:, vc, 1, 0] += _hc(dc_dp2, dc_dp1, d2c_dp12) + _hd(
                dd_dp2, dd_dp1, d2d_dp12
            )
            H_diag[:, vc, 1, 1] += _hc(dc_dp2, dc_dp2, d2c_dp22) + _hd(
                dd_dp2, dd_dp2, d2d_dp22
            )

        # Prior contributions (per var_comp; scale by n_isos-1 for share_variance)
        prior_scale = (n_isos - 1) if self.share_variance else 1
        if self.var_prior_model != "none":
            alpha = self.var_prior_model_params["alpha"]
            beta = self.var_prior_model_params["beta"]
            # Prior on sigma_total = sigma; theta_logit has no effect on prior
            s_all = self.sigma  # (n_genes, n_var_comps)
            if self.var_prior_model == "inv_gamma":
                # InverseGamma on u=sigma^2: d²logp/dσ² = 2(α+1)/σ² - 6β/σ⁴
                u = s_all**2
                d2logp_du2 = (alpha + 1) / u**2 - 2 * beta / u**3
                dlogp_du = -(alpha + 1) / u + beta / u**2
                d2prior_dp1 = d2logp_du2 * (2 * s_all) ** 2 + dlogp_du * 2
            else:  # gamma on sigma
                v = s_all.abs()
                d2prior_dp1 = -(alpha - 1) / v**2
            H_diag[:, :, 0, 0] += prior_scale * d2prior_dp1
            # [0,1], [1,0], [1,1] entries: 0 (prior doesn't depend on theta_logit)

        # Build block-diagonal output (n_genes, n_var_comps*2, n_var_comps*2)
        if n_var_comps == 1:
            H_full = H_diag[:, 0, :, :]  # (n_genes, 2, 2)
        else:
            H_full = torch.zeros(
                n_genes,
                n_var_comps * 2,
                n_var_comps * 2,
                device=self.device,
                dtype=dtype,
            )
            for vc in range(n_var_comps):
                H_full[:, vc * 2 : vc * 2 + 2, vc * 2 : vc * 2 + 2] = H_diag[
                    :, vc, :, :
                ]

        return H_full / n_genes

    def _get_log_lik_hessian_sigma_expand(self):
        """Get the Hessian matrix of the log mvn wrt the variance components."""
        sigma_expand = torch.stack(
            [self.sigma, self.theta_logit], dim=-1
        )  # (n_genes, n_var_components, 2)

        # calculate the hessian using pytorch's functional
        # x = sigma_expand.clone().detach().requires_grad_(True)
        hessian_sigma_expand = jacobian(
            self._get_sum_of_grad_log_prob_mvn_wrt_sigma, sigma_expand, vectorize=True
        ).permute(
            2, 0, 1, 3, 4
        )  # (n_genes, n_var_components, 2, n_genes, n_var_components, 2)

        # hessian_sigma_expand = hessian(
        #     self._calc_log_prob_mvn_wrt_sigma, sigma_expand
        # )  # (n_genes, n_var_components, 2, n_genes, n_var_components, 2)

        n_var_comps = hessian_sigma_expand.shape[-2]
        hessian_sigma_expand = hessian_sigma_expand.reshape(
            self.n_genes, n_var_comps * 2, n_var_comps * 2
        )  # (n_genes, n_var_components * 2, n_var_components * 2)

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
        sigma_expand = torch.stack(
            [self.sigma, self.theta_logit], dim=-1
        )  # (n_genes, n_var_components, 2)
        sigma_expand_grad = torch.stack(
            [self.sigma.grad, self.theta_logit.grad], dim=-1
        )  # (n_genes, n_var_components, 2)

        hessian_sigma_expand = (
            -self._get_log_lik_hessian_sigma_expand_analytic()
        )  # (n_genes, n_var_components * 2, n_var_components * 2)
        hessian_sigma_expand += 1e-5 * torch.eye(
            hessian_sigma_expand.shape[-1]
        ).unsqueeze(0).to(
            self.device
        )  # for stability
        right = hessian_sigma_expand.matmul(
            sigma_expand.reshape(n_genes, -1, 1)
        ) - sigma_expand_grad.reshape(
            n_genes, -1, 1
        )  # (n_genes, n_var_components * 2, 1)
        sigma_expand_new = torch.linalg.solve(
            hessian_sigma_expand, right
        )  # (n_genes, n_var_components * 2, 1)
        sigma_expand_new = sigma_expand_new.reshape(
            sigma_expand.shape
        )  # (n_genes, n_var_components, 2)

        # update the parameters and clear the gradients
        with torch.no_grad():
            self.sigma.copy_(sigma_expand_new[..., 0])
            self.theta_logit.copy_(sigma_expand_new[..., 1])
            self.sigma.grad.zero_()
            self.theta_logit.grad.zero_()

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
            self.n_genes,
            self.n_spots,
            self.n_isos,
            self.n_factors,
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
        gradient_nu = self.nu.grad.transpose(1, 2).reshape(
            n_genes, -1, 1
        )  # (n_genes, n_spots * (n_isos - 1), 1)
        right = (
            hessian_nu.matmul(self.nu.transpose(1, 2).reshape(n_genes, -1, 1))
            - step * gradient_nu
        )  # (n_genes, n_spots * (n_isos - 1), -1)
        nu_new = (
            torch.linalg.solve(hessian_nu, right)
            .reshape(n_genes, n_isos - 1, n_spots)
            .transpose(1, 2)
        )
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
        hessian_beta_expand += 1e-5 * torch.eye(
            hessian_beta_expand.shape[-1]
        ).unsqueeze(0).to(self.device)

        # combine beta and bias_eta into a single (n_genes, n_factors+1, n_isos-1) tensor
        # gradient w.r.t. neg log-likelihood = -grad w.r.t. log-likelihood
        gradient_beta_expand = -torch.cat(
            [self.beta.grad, self.bias_eta.grad.unsqueeze(1)], dim=1
        )  # (n_genes, n_factors + 1, n_isos - 1)
        beta_expand = torch.cat([self.beta, self.bias_eta.unsqueeze(1)], dim=1)
        # Newton step: β_new = β − H⁻¹ ∇L(β)
        # Equivalent form: H β_new = H β − ∇L  ⟹  right = H β − ∇L
        right = hessian_beta_expand.matmul(
            beta_expand.transpose(1, 2).reshape(n_genes, -1, 1)
        ) - step * gradient_beta_expand.transpose(1, 2).reshape(
            n_genes, -1, 1
        )  # (n_genes, (n_factors + 1) * (n_isos - 1), 1)
        beta_expand_new = (
            torch.linalg.solve(hessian_beta_expand, right)
            .reshape(n_genes, n_isos - 1, n_factors + 1)
            .transpose(1, 2)
        )  # (n_genes, n_factors + 1, n_isos - 1)

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
            chol = (
                self._chol_hessian_nu
            )  # (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1))
        except AttributeError:
            # if cholesky is not available, compute it on the fly
            chol = torch.linalg.cholesky(-self._get_log_lik_hessian_nu())

        n_genes, n_spots, n_isos = self.n_genes, self.n_spots, self.n_isos
        step = 1  # step size

        # Use cholesky_solve to compute H⁻¹ g without materialising the full inverse.
        # This avoids an O(n_genes × (n_spots*(n_isos-1))²) memory allocation.
        gradient_nu = self.nu.grad.transpose(1, 2).reshape(
            n_genes, -1, 1
        )  # (n_genes, n_spots * (n_isos - 1), 1)
        # delta_nu = H⁻¹ g, solved via the stored Cholesky factor
        delta_nu = torch.cholesky_solve(
            gradient_nu, chol
        )  # (n_genes, n_spots * (n_isos - 1), 1)
        nu_new = self.nu.transpose(1, 2).reshape(n_genes, -1, 1) - step * delta_nu
        nu_new = nu_new.reshape(n_genes, n_isos - 1, n_spots).transpose(1, 2)
        # update the parameters and clear the gradients
        with torch.no_grad():
            self.nu.copy_(nu_new)
            self.nu.grad.zero_()

        if return_variables:
            return nu_new

    def _fit(
        self,
        store_param_history: bool = False,
        verbose: bool = False,
        quiet: bool = False,
        random_seed=None,
    ):
        """Main fitting function to find the MAP estimates using specified fitting method."""
        # extract configs
        fitting_method = self.fitting_method
        optim = self.fitting_configs["optim"]
        max_epochs = self.fitting_configs["max_epochs"]
        patience = self.fitting_configs["patience"]
        tol = self.fitting_configs["tol"]
        tol_relative = self.fitting_configs.get("tol_relative", False)
        max_epochs = 10000 if max_epochs == -1 else max_epochs
        patience = patience if patience > 0 else 1

        if random_seed is not None:  # set random seed for reproducibility
            torch.manual_seed(random_seed)

        batch_size = self.n_genes
        t_start = timer()
        logger = PatienceLogger(
            batch_size,
            patience,
            min_delta=tol,
            tol_relative=tol_relative,
            store_param_history=store_param_history,
        )

        # start training
        self.train()

        while logger.epoch < max_epochs and not logger.convergence.all():
            self.optimizer.zero_grad()

            # minimize the negative log-likelihood or the negative log-marginal-likelihood
            neg_log_prob = -self()  # (n_genes,)
            neg_log_prob.mean().backward()  # backpropagate gradients

            if fitting_method == "joint_newton":
                # update nu, beta, bias, and sigmas using Newton's method
                self._update_joint_newton()

            elif fitting_method == "marginal_newton":
                # update nu every k epochs using Newton's method;
                # use logger.epoch (canonical counter) — NOT a stale local variable
                if logger.epoch % self.fitting_configs["update_nu_every_k"] == 0:
                    self._update_marginal_nu_newton()
                # skip gradient descent update for nu
                self.nu.grad.zero_()

            # gradient-based updates
            if optim == "lbfgs":
                # update all parameters using L-BFGS
                self.optimizer.zero_grad()
                neg_log_prob = self.optimizer.step(self._closure)
            else:
                # update the remaining parameters with non-zero gradients using gradient descent
                self.optimizer.step()

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
                    stacklevel=3,
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

        # make sure constraints are satisfied
        self._final_sanity_check()

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

    @torch.no_grad()
    def _final_sanity_check(self):
        """Make sure constraints are satisfied."""
        # ensure positive parameters
        self.sigma.abs_()

    def fit(
        self,
        store_param_history: bool = False,
        verbose: bool = False,
        quiet: bool = False,
        random_seed: Optional[int] = None,
        diagnose: Optional[bool] = None,
    ) -> Optional[dict]:
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

        self._configure_optimizer(verbose=verbose)
        diag_outputs = self._fit(
            store_param_history=store_param_history,
            verbose=verbose,
            quiet=quiet,
            random_seed=random_seed,
        )
        if store_param_history:
            return diag_outputs

    def clone(self) -> "MultinomGLMM":
        """Clone a Multinomial GLMM model with the same set of parameters."""
        new_model = type(self)(
            share_variance=self.share_variance,
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
            device=self.device.type,
        )
        new_model.load_state_dict(self.state_dict())

        return new_model
