"""Private worker functions and model subclasses for SplisosmGLMM.

Not part of the public API.  Imported by :mod:`splisosm.hyptest.glmm`.
"""

from __future__ import annotations

import torch
from torch import nn

from splisosm.glmm.model import MultinomGLM, MultinomGLMM

# ---------------------------------------------------------------------------
# Model subclasses
# ---------------------------------------------------------------------------


class IsoFullModel(MultinomGLMM):
    """The full model with all parameters.

    See model.MultinomGLMM for more details.

    Examples
    --------
    Direct training:

    >>> model = IsoFullModel()
    >>> model.setup_data(counts, cov_sp, design_mat)
    >>> model.fit()

    Initialize from a trained null model without spatial variance:

    >>> model = IsoFullModel.from_trained_null_sp_var_model(null_model)
    >>> model.fit()

    Initialize from a trained null model without a given factor:

    >>> model = IsoFullModel.from_trained_null_no_beta_model(null_model, new_X_spot_col, factor_idx)
    >>> model.fit()
    """

    @classmethod
    def from_trained_null_no_sp_var_model(
        cls, null_model: MultinomGLMM
    ) -> "IsoFullModel":
        # clone the model and convert it
        new_model = null_model.clone()
        new_model.__class__ = cls

        # clear the fitting history
        new_model.fitting_time = 0
        new_model.register_buffer(
            "convergence", torch.zeros(new_model.n_genes, dtype=bool)
        )

        # set fitting methods to gradient descent
        if new_model.fitting_method == "joint_newton":
            new_model.fitting_method = "joint_gd"
        elif new_model.fitting_method == "marginal_newton":
            new_model.fitting_method = "marginal_gd"

        new_model.fitting_configs.update({"lr": 1e-2, "optim": "adam", "patience": 5})

        # Re-initialize sigma from the fitted nu magnitude (prevents sigma collapse)
        nu_std = new_model.nu.detach().std(dim=1)  # (n_genes, n_isos-1)
        if new_model.share_variance:
            sigma_reinit = nu_std.max(dim=-1, keepdim=True).values.clamp(min=0.1)
        else:
            sigma_reinit = nu_std.clamp(min=0.1)
        new_model.sigma.detach_().copy_(sigma_reinit)

        # turn the gradient of the spatial variance term back on
        new_model.theta_logit.detach_().fill_(-3.0).requires_grad_(True)

        return new_model


class IsoNullNoSpVar(MultinomGLMM):
    """The null model without spatial variance.

    See model.MultinomGLMM for more details.

    Examples
    --------
    Direct training:

    >>> model = IsoNullNoSpVar()
    >>> model.setup_data(counts, cov_sp, design_mat)
    >>> model.fit()

    Initialize from a trained full model:

    >>> model = IsoNullNoSpVar.from_trained_full_model(full_model)
    >>> model.fit()
    """

    _supported_fitting_methods = ["joint_gd", "marginal_gd", "marginal_newton"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # currently only support fitting methods that update the variance using gradient descent
        if self.fitting_method not in IsoNullNoSpVar._supported_fitting_methods:
            raise ValueError(
                f"The fitting method must be one of {IsoNullNoSpVar._supported_fitting_methods}."
            )

    def _configure_learnable_variables(self):
        super()._configure_learnable_variables()
        # make sure the spatial variance is turned off after configuration
        self._turn_off_spatial_variance()

    def _initialize_params(self):
        super()._initialize_params()
        # make sure the spatial variance is turned off after initialization
        self._turn_off_spatial_variance()

    def _turn_off_spatial_variance(self):
        """Set spatial variance to zero and don't update it."""
        self.theta_logit.detach_().fill_(-torch.inf).requires_grad_(False)

    @classmethod
    def from_trained_full_model(cls, full_model: MultinomGLMM) -> "IsoNullNoSpVar":
        """Initialize an IsoNullNoSpVar model from a trained full model."""
        # clone the model and convert it to the NullNoSpVar class
        new_model = full_model.clone()
        new_model.__class__ = cls

        # clear the fitting history
        new_model.fitting_time = 0
        new_model.register_buffer(
            "convergence", torch.zeros(new_model.n_genes, dtype=bool)
        )

        # set fitting methods to gradient descent
        if new_model.fitting_method == "joint_newton":
            new_model.fitting_method = "joint_gd"
        elif new_model.fitting_method == "marginal_newton":
            new_model.fitting_method = "marginal_gd"

        new_model.fitting_configs.update({"lr": 1e-2, "optim": "adam", "patience": 5})

        # remove spatial variance
        new_model._turn_off_spatial_variance()

        return new_model


# ---------------------------------------------------------------------------
# Single-gene fitting workers (used by joblib / multiprocessing dispatch)
# ---------------------------------------------------------------------------


def _fit_model_one_gene(
    model_configs,
    model_type,
    counts,
    corr_sp_eigvals,
    corr_sp_eigvecs,
    design_mtx,
    quiet=True,
    random_seed=None,
    device: str = "cpu",
):
    """Fit the MultinomGLMM model to the data.

    This is a worker function for multiprocessing.

    Parameters
    ----------
    model_configs : dict
        The fitting configurations for the model.
    model_type : str
        The model type to fit. Can be one of 'glmm-full', 'glmm-null', 'glm'.
    counts : torch.Tensor
        Isoform counts.
    corr_sp_eigvals : torch.Tensor
        Eigenvalues of spatial covariance matrix.
    corr_sp_eigvecs : torch.Tensor
        Eigenvectors of spatial covariance matrix.
    design_mtx : torch.Tensor or None
        Design matrix.
    quiet : bool, optional
        Suppress fitting logs.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pars : dict
        The fitted parameters extracted.
    """
    assert model_type in ["glmm-full", "glmm-null", "glm"]

    if counts.is_sparse:
        counts = counts.to_dense()

    # initialize and setup the model
    if model_type == "glm":
        model = MultinomGLM()
        model.setup_data(counts, design_mtx=design_mtx, device=device)
        return_par_names = ["beta", "bias_eta"]
    else:
        if model_type == "glmm-full":
            model = IsoFullModel(**model_configs)
        elif model_type == "glmm-null":
            model = IsoNullNoSpVar(**model_configs)
        else:
            raise ValueError(f"Invalid model type {model_type}.")

        model.setup_data(
            counts,
            design_mtx=design_mtx,
            corr_sp_eigvals=corr_sp_eigvals,
            corr_sp_eigvecs=corr_sp_eigvecs,
            device=device,
        )
        return_par_names = [
            "nu",
            "beta",
            "bias_eta",
            "sigma",
            "theta_logit",
        ]

    # fit the model
    model.fit(quiet=quiet, verbose=False, random_seed=random_seed)

    # extract and return the fitted parameters
    pars = {
        k: v.detach() for k, v in model.state_dict().items() if k in return_par_names
    }

    return pars


def _fit_null_full_sv_one_gene(
    model_configs,
    counts,
    corr_sp_eigvals,
    corr_sp_eigvecs,
    design_mtx,
    refit_null=True,
    quiet=True,
    random_seed=None,
    device: str = "cpu",
):
    """Fit the null and full model to the data.

    This is a worker function for multiprocessing. See splisosm.fit_null_full_sv() for more details.

    Parameters
    ----------
    model_configs : dict
        Model configuration dictionary.
    counts : torch.Tensor
        Isoform counts.
    corr_sp_eigvals : torch.Tensor
        Eigenvalues of spatial covariance.
    corr_sp_eigvecs : torch.Tensor
        Eigenvectors of spatial covariance.
    design_mtx : torch.Tensor or None
        Design matrix.
    refit_null : bool, optional
        Whether to refit null model.
    quiet : bool, optional
        Suppress logs.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    tuple of dict
        (null_pars, full_pars), the fitted parameters of the null and full models.
    """
    if counts.is_sparse:
        counts = counts.to_dense()

    # fit the null model
    null = IsoNullNoSpVar(**model_configs)
    null.setup_data(
        counts,
        design_mtx=design_mtx,
        corr_sp_eigvals=corr_sp_eigvals,
        corr_sp_eigvecs=corr_sp_eigvecs,
        device=device,
    )
    null.fit(quiet=quiet, verbose=False, random_seed=random_seed)

    # fit the full model from the null
    full = IsoFullModel.from_trained_null_no_sp_var_model(null)
    full.fit(quiet=quiet, verbose=False, random_seed=random_seed)

    # refit the null model if needed
    if refit_null:
        null_refit = IsoNullNoSpVar.from_trained_full_model(full)
        null_refit.fit(quiet=quiet, verbose=False, random_seed=random_seed)

        # update the null if larger log-likelihood
        if null_refit().mean() > null().mean():  # null() returns shape of (n_genes,)
            null = null_refit

        # refit the full model from the null if likelihood decreases
        if null().mean() > full().mean():
            full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
            full_refit.fit(quiet=quiet, verbose=False, random_seed=random_seed)
            if full_refit().mean() > full().mean():
                full = full_refit

    return_par_names = [
        "nu",
        "beta",
        "bias_eta",
        "sigma",
        "theta_logit",
    ]
    null_pars = {
        k: v.detach() for k, v in null.state_dict().items() if k in return_par_names
    }
    full_pars = {
        k: v.detach() for k, v in full.state_dict().items() if k in return_par_names
    }

    return (null_pars, full_pars)


def _fit_perm_one_gene(
    perm_idx,
    model_configs,
    counts,
    corr_sp_eigvals,
    corr_sp_eigvecs,
    design_mtx,
    refit_null,
    random_seed=None,
    device: str = "cpu",
):
    """Calculate the likelihood ratio statistic for spatial variability using permutation.

    This is a worker function for multiprocessing. See splisosm.fit_perm_sv_llr() for more details.

    Parameters
    ----------
    perm_idx : torch.Tensor
        Permutation indices.
    model_configs : dict
        Model configurations.
    counts : torch.Tensor
        Isoform counts.
    corr_sp_eigvals : torch.Tensor
        Eigenvalues of spatial covariance.
    corr_sp_eigvecs : torch.Tensor
        Eigenvectors of spatial covariance.
    design_mtx : torch.Tensor or None
        Design matrix.
    refit_null : bool
        Whether to refit null model.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    _sv_llr : torch.Tensor
        Shape (1,), the likelihood ratio statistic.
    """

    if counts.is_sparse:
        counts = counts.to_dense()

    # permute the data coordinates
    counts_perm = counts[:, perm_idx, :]  # (n_genes, n_spots, n_isos)
    design_mtx_perm = design_mtx[perm_idx, :] if design_mtx is not None else None

    # fit the null model
    null = IsoNullNoSpVar(**model_configs)
    null.setup_data(
        counts_perm,
        design_mtx=design_mtx_perm,
        corr_sp_eigvals=corr_sp_eigvals,
        corr_sp_eigvecs=corr_sp_eigvecs,
        device=device,
    )
    null.fit(quiet=True, verbose=False, random_seed=random_seed)

    # fit the full model from the null
    full = IsoFullModel.from_trained_null_no_sp_var_model(null)
    full.fit(quiet=True, verbose=False, random_seed=random_seed)

    # refit the null model if needed
    if refit_null:
        null_refit = IsoNullNoSpVar.from_trained_full_model(full)
        null_refit.fit(quiet=True, verbose=False, random_seed=random_seed)

        # update the null if larger log-likelihood
        if null_refit().mean() > null().mean():  # null() returns shape of (n_genes,)
            null = null_refit

        # refit the full model from the null if likelihood decreases
        if null().mean() > full().mean():
            full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
            full_refit.fit(quiet=True, verbose=False, random_seed=random_seed)
            if full_refit().mean() > full().mean():
                full = full_refit

    # calculate the likelihood ratio statistic
    # use marginal likelihood for stability
    _sv_llr, _ = _calc_llr_spatial_variability(null, full)

    return _sv_llr


# ---------------------------------------------------------------------------
# Test-statistic helpers
# ---------------------------------------------------------------------------


def _calc_llr_spatial_variability(null_model: IsoNullNoSpVar, full_model: IsoFullModel):
    """Calculate the likelihood ratio statistic for spatial variability.

    Parameters
    ----------
    null_model : IsoNullNoSpVar
        The fitted null model of one gene (theta = 0, i.e. no spatial variance).
    full_model : IsoFullModel
        The fitted full model of one gene (theta != 0).

    Returns
    -------
    sv_llr : torch.Tensor
        Shape (n_genes,), the likelihood ratio statistic per gene.
    df : int
        The degrees of freedom for the likelihood ratio statistic (equal to the number of variance components).
    """
    # calculate the likelihood ratio statistic
    # use marginal likelihood for stability
    sv_llr = (
        full_model._calc_log_prob_marginal().detach()
        - null_model._calc_log_prob_marginal().detach()
    ) * 2  # (n_genes,)

    # the degrees of freedom for the likelihood ratio statistic
    n_var_comps = 1 if full_model.share_variance else full_model.n_isos - 1

    return sv_llr, n_var_comps


def _calc_wald_differential_usage(fitted_full_model: MultinomGLM):
    """Calculate the Wald statistic for differential usage.

    H_0: beta[p,:] = 0
    H_1: beta[p,:] != 0

    Parameters
    ----------
    fitted_full_model : MultinomGLM
        The fitted full model of one gene.

    Returns
    -------
    wald_stat : torch.Tensor
        Shape (n_genes, n_factors), the Wald statistic for each factor per gene.
    df : int
        The degrees of freedom for the Wald statistic (equal to n_isos - 1).
    """
    n_factors, n_isos = (
        fitted_full_model.n_factors,
        fitted_full_model.n_isos,
    )
    assert n_factors > 0, "No factor is included in the model."

    # extract the Hessian for beta per factor
    # beta_bias_hess.shape = (n_genes, (n_factors + 1)*(n_isos - 1), (n_factors + 1)*(n_isos - 1))
    beta_bias_hess = fitted_full_model._get_log_lik_hessian_beta_bias().detach()
    fisher_info = []  # -> (n_genes, n_factors, n_isos - 1, n_isos - 1)
    for i in range(n_factors):
        # retrieve the Hessian for beta per factor
        beta_idx_per_factor = [i + j * (n_factors + 1) for j in range(n_isos - 1)]
        # beta_hess.shape = (n_genes, n_isos - 1, n_isos - 1)
        beta_hess = beta_bias_hess[:, beta_idx_per_factor, :][:, :, beta_idx_per_factor]
        # the Fisher information matrix is the negative Hessian
        fisher_info.append(-beta_hess)  # (n_genes, n_isos - 1, n_isos - 1)
    fisher_info = torch.stack(
        fisher_info, dim=1
    )  # (n_genes, n_factors, n_isos - 1, n_isos - 1)

    # extract the beta estimates
    beta_est = fitted_full_model.beta.detach()  # (n_genes, n_factors, n_isos - 1)

    # calculate the Wald statistic
    wald_stat = (
        beta_est.unsqueeze(-2) @ fisher_info @ beta_est.unsqueeze(-1)
    )  # (n_genes, n_factors, 1, 1)

    return wald_stat.squeeze(), n_isos - 1


def _calc_score_differential_usage(fitted_full_model: MultinomGLM, covar_to_test):
    """Calculate the score statistic for differential usage.

    H_0: beta[p,:] = 0
    H_1: beta[p,:] != 0

    Parameters
    ----------
    fitted_full_model : MultinomGLM
        The fitted full model of one gene without covariates.
    covar_to_test : torch.Tensor
        Shape (n_spots, n_covars), the design matrix of the covariates to test.

    Returns
    -------
    score_stat : torch.Tensor
        Shape (n_factors,), the score statistic for each factor.
    df : int
        The degrees of freedom for the score statistic (equal to n_isos - 1).
    """
    n_genes, n_factors_design, n_isos = (
        fitted_full_model.n_genes,
        fitted_full_model.n_factors,
        fitted_full_model.n_isos,
    )
    # assert n_factors == 0, "No factor should be included in the model."

    # in case of a single covariate, expand the design matrix
    if covar_to_test.dim() == 1:
        covar_to_test = covar_to_test.unsqueeze(-1)  # (n_spots, 1)
    elif covar_to_test.dim() > 2:
        raise ValueError("The covariate design matrix must be 2D.")

    n_factors_covar = covar_to_test.shape[-1]
    n_factors = n_factors_design + n_factors_covar

    # clone the full model and reset the design matrix
    m_full = fitted_full_model.clone()
    with torch.no_grad():
        # merge the design matrix with the covariates to test -> (1, n_spots, n_factors)
        m_full.X_spot = torch.concat(
            [m_full.X_spot, covar_to_test.unsqueeze(0)], axis=-1
        )

        # merge the coefficients with zeros for the covariates to test -> (n_genes, n_factors, n_isos - 1)
        m_full.beta = nn.Parameter(
            torch.concat(
                [
                    m_full.beta,
                    torch.zeros(
                        n_genes, n_factors_covar, n_isos - 1, device=m_full.beta.device
                    ),
                ],
                axis=1,
            ),
            requires_grad=True,
        )

    # calculate the score aka the gradient of the log-joint-likelihood
    d_l_d_eta = m_full.counts - m_full._alpha() * m_full.counts.sum(
        axis=-1, keepdim=True
    )  # (n_genes, n_spots, n_isos)
    score = (
        covar_to_test.T.unsqueeze(0) @ d_l_d_eta.detach()[..., :-1]
    )  # (n_genes, n_factors_covar, n_isos - 1)

    # calculate the Fisher information matrix
    # beta_bias_hess.shape = (n_genes, (n_factors + 1)*(n_isos - 1), (n_factors + 1)*(n_isos - 1))
    beta_bias_hess = m_full._get_log_lik_hessian_beta_bias().detach()
    fisher_info = []  # -> (n_genes, n_factors, n_isos - 1, n_isos - 1)
    for i in range(n_factors):
        # retrieve the Hessian for beta per factor
        beta_idx_per_factor = [i + j * (n_factors + 1) for j in range(n_isos - 1)]
        # beta_hess.shape = (n_genes, n_isos - 1, n_isos - 1)
        beta_hess = beta_bias_hess[:, beta_idx_per_factor, :][:, :, beta_idx_per_factor]

        # add a small value to the diagonal to ensure invertibility
        beta_hess += 1e-5 * torch.eye(
            beta_hess.shape[-1], device=beta_hess.device
        ).unsqueeze(0)

        # the Fisher information matrix is the negative Hessian
        fisher_info.append(-beta_hess)  # (n_genes, n_isos - 1, n_isos - 1)

    fisher_info = torch.stack(
        fisher_info, dim=1
    )  # (n_genes, n_factors, n_isos - 1, n_isos - 1)
    fisher_info = fisher_info[
        :, n_factors_design:, :, :
    ]  # exclude the design matrix in the fitted model

    # Score statistic: s^T F^{-1} s per (gene, factor).
    # Compute F^{-1} s via solve.  Per-factor loop avoids a PyTorch
    # deprecation warning about output-tensor resizing in batched solve
    # when the gene dimension is 1 (common for reconstructed per-gene models).
    score_stat = torch.empty(n_genes, n_factors_covar, device=score.device)
    for f in range(n_factors_covar):
        # fisher_info[:, f] shape: (n_genes, n_isos-1, n_isos-1)
        # score[:, f]       shape: (n_genes, n_isos-1)
        _sol = torch.linalg.solve(fisher_info[:, f], score[:, f])  # (n_genes, n_isos-1)
        score_stat[:, f] = (score[:, f] * _sol).sum(-1)

    return score_stat, n_isos - 1
