import warnings
from timeit import default_timer as timer
import re

import pandas as pd
import numpy as np
from scipy.stats import chi2, ttest_ind, combine_pvalues
import torch
import torch.multiprocessing as mp
from torch import nn
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from splisosm.utils import get_cov_sp, counts_to_ratios, false_discovery_control
from splisosm.dataset import IsoDataset
from splisosm.model import MultinomGLM, MultinomGLMM
from splisosm.likelihood import liu_sf

class IsoFullModel(MultinomGLMM):
    """The full model with all parameters.

    See model.MultinomGLMM for more details.

    Usages:
    - Direct training:
        model = IsoFullModel()
        model.setup_data(counts, cov_sp, design_mat)
        model.fit()
    - Initialize from a trained null model without spatial variance:
        model = IsoFullModel.from_trained_null_sp_var_model(null_model)
        model.fit()
    - Initialize from a trained null model without a given factor:
        model = IsoFullModel.from_trained_null_no_beta_model(null_model, new_X_spot_col, factor_idx)
        model.fit()
    """
    @classmethod
    def from_trained_null_no_sp_var_model(cls, null_model):
        # clone the model and convert it
        new_model = null_model.clone()
        new_model.__class__ = cls

        # clear the fitting history
        new_model.fitting_time = 0
        new_model.register_buffer("convergence", torch.zeros(new_model.n_genes, dtype=bool))

        # set fitting methods to gradient descent
        if new_model.fitting_method == 'joint_newton':
            new_model.fitting_method = 'joint_gd'
        elif new_model.fitting_method == 'marginal_newton':
            new_model.fitting_method = 'marginal_gd'

        new_model.fitting_configs.update({
            'lr': 1e-3,
            'optim': "adam",
            'patience': 3
        })

        # turn the gradient of the spatial variance term back on
        if new_model.var_parameterization_sigma_theta:
            new_model.theta_logit.detach_().fill_(-5.0).requires_grad_(True)
        else:
            new_model.sigma_sp.requires_grad_(True)

        return new_model


class IsoNullNoSpVar(MultinomGLMM):
    """The null model without spatial variance.

    See model.MultinomGLMM for more details.

    Usages:
    - Direct training:
        model = IsoNullNoSpVar()
        model.setup_data(counts, cov_sp, design_mat)
        model.fit()
    - Initialize from a trained full model:
        model = IsoNullNoSpVar.from_trained_full_model(full_model)
        model.fit()
    """
    _supported_fitting_methods = ['joint_gd', 'marginal_gd', 'marginal_newton']

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
        if self.var_parameterization_sigma_theta:
            self.theta_logit.detach_().fill_(-torch.inf).requires_grad_(False)
        else:
            self.sigma_sp.detach_().fill_(0.0).requires_grad_(False)

    @classmethod
    def from_trained_full_model(cls, full_model):
        """Initialize an IsoNullNoSpVar model from a trained full model."""
        # clone the model and convert it to the NullNoSpVar class
        new_model = full_model.clone()
        new_model.__class__ = cls

        # clear the fitting history
        new_model.fitting_time = 0
        new_model.register_buffer("convergence", torch.zeros(new_model.n_genes, dtype=bool))

        # set fitting methods to gradient descent
        if new_model.fitting_method == 'joint_newton':
            new_model.fitting_method = 'joint_gd'
        elif new_model.fitting_method == 'marginal_newton':
            new_model.fitting_method = 'marginal_gd'

        new_model.fitting_configs.update({
            'lr': 1e-3,
            'optim': "adam",
            'patience': 3
        })

        # remove spatial variance
        new_model._turn_off_spatial_variance()

        return new_model


def _fit_model_one_gene(model_configs, model_type, counts, corr_sp_eigvals, corr_sp_eigvecs, design_mtx,
                        quiet=True, random_seed=None):
    """Fit the MultinomGLMM model to the data.

    This is a worker function for multiprocessing.

    Args:
        model_configs: dict, the fitting configurations for the model.
        model_type: str, the model type to fit. Can be one of 'glmm-full', 'glmm-null', 'glm'.

    Returns:
        pars: dict, the fitted parameters extracted.
    """
    assert model_type in ['glmm-full', 'glmm-null', 'glm']

    # initialize and setup the model
    if model_type == 'glm':
        model = MultinomGLM()
        model.setup_data(counts, design_mtx=design_mtx)
        return_par_names = ['beta', 'bias_eta']
    else:
        if model_type == 'glmm-full':
            model = IsoFullModel(**model_configs)
        elif model_type == 'glmm-null':
            model = IsoNullNoSpVar(**model_configs)
        else:
            raise ValueError(f"Invalid model type {model_type}.")

        model.setup_data(
            counts, design_mtx=design_mtx,
            corr_sp_eigvals=corr_sp_eigvals, corr_sp_eigvecs=corr_sp_eigvecs
        )
        return_par_names = ['nu', 'beta', 'bias_eta', 'sigma', 'theta_logit', 'sigma_sp', 'sigma_nsp']

    # fit the model
    model.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

    # extract and return the fitted parameters
    pars = {k: v.detach() for k, v in model.state_dict().items() if k in return_par_names}

    return pars

def _fit_null_full_sv_one_gene(model_configs, counts, corr_sp_eigvals, corr_sp_eigvecs,
                               design_mtx, refit_null = True, quiet=True, random_seed=None):
    """Fit the null and full model to the data.

    This is a worker function for multiprocessing. See splisosm.fit_null_full_sv() for more details.

    Returns:
        (null_pars, full_pars): dicts, the fitted parameters of the null and full models.
    """
    # fit the null model
    null = IsoNullNoSpVar(**model_configs)
    null.setup_data(
        counts, design_mtx = design_mtx,
        corr_sp_eigvals = corr_sp_eigvals, corr_sp_eigvecs = corr_sp_eigvecs
    )
    null.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

    # fit the full model from the null
    full = IsoFullModel.from_trained_null_no_sp_var_model(null)
    full.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

    # refit the null model if needed
    if refit_null:
        null_refit = IsoNullNoSpVar.from_trained_full_model(full)
        null_refit.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

        # update the null if larger log-likelihood
        if null_refit().mean() > null().mean(): # null() returns shape of (n_genes,)
            null = null_refit

        # refit the full model from the null if likelihood decreases
        if null().mean() > full().mean():
            full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
            full_refit.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)
            if full_refit().mean() > full().mean():
                full = full_refit

    return_par_names = ['nu', 'beta', 'bias_eta', 'sigma', 'theta_logit', 'sigma_sp', 'sigma_nsp']
    null_pars = {k: v.detach() for k, v in null.state_dict().items() if k in return_par_names}
    full_pars = {k: v.detach() for k, v in full.state_dict().items() if k in return_par_names}

    return (null_pars, full_pars)

def _fit_perm_one_gene(perm_idx, model_configs, counts, corr_sp_eigvals, corr_sp_eigvecs,
                       design_mtx, refit_null, random_seed=None):
    """Calculate the likelihood ratio statistic for spatial variability using permutation.

    This is a worker function for multiprocessing. See splisosm.fit_perm_sv_llr() for more details.

    Returns:
        _sv_llr: tensor(1), the likelihood ratio statistic.
    """
    # permute the data coordinates
    counts_perm = counts[:, perm_idx, :] # (n_genes, n_spots, n_isos)
    design_mtx_perm = design_mtx[perm_idx, :] if design_mtx is not None else None

    # fit the null model
    null = IsoNullNoSpVar(**model_configs)
    null.setup_data(
        counts_perm, design_mtx = design_mtx_perm,
        corr_sp_eigvals = corr_sp_eigvals, corr_sp_eigvecs = corr_sp_eigvecs
    )
    null.fit(quiet=True, verbose=False, diagnose=False, random_seed=random_seed)

    # fit the full model from the null
    full = IsoFullModel.from_trained_null_no_sp_var_model(null)
    full.fit(quiet=True, verbose=False, diagnose=False, random_seed=random_seed)

    # refit the null model if needed
    if refit_null:
        null_refit = IsoNullNoSpVar.from_trained_full_model(full)
        null_refit.fit(quiet=True, verbose=False, diagnose=False, random_seed=random_seed)

        # update the null if larger log-likelihood
        if null_refit().mean() > null().mean(): # null() returns shape of (n_genes,)
            null = null_refit

        # refit the full model from the null if likelihood decreases
        if null().mean() > full().mean():
            full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
            full_refit.fit(quiet=True, verbose=False, diagnose=False, random_seed=random_seed)
            if full_refit().mean() > full().mean():
                full = full_refit

    # calculate the likelihood ratio statistic
    # use marginal likelihood for stability
    # _sv_llr = (
    # 	full._calc_log_prob_marginal().detach() - null._calc_log_prob_marginal().detach()
    # ) * 2
    _sv_llr, _ = _calc_llr_spatial_variability(null, full)

    return _sv_llr

def _calc_llr_spatial_variability(null_model: IsoNullNoSpVar, full_model: IsoFullModel):
    """Calculate the likelihood ratio statistic for spatial variability.

    Args:
        null_model: the fitted null model of one gene (sigma_sp = 0).
        full_model: the fitted full model of one gene (sigma_sp != 0).

    Returns:
        sv_llr: tensor(n_genes,), the likelihood ratio statistic per gene.
        df: int, the degrees of freedom for the likelihood ratio statistic (equal to the number of variance components).
    """
    # calculate the likelihood ratio statistic
    # use marginal likelihood for stability
    sv_llr = (
        full_model._calc_log_prob_marginal().detach() - null_model._calc_log_prob_marginal().detach()
    ) * 2 # (n_genes,)

    # the degrees of freedom for the likelihood ratio statistic
    n_var_comps = 1 if full_model.share_variance else full_model.n_isos - 1

    return sv_llr, n_var_comps

def _calc_wald_differential_usage(fitted_full_model: MultinomGLM):
    """Calculate the Wald statistic for differential usage.

    H_0: beta[p,:] = 0
    H_1: beta[p,:] != 0

    Args:
        full_model_fitted: the fitted full model of one gene.

    Returns:
        wald_stat: tensor(n_genes, n_factors), the Wald statistic for each factor per gene.
        df: int, the degrees of freedom for the Wald statistic (equal to n_isos - 1).
    """
    n_genes, n_factors, n_isos = (
        fitted_full_model.n_genes, fitted_full_model.n_factors, fitted_full_model.n_isos
    )
    assert n_factors > 0, "No factor is included in the model."

    # extract the Hessian for beta per factor
    # beta_bias_hess.shape = (n_genes, (n_factors + 1)*(n_isos - 1), (n_factors + 1)*(n_isos - 1))
    beta_bias_hess = fitted_full_model._get_log_lik_hessian_beta_bias().detach()
    fisher_info = [] # -> (n_genes, n_factors, n_isos - 1, n_isos - 1)
    for i in range(n_factors):
        # retrieve the Hessian for beta per factor
        beta_idx_per_factor = [i + j * (n_factors + 1) for j in range(n_isos - 1)]
        # beta_hess.shape = (n_genes, n_isos - 1, n_isos - 1)
        beta_hess = beta_bias_hess[:, beta_idx_per_factor, :][:, :, beta_idx_per_factor]
        # the Fisher information matrix is the negative Hessian
        fisher_info.append( - beta_hess) # (n_genes, n_isos - 1, n_isos - 1)
    fisher_info = torch.stack(fisher_info, dim=1) # (n_genes, n_factors, n_isos - 1, n_isos - 1)

    # extract the beta estimates
    beta_est = fitted_full_model.beta.detach() # (n_genes, n_factors, n_isos - 1)

    # calculate the Wald statistic
    wald_stat = beta_est.unsqueeze(-2) @ fisher_info @ beta_est.unsqueeze(-1) # (n_genes, n_factors, 1, 1)

    return wald_stat.squeeze(), n_isos - 1

def _calc_score_differential_usage(fitted_full_model: MultinomGLM, covar_to_test):
    """Calculate the score statistic for differential usage.

    H_0: beta[p,:] = 0
    H_1: beta[p,:] != 0

    Args:
        full_model_fitted: the fitted full model of one gene without covariates.
        covar_to_test: tensor(n_spots, n_covars), the design matrix of the covariates to test.

    Returns:
        score_stat: tensor(n_factors), the score statistic for each factor.
        df: int, the degrees of freedom for the score statistic (equal to n_isos - 1).
    """
    n_genes, n_factors_design, n_isos = (
        fitted_full_model.n_genes,
        fitted_full_model.n_factors,
        fitted_full_model.n_isos
    )
    # assert n_factors == 0, "No factor should be included in the model."

    # in case of a single covariate, expand the design matrix
    if covar_to_test.dim() == 1:
        covar_to_test = covar_to_test.unsqueeze(-1) # (n_spots, 1)
    elif covar_to_test.dim() > 2:
        raise ValueError("The covariate design matrix must be 2D.")

    n_factors_covar = covar_to_test.shape[-1]
    n_factors = n_factors_design + n_factors_covar

    # clone the full model and reset the design matrix
    m_full = fitted_full_model.clone()
    with torch.no_grad():
        # merge the design matrix with the covariates to test -> (1, n_spots, n_factors)
        m_full.X_spot = torch.concat([m_full.X_spot, covar_to_test.unsqueeze(0)], axis = -1)

        # merge the coefficients with zeros for the covariates to test -> (n_genes, n_factors, n_isos - 1)
        m_full.beta = nn.Parameter(
            torch.concat([m_full.beta, torch.zeros(n_genes, n_factors_covar, n_isos - 1)], axis = 1),
            requires_grad=True
        )
    # # calculate gradient using autograd (when the model is fitted with marginal likelihood)
    # log_prob = m_full()
    # log_prob.backward()
    # score = m_full.beta.grad.detach() # n_factors x (n_isos - 1)
    # score = score[n_factors_design:, :] # exclude the design matrix in the fitted model

    # calculate the score aka the gradient of the log-joint-likelihood
    d_l_d_eta = m_full.counts - m_full._alpha() * m_full.counts.sum(axis=-1, keepdim = True) # (n_genes, n_spots, n_isos)
    score = covar_to_test.T.unsqueeze(0) @ d_l_d_eta.detach()[..., :-1] # (n_genes, n_factors_covar, n_isos - 1)

    # calculate the Fisher information matrix
    # beta_bias_hess.shape = (n_genes, (n_factors + 1)*(n_isos - 1), (n_factors + 1)*(n_isos - 1))
    beta_bias_hess = m_full._get_log_lik_hessian_beta_bias().detach()
    fisher_info = [] # -> (n_genes, n_factors, n_isos - 1, n_isos - 1)
    for i in range(n_factors):
        # retrieve the Hessian for beta per factor
        beta_idx_per_factor = [i + j * (n_factors + 1) for j in range(n_isos - 1)]
        # beta_hess.shape = (n_genes, n_isos - 1, n_isos - 1)
        beta_hess = beta_bias_hess[:, beta_idx_per_factor, :][:, :, beta_idx_per_factor]

        # add a small value to the diagonal to ensure invertibility
        beta_hess += 1e-5 * torch.eye(beta_hess.shape[-1]).unsqueeze(0)

        # the Fisher information matrix is the negative Hessian
        fisher_info.append( - beta_hess) # (n_genes, n_isos - 1, n_isos - 1)

    fisher_info = torch.stack(fisher_info, dim=1) # (n_genes, n_factors, n_isos - 1, n_isos - 1)
    fisher_info = fisher_info[:, n_factors_design:, :, :] # exclude the design matrix in the fitted model

    # calculate the score statistic
    score_stat = score.unsqueeze(-2) @ torch.linalg.inv(fisher_info) @ score.unsqueeze(-1) # (n_genes, n_factors_covar, 1, 1)

    return score_stat.squeeze(), n_isos - 1


class SplisosmGLMM():
    """Parametric spatial isoform statistical modeling using GLMM.

    Usages:
    - Model fitting:
        model = SplisosmGLMM()
        model.setup_data(data, coordinates)
        model.fit(with_design_matrix = False, ...)

    - Spatial variability test:
        model.test_spatial_variability(...)

    - Differential usage test:
        model.test_diffential_usage(...)

    - Retreive results:
        sv_results = model.get_formatted_test_results('sv')
        du_results = model.get_formatted_test_results('du')
    """
    def __init__(
        self,
        model_type = 'glmm-full',
        share_variance: bool = True,
        var_parameterization_sigma_theta: bool = True,
        var_fix_sigma: bool = False,
        var_prior_model: str = "none",
        var_prior_model_params: dict = {},
        init_ratio: str = "observed",
        fitting_method: str = "joint_gd",
        fitting_configs: dict = {'max_epochs': -1}
    ):
        # specify the model type to fit
        assert model_type in ['glmm-full', 'glmm-null', 'glm']
        self.model_type = model_type

        self.model_configs = {
            'share_variance': share_variance,
            'var_parameterization_sigma_theta': var_parameterization_sigma_theta,
            'var_fix_sigma': var_fix_sigma,
            'var_prior_model': var_prior_model,
            'var_prior_model_params': var_prior_model_params,
            'init_ratio': init_ratio,
            'fitting_method': fitting_method,
            'fitting_configs': fitting_configs
        }

        # to be set after running setup_data()
        self.n_genes = None # number of genes
        self.n_spots = None # number of spots
        self.n_isos_per_gene = None # list of number of isoforms for each gene
        self.n_factors = None # number of covariates to test for differential usage

        # to store the fitted models after running fit()
        self._is_trained = False
        self.fitting_results = {
            'models_glmm-full': [],
            'models_glmm-null': [],
            'models_glm': []
        }

        # to store the spatial variability test results after running test_spatial_variability()
        self.sv_test_results = {}

        # to store the differential usage test results after running test_differential_usage()
        self.du_test_results = {}

    def __str__(self):
        _sv_status = f"Completed ({self.sv_test_results['method']})" if len(self.sv_test_results) > 0 else "NA"
        _du_status = f"Completed ({self.du_test_results['method']})" if len(self.du_test_results) > 0 else "NA"
        return f"=== Parametric SPLISOSM model for spatial isoform testings\n" + \
                f"- Number of genes: {self.n_genes}\n" + \
                f"- Number of spots: {self.n_spots}\n" + \
                f"- Number of covariates: {self.n_factors}\n" + \
                f"- Average number of isoforms per gene: {np.mean(self.n_isos_per_gene) if self.n_isos_per_gene is not None else None}\n" + \
                 "=== Model configurations\n" + \
                f"- Model type: {self.model_type}\n" + \
                f"- Parameterized using sigma and theta: {self.model_configs['var_parameterization_sigma_theta']}\n" + \
                f"- Learnable variance: {not self.model_configs['var_fix_sigma']}\n" + \
                f"- Same variance across classes: {self.model_configs['share_variance']}\n" + \
                f"- Prior on total variance: {self.model_configs['var_prior_model']}\n" + \
                f"- Initialization method: {self.model_configs['init_ratio']}\n" + \
                 "=== Fitting configurations \n" + \
                f"- Trained: {self._is_trained}\n" + \
                f"- Fitting methods: {self.model_configs['fitting_method']}\n" + \
                f"- Parameters: {self.model_configs['fitting_configs']}\n" + \
                 "=== Test results\n" + \
                f"- Spatial variability test: {_sv_status}\n" + \
                f"- Differential usage test: {_du_status}"

    def setup_data(self, data, coordinates, design_mtx = None,
                   gene_names = None, group_gene_by_n_iso = False, covariate_names = None):
        """Setup the data for the model.

        Args:
            data: list of tensor(n_spots, n_isos), the observed isoform counts for each gene.
            coordinates: tensor(n_spots, 2), the spatial coordinates.
            design_mtx: tensor(n_spots, n_factors), the design matrix for the fixed effects.
            gene_names: list of str, the gene names.
            group_gene_by_n_iso: bool, whether to group genes by the number of isoforms.
        """
        # create the dataset and extract statistics
        _dataset = IsoDataset(data, gene_names, group_gene_by_n_iso)
        self.n_genes, self.n_spots, self.n_isos_per_gene = (
            _dataset.n_genes, _dataset.n_spots, _dataset.n_isos_per_gene
        )
        self.gene_names = _dataset.gene_names # list of gene names
        self.dataset = _dataset # call self.dataset.get_dataloader() to access the data
        self.group_gene_by_n_iso = group_gene_by_n_iso

        # create spatial covariance matrix from the coordinates
        assert coordinates.shape[0] == self.n_spots
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.from_numpy(coordinates)
        self.coordinates = coordinates
        self.corr_sp = get_cov_sp(coordinates, k = 4, rho=0.99) # (n_spots, n_spots)

        # check and store the design matrix
        if design_mtx is not None:
            if design_mtx.dim() == 1: # in case of a single covariate
                design_mtx = design_mtx.unsqueeze(1)

            if isinstance(design_mtx, np.ndarray): # convert to tensor if numpy array
                design_mtx = torch.from_numpy(design_mtx)

            assert design_mtx.shape[0] == self.n_spots, "Design matrix must match the number of spots."

            if covariate_names is not None:	# set default names
                assert len(covariate_names) == design_mtx.shape[1], "Covariate names must match the number of factors."
            else:
                covariate_names = [f"factor_{str(i + 1).zfill(4)}" for i in range(design_mtx.shape[1])]

            # check for constant covariates
            _ind = torch.where(design_mtx.std(0) < 1e-5)[0]
            for _i in _ind:
                warnings.warn(f"{covariate_names[_i]} has zero variance. Please remove it.")

        self.n_factors = design_mtx.shape[1] if design_mtx is not None else 0
        self.design_mtx = design_mtx # (n_spots, n_factors) or None
        self.covariate_names = covariate_names

        # store the eigendecomposition of the spatial covariance matrix
        try:
            corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.corr_sp)
        except RuntimeError:
            # fall back to eig if eigh fails
            # related to a pytorch bug on M1 macs, see https://github.com/pytorch/pytorch/issues/83818
            corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eig(self.corr_sp)
            corr_sp_eigvals = torch.real(corr_sp_eigvals)
            corr_sp_eigvecs = torch.real(corr_sp_eigvecs)

        self._corr_sp_eigvals = corr_sp_eigvals # (n_spots,)
        self._corr_sp_eigvecs = corr_sp_eigvecs # (n_spots, n_spots)

    def get_formatted_test_results(self, test_type):
        """Get the formatted test results as data frame."""
        assert test_type in ['sv', 'du'], "Invalid test type. Must be one of 'sv' or 'du'."
        if test_type == 'sv':
            # check if the spatial variability test has been run
            assert len(self.sv_test_results) > 0, "No spatial variability test results found. Please run test_spatial_variability() first."
            # format the results
            res = pd.DataFrame({
                'gene': self.gene_names,
                'statistic': self.sv_test_results['statistic'],
                'pvalue': self.sv_test_results['pvalue'],
                'pvalue_adj': self.sv_test_results['pvalue_adj'],
            })
            return res
        else:
            # check if the differential usage test has been run
            assert len(self.du_test_results) > 0, "No differential usage test results found. Please run test_differential_usage() first."
            # format the results
            res = pd.DataFrame({
                'gene': np.repeat(self.gene_names, self.n_factors),
                'covariate': np.tile(self.covariate_names, self.n_genes),
                'statistic': self.du_test_results['statistic'].reshape(-1),
                'pvalue': self.du_test_results['pvalue'].reshape(-1),
                'pvalue_adj': self.du_test_results['pvalue_adj'].reshape(-1),
            })
            return res

    def fit(self, n_jobs = 1, batch_size = 1, quiet=True, print_progress=True,
            with_design_mtx = False, from_null = False, refit_null = True, random_seed = None):
        """Fit the full model to the data.

        Args:
            n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
            batch_size: int, the maximum number of genes per job to fit in parallel. Default to 1.
            quiet: bool, whether to suppress the fitting logs. Default to True.
            print_progress: bool, whether to show the progress bar. Default to True.
            with_design_mtx: bool, whether to include the design matrix for the fixed effects. Default to False.
            from_null: bool, whether to initialize the full model from a null model
                with zero spatial variability (random effect).
            refit_null: bool, whether to refit the null model after fitting the full model.
            random_seed: int, the random seed for reproducibility. Default to None.
        """

        if batch_size > 1 and not self.group_gene_by_n_iso:
            warnings.warn("Ignoring batch size argument since the dataset is not grouped. " +
                "For batch fitting please set 'group_gene_by_n_iso = True' when setup_data()")
            batch_size = 1

        if from_null:
            # fit the null and full model sequentially
            self._fit_null_full_sv(
                n_jobs=n_jobs, batch_size=batch_size, quiet=quiet, print_progress=print_progress,
                refit_null=refit_null, with_design_mtx=with_design_mtx, random_seed=random_seed
            )
        else:
            # fit the full model only de novo
            self._fit_denovo(
                n_jobs=n_jobs, batch_size=batch_size, quiet=quiet, print_progress=print_progress,
                with_design_mtx=with_design_mtx, random_seed=random_seed
            )

        # store the fitting configurations
        self.model_configs['fitting_configs'].update({
            'with_design_mtx': with_design_mtx,
            'from_null': from_null,
            'refit_null': refit_null,
            'batch_size': batch_size
        })

        self._is_trained = True

    def save(self, path):
        """Save the fitted models to a file.

        Saving using torch.save() directly will save n_genes copies of self.corr_sp because the matrix
        is reconstructed per fitted model using self._corr_sp_eigvals and self._corr_sp_eigvecs.

        """
        for key in ['models_glmm-full', 'models_glmm-null']:
            if key in self.fitting_results and len(self.fitting_results[key]) > 0:
                # update model.corr_sp as a reference to self.corr_sp
                fitted_models = self.fitting_results[key]
                for model in fitted_models:
                    model.corr_sp = self.corr_sp

        torch.save(self, path)

    def get_fitted_models(self):
        """Get the fitted models after running fit().

        Returns:
            models: list of IsoFullModel or IsoNullNoSpVar, the fitted models.
        """
        if self.model_type == 'glmm-full':
            return self.fitting_results['models_glmm-full']
        elif self.model_type == 'glmm-null':
            return self.fitting_results['models_glmm-null']
        elif self.model_type == 'glm':
            return self.fitting_results['models_glm']
        else:
            raise ValueError(f"Invalid model type {self.model_type}.")

    def _ungroup_fitted_models(self, fitted_models, batch_size, with_design_mtx):
        """Ungroup the fitted models to match the original gene names.

        Args:
            fitted_models: list of length n_batches
            batch_size: int, the maximum number of genes per job to fit in parallel.
            with_design_mtx: bool, whether to include the design matrix for the fixed effects.

        Returns:
            fitted_models_ungrouped: list of length n_genes, the fitted models in the original order.
        """
        data = self.dataset.get_dataloader(batch_size=batch_size)

        fitted_models_ungrouped = []
        gene_names_ungroupped = []

        # loop over each batch
        for grouped_m, batch in zip(fitted_models, data):
            # unwrap the batch
            b_n_isos, b_counts, b_gene_names = (
                batch['n_isos'], batch['x'], batch['gene_name']
            )
            assert b_n_isos[0] == grouped_m.n_isos

            # add the gene names to the list
            gene_names_ungroupped.extend(b_gene_names)

            # extract batched parameters
            return_par_names = ['nu', 'beta', 'bias_eta', 'sigma', 'theta_logit', 'sigma_sp', 'sigma_nsp']
            pars = {k: v.detach() for k, v in grouped_m.state_dict().items() if k in return_par_names}

            # loop over each gene in the batch
            for _g in range(b_counts.shape[0]):
                # extract the counts and fitted parameters for the gene
                _g_counts = b_counts[_g:(_g + 1), ...] # (1, n_spots, b_n_isos)
                _g_pars = {k: v[_g:(_g + 1), ...] for k, v in pars.items()}

                # initialize and setup the model
                if self.model_type == 'glm':
                    model = MultinomGLM()
                    model.setup_data(_g_counts, design_mtx=self.design_mtx if with_design_mtx else None)
                else:
                    if self.model_type == 'glmm-full':
                        model = IsoFullModel(**self.model_configs)
                    elif self.model_type == 'glmm-null':
                        model = IsoNullNoSpVar(**self.model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self.model_type}.")

                    model.setup_data(
                        _g_counts, design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
                    )

                # update model parameters
                model.update_params_from_dict(_g_pars)
                fitted_models_ungrouped.append(model)

        # reorder the fitted models by gene names
        _order = [gene_names_ungroupped.index(_g) for _g in self.gene_names]
        fitted_models_ungrouped = [fitted_models_ungrouped[i] for i in _order]

        return fitted_models_ungrouped

    def _fit_denovo(self, n_jobs = 1, batch_size = 1, quiet=True, print_progress=True, with_design_mtx = False,
                    random_seed = None):
        """Fit the selected model to the data de novo.

        Args:
            n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
            batch_size: int, the maximum number of genes per job to fit in parallel. Default to 1.
            quiet: bool, whether to suppress the fitting logs. Default to True.
            print_progress: bool, whether to show the progress bar. Default to True.
            with_design_mtx: bool, whether to include the design matrix for the fixed effects. Default to False.
            random_seed: int, the random seed for reproducibility. Default to None.
        """
        # empty existing models before the new run
        fitted_models = []

        # decide whether to use multiprocessing
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        # start timer
        t_start = timer()

        # extract the dataloader
        n_batches = sum(1 for _ in self.dataset.get_dataloader(batch_size=batch_size))
        data = self.dataset.get_dataloader(batch_size=batch_size)

        if n_jobs == 1: # use single core
            if print_progress:
                print(f"Fitting with single core for {self.n_genes} genes (batch_size={batch_size}).")

            # iterate over genes and fit the selected model
            for batch in tqdm(data, disable=not print_progress, total=n_batches):
                _, b_counts, _ = (
                    batch['n_isos'], batch['x'], batch['gene_name']
                )

                # initialize and setup the model
                if self.model_type == 'glm':
                    model = MultinomGLM()
                    model.setup_data(b_counts, design_mtx=self.design_mtx if with_design_mtx else None)
                else:
                    if self.model_type == 'glmm-full':
                        model = IsoFullModel(**self.model_configs)
                    elif self.model_type == 'glmm-null':
                        model = IsoNullNoSpVar(**self.model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self.model_type}.")
                    model.setup_data(
                        b_counts, design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
                    )

                # fit the model
                model.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)
                fitted_models.append(model)
        else:
            if print_progress:
                print(f"Fitting with {n_jobs} cores for {self.n_genes} genes (batch_size={batch_size}).")
                print("Note: the progress bar is updated before each fitting, rather than when it finishes.")

            # Prepare tasks with delayed to ensure they're ready for parallel execution
            tasks_gen = (
                delayed(_fit_model_one_gene)(
                    self.model_configs, self.model_type,
                    batch['x'], self._corr_sp_eigvals, self._corr_sp_eigvecs,
                    self.design_mtx if with_design_mtx else None,
                    quiet, random_seed
                ) for batch in data
            )

            fitted_pars = Parallel(n_jobs=n_jobs)(tqdm(tasks_gen, total=n_batches, disable=not print_progress))

            # convert the fitted parameters to models
            for batch, pars in zip(self.dataset.get_dataloader(batch_size=batch_size), fitted_pars):
                # unwrap the batch
                b_n_isos, b_counts, b_gene_names = (
                    batch['n_isos'], batch['x'], batch['gene_name']
                )

                # initialize and setup the model
                if self.model_type == 'glm':
                    model = MultinomGLM()
                    model.setup_data(b_counts, design_mtx=self.design_mtx if with_design_mtx else None)
                else:
                    if self.model_type == 'glmm-full':
                        model = IsoFullModel(**self.model_configs)
                    elif self.model_type == 'glmm-null':
                        model = IsoNullNoSpVar(**self.model_configs)
                    else:
                        raise ValueError(f"Invalid model type {self.model_type}.")
                    model.setup_data(
                        b_counts, design_mtx=self.design_mtx if with_design_mtx else None,
                        corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
                    )

                # update model parameters
                model.update_params_from_dict(pars)
                fitted_models.append(model)

        # ungroup the fitted models to match the original gene names
        if batch_size > 1:
            fitted_models = self._ungroup_fitted_models(fitted_models, batch_size, with_design_mtx)

        # store the fitted models
        if self.model_type == 'glmm-full':
            self.fitting_results['models_glmm-full'] = fitted_models
        elif self.model_type == 'glmm-null':
            self.fitting_results['models_glmm-null'] = fitted_models
        elif self.model_type == 'glm':
            self.fitting_results['models_glm'] = fitted_models
        else:
            raise ValueError(f"Invalid model type {self.model_type}.")

        # stop timer
        t_end = timer()

        if print_progress:
            print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

    def _fit_null_full_sv(self, refit_null = True, n_jobs = 1, batch_size = 1,
                          quiet=True, print_progress=True, with_design_mtx = True, random_seed = None):
        """Fit the null (no spatial random effect) and the full model to the data sequentially.

        Args:
            refit_null: bool, whether to refit the null model after fitting the full model.
                Default to True.
            n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
            batch_size: int, the maximum number of genes per job to fit in parallel. Default to 1.
            quiet: bool, whether to suppress the fitting logs. Default to True.
            print_progress: bool, whether to show the progress bar. Default to True.
                Only applicable when n_jobs = 1.
            with_design_mtx: bool, whether to include the design matrix for the fixed effects. Default to True.
            random_seed: int, the random seed for reproducibility. Default to None.
        """
        # empty existing models before the new run
        fitted_null_models_sv = []
        fitted_full_models = []

        # decide whether to use multiprocessing
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        # start timer
        t_start = timer()

        # extract the dataloader
        n_batches = sum(1 for _ in self.dataset.get_dataloader(batch_size=batch_size))
        data = self.dataset.get_dataloader(batch_size=batch_size)

        if n_jobs == 1: # use single core
            if print_progress:
                print(f"Fitting with single core for {self.n_genes} genes (batch_size={batch_size}).")

            # iterate over genes and fit the selected model
            for batch in tqdm(data, disable=not print_progress, total=n_batches):
                _, b_counts, _ = (
                    batch['n_isos'], batch['x'], batch['gene_name']
                )

                # fit the null model
                null = IsoNullNoSpVar(**self.model_configs)
                null.setup_data(
                    b_counts, design_mtx=self.design_mtx if with_design_mtx else None,
                    corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
                )
                null.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

                # fit the full model from the null
                full = IsoFullModel.from_trained_null_no_sp_var_model(null)
                full.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

                # refit the null model if needed
                if refit_null:
                    null_refit = IsoNullNoSpVar.from_trained_full_model(full)
                    null_refit.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)

                    # update the null if larger log-likelihood
                    if null_refit().mean() > null().mean(): # null() returns shape of (n_genes,)
                        null = null_refit

                    # refit the full model from the null if likelihood decreases
                    if null().mean() > full().mean():
                        full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
                        full_refit.fit(quiet=quiet, verbose=False, diagnose=False, random_seed=random_seed)
                        if full_refit().mean() > full().mean():
                            full = full_refit

                fitted_null_models_sv.append(null)
                fitted_full_models.append(full)

        else: # use multiprocessing
            if print_progress:
                print(f"Fitting with {n_jobs} cores for {self.n_genes} genes (batch_size={batch_size}).")
                print("Note: the progress bar is updated before each fitting, rather than when it finishes.")

            # Prepare tasks with delayed to ensure they're ready for parallel execution
            tasks_gen = (
                delayed(_fit_null_full_sv_one_gene)(
                    self.model_configs, batch['x'],
                    self._corr_sp_eigvals, self._corr_sp_eigvecs,
                    self.design_mtx if with_design_mtx else None,
                    quiet, random_seed
                ) for batch in data
            )

            fitted_pars = Parallel(n_jobs=n_jobs)(tqdm(tasks_gen, total=n_batches, disable=not print_progress))

            # convert the fitted parameters to models
            for batch, (n_par, f_par) in zip(self.dataset.get_dataloader(batch_size=batch_size), fitted_pars):
                # unwrap the batch
                b_n_isos, b_counts, b_gene_names = (
                    batch['n_isos'], batch['x'], batch['gene_name']
                )

                # null models
                null = IsoNullNoSpVar(**self.model_configs)
                null.setup_data(
                    b_counts, design_mtx=self.design_mtx if with_design_mtx else None,
                    corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
                )
                # update model parameters
                null.update_params_from_dict(n_par)

                # full models
                full = IsoFullModel.from_trained_null_no_sp_var_model(null)
                full.update_params_from_dict(f_par)

                fitted_null_models_sv.append(null)
                fitted_full_models.append(full)

        # ungroup the fitted models to match the original gene names
        if batch_size > 1:
            fitted_null_models_sv = self._ungroup_fitted_models(fitted_null_models_sv, batch_size, with_design_mtx)
            fitted_full_models = self._ungroup_fitted_models(fitted_full_models, batch_size, with_design_mtx)

        # store the fitted models
        self.fitting_results['models_glmm-null'] = fitted_null_models_sv
        self.fitting_results['models_glmm-full'] = fitted_full_models

        # stop timer
        t_end = timer()

        if print_progress:
            print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

    def _fit_sv_llr_perm(self, n_perms = 20, n_jobs = 1,
                         print_progress = True, random_seed = None):
        """Calculate the null distribution of likelihood ratio using permutation.

        Args:
            n_perms: int, the number of permutations to run per gene. Default to 20.
            n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
            print_progress: bool, whether to show the progress bar. Default to True.
            random_seed: int, the random seed for reproducibility. Default to None.
        """
        # fit permutated data using the same null model
        fitting_configs = self.model_configs['fitting_configs']

        try:
            with_design_mtx = fitting_configs['with_design_mtx']
            refit_null = fitting_configs['refit_null']
            batch_size = fitting_configs['batch_size']
        except KeyError:
            raise ValueError("Null models not found. Please run fit() with from_null = True first.")

        if random_seed is not None: # set random seed for reproducibility
            torch.manual_seed(random_seed)

        # extract the likelihood ratio statistics from each permutation
        _sv_llr_perm_stats = []

        # decide whether to use multiprocessing
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        # start timer
        t_start = timer()

        if n_jobs == 1: # use single core
            if print_progress:
                print(f"Running permutation with single core for {self.n_genes} genes "
                      f"(n_perms={n_perms}, batch_size={batch_size}).")

            # run n_perms permutations for each gene
            for _ in tqdm(range(n_perms), disable=not print_progress):
                # randomly shuffle the spatial locations
                perm_idx = torch.randperm(self.n_spots)

                # fit a new SplisosmGLMM model
                new_model = SplisosmGLMM(**self.model_configs)
                new_design_mtx = self.design_mtx[perm_idx, :] if (self.design_mtx is not None and with_design_mtx) else None
                new_data = [_d[perm_idx, :] for _d in self.dataset.data]
                new_model.setup_data(
                    new_data, self.coordinates, new_design_mtx,
                    group_gene_by_n_iso=self.group_gene_by_n_iso
                )
                new_model._fit_null_full_sv(
                    refit_null=refit_null, n_jobs=1, batch_size=batch_size, quiet=True,
                    print_progress=False, with_design_mtx=with_design_mtx,
                    random_seed=random_seed
                )

                # calculate the likelihood ratio statistic
                _sv_llr_stats = []
                for full_m, null_m in zip(
                    new_model.fitting_results['models_glmm-full'], new_model.fitting_results['models_glmm-null']
                ):
                    # use marginal likelihood for stability
                    llr, _ = _calc_llr_spatial_variability(null_m, full_m)
                    _sv_llr_stats.append(llr)

                _sv_llr_stats = torch.tensor(_sv_llr_stats)
                _sv_llr_perm_stats.append(_sv_llr_stats)

            # save the llr statistics from permutated data
            self.fitting_results['sv_llr_perm_stats'] = torch.concat(_sv_llr_perm_stats, dim=0)

        else: # use multiprocessing
            if print_progress:
                print(f"Running permutation with {n_jobs} cores for {self.n_genes} genes "
                      f"(n_perms={n_perms}, batch_size={batch_size}).")
                print("Note: the progress bar is updated before each fitting, rather than when it finishes.")

            # extract the dataloader
            n_batches = sum(1 for _ in self.dataset.get_dataloader(batch_size=batch_size))
            data = self.dataset.get_dataloader(batch_size=batch_size)

            # Prepare tasks with delayed to ensure they're ready for parallel execution
            tasks_gen = (
                delayed(_fit_perm_one_gene)(
                    torch.randperm(self.n_spots),
                    self.model_configs,
                    batch['x'],
                    self._corr_sp_eigvals, self._corr_sp_eigvecs,
                    self.design_mtx if with_design_mtx else None,
                    refit_null, random_seed
                ) for batch in data
                for _ in range(n_perms)
            )

            _sv_llr_perm_stats = Parallel(n_jobs=n_jobs)(
                tqdm(tasks_gen, total=(n_batches * n_perms), disable=not print_progress)
            )

            self.fitting_results['sv_llr_perm_stats'] = torch.concat(_sv_llr_perm_stats, dim=0)

        # stop timer
        t_end = timer()

        if print_progress:
            print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

    def test_spatial_variability(self, method = "llr", use_perm_null = False, return_results = False,
                                 print_progress = True, n_perms_per_gene = None, **kwargs):
        """Parametric test for spatial variability.

        Note: the likelihood ratio test statistic is not well-calibrated for sparse data.
            We recommend using the non-parametric HSIC tests of SplisosmNP instead.

        Args:
            method: str, the test method.
                Currently only support "llr", the likelihood ratio test (H_0: sigma_sp = 0).
            use_perm_null: bool, whether to generate the null distribution from permutation.
                If False, use the chi-square with df = n_var_components as the null.
            return_results: bool, whether to return the test statistics and p-values.
            print_progress: bool, whether to show the progress bar for permutation. Default to True.
            kwargs: additional arguments passed to _fit_sv_llr_perm() if use_perm_null = True.
        """

        valid_methods = ["llr"]
        assert method in valid_methods, f"Invalid method. Must be one of {valid_methods}."

        # Parametric likelihood ratio test for spatial variability. Need to fit the null and full models.
        if len(self.fitting_results['models_glmm-null']) == 0:
            raise ValueError("Null models not found. Please run fit() with from_null = True first.")

        _sv_llr_stats, _sv_llr_dfs = [], []
        # iterate over genes and calculate the likelihood ratio statistic
        for full_m, null_m in zip(
            self.fitting_results['models_glmm-full'], self.fitting_results['models_glmm-null']
        ):
            # use marginal likelihood for stability
            llr, df = _calc_llr_spatial_variability(null_m, full_m)
            _sv_llr_stats.append(llr)
            _sv_llr_dfs.append(df)

        _sv_llr_stats = torch.tensor(_sv_llr_stats)
        _sv_llr_dfs = torch.tensor(_sv_llr_dfs)

        if use_perm_null:
            # use permutation to calculate the p-value.
            if not 'sv_llr_perm_stats' in self.fitting_results.keys():
                self._fit_sv_llr_perm(
                    n_perms=n_perms_per_gene if n_perms_per_gene is not None else 20,
                    print_progress=print_progress,
                    **kwargs
                )
            else: # use the cached results if available
                print("Using cached permutation results...")

            _sv_llr_perm = self.fitting_results['sv_llr_perm_stats']
            _sv_llr_pvals = 1 - (_sv_llr_stats[:, None] > _sv_llr_perm[None, :]).sum(1) / len(_sv_llr_perm)
        else:
            # calculate the p-value using chi-square distribution
            _sv_llr_pvals = 1 - chi2.cdf(_sv_llr_stats, df=_sv_llr_dfs)
            _sv_llr_pvals = torch.tensor(_sv_llr_pvals)

        # store the results
        self.sv_test_results = {
            'statistic': _sv_llr_stats.numpy(),
            'pvalue': _sv_llr_pvals.numpy(),
            'df': _sv_llr_dfs.numpy(),
            'method': method,
            'use_perm_null': use_perm_null,
        }

        # calculate adjusted p-values
        self.sv_test_results['pvalue_adj'] = false_discovery_control(self.sv_test_results['pvalue'])

        # return results
        if return_results:
            return self.sv_test_results

    def test_differential_usage(self, method = 'score', print_progress = True, return_results = False):
        """Parametric test for spatial isoform differential usage.

        Args:
            method: str, the test method. Must be one of "score", "wald"
                - "wald": Wald test for isoform differential usage along each factor in the design matrix.
                    Model fitting using fit(..., with_design_mtx = True) is required.
                - "score": Score test for isoform differential usage along each factor in the design matrix.
                    Model fitting using fit(..., with_design_mtx = False) is required.
            print_progress: bool, whether to show the progress bar. Default to True.
            return_results: bool, whether to return the test statistics and p-values.
        """
        if self.design_mtx is None:
            raise ValueError("No design matrix is provided. Run setup_data() first.")

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["wald", "score"]
        assert method in valid_methods, f"Invalid method. Must be one of {valid_methods}."

        if method == 'score': # Score test
            # extract the fitted full models
            fitted_models = self.get_fitted_models()
            if len(fitted_models) == 0:
                raise ValueError("Fitted full models not found. Run fit(..., with_design_mtx = False) first.")
            if self.model_configs['fitting_configs']['with_design_mtx']:
                raise ValueError(
                    "Design matrix is included in the fitted models. "
                    "Perhaps you want to use the wald test. Otherwise please run fit() with with_design_mtx = False."
                )

            _du_score_stats, _du_score_dfs = [], []
            # iterate over genes and calculate the score statistic
            for m in tqdm(fitted_models, disable=(not print_progress)):
                score_stat, score_df = _calc_score_differential_usage(m, self.design_mtx)
                _du_score_stats.append(score_stat)
                _du_score_dfs.append(score_df)

            _du_score_stats = torch.stack(_du_score_stats, dim=0).reshape(-1, n_factors) # (n_genes, n_factors)
            _du_score_dfs = torch.tensor(_du_score_dfs).unsqueeze(-1).expand(-1, n_factors) # (n_genes, n_factors)

            # calculate the p-value using chi-square distribution
            _du_score_pvals = 1 - chi2.cdf(_du_score_stats, df=_du_score_dfs)
            _du_score_pvals = torch.tensor(_du_score_pvals)

            # store the results
            self.du_test_results = {
                'statistic': _du_score_stats, # (n_genes, n_factors)
                'pvalue': _du_score_pvals, # (n_genes, n_factors)
                'method': method,
            }

        else: # method == 'wald', Wald test (anti-conservative)
            # extract the fitted full models
            fitted_models = self.get_fitted_models()
            if len(fitted_models) == 0:
                raise ValueError("Fitted full models not found. Run fit() first.")
            if not self.model_configs['fitting_configs']['with_design_mtx']:
                raise ValueError(
                    "Design matrix is not included in the fitted models. "
                    "Perhaps you want to use the score test. Otherwise please run fit() with with_design_mtx = True."
                )

            _du_wald_stats, _du_wald_dfs = [], []
            # iterate over genes and calculate the Wald statistic
            for m in tqdm(fitted_models, disable=(not print_progress)):
                wald_stat, wald_df = _calc_wald_differential_usage(m)
                _du_wald_stats.append(wald_stat)
                _du_wald_dfs.append(wald_df)

            _du_wald_stats = torch.stack(_du_wald_stats, dim=0).reshape(-1, n_factors) # (n_genes, n_factors)
            _du_wald_dfs = torch.tensor(_du_wald_dfs).unsqueeze(-1).expand(-1, n_factors) # (n_genes, n_factors)

            # calculate the p-value using chi-square distribution
            _du_wald_pvals = 1 - chi2.cdf(_du_wald_stats, df=_du_wald_dfs)
            _du_wald_pvals = torch.tensor(_du_wald_pvals)

            # store the results
            self.du_test_results = {
                'statistic': _du_wald_stats, # (n_genes, n_factors)
                'pvalue': _du_wald_pvals, # (n_genes, n_factors)
                'method': method,
            }

        # calculate adjusted p-values (independently for each factor)
        self.du_test_results['pvalue_adj'] = false_discovery_control(
            self.du_test_results['pvalue'], axis=0
        )

        # return the results if needed
        if return_results:
            return self.du_test_results
