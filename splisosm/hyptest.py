import warnings
from timeit import default_timer as timer
import re

import numpy as np
from scipy.stats import chi2, ttest_ind, combine_pvalues
import torch
import torch.multiprocessing as mp
from torch import nn
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from splisosm.utils import counts_to_ratios, false_discovery_control
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


def _fit_model_one_gene(model_configs, model_type, counts, corr_sp_eigvals, corr_sp_eigvecs, design_mtx, quiet=True):
	"""Fit the MultinomGLMM model to the data.

	This is a worker function for multiprocessing.

	Args:
		model_configs: dict, the fitting configurations for the model.
		model_type: str, the model type to fit. Can be one of 'full', 'null', 'glm'.

	Returns:
		pars: dict, the fitted parameters extracted.
	"""
	assert model_type in ['full', 'null', 'glm']

	# initialize and setup the model
	if model_type == 'glm':
		model = MultinomGLM()
		model.setup_data(counts, design_mtx=design_mtx)
		return_par_names = ['beta', 'bias_eta']
	else:
		if model_type == 'full':
			model = IsoFullModel(**model_configs)
		elif model_type == 'null':
			model = IsoNullNoSpVar(**model_configs)
		else:
			raise ValueError(f"Invalid model type {model_type}.")

		model.setup_data(
			counts, design_mtx=design_mtx,
			corr_sp_eigvals=corr_sp_eigvals, corr_sp_eigvecs=corr_sp_eigvecs
		)
		return_par_names = ['nu', 'beta', 'bias_eta', 'sigma', 'theta_logit', 'sigma_sp', 'sigma_nsp']

	# fit the model
	model.fit(quiet=quiet, verbose=False, diagnose=False)

	# extract and return the fitted parameters
	pars = {k: v.detach() for k, v in model.state_dict().items() if k in return_par_names}

	return pars

def _fit_null_full_sv_one_gene(model_configs, counts, corr_sp_eigvals, corr_sp_eigvecs,
							   design_mtx, refit_null = True, quiet=True):
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
	null.fit(quiet=quiet, verbose=False, diagnose=False)

	# fit the full model from the null
	full = IsoFullModel.from_trained_null_no_sp_var_model(null)
	full.fit(quiet=quiet, verbose=False, diagnose=False)

	# refit the null model if needed
	if refit_null:
		null_refit = IsoNullNoSpVar.from_trained_full_model(full)
		null_refit.fit(quiet=quiet, verbose=False, diagnose=False)

		# update the null if larger log-likelihood
		if null_refit() > null():
			null = null_refit

		# refit the full model from the null if likelihood decreases
		if null() > full():
			full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
			full_refit.fit(quiet=quiet, verbose=False, diagnose=False)
			if full_refit() > full():
				full = full_refit

	return_par_names = ['nu', 'beta', 'bias_eta', 'sigma', 'theta_logit', 'sigma_sp', 'sigma_nsp']
	null_pars = {k: v.detach() for k, v in null.state_dict().items() if k in return_par_names}
	full_pars = {k: v.detach() for k, v in full.state_dict().items() if k in return_par_names}

	return (null_pars, full_pars)


def _fit_perm_one_gene(perm_idx, model_configs, counts, corr_sp_eigvals, corr_sp_eigvecs,
					   design_mtx, refit_null):
	"""Calculate the likelihood ratio statistic for spatial variability using permutation.

	This is a worker function for multiprocessing. See splisosm.fit_perm_sv_llr() for more details.

	Returns:
		_sv_llr: tensor(1), the likelihood ratio statistic.
	"""
	# permute the data coordinates
	counts_perm = counts[perm_idx, :]
	design_mtx_perm = design_mtx[perm_idx, :] if design_mtx is not None else None

	# fit the null model
	null = IsoNullNoSpVar(**model_configs)
	null.setup_data(
		counts_perm, design_mtx = design_mtx_perm,
		corr_sp_eigvals = corr_sp_eigvals, corr_sp_eigvecs = corr_sp_eigvecs
	)
	null.fit(quiet=True, verbose=False, diagnose=False)

	# fit the full model from the null
	full = IsoFullModel.from_trained_null_no_sp_var_model(null)
	full.fit(quiet=True, verbose=False, diagnose=False)

	# refit the null model if needed
	if refit_null:
		null_refit = IsoNullNoSpVar.from_trained_full_model(full)
		null_refit.fit(quiet=True, verbose=False, diagnose=False)

		# update the null if larger log-likelihood
		if null_refit() > null():
			null = null_refit

		# refit the full model from the null if likelihood decreases
		if null() > full():
			full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
			full_refit.fit(quiet=True, verbose=False, diagnose=False)
			if full_refit() > full():
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
		sv_llr: tensor(1), the likelihood ratio statistic.
		df: int, the degrees of freedom for the likelihood ratio statistic (equal to the number of variance components).
	"""
	# calculate the likelihood ratio statistic
	# use marginal likelihood for stability
	sv_llr = (
		full_model._calc_log_prob_marginal().detach() - null_model._calc_log_prob_marginal().detach()
	) * 2

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
		wald_stat: tensor(n_factors), the Wald statistic for each factor.
		df: int, the degrees of freedom for the Wald statistic (equal to n_isos - 1).
	"""
	n_factors, n_isos = fitted_full_model.n_factors, fitted_full_model.n_isos
	assert n_factors > 0, "No factor is included in the model."

	# extract the Hessian for beta per factor
	# beta_bias_hess: (n_factors + 1)(n_isos - 1) x (n_factors + 1)(n_isos - 1)
	beta_bias_hess = fitted_full_model._get_log_lik_hessian_beta_bias().detach()
	fisher_info = [] # n_factors x (n_isos - 1) x (n_isos - 1)
	for i in range(n_factors):
		beta_idx_per_factor = [i + j * (n_factors + 1) for j in range(n_isos - 1)]
		beta_hess = beta_bias_hess[beta_idx_per_factor][:, beta_idx_per_factor]
		# the Fisher information matrix is the negative Hessian
		fisher_info.append( - beta_hess)
	fisher_info = torch.stack(fisher_info, dim=0) # n_factors x (n_isos - 1) x (n_isos - 1)

	# extract the beta estimates
	beta_est = fitted_full_model.beta.detach() # n_factors x n_isos

	# calculate the Wald statistic
	wald_stat = beta_est.unsqueeze(1) @ fisher_info @ beta_est.unsqueeze(2) # n_factors x 1 x 1

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
	n_factors_design, n_isos = fitted_full_model.n_factors, fitted_full_model.n_isos
	# assert n_factors == 0, "No factor should be included in the model."

	# in case of a single covariate, expand the design matrix
	if covar_to_test.dim() == 1:
		covar_to_test = covar_to_test.unsqueeze(1)
	n_factors_covar = covar_to_test.shape[1]
	n_factors = n_factors_design + n_factors_covar

	# clone the full model and reset the design matrix
	m_full = fitted_full_model.clone()
	with torch.no_grad():
		m_full.X_spot = torch.concat([m_full.X_spot, covar_to_test], axis = 1)
		m_full.beta = nn.Parameter(
			torch.concat([m_full.beta, torch.zeros(n_factors_covar, n_isos - 1)], axis = 0),
			requires_grad=True
		)
	# # calculate gradient using autograd (when the model is fitted with marginal likelihood)
	# log_prob = m_full()
	# log_prob.backward()
	# score = m_full.beta.grad.detach() # n_factors x (n_isos - 1)
	# score = score[n_factors_design:, :] # exclude the design matrix in the fitted model

	# calculate the score aka the gradient of the log-joint-likelihood
	d_l_d_eta = m_full.counts - m_full._alpha() * m_full.counts.sum(axis=1, keepdim = True) # n_spots x n_isos
	score = covar_to_test.T @ d_l_d_eta.detach()[:, :-1] # n_factors_covar x (n_isos - 1)

	# calculate the Fisher information matrix
	beta_bias_hess = m_full._get_log_lik_hessian_beta_bias().detach()
	fisher_info = [] # n_factors x (n_isos - 1) x (n_isos - 1)
	for i in range(n_factors):
		# retrieve the Hessian for beta per factor
		beta_idx_per_factor = [i + j * (n_factors + 1) for j in range(n_isos - 1)]
		beta_hess = beta_bias_hess[beta_idx_per_factor][:, beta_idx_per_factor]

		# add a small value to the diagonal to ensure invertibility
		beta_hess += 1e-5 * torch.eye(beta_hess.shape[0])

		# the Fisher information matrix is the negative Hessian
		fisher_info.append( - beta_hess)

	fisher_info = torch.stack(fisher_info, dim=0) # n_factors x (n_isos - 1) x (n_isos - 1)
	fisher_info = fisher_info[n_factors_design:, :, :] # exclude the design matrix in the fitted model

	# calculate the score statistic
	score_stat = score.unsqueeze(1) @ torch.linalg.inv(fisher_info) @ score.unsqueeze(2) # n_factors_covar x 1 x 1

	return score_stat.squeeze(), n_isos - 1


def _calc_ttest_differential_usage(data, groups, combine_pval = True, combine_method = 'tippett'):
	"""Calculate the two-sample t-test statistic for differential usage.

	The t-test is applied to each isoform independently and combined if combine_pval is True.

	Args:
		data: tensor(n_spots, n_isos), the observed counts for each gene.
		groups: tensor(n_spots), the binary group labels for each spot.
		combine_pval: bool, whether to combine p-values across isoforms.
		combine_method: str, the method to combine p-values.
  			See scipy.stats.combine_pvalues() for more details.

	Returns:
		stats: tensor(n_isos) or 1, the t-test statistic.
		pval: tensor(n_isos) or 1, the p-value.
	"""
	# check if groups contains more than two unique values
	_g = torch.unique(groups) # group labels
	if len(_g) > 2:
		raise ValueError("More than two groups detected. Only two are allowed for the two-sample t-test.")

	# run t-test per isoform
	t1 = data[groups == _g[0], :] # k x n_isos
	t2 = data[groups == _g[1], :] # (n_spots - k) x n_isos
	stats, pval = ttest_ind(t1, t2, axis=0) # each of len n_isos

	# combine p-values across isoforms
	if combine_pval:
		stats, pval = combine_pvalues(pval, method = combine_method) # each of len 1

	return stats, pval


class IsoSDE():
	"""Spatial differential expression analysis at the isoform-levl.

	Usages:
	- Set up and fit models for all genes:
		isosde = IsoSDE(...) # specify model configurations
		splisosm.setup_data(data, corr_sp, design_mtx)
		splisosm.fit()

	- Spatial variability test:
		isosde = IsoSDE(...) # specify model configurations
		splisosm.setup_data(data, corr_sp, design_mtx)
		- HSIC-based tests
			splisosm.test_spatial_variability(method = 'hsic-ratio', use_perm_null = True)
		- Likelihood ratio test (not recommended):
			splisosm.fit(from_null = True, refit_null = True)
			splisosm.test_spatial_variability(method = 'llr', use_perm_null = True)

	- Differential usage test:
		iso_sde = IsoSDE(...) # specify model configurations
		iso_sde.setup_data(data, corr_sp, design_mtx)
		-  Wald test:
			splisosm.fit()
			iso_sde.test_differential_usage(method = 'wald')
	"""
	def __init__(
		self,
		model_type = 'full',
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
		assert model_type in ['full', 'null', 'glm']
		self.model_type = model_type # 'full', 'null', 'glm'

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
		self.n_isos = None # list of number of isoforms for each gene
		self.n_factors = None # number of covariates to test for differential usage

		# to store the fitted models after running fit()
		self.fitting_results = {
			'models_full': [],
			'models_null': [],
			'models_glm': []
		}

		# to store the spatial variability test results after running test_spatial_variability()
		self.sv_test_results = {}

		# to store the differential usage test results after running test_differential_usage()
		self.du_test_results = {}

	def __str__(self):
		return f"An IsoSDE model for isoform-level spatial expression testings\n" + \
				f"- Number of genes: {self.n_genes}\n" + \
				f"- Number of spots: {self.n_spots}\n" + \
				f"- Number of covariates: {self.n_factors}\n" + \
				f"- Average number of isoforms per gene: {np.mean(self.n_isos) if self.n_isos is not None else None}\n" + \
				f"- Model configurations:\n" + \
				f"\t* Model type: {self.model_type}\n" + \
				f"\t* Fitting method: {self.model_configs['fitting_method']}\n" + \
				f"\t* Parameterized using sigma and theta: {self.model_configs['var_parameterization_sigma_theta']}\n" + \
				f"\t* Learnable variance: {not self.model_configs['var_fix_sigma']}\n" + \
				f"\t* Same variance across classes: {self.model_configs['share_variance']}\n" + \
				f"\t* Prior on total variance: {self.model_configs['var_prior_model']}\n" + \
				f"\t* Initialization method: {self.model_configs['init_ratio']}\n"

	def setup_data(self, data, corr_sp, design_mtx = None, gene_names = None):
		"""Setup the data for the model.

		Args:
			data: list of tensor(n_spots, n_isos), the observed counts for each gene.
			corr_sp: (n_spots, n_spots), the spatial covariance matrix.
			design_mtx: (n_spots, n_factors), the design matrix for the fixed effects.
			gene_names: list of str, the gene names.
		"""
		self.n_genes = len(data) # number of genes
		self.n_spots = len(data[0]) # number of spots
		self.n_isos = [data_g.shape[1] for data_g in data] # number of isoforms for each gene
		self.gene_names = gene_names
		assert len(gene_names) == self.n_genes if gene_names is not None else True
		assert min(self.n_isos) > 1, "At least two isoforms are required for each gene."

		# convert numpy.array to torch.tensor float if not already
		_data = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in data]
		self.data = [data_g.float() for data_g in _data] # [tensor(n_spots, n_isos)] * n_genes

		# check the spatial covariance matrix
		assert corr_sp.shape[0] == self.n_spots
		self.corr_sp = corr_sp

		# check the design matrix
		if design_mtx is not None:
			assert design_mtx.shape[0] == self.n_spots
			if design_mtx.dim() == 1: # in case of a single covariate
				design_mtx = design_mtx.unsqueeze(1)

		self.design_mtx = design_mtx
		self.n_factors = design_mtx.shape[1] if design_mtx is not None else 0

		# store the eigendecomposition of the spatial covariance matrix
		try:
			corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eigh(self.corr_sp)
		except RuntimeError:
			# fall back to eig if eigh fails
			# related to a pytorch bug on M1 macs, see https://github.com/pytorch/pytorch/issues/83818
			corr_sp_eigvals, corr_sp_eigvecs = torch.linalg.eig(self.corr_sp)
			corr_sp_eigvals = torch.real(corr_sp_eigvals)
			corr_sp_eigvecs = torch.real(corr_sp_eigvecs)

		self._corr_sp_eigvals = corr_sp_eigvals
		self._corr_sp_eigvecs = corr_sp_eigvecs

	def fit(self, n_jobs = 1, quiet=True, print_progress=True,
			with_design_mtx = True, from_null = False, refit_null = True):
		"""Fit the full model to the data.

		Args:
			n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
			quiet: bool, whether to suppress the fitting logs. Default to True.
			print_progress: bool, whether to show the progress bar. Default to True.
			with_design_mtx: bool, whether to include the design matrix for the fixed effects. Default to True.
			from_null: bool, whether to initialize the full model from a null model
				with zero spatial variability (random effect).
			refit_null: bool, whether to refit the null model after fitting the full model.
		"""
		if from_null:
			# fit the null and full model sequentially
			self._fit_null_full_sv(
				n_jobs=n_jobs, quiet=quiet, print_progress=print_progress,
				refit_null=refit_null, with_design_mtx=with_design_mtx
			)
		else:
			# fit the full model only de novo
			self._fit_denovo(
				n_jobs=n_jobs, quiet=quiet, print_progress=print_progress,
				with_design_mtx=with_design_mtx
			)

		# store the fitting configurations
		self.fitting_configs = {
			'with_design_mtx': with_design_mtx,
			'from_null': from_null,
			'refit_null': refit_null
		}

	def get_fitted_models(self):
		"""Get the fitted models after running fit().

		Returns:
			models: list of IsoFullModel or IsoNullNoSpVar, the fitted models.
		"""
		if self.model_type == 'full':
			return self.fitting_results['models_full']
		elif self.model_type == 'null':
			return self.fitting_results['models_null']
		elif self.model_type == 'glm':
			return self.fitting_results['models_glm']
		else:
			raise ValueError(f"Invalid model type {self.model_type}.")

	def _fit_denovo(self, n_jobs = 1, quiet=True, print_progress=True, with_design_mtx = True):
		"""Fit the selected model to the data de novo.

		Args:
			n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
			quiet: bool, whether to suppress the fitting logs. Default to True.
			print_progress: bool, whether to show the progress bar. Default to True.
			with_design_mtx: bool, whether to include the design matrix for the fixed effects. Default to True.
		"""
		# empty existing models before the new run
		fitted_models = []

		# decide whether to use multiprocessing
		n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

		# start timer
		t_start = timer()

		if n_jobs == 1: # use single core
			if print_progress:
				print(f"Running with single core for fitting of {self.n_genes} genes.")

			# iterate over genes and fit the selected model
			# for counts in tqdm(self.data, disable=not print_progress):
				# initialize and setup the model
				if self.model_type == 'glm':
					model = MultinomGLM()
					model.setup_data(counts, design_mtx=self.design_mtx if with_design_mtx else None)
				else:
					if self.model_type == 'full':
						model = IsoFullModel(**self.model_configs)
					elif self.model_type == 'null':
						model = IsoNullNoSpVar(**self.model_configs)
					else:
						raise ValueError(f"Invalid model type {self.model_type}.")
					model.setup_data(
						counts, design_mtx=self.design_mtx if with_design_mtx else None,
						corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
					)

				# fit the model
				model.fit(quiet=quiet, verbose=False, diagnose=False)
				fitted_models.append(model)
		else:
			if print_progress:
				print(f"Running with {n_jobs} cores for parallel fitting of {self.n_genes} genes.")
				print("Note: the progress bar is updated before each fitting, rather than when it finishes.")

			# Prepare tasks with delayed to ensure they're ready for parallel execution
			tasks_gen = (
				delayed(_fit_model_one_gene)(
					self.model_configs, self.model_type,
					counts, self._corr_sp_eigvals, self._corr_sp_eigvecs,
					self.design_mtx if with_design_mtx else None,
					quiet
				) for counts in self.data
			)

			fitted_pars = Parallel(n_jobs=n_jobs)(tqdm(tasks_gen, total=len(self.data)))

			# convert the fitted parameters to models
			for counts, pars in zip(self.data, fitted_pars):
				# initialize and setup the model
				if self.model_type == 'glm':
					model = MultinomGLM()
					model.setup_data(counts, design_mtx=self.design_mtx if with_design_mtx else None)
				else:
					if self.model_type == 'full':
						model = IsoFullModel(**self.model_configs)
					elif self.model_type == 'null':
						model = IsoNullNoSpVar(**self.model_configs)
					else:
						raise ValueError(f"Invalid model type {self.model_type}.")
					model.setup_data(
						counts, design_mtx=self.design_mtx if with_design_mtx else None,
						corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
					)

				# update model parameters
				model.update_params_from_dict(pars)
				fitted_models.append(model)

		# store the fitted models
		if self.model_type == 'full':
			self.fitting_results['models_full'] = fitted_models
		elif self.model_type == 'null':
			self.fitting_results['models_null'] = fitted_models
		elif self.model_type == 'glm':
			self.fitting_results['models_glm'] = fitted_models
		else:
			raise ValueError(f"Invalid model type {self.model_type}.")

		# stop timer
		t_end = timer()

		if print_progress:
			print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

	def _fit_null_full_sv(self, refit_null = True, n_jobs = 1, quiet=True, print_progress=True, with_design_mtx = True):
		"""Fit the null (no spatial random effect) and the full model to the data sequentially.

		Args:
			refit_null: bool, whether to refit the null model after fitting the full model.
				Default to True.
			n_jobs: int, the number of cores to use for parallel fitting. Default to 1.
			quiet: bool, whether to suppress the fitting logs. Default to True.
			print_progress: bool, whether to show the progress bar. Default to True.
				Only applicable when n_jobs = 1.
			with_design_mtx: bool, whether to include the design matrix for the fixed effects. Default to True.
		"""
		# empty existing models before the new run
		fitted_null_models_sv = []
		fitted_full_models = []

		# decide whether to use multiprocessing
		n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

		# start timer
		t_start = timer()

		if n_jobs == 1: # use single core
			if print_progress:
				print(f"Running with single core for fitting of {self.n_genes} genes.")

			# iterate over genes and fit the models
			for counts in tqdm(self.data, disable=not print_progress):
				# fit the null model
				null = IsoNullNoSpVar(**self.model_configs)
				null.setup_data(
					counts, design_mtx=self.design_mtx if with_design_mtx else None,
					corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
				)
				null.fit(quiet=quiet, verbose=False, diagnose=False)

				# fit the full model from the null
				full = IsoFullModel.from_trained_null_no_sp_var_model(null)
				full.fit(quiet=quiet, verbose=False, diagnose=False)

				# refit the null model if needed
				if refit_null:
					null_refit = IsoNullNoSpVar.from_trained_full_model(full)
					null_refit.fit(quiet=quiet, verbose=False, diagnose=False)

					# update the null if larger log-likelihood
					if null_refit() > null():
						null = null_refit

					# refit the full model from the null if likelihood decreases
					if null() > full():
						full_refit = IsoFullModel.from_trained_null_no_sp_var_model(null)
						full_refit.fit(quiet=quiet, verbose=False, diagnose=False)
						if full_refit() > full():
							full = full_refit

				fitted_null_models_sv.append(null)
				fitted_full_models.append(full)

		else: # use multiprocessing
			if print_progress:
				print(f"Running with {n_jobs} cores for parallel fitting of {self.n_genes} genes.")
				print("Note: the progress bar is updated before each fitting, rather than when it finishes.")

			# use multiprocessing Pool and fit the models in parallel
			tasks = [
				(self.model_configs, counts, self._corr_sp_eigvals, self._corr_sp_eigvecs,
				 self.design_mtx if with_design_mtx else None,
				 refit_null, quiet)
				for counts in self.data
			]
			with mp.Pool(processes=n_jobs) as pool:
				# Map the data to the pool
				fitted_pars = pool.starmap(
					_fit_null_full_sv_one_gene,
					tqdm(tasks, total=len(tasks), disable=not print_progress)
				)

			# convert the fitted parameters to models
			fitted_full_models = []
			for counts, (n_par, f_par) in zip(self.data, fitted_pars):
				# null models
				null = IsoNullNoSpVar(**self.model_configs)
				null.setup_data(
					counts, design_mtx=self.design_mtx if with_design_mtx else None,
					corr_sp_eigvals=self._corr_sp_eigvals, corr_sp_eigvecs=self._corr_sp_eigvecs
				)
				# update model parameters
				null.update_params_from_dict(n_par)

				# full models
				full = IsoFullModel.from_trained_null_no_sp_var_model(null)
				full.update_params_from_dict(f_par)

				fitted_null_models_sv.append(null)
				fitted_full_models.append(full)

		# store the fitted models
		self.fitting_results['models_null'] = fitted_null_models_sv
		self.fitting_results['models_full'] = fitted_full_models

		# stop timer
		t_end = timer()

		if print_progress:
			print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

	def _fit_sv_llr_perm(self, n_perms = 20, n_jobs = 1, print_progress = True, with_design_mtx = True):
		"""Calculate the null distribution of likelihood ratio using permutation."""
		# fit permutated data using the same null model
		if not hasattr(self, 'fitting_configs') or not self.fitting_configs['from_null']:
			raise ValueError("Null models not found. Please run fit() with from_null = True first.")
		refit_null = self.fitting_configs['refit_null']

		# extract the likelihood ratio statistics from each permutation
		_sv_llr_perm_stats = []

		# decide whether to use multiprocessing
		n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

		# start timer
		t_start = timer()

		if n_jobs == 1: # use single core
			if print_progress:
				print(f"Running with single core for fitting of {self.n_genes} genes "
					  f"with {n_perms} permutations.")

			for _ in tqdm(range(n_perms), disable=not print_progress):
				# randomly shuffle the spatial locations
				perm_idx = torch.randperm(self.n_spots)

				# fit the new model
				new_model = IsoSDE(**self.model_configs)
				new_design_mtx = self.design_mtx[perm_idx, :] if (self.design_mtx is not None and with_design_mtx) else None
				new_data = [data[perm_idx, :] for data in self.data]
				new_model.setup_data(
					new_data, self.corr_sp, new_design_mtx
				)
				new_model._fit_null_full_sv(refit_null=refit_null, print_progress=False)

				# iterate over genes and calculate the likelihood ratio statistic
				_llr_all_genes = []
				for full_m, null_m in zip(new_model.fitted_full_models, new_model.fitted_null_models_sv):
					# use marginal likelihood for stability
					llr, _ = _calc_llr_spatial_variability(null_m, full_m)
					_llr_all_genes.append(llr)

				_llr_all_genes = torch.tensor(_llr_all_genes)
				_sv_llr_perm_stats.append(_llr_all_genes)

			self.fitting_results['sv_llr_perm_stats'] = torch.concat(_sv_llr_perm_stats, dim=0)

		else: # use multiprocessing
			if print_progress:
				print(f"Running with {n_jobs} cores for parallel fitting of {self.n_genes} genes "
					  f"with {n_perms} permutations.")
				print("Note: the progress bar is updated before each fitting, rather than when it finishes.")

			# use multiprocessing Pool and fit the models in parallel
			tasks = [
				(torch.randperm(self.n_spots), self.model_configs, counts,
				 self._corr_sp_eigvals, self._corr_sp_eigvecs,
				 self.design_mtx if with_design_mtx else None, refit_null)
				for counts in self.data
				for _ in range(n_perms)
			]
			with mp.Pool(processes=n_jobs) as pool:
				# Map the data to the pool
				_sv_llr_perm_stats = pool.starmap(
					_fit_perm_one_gene,
					tqdm(tasks, total=len(tasks), disable=not print_progress)
				)

			self.fitting_results['sv_llr_perm_stats'] = torch.tensor(_sv_llr_perm_stats)

		# stop timer
		t_end = timer()

		if print_progress:
			print(f"Fitting finished. Time elapsed: {t_end - t_start:.2f} seconds.")

	def test_spatial_variability(self, method = "hsic-ratio", ratio_transformation = 'radial',
								 use_perm_null = False, return_results = False,
								 print_progress = True, n_perms_per_gene = None, **kwargs):
		"""Test for spatial variability.

		Args:
			method: str, the test method. Must be one of "hsic-count", "hsic-ratio", "llr".
				Non-parametric tests:
					- "hsic-count": HSIC-based dependence test between isoform counts and locations.
					- "hsic-ratio": HSIC-based dependence test between isoform ratios and locations.
				Parametric tests:
					- "llr": Likelihood ratio test for spatial variability (H_0: sigma_sp = 0).
			ratio_transformation: str, if using the isoform ratio, what compositional transformation to use.
				Can be one of 'none', 'clr', 'ilr', 'alr', 'radial'.
			use_perm_null: bool, whether to generate the null distribution from permutation.
				If False, use the following asymptotic distributions:
					- "hsic-count" and "hsic-ratio": mixtures of chi-square with df = 1.
						See Zhang, Kun, et al. "Kernel-based conditional independence test and application in causal discovery."
	  					arXiv preprint arXiv:1202.3775 (2012).
					- "llr": chi-square with df = 1 if share_variance else n_isos - 1.
			return_results: bool, whether to return the test statistics and p-values.
			print_progress: bool, whether to show the progress bar. Default to True.
			kwargs: additional arguments passed to _fit_sv_llr_perm() if method = "llr" and use_perm_null = True.
		"""

		valid_methods = ["hsic-count", "hsic-ratio", "llr"]
		assert method in valid_methods, f"Invalid method. Must be one of {valid_methods}."

		if method in ["hsic-count", "hsic-ratio"]:
			# (1) Non-parametric HSIC-based spatial variability test. No need to fit the full model.
			n_spots = self.n_spots
			n_isos_list = self.n_isos # list of number of isoforms for each gene

			# center the spatial covariance matrix
			H = torch.eye(n_spots) - 1/n_spots
			cov_sp = H @ self.corr_sp @ H

			# prepare inputs for generating the null distribution
			if use_perm_null:
				n_nulls = n_perms_per_gene if n_perms_per_gene is not None else 1000
			else: # use chi-square mixture using the Liu's method
				lambda_s = torch.linalg.eigvalsh(cov_sp) # eigenvalues of length n_spots

			# iterate over genes and calculate the HSIC statistic
			hsic_list, pvals_list = [], []
			for counts in tqdm(self.data, disable=(not print_progress)):
				if method == 'hsic-count': # use count data
					y = counts - counts.mean(0, keepdim=True) # centering per isoform
				else: # use isoform ratio
					y = counts_to_ratios(counts, transformation = ratio_transformation)
					y = y - y.mean(0, keepdim=True) # centering per isoform

				# calculate the HSIC statistic
				hsic_scaled = torch.trace(y.T @ cov_sp @ y)
				hsic_list.append(hsic_scaled / n_spots ** 2)

				if use_perm_null: # permutation-based null distribution
					# randomly shuffle the spatial locations
					perm_idx = torch.stack([torch.randperm(n_spots) for _ in range(n_nulls)])
					yy = y[perm_idx,:] # n_nulls x n_spots x n_isos
					null_m = torch.einsum('bii->b', (yy.transpose(1,2) @ cov_sp.unsqueeze(0) @ yy))

					# calculate the p-value
					pvals = (null_m > hsic_scaled ).sum() / n_nulls

				else: # asymptotic null distribution
					lambda_y = torch.linalg.eigvalsh(y.T @ y) # length of n_isos
					lambda_sy = (lambda_s.unsqueeze(0) * lambda_y.unsqueeze(1)).reshape(-1) # n_spots * n_isos
					pvals = liu_sf((hsic_scaled * n_spots).numpy(), lambda_sy.numpy())

				pvals_list.append(pvals)

			# store the results
			self.sv_test_results = {
				'statistic': torch.tensor(hsic_list).numpy(),
				'pvalue': torch.tensor(pvals_list).numpy(),
				'method': method,
				'use_perm_null': use_perm_null,
			}

		else:
			# (3) Parametric likelihood ratio test for spatial variability. Need to fit the null and full models.
			if len(self.fitting_results['models_null']) == 0:
				raise ValueError("Null models not found. Please run fit() with from_null = True first.")

			_sv_llr_stats, _sv_llr_dfs = [], []
			# iterate over genes and calculate the likelihood ratio statistic
			for full_m, null_m in zip(
				self.fitting_results['models_full'], self.fitting_results['models_null']
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
					self._fit_sv_llr_perm(with_design_mtx=self.fitting_configs['with_design_mtx'], **kwargs)
				else: # use the cached results if available
					warnings.warn("Using cached permutation results.")

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


	def test_differential_usage(self, method = "hsic", ratio_transformation = 'none', 
                            	print_progress = True, return_results = False):
		"""Test for spatial isoform differential usage.

		Args:
			method: str, the test method. Must be one of "wald", "hsic", "t-fisher", "t-tippett".
				Non-parametric tests:
					- "hsic": HSIC-based test for isoform differential usage along each factor in the design matrix.
						- For continuous factors, it is equivalent to the pearson correlation test.
						- For binary factors, it is equivalent to the two-sample t-test.
					- "t-fisher", "t-tippett": two-sample t-test for isoform differential usage along each factor in the design matrix.
						T-test is applied to the ratio of each isoform independently and combined using one of 'fisher' or 'tippett'.
				Parametric tests:
					- "wald": Wald test for isoform differential usage along each factor in the design matrix.
						Model fitting using fit(..., with_design_mtx = True) is required.
					- "score": Score test for isoform differential usage along each factor in the design matrix.
						Model fitting using fit(..., with_design_mtx = False) is required.
			ratio_transformation: str, if using the isoform ratio, what compositional transformation to use.
				Only applicable for non-parametric tests. Can be one of 'none', 'clr', 'ilr', 'alr', 'radial'.
			print_progress: bool, whether to show the progress bar. Default to True.
			return_results: bool, whether to return the test statistics and p-values.
		"""
		if self.design_mtx is None:
			raise ValueError("No design matrix is provided. Run setup_data() first.")

		n_spots, n_factors = self.design_mtx.shape

		# check the validity of the specified method and transformation
		valid_methods = ["wald", "score", "hsic", "t-fisher", "t-tippett"]
		valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
		assert method in valid_methods, f"Invalid method. Must be one of {valid_methods}."
		assert ratio_transformation in valid_transformations, f"Invalid transformation. Must be one of {valid_transformations}."

		if method == 'wald': # Wald test (anti-conservative)
			# extract the fitted full models
			fitted_models = self.get_fitted_models()
			if len(fitted_models) == 0:
				raise ValueError("Fitted full models not found. Run fit() first.")
			if not self.fitting_configs['with_design_mtx']:
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

			_du_wald_stats = torch.stack(_du_wald_stats, dim=0).reshape(-1, n_factors) # n_genes x n_factors
			_du_wald_dfs = torch.tensor(_du_wald_dfs).unsqueeze(-1).expand(-1, n_factors) # n_genes x n_factors

			# calculate the p-value using chi-square distribution
			_du_wald_pvals = 1 - chi2.cdf(_du_wald_stats, df=_du_wald_dfs)
			_du_wald_pvals = torch.tensor(_du_wald_pvals)

			# store the results
			self.du_test_results = {
				'statistic': _du_wald_stats.numpy(), # n_genes x n_factors
				'pvalue': _du_wald_pvals.numpy(), # n_genes x n_factors
				'method': method,
			}

		elif method == 'score': # Score test (conservative)
			# extract the fitted full models
			fitted_models = self.get_fitted_models()
			if len(fitted_models) == 0:
				raise ValueError("Fitted full models not found. Run fit(..., with_design_mtx = False) first.")
			if self.fitting_configs['with_design_mtx']:
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

			_du_score_stats = torch.stack(_du_score_stats, dim=0).reshape(-1, n_factors) # n_genes x n_factors
			_du_score_dfs = torch.tensor(_du_score_dfs).unsqueeze(-1).expand(-1, n_factors) # n_genes x n_factors

			# calculate the p-value using chi-square distribution
			_du_score_pvals = 1 - chi2.cdf(_du_score_stats, df=_du_score_dfs)
			_du_score_pvals = torch.tensor(_du_score_pvals)

			# store the results
			self.du_test_results = {
				'statistic': _du_score_stats.numpy(), # n_genes x n_factors
				'pvalue': _du_score_pvals.numpy(), # n_genes x n_factors
				'method': method,
			}

		elif method == 'hsic': # HSIC-based test
			hsic_list, pvals_list = [], []
			# iterate over factors
			for _ind in range(n_factors):
				# center the factor of interest
				z = self.design_mtx[:, _ind] # len of n_spots
				z = z - z.mean()
				lambda_z = z @ z # the eigenvalue of z[:, None] @ z[None, :]

				_hsic_ind, _pvals_ind = [], []
				# iterate over genes and calculate the HSIC statistic
				for counts in self.data:
					# calculate isoform usage ratio (n_spots, n_isos)
					y = counts_to_ratios(counts, transformation = ratio_transformation)
					y = y - y.mean(0, keepdim=True) # centering per isoform

					# calculate the HSIC statistic
					hsic_scaled = torch.norm(y.T @ z, p = 'fro').pow(2)
					_hsic_ind.append(hsic_scaled / n_spots ** 2)

					# asymptotic null distribution
					lambda_zy = torch.linalg.eigvalsh(y.T @ y) * lambda_z # length of n_isos
					pvals = liu_sf((hsic_scaled * n_spots).numpy(), lambda_zy.numpy())
					_pvals_ind.append(pvals)

				# stack the results
				hsic_list.append(torch.tensor(_hsic_ind))
				pvals_list.append(torch.tensor(_pvals_ind))

			# combine results
			hsic_all = torch.stack(hsic_list, dim=1)
			pvals_all = torch.stack(pvals_list, dim=1)

			# store the results
			self.du_test_results = {
				'statistic': hsic_all.numpy(), # n_genes x n_factors
				'pvalue': pvals_all.numpy(), # n_genes x n_factors
				'method': method,
			}


		else: # two-sample t-test
			# method to combine p-values across isoforms, either 'fisher' or 'tippett'
			combine_method = re.findall(r'^t-(.+)', method)[0]

			_du_ttest_stats_all, _du_ttest_pvals_all = [], []

			# iterate over factors
			for _ind in range(n_factors):
				# iterate over genes and calculate the t-test statistic
				_du_ttest_stats, _du_ttest_pvals = [], []
				for counts in self.data:
					# calculate isoform usage ratio (n_spots, n_isos)
					ratios = counts_to_ratios(counts, transformation = ratio_transformation)

					# run t-test and combine p-values
					_stats, _pvals = _calc_ttest_differential_usage(
						ratios, self.design_mtx[:, _ind],
						combine_pval=True, combine_method=combine_method
					)
					_du_ttest_stats.append(_stats)
					_du_ttest_pvals.append(_pvals)

				_du_ttest_stats = np.stack(_du_ttest_stats, axis=0)
				_du_ttest_pvals = np.stack(_du_ttest_pvals, axis=0)

				_du_ttest_stats_all.append(_du_ttest_stats)
				_du_ttest_pvals_all.append(_du_ttest_pvals)

			# combine results
			_du_ttest_stats_all = np.stack(_du_ttest_stats_all, axis=1)
			_du_ttest_pvals_all = np.stack(_du_ttest_pvals_all, axis=1)

			# store the results
			self.du_test_results = {
				'statistic': _du_ttest_stats_all, # n_genes x n_factors
				'pvalue': _du_ttest_pvals_all, # n_genes x n_factors
				'method': method,
			}

		# calculate adjusted p-values (independently for each factor)
		self.du_test_results['pvalue_adj'] = false_discovery_control(
			self.du_test_results['pvalue'], axis=0
		)

		# return the results if needed
		if return_results:
			return self.du_test_results

