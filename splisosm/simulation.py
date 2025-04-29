import itertools
import sys

import numpy as np
import pandas as pd
try:
    import scanpy as sc
except ImportError:
    pass
import torch
from torch.distributions import Multinomial, MultivariateNormal, Poisson
from tqdm.auto import tqdm

from splisosm.utils import get_cov_sp

def _sample_multinom_sp_single_gene(
    iso_ratio_expected, total_counts_expected):
    """Sample isoform counts from Multinomial distribution.

    Given the expected isoform ratio per spot and total counts, simulate the isoform counts
    from Y[s:] ~ Multinomial(N[s], iso_ratio_expected[s:]) where N[s] ~ Poisson(total_counts_expected).

    Args:
        iso_ratio_expected: n_spots x n_isos
        total_counts_expected: int or vector of length n_spots
    """
    n_spots, n_isos = iso_ratio_expected.shape

    # simulate total counts per location from Poisson
    if isinstance(total_counts_expected, int):
        total_counts = Poisson(total_counts_expected).sample(sample_shape=(n_spots,))
    else: # use the provided total counts
        assert len(total_counts_expected) == n_spots
        total_counts = Poisson(total_counts_expected).sample()

    counts = []

    for spot, total_spot in enumerate(total_counts):  # iterate over spots
        if total_spot > 0:
            # dm = DirichletMultinomial(alpha[:, i], total_count)
            # counts.append(dm.sample(sample_shape=(1,)))
            m = Multinomial(total_spot.int().item(), iso_ratio_expected[spot,:])
            counts.append(m.sample(sample_shape=(1,)))
        else:
            counts.append(torch.zeros(1, n_isos))

    counts = torch.concat(counts, dim=0)  # n_spots x n_isos

    return counts


def simulate_isoform_counts_single_gene(
    grid_size=(30, 30),
    n_isos=3,
    total_counts_expected=100,
    var_sp=0,
    var_nsp=1,
    rho=0.99,
    design_mtx = None, # [n_spots, n_factors]
    beta_true = None, # [n_factors, n_isoforms - 1]
    return_params=True,
):
    """Generate isoform x spot count matrix for a given gene. Each gene is simulated
        independently. For simplicity we constrain all genes to have the same number of
        isoforms. (In fact in the current implementation all genes have the exact same
        distribution.)

    Args:
        grid_size (tuple): Shape of the spatial grid.
        n_isos (int): Number of isoforms.
        total_counts_expected (int or vector of length n_spots): Expected total gene counts per spot.
        var_sp (float): Variance explained by spatial structure.
        var_nsp (float): Unstructured variance (white noise).
        design_mtx (torch.Tensor of [n_spots, n_factors]): Design matrix for the isoform usage ratio.
        beta_true (torch.Tensor of [n_factors, n_isoforms - 1]): Factor coefficients for the design matrix.
        return_params (bool): Whether to return simulation parameters.
    """
    n_x, n_y = grid_size
    n_spots = n_x * n_y

    ## calculate expected isoform ratio
    if design_mtx is not None or beta_true is not None:
        n_factors = design_mtx.shape[1]

        assert beta_true.shape == (n_factors, n_isos - 1)
        assert n_spots == design_mtx.shape[0]

        eta_fixed = design_mtx @ beta_true # n_spots x (n_isos - 1)
    else:
        eta_fixed = torch.zeros((n_spots, n_isos - 1))

    ## generate spatial grid and the spatial covariance matrix
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    coords = np.array(list(itertools.product(x, y)))
    cov_sp = get_cov_sp(coords, rho=rho)
    cov_sp = var_sp * cov_sp + var_nsp * torch.eye(n_spots)

    ## sample random effects from mvn
    for q in range(n_isos - 1):
        mvn = MultivariateNormal(torch.zeros(n_spots), covariance_matrix=cov_sp)
        eta_fixed[:,q] += (mvn.sample(sample_shape=(1,))).squeeze()

    ## convert linear predictor to proportions
    props = torch.concat([eta_fixed, torch.zeros(n_spots, 1)], dim=1)
    props = torch.softmax(props, dim=1) # n_spots x n_isos

    ## sample from multinomial distributions
    counts = _sample_multinom_sp_single_gene(props, total_counts_expected)

    ## store simulated data and ground truth
    data = {"counts": counts, "coords": coords, "cov_sp": cov_sp, "design_mtx": design_mtx}

    ## return simulation parameters
    if return_params:
        params = {
            "grid_size": grid_size,
            "n_spots": n_spots,
            "n_isos": n_isos,
            "total_counts_expected": total_counts_expected,
            "rho": rho,
            "var_sp": var_sp,
            "var_nsp": var_nsp,
            "beta_true": beta_true,
            "iso_ratio_expected": props,
        }
        return data, params
    else:
        return data


def simulate_isoform_counts(
    n_genes=1,
    grid_size=(30, 30),
    n_isos=3,
    total_counts_expected=100,
    var_sp=0,
    var_nsp=1,
    rho=0.99,
    design_mtx = None, # [n_spots, n_factors]
    beta_true = None, # [n_factors, n_isoforms - 1]
    return_params=True,
):
    """Generate isoform x spot count matrix for a given gene. Each gene is simulated
        independently. For simplicity we constrain all genes to have the same number of
        isoforms. (In fact in the current implementation all genes have the exact same
        distribution.)

    Args:
        grid_size (tuple): Shape of the spatial grid.
        n_isos (int): Number of isoforms.
        total_counts_expected (int or vector of length n_spots): Expected total gene counts per spot.
        var_sp (float): Variance explained by spatial structure.
        var_nsp (float): Unstructured variance (white noise).
        design_mtx (torch.Tensor of [n_spots, n_factors]): Design matrix for the isoform usage ratio.
        beta_true (torch.Tensor of [n_factors, n_isoforms - 1]): Factor coefficients for the design matrix.
        return_params (bool): Whether to return simulation parameters.
    """
    n_x, n_y = grid_size
    n_spots = n_x * n_y
    n_isos = int(n_isos)

    ## calculate expected isoform ratio
    if design_mtx is not None or beta_true is not None:
        n_factors = design_mtx.shape[1]

        assert beta_true.shape == (n_factors, n_isos - 1)
        assert n_spots == design_mtx.shape[0]

        eta_fixed = design_mtx @ beta_true # n_spots x (n_isos - 1)
    else:
        eta_fixed = torch.zeros((n_spots, n_isos - 1))

    ## expand to n_genes
    eta_fixed = eta_fixed.unsqueeze(0).repeat(n_genes, 1, 1) # n_genes x n_spots x (n_isos - 1)

    ## generate spatial grid and the spatial covariance matrix
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    coords = np.array(list(itertools.product(x, y)))
    cov_sp = get_cov_sp(coords, rho=rho)
    cov_sp = var_sp * cov_sp + var_nsp * torch.eye(n_spots)

    ## sample random effects from mvn
    for g in range(n_genes):
        for q in range(n_isos - 1):
            mvn = MultivariateNormal(torch.zeros(n_spots), covariance_matrix=cov_sp)
            eta_fixed[g][:,q] += (mvn.sample(sample_shape=(1,))).squeeze()

    ## convert linear predictor to proportions
    props = torch.concat([eta_fixed, torch.zeros(n_genes, n_spots, 1)], dim=-1)
    props = torch.softmax(props, dim=-1) # n_genes x n_spots x n_isos

    ## sample from multinomial distributions
    counts = []
    for p in props:
        counts.append(_sample_multinom_sp_single_gene(p, total_counts_expected))
    counts = torch.stack(counts, dim = 0) # n_genes x n_spots x n_isos

    ## store simulated data and ground truth
    data = {"counts": counts, "coords": coords, "cov_sp": cov_sp, "design_mtx": design_mtx}

    ## return simulation parameters
    if return_params:
        params = {
            "grid_size": grid_size,
            "n_spots": n_spots,
            "n_isos": n_isos,
            "total_counts_expected": total_counts_expected,
            "rho": rho,
            "var_sp": var_sp,
            "var_nsp": var_nsp,
            "design_mtx": design_mtx,
            "beta_true": beta_true,
            "iso_ratio_expected": props,
        }
        return data, params
    else:
        return data
