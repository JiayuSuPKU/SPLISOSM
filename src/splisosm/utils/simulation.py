"""Simulation utilities for generating spatial isoform count data."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

import torch
from torch.distributions import Multinomial, Poisson

from splisosm.utils.preprocessing import get_cov_sp

__all__ = ["simulate_isoform_counts_single_gene", "simulate_isoform_counts"]


def _sample_multinom_sp(
    iso_ratio_expected: torch.Tensor, total_counts_expected: int | torch.Tensor
) -> torch.Tensor:
    """Sample isoform counts from Multinomial distribution (vectorised).

    Parameters
    ----------
    iso_ratio_expected : torch.Tensor
        Shape ``(n_spots, n_isos)`` or ``(n_genes, n_spots, n_isos)``.
    total_counts_expected : int or torch.Tensor
        Scalar int or vector of shape ``(n_spots,)``.

    Returns
    -------
    torch.Tensor
        Same shape as ``iso_ratio_expected``, integer counts.
    """
    n_spots = iso_ratio_expected.shape[-2]
    # n_isos = iso_ratio_expected.shape[-1]

    # Simulate total counts per spot from Poisson
    if isinstance(total_counts_expected, int):
        total_counts = Poisson(float(total_counts_expected)).sample(
            sample_shape=(n_spots,)
        )
    else:
        total_counts = Poisson(total_counts_expected.float()).sample()

    total_counts = total_counts.int()  # (n_spots,)

    # Vectorised Multinomial sampling (no per-spot Python loop)
    # Multinomial requires total_count to be the same for all samples in a batch,
    # so we group spots by their total count and sample each group.
    if iso_ratio_expected.ndim == 2:
        # Single gene: (n_spots, n_isos)
        counts = torch.zeros_like(iso_ratio_expected)
        unique_totals = total_counts.unique()
        for tc in unique_totals:
            tc_val = tc.item()
            if tc_val <= 0:
                continue
            mask = total_counts == tc
            probs = iso_ratio_expected[mask]  # (n_masked, n_isos)
            # Clamp to avoid negative probabilities from numerical noise
            probs = probs.clamp(min=0)
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-12)
            m = Multinomial(tc_val, probs)
            counts[mask] = m.sample()
        return counts
    else:
        # Multi-gene: (n_genes, n_spots, n_isos)
        n_genes = iso_ratio_expected.shape[0]
        counts = torch.zeros_like(iso_ratio_expected)
        unique_totals = total_counts.unique()
        for tc in unique_totals:
            tc_val = tc.item()
            if tc_val <= 0:
                continue
            mask = total_counts == tc  # (n_spots,)
            probs = iso_ratio_expected[:, mask, :]  # (n_genes, n_masked, n_isos)
            probs = probs.clamp(min=0)
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-12)
            # Sample each gene independently
            for g in range(n_genes):
                m = Multinomial(tc_val, probs[g])
                counts[g, mask] = m.sample()
        return counts


# Keep the old name for backward compatibility in tests
_sample_multinom_sp_single_gene = _sample_multinom_sp


def _sample_mvn_cholesky(chol_cov: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Sample from MVN(0, C) using pre-computed Cholesky factor L.

    Parameters
    ----------
    chol_cov : torch.Tensor
        Lower-triangular Cholesky factor of C, shape ``(n_spots, n_spots)``.
    n_samples : int
        Number of independent samples.

    Returns
    -------
    torch.Tensor
        Shape ``(n_samples, n_spots)``.
    """
    n_spots = chol_cov.shape[0]
    z = torch.randn(n_spots, n_samples)  # (n_spots, n_samples)
    samples = chol_cov @ z  # (n_spots, n_samples)
    return samples.T  # (n_samples, n_spots)


def simulate_isoform_counts_single_gene(
    grid_size: tuple[int, int] = (30, 30),
    n_isos: int = 3,
    total_counts_expected: int | torch.Tensor = 100,
    var_sp: float = 0,
    var_nsp: float = 1,
    rho: float = 0.99,
    design_mtx: Optional[torch.Tensor] = None,
    beta_true: Optional[torch.Tensor] = None,
    return_params: bool = True,
) -> Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    """Generate isoform-by-spot count matrix for a given gene.

    Each gene is simulated independently. For simplicity we constrain all genes
    to have the same number of isoforms. (In fact in the current implementation all genes
    have the exact same distribution.)

    Parameters
    ----------
    grid_size
        Shape of the spatial grid.
    n_isos
        Number of isoforms.
    total_counts_expected
        Expected total gene counts per spot.
    var_sp
        Variance explained by spatial structure.
    var_nsp
        Unstructured variance (white noise).
    rho
        Spatial autocorrelation parameter.
    design_mtx
        Shape (n_spots, n_factors). Design matrix for the isoform usage ratio.
    beta_true
        Shape (n_factors, n_isoforms - 1). Factor coefficients for the design matrix.
    return_params
        Whether to return simulation parameters.

    Returns
    -------
    dict or tuple[dict]
        If `return_params` is False, returns `datadict` with ``'counts'``, ``'coords'``, ``'cov_sp'``, ``'design_mtx'``.
        If `return_params` is True, returns (`data_dict`, `params_dict`).
    """
    n_x, n_y = grid_size
    n_spots = n_x * n_y

    ## calculate expected isoform ratio
    if design_mtx is not None or beta_true is not None:
        n_factors = design_mtx.shape[1]
        if beta_true.shape != (n_factors, n_isos - 1):
            raise ValueError(
                "`beta_true` must have shape "
                f"({n_factors}, {n_isos - 1}); got {tuple(beta_true.shape)}."
            )
        if n_spots != design_mtx.shape[0]:
            raise ValueError(
                "`design_mtx` row count must match the number of spots "
                f"({n_spots}); got {design_mtx.shape[0]}."
            )
        eta_fixed = design_mtx @ beta_true  # n_spots x (n_isos - 1)
    else:
        eta_fixed = torch.zeros((n_spots, n_isos - 1))

    ## generate spatial grid and the spatial covariance matrix
    gy, gx = np.mgrid[0 : 1 : n_x * 1j, 0 : 1 : n_y * 1j]
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    cov_sp = get_cov_sp(coords, rho=rho)
    cov_sp = var_sp * cov_sp + var_nsp * torch.eye(n_spots)

    ## sample random effects from MVN — single Cholesky, all isoforms at once
    chol_cov = torch.linalg.cholesky(cov_sp)
    nu = _sample_mvn_cholesky(chol_cov, n_isos - 1)  # (n_isos-1, n_spots)
    eta_fixed += nu.T  # (n_spots, n_isos-1)

    ## convert linear predictor to proportions
    props = torch.cat([eta_fixed, torch.zeros(n_spots, 1)], dim=1)
    props = torch.softmax(props, dim=1)  # n_spots x n_isos

    ## sample from multinomial distributions
    counts = _sample_multinom_sp(props, total_counts_expected)

    ## store simulated data and ground truth
    data = {
        "counts": counts,
        "coords": coords,
        "cov_sp": cov_sp,
        "design_mtx": design_mtx,
    }

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
    n_genes: int = 1,
    grid_size: tuple[int, int] = (30, 30),
    n_isos: int = 3,
    total_counts_expected: int | torch.Tensor = 100,
    var_sp: float = 0,
    var_nsp: float = 1,
    rho: float = 0.99,
    design_mtx: Optional[torch.Tensor] = None,
    beta_true: Optional[torch.Tensor] = None,
    return_params: bool = True,
) -> Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    """Generate isoform-by-spot count matrix for multiple genes.

    Each gene is simulated independently. For simplicity we constrain all genes
    to have the same number of isoforms. (In fact in the current implementation all genes
    have the exact same distribution.)

    Parameters
    ----------
    n_genes
        Number of genes to simulate.
    grid_size
        Shape of the spatial grid.
    n_isos
        Number of isoforms.
    total_counts_expected
        Expected total gene counts per spot.
    var_sp
        Variance explained by spatial structure.
    var_nsp
        Unstructured variance (white noise).
    rho
        Spatial autocorrelation parameter.
    design_mtx
        Shape (n_spots, n_factors). Design matrix for the isoform usage ratio.
    beta_true
        Shape (n_factors, n_isoforms - 1). Factor coefficients for the design matrix.
    return_params
        Whether to return simulation parameters.

    Returns
    -------
    dict or tuple[dict]
        If `return_params` is False, returns `datadict` with ``'counts'``, ``'coords'``, ``'cov_sp'``, ``'design_mtx'``.
        If `return_params` is True, returns (`data_dict`, `params_dict`).
    """
    n_x, n_y = grid_size
    n_spots = n_x * n_y
    n_isos = int(n_isos)

    ## calculate expected isoform ratio
    if design_mtx is not None or beta_true is not None:
        n_factors = design_mtx.shape[1]
        if beta_true.shape != (n_factors, n_isos - 1):
            raise ValueError(
                "`beta_true` must have shape "
                f"({n_factors}, {n_isos - 1}); got {tuple(beta_true.shape)}."
            )
        if n_spots != design_mtx.shape[0]:
            raise ValueError(
                "`design_mtx` row count must match the number of spots "
                f"({n_spots}); got {design_mtx.shape[0]}."
            )
        eta_fixed = design_mtx @ beta_true  # n_spots x (n_isos - 1)
    else:
        eta_fixed = torch.zeros((n_spots, n_isos - 1))

    ## expand to n_genes
    eta_fixed = (
        eta_fixed.unsqueeze(0).expand(n_genes, -1, -1).clone()
    )  # n_genes x n_spots x (n_isos - 1)

    ## generate spatial grid and the spatial covariance matrix
    gy, gx = np.mgrid[0 : 1 : n_x * 1j, 0 : 1 : n_y * 1j]
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    cov_sp = get_cov_sp(coords, rho=rho)
    cov_sp = var_sp * cov_sp + var_nsp * torch.eye(n_spots)

    ## sample random effects from MVN — single Cholesky for all genes × isoforms
    chol_cov = torch.linalg.cholesky(cov_sp)  # O(n³) once
    n_total_samples = n_genes * (n_isos - 1)
    all_nu = _sample_mvn_cholesky(chol_cov, n_total_samples)  # (n_total, n_spots)
    all_nu = all_nu.reshape(
        n_genes, n_isos - 1, n_spots
    )  # (n_genes, n_isos-1, n_spots)
    eta_fixed += all_nu.permute(0, 2, 1)  # (n_genes, n_spots, n_isos-1)

    ## convert linear predictor to proportions
    props = torch.cat([eta_fixed, torch.zeros(n_genes, n_spots, 1)], dim=-1)
    props = torch.softmax(props, dim=-1)  # n_genes x n_spots x n_isos

    ## sample from multinomial distributions
    counts = _sample_multinom_sp(props, total_counts_expected)

    ## store simulated data and ground truth
    data = {
        "counts": counts,
        "coords": coords,
        "cov_sp": cov_sp,
        "design_mtx": design_mtx,
    }

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
