"""Likelihood functions and pvalue-approximation utilities."""

import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
from scipy.stats import ncx2
import torch

__all__ = [
    "log_prob_fastmult",
    "log_prob_fastmult_batched",
    "log_prob_fastdm",
    "log_prob_fastmvn",
    "log_prob_fastmvn_batched",
    "liu_sf",
]

try:
    from pyro.distributions import Multinomial, DirichletMultinomial, MultivariateNormal
except ImportError:
    from torch.distributions import Multinomial, MultivariateNormal

    DirichletMultinomial = None  # Placeholder if DirichletMultinomial is not available

_DELTA = 1e-10


def log_prob_mult(probs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Compute multinomial log-likelihood for one or multiple spots.

    Parameters
    ----------
    probs
        Isoform ratio tensor of shape (n_isos,) or (n_isos, n_spots).
    counts
        Isoform count tensor with the same shape as `probs`.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood.

    Raises
    ------
    ValueError
        If `probs` and `counts` shapes do not match.
    """
    if probs.shape != counts.shape:
        raise ValueError("`probs` and `counts` must have the same shape.")

    if probs.dim() == 1:  # only one sample (spot)
        log_prob = Multinomial(
            total_count=counts.sum().int().item(), probs=probs
        ).log_prob(counts)
    else:
        log_prob = 0
        total_counts = counts.sum(dim=0).int()  # vector of length n_spots
        for i, total_counts_s in enumerate(total_counts):
            # iterate over samples (spots)
            m = Multinomial(total_count=total_counts_s.item(), probs=probs[:, i])
            log_prob += m.log_prob(counts[:, i])

    return log_prob


def log_prob_fastmult(
    probs: torch.Tensor,
    counts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute optimized multinomial log-likelihood.

    Parameters
    ----------
    probs
        Isoform ratio tensor of shape (n_isos, n_spots).
    counts
        Isoform count tensor of shape (n_isos, n_spots).
    mask
        Spot mask of shape (n_spots,). Entries with value ``1``
        are treated as masked and are ignored when computing likelihood.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood.
    """
    n_total = counts.sum(0)  # n_spots
    if mask is not None:
        mask = (1 - mask.int()).bool()
        probs = probs[mask.expand_as(probs)]
        counts = counts[mask.expand_as(counts)]
        n_total = n_total[mask]
    log_prob = (
        (torch.log(probs) * counts).sum()
        + torch.lgamma(n_total + 1).sum()
        - torch.lgamma(counts + 1).sum()
    )

    return log_prob


def log_prob_fastmult_batched(
    probs: torch.Tensor,
    counts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute batched optimized multinomial log-likelihood.

    Parameters
    ----------
    probs
        Batched isoform ratio tensor of shape (batch_size, n_isos, n_spots).
    counts
        Batched isoform count tensor of shape (batch_size, n_isos, n_spots).
    mask
        Mask tensor of shape (batch_size, n_spots). Entries with value ``1``
        are treated as masked and are ignored when computing likelihood.

    Returns
    -------
    torch.Tensor
        Log-likelihood values of shape (batch_size,).
    """
    batch_size, num_isoforms, num_spots = probs.shape
    if mask is not None:
        mask = (1 - mask.int()).unsqueeze(1)
    else:
        mask = torch.ones(
            batch_size, 1, num_spots, device=probs.device, dtype=probs.dtype
        )
    log_prob = (
        probs.log().mul(counts).mul(mask).sum(dim=[1, 2])
        + counts.sum(dim=1, keepdim=True).add(1).lgamma().mul(mask).sum(dim=[1, 2])
        - counts.add(1).lgamma().mul(mask).sum(dim=[1, 2])
    )
    return log_prob


def log_prob_dm(concentration: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Compute Dirichlet-multinomial log-likelihood.

    Parameters
    ----------
    concentration
        Isoform concentration tensor of shape (n_isos,) or (n_isos, n_spots).
    counts
        Isoform count tensor with the same shape as `concentration`.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood.

    Raises
    ------
    ValueError
        If `concentration` and `counts` shapes do not match.
    ImportError
        If `pyro` is not installed and Dirichlet-multinomial likelihood is
        requested.
    """
    if DirichletMultinomial is None:
        raise ImportError(
            "DirichletMultinomial requires `pyro-ppl`. Install `pyro-ppl` to use log_prob_dm()."
        )
    if concentration.shape != counts.shape:
        raise ValueError("`concentration` and `counts` must have the same shape.")

    if concentration.dim() == 1:  # only one sample (spot)
        log_prob = DirichletMultinomial(concentration, counts.sum()).log_prob(counts)
    else:
        log_prob = 0
        total_counts = counts.sum(dim=0)  # vector of length n_spots
        for i, total_counts_s in enumerate(total_counts):
            # iterate over samples (spots)
            dm = DirichletMultinomial(concentration[:, i], total_counts_s)
            log_prob += dm.log_prob(counts[:, i])

    return log_prob


def log_prob_fastdm(
    concentration: torch.Tensor,
    counts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute optimized Dirichlet-multinomial log-likelihood.

    Parameters
    ----------
    concentration
        Isoform concentration tensor of shape (n_isos, n_spots).
    counts
        Isoform count tensor of shape (n_isos, n_spots).
    mask
        Spot mask of shape (n_spots,). Entries with value ``1``
        are treated as masked and are ignored when computing likelihood.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood.
    """
    n_total_conc = concentration.sum(0)  # n_spots
    n_total_counts = counts.sum(0)  # n_spots

    if mask is not None:
        mask = (1 - mask.int()).bool()
        concentration = concentration[:, mask]
        counts = counts[:, mask]
        n_total_conc = n_total_conc[mask]
        n_total_counts = n_total_counts[mask]

    log_prob = (
        torch.lgamma(concentration + counts).sum()
        - torch.lgamma(concentration).sum()
        - torch.lgamma(counts + 1).sum()
        + torch.lgamma(n_total_conc).sum()
        + torch.lgamma(n_total_counts + 1).sum()
        - torch.lgamma(n_total_conc + n_total_counts).sum()
    )

    return log_prob


def log_prob_mvn(
    locs: torch.Tensor,
    covs: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """Compute multivariate normal log-likelihood.

    When multiple classes (e.g., isoforms) are present, they are treated as independent,
    and the log-likelihood is computed as the sum of log-likelihoods for each class.

    Parameters
    ----------
    locs
        Mean tensor of shape (n_spots,) or (n_classes, n_spots).
    covs
        Covariance tensor of shape (n_spots, n_spots) or
        (n_classes, n_spots, n_spots).
    data
        Observations of shape (n_spots,) or (n_classes, n_spots).

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood.

    Raises
    ------
    ValueError
        If batched dimensions are inconsistent.
    """
    if data.dim() == 1:  # only one sample (isoform)
        mvn = MultivariateNormal(locs, covariance_matrix=covs)
        log_prob = mvn.log_prob(data)
    else:
        n_isos = data.shape[0]
        if len(locs) != n_isos or len(covs) != n_isos:
            raise ValueError(
                "For batched input, `locs` and `covs` must match the number of isoforms in `data`."
            )

        log_prob = 0
        for mu_i, cov_i, gamma_i in zip(locs, covs, data):
            # iterate over samples (isoforms)
            mvn = MultivariateNormal(mu_i, covariance_matrix=cov_i)
            log_prob += mvn.log_prob(gamma_i)

    return log_prob


def log_prob_fastmvn(
    locs: torch.Tensor,
    cov_eigvals: torch.Tensor,
    cov_eigvecs: torch.Tensor,
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute multivariate normal log-likelihood with eigendecomposition.

    Classes along the first axis are treated as independent,
    and the log-likelihood is computed as the sum of log-likelihoods for each class.

    Parameters
    ----------
    locs
        Mean tensor of shape (n_classes, n_spots).
    cov_eigvals
        Covariance eigenvalues of shape (n_classes, n_spots) or (1, n_spots).
    cov_eigvecs
        Covariance eigenvectors of shape (n_classes, n_spots, n_spots) or (1, n_spots, n_spots).
    data
        Observation tensor of shape (n_classes, n_spots).
    mask
        Spot mask of shape (n_spots,). Entries with value ``1``
        are treated as masked and are ignored when computing likelihood.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood.

    Raises
    ------
    ValueError
        If covariance eigendecomposition shapes are incompatible with `data`.
    """
    n_classes, n_spots = data.shape
    if cov_eigvals.shape not in {(n_classes, n_spots), (1, n_spots)}:
        raise ValueError(
            "`cov_eigvals` must have shape (n_classes, n_spots) or (1, n_spots)."
        )
    if cov_eigvecs.shape not in {(n_classes, n_spots, n_spots), (1, n_spots, n_spots)}:
        raise ValueError(
            "`cov_eigvecs` must have shape (n_classes, n_spots, n_spots) or (1, n_spots, n_spots)."
        )

    if mask is not None:
        mask = (1 - mask.int()).unsqueeze(0)
        locs = locs * mask
        data = data * mask
        n_spots = mask.sum()

    # calculate (x - mu).T @ (VLV.T)^-1 @ (x - mu)
    res = ((data - locs).unsqueeze(1) @ cov_eigvecs).squeeze(1)  # n_classes x n_spots
    quad_gamma = (res**2 / cov_eigvals).sum()

    # calculate log_det
    log_det_cov = torch.log(cov_eigvals).sum() * (
        n_classes if cov_eigvals.shape[0] == 1 else 1
    )

    log_prob = -0.5 * (
        n_spots * n_classes * np.log(2 * np.pi) + log_det_cov + quad_gamma
    )

    return log_prob


def log_prob_fastmvn_batched(
    locs: torch.Tensor,
    cov_eigvals: torch.Tensor,
    cov_eigvecs: torch.Tensor,
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute batched multivariate normal log-likelihood with eigendecomposition.

    Classes along the second axis are treated as independent,
    and the log-likelihood is computed as the sum of log-likelihoods for each class.

    Parameters
    ----------
    locs
        Batched mean tensor of shape (batch_size, n_classes, n_spots).
    cov_eigvals
        Batched eigenvalue tensor of shape (batch_size, n_classes, n_spots).
    cov_eigvecs
        Batched eigenvector tensor of shape (batch_size, n_classes, n_spots, n_spots).
    data
        Batched observation of shape (batch_size, n_classes, n_spots).
    mask : torch.Tensor, optional
        Batched mask tensor of shape (batch_size, n_spots). Entries with value ``1``
        are treated as masked and are ignored when computing likelihood.

    Returns
    -------
    torch.Tensor
        Log-likelihood values of shape (batch_size,).

    Raises
    ------
    ValueError
        If covariance decomposition shapes are incompatible with input sizes.
    """
    batch_size, num_classes, num_spots = data.shape
    if mask is not None:
        mask = (1 - mask.int()).unsqueeze(1)
        num_spots_non_mask = mask.sum(dim=[1, 2])
    else:
        num_spots_non_mask = num_spots
        mask = torch.ones(
            batch_size, 1, num_spots, device=data.device, dtype=data.dtype
        )

    if cov_eigvals.shape[1:] != (num_classes, num_spots):
        raise ValueError(f"Invalid shape of cov_eigvals: {cov_eigvals.shape}")
    if cov_eigvecs.shape[1:] != (num_classes, num_spots, num_spots):
        raise ValueError(f"Invalid shape of cov_eigvecs: {cov_eigvecs.shape}")

    # calculate (x - mu).T @ (VLV.T)^-1 @ (x - mu)
    # batch_size x n_classes x n_spots
    # res = ((data - locs).mul(mask).unsqueeze(2) @ cov_eigvecs).squeeze(2)
    res = torch.einsum("bis,bist->bit", (data - locs).mul(mask), cov_eigvecs)
    quad_gamma = (res**2 / cov_eigvals).sum(dim=[1, 2])

    # calculate log_det
    log_det_cov = cov_eigvals.log().sum(dim=[1, 2])
    log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=data.device, dtype=data.dtype))
    log_prob = -0.5 * (
        num_spots_non_mask * num_classes * log_2pi + log_det_cov + quad_gamma
    )

    return log_prob


def liu_sf(
    t: ArrayLike,
    lambs: ArrayLike,
    dofs: Optional[ArrayLike] = None,
    deltas: Optional[ArrayLike] = None,
    kurtosis: bool = False,
) -> np.ndarray:
    """Compute p-values for weighted sums of chi-squared variables using the Liu approximation.

    Let :math:`X = \\sum_{i=1}^{d} \\lambda_i * \\chi^2(h_i, \\delta_i)` be a linear combination of
    `d` chi-squared random variables, each with degree of freedom :math:`h_i` and noncentrality
    :math:`\\delta_i`, this function approximates :math:``Pr(X > t)`` using the
    Liu moment-matching approach :cite:`liu2009new`.

    Parameters
    ----------
    t
        Shape (n,), observed test statistic.
    lambs
        Shape (d,), weights of each chi-squared component.
    dofs
        Shape (d,), degrees of freedom for each component. Defaults to all ones.
    deltas
        Shape (d,), noncentrality parameters for each component. Defaults to all zeros.
    kurtosis
        If True, uses kurtosis matching proposed in :cite:`lee2012optimal`; otherwise uses the original
        skewness matching :cite:`liu2009new`.

    Returns
    -------
    numpy.ndarray
        P-values of shape (n,) computed as ``Pr(X > t)``.

    Notes
    -----
    From https://github.com/limix/chiscore/blob/master/chiscore/_liu.py
    """
    if dofs is None:
        dofs = np.ones_like(lambs)
    if deltas is None:
        deltas = np.zeros_like(lambs)

    t = np.asarray(t, float)
    lambs = np.asarray(lambs, float)
    dofs = np.asarray(dofs, float)
    deltas = np.asarray(deltas, float)

    lambs = {i: lambs**i for i in range(1, 5)}

    c = {
        i: np.sum(lambs[i] * dofs) + i * np.sum(lambs[i] * deltas) for i in range(1, 5)
    }

    s1 = c[3] / (np.sqrt(c[2]) ** 3 + _DELTA)
    s2 = c[4] / (c[2] ** 2 + _DELTA)

    s12 = s1**2
    if s12 > s2:
        a = 1 / (s1 - np.sqrt(s12 - s2))
        delta_x = s1 * a**3 - a**2
        dof_x = a**2 - 2 * delta_x
    else:
        delta_x = 0
        if kurtosis:
            a = 1 / np.sqrt(s2)
            dof_x = 1 / s2
        else:
            a = 1 / (s1 + _DELTA)
            dof_x = 1 / (s12 + _DELTA)

    mu_q = c[1]
    sigma_q = np.sqrt(2 * c[2])

    mu_x = dof_x + delta_x
    sigma_x = np.sqrt(2 * (dof_x + 2 * delta_x))

    t_star = (t - mu_q) / (sigma_q + _DELTA)
    tfinal = t_star * sigma_x + mu_x

    q = ncx2.sf(tfinal, dof_x, np.maximum(delta_x, 1e-9))

    return q
