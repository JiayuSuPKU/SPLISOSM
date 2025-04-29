import numpy as np
from scipy.stats import ncx2
import torch

try:
    from pyro.distributions import Multinomial, DirichletMultinomial, MultivariateNormal
except ImportError:
    from torch.distributions import Multinomial, MultivariateNormal
    DirichletMultinomial = None  # Placeholder if DirichletMultinomial is not available

_DELTA = 1e-10


def log_prob_mult(probs, counts):
    """Multinomial log-likelihood.

    Args:
            probs: n_isoforms (1D) or n_isoforms x n_spots (2D).
            counts: n_isoforms (1D) or n_isoforms x n_spots (2D).
    """
    assert probs.shape == counts.shape

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


def log_prob_fastmult(probs, counts, mask=None):
    """Custom multinomial log-likelihood function.

    Args:
            probs: n_isoforms x n_spots (2D).
            counts: n_isoforms x n_spots (2D).
            mask: n_spots (1D). Masked spots (True) are ignored in likelihood calculation.
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
    probs: torch.Tensor, counts: torch.tensor, mask: torch.tensor = None
) -> torch.tensor:
    """Batched custom multinomial log-likelihood function.

    Args:
        probs: [batch_size, num_isoforms, num_spots]
        counts: [batch_size, num_isoforms, num_spots]
        mask: [batch_size, num_spots], 1 for masked, 0 for unmasked
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


def log_prob_dm(concentration, counts):
    """Dirichlet-Multinomial log-likelihood.

    This is a wrapper around Pyro's DirichletMultinomial log_prob function.

    Args:
            concentration: n_isoforms (1D) or n_isoforms x n_spots (2D).
            counts: n_isoforms (1D) or n_isoforms x n_spots (2D).
    """
    assert concentration.shape == counts.shape

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


def log_prob_fastdm(concentration, counts, mask=None):
    """Custom Dirichlet-Multinomial log-likelihood function.

    Args:
            concentration: n_isoforms x n_spots (2D).
            counts: n_isoforms x n_spots (2D).
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


def log_prob_mvn(locs, covs, data):
    """Multivariate normal log-likelihood.

    If data.dim() == 2, the first dimension is assumed to be independent MVN samples (isoforms).

    Args:
            data: n_spots (1D) or n_isoforms x n_spots (2D). The observed MVN data.
            locs: n_spots (1D) or n_isoforms x n_spots (2D). Mean.
            covs: n_spots x n_spots (2D) or n_isoforms x n_spots x n_spots (3D). Covariance.

    Returns:
            log_prob: scalar, log probability.
    """
    if data.dim() == 1:  # only one sample (isoform)
        mvn = MultivariateNormal(locs, covariance_matrix=covs)
        log_prob = mvn.log_prob(data)
    else:
        n_isos = data.shape[0]
        assert len(locs) == n_isos
        assert len(covs) == n_isos

        log_prob = 0
        for mu_i, cov_i, gamma_i in zip(locs, covs, data):
            # iterate over samples (isoforms)
            mvn = MultivariateNormal(mu_i, covariance_matrix=cov_i)
            log_prob += mvn.log_prob(gamma_i)

    return log_prob


def log_prob_fastmvn(locs, cov_eigvals, cov_eigvecs, data, mask=None):
    """Multivariate normal log-likelihood with pre-decomposed covariance.

    - An order of magnitude faster than pytorch's likelihood.
    - The first dimension (isoforms) is assumed to be independent MVN samples.
    - If len(cov_eigvals) == len(cov_eigvecs) == 1, the covariance is
            assumed to be the same for all isoforms.

    Args:
            data: n_isoforms x n_spots (2D).
            locs: n_isoforms x n_spots (2D).
            cov_eigvals: (n_isoforms or 1) x n_spots (2D).
            cov_eigvecs: (n_isoforms or 1) x n_spots x n_spots (3D).
            mask: n_spots (1D)

    Returns:
            log_prob: scalar, log probability.
    """
    n_isos, n_spots = data.shape
    assert cov_eigvals.shape == (n_isos, n_spots) or cov_eigvals.shape == (1, n_spots)
    assert cov_eigvecs.shape == (n_isos, n_spots, n_spots) or cov_eigvecs.shape == (
        1,
        n_spots,
        n_spots,
    )

    if mask is not None:
        mask = (1 - mask.int()).unsqueeze(0)
        locs = locs * mask
        data = data * mask
        n_spots = mask.sum()

    # calculate (x - mu).T @ (VLV.T)^-1 @ (x - mu)
    res = ((data - locs).unsqueeze(1) @ cov_eigvecs).squeeze(1)  # n_isoforms x n_spots
    quad_gamma = (res**2 / cov_eigvals).sum()

    # calculate log_det
    log_det_cov = torch.log(cov_eigvals).sum() * (
        n_isos if cov_eigvals.shape[0] == 1 else 1
    )

    log_prob = -0.5 * (n_spots * n_isos * np.log(2 * np.pi) + log_det_cov + quad_gamma)

    # ===== below is the anwser given by GPT4 =====
    # diff = gamma - locs[:, None]  # Shape: n_isoforms x 1 x n_spots

    # # Compute the inverse of the covariance matrices using the eigendecomposition
    # # n_isoforms x n_spots x n_spots
    # inv_covs = cov_eigvecs @ (cov_eigvecs.transpose(1, 2) / cov_eigvals.unsqueeze(-1))

    # # Compute the Mahalanobis distance
    # # do matrix multiplication and add up diagonal elements
    # mahal_dist = torch.einsum("ijk,ikj->i", diff @ inv_covs, diff.transpose(-1, -2))

    # # Calculate the log determinant
    # log_det = torch.sum(torch.log(cov_eigvals), dim=-1)

    # # Calculate the log probability
    # log_prob = -0.5 * (
    #     mahal_dist
    #     + log_det
    #     + n_spots * np.log(2 * torch.pi)
    # )

    # return torch.sum(log_prob)

    return log_prob


def log_prob_fastmvn_batched(
    locs: torch.tensor,
    cov_eigvals: torch.tensor,
    cov_eigvecs: torch.tensor,
    data: torch.tensor,
    mask: torch.tensor = None,
):
    """Batched multivariate normal log-likelihood with pre-decomposed covariance.

    - An order of magnitude faster than pytorch's likelihood.
    - The first dimension (isoforms) is assumed to be independent MVN samples.
    - If len(cov_eigvals) == len(cov_eigvecs) == 1, the covariance is
                    assumed to be the same for all isoforms.

    Args:
        data: [batch_size, num_isoforms, num_spots]
        locs: [batch_size, num_isoforms, num_spots]
        cov_eigvals: [batch_size, num_isoforms, num_spots]
        cov_eigvecs: [batch_size, num_isoforms, num_spots, num_spots]
        mask: [batch_size, num_spots], 1 for masked, 0 for unmasked

    Returns:
        log_prob: [batch_size], log probability.
    """
    batch_size, num_isoforms, num_spots = data.shape
    if mask is not None:
        mask = (1 - mask.int()).unsqueeze(1)
        num_spots_non_mask = mask.sum(dim=[1, 2])
    else:
        num_spots_non_mask = num_spots
        mask = torch.ones(
            batch_size, 1, num_spots, device=data.device, dtype=data.dtype
        )

    assert cov_eigvals.shape[1:] == (
        num_isoforms,
        num_spots,
    ), f"Invalid shape of cov_eigvals: {cov_eigvals.shape}"
    assert cov_eigvecs.shape[1:] == (
        num_isoforms,
        num_spots,
        num_spots,
    ), f"Invalid shape of cov_eigvecs: {cov_eigvecs.shape}"

    # calculate (x - mu).T @ (VLV.T)^-1 @ (x - mu)
    # batch_size x n_isos x n_spots
    # res = ((data - locs).mul(mask).unsqueeze(2) @ cov_eigvecs).squeeze(2)
    res = torch.einsum("bis,bist->bit", (data - locs).mul(mask), cov_eigvecs)
    quad_gamma = (res**2 / cov_eigvals).sum(dim=[1, 2])

    # calculate log_det
    log_det_cov = cov_eigvals.log().sum(dim=[1, 2])
    log_prob = -0.5 * (
        num_spots_non_mask * num_isoforms * torch.log(2 * torch.tensor(torch.pi))
        + log_det_cov
        + quad_gamma
    )

    return log_prob


def liu_sf(t, lambs, dofs=None, deltas=None, kurtosis=False):
    """
    Liu approximation to linear combination of noncentral chi-squared variables.

    From https://github.com/limix/chiscore/blob/master/chiscore/_liu.py

    Let

            𝑋 = ∑λᵢχ²(hᵢ, 𝛿ᵢ)

    be a linear combination of noncentral chi-squared random variables, where λᵢ, hᵢ,
    and 𝛿ᵢ are the weights, degrees of freedom, and noncentrality parameters.
    [1] proposes a method that approximates 𝑋 by a single noncentral chi-squared
    random variable, χ²(l, 𝛿):

            Pr(𝑋 > 𝑡) ≈ Pr(χ²(𝑙, 𝛿) > 𝑡⁺𝜎ₓ + 𝜇ₓ),

    where 𝑡⁺ = (𝑡 - 𝜇₀) / 𝜎₀, 𝜇ₓ = 𝑙 + 𝛿, and 𝜎ₓ = √(𝑙 + 2𝛿).
    The mean and standard deviation of 𝑋 are given by 𝜇₀ and 𝜎₀.

    Then ``kurtosis=True``, the approximation is done by matching the kurtosis, rather
    than the skewness, as derived in [2].

    Parameters
    ----------
    t : array_like
            Points at which the survival function will be applied, Pr(𝑋 > 𝑡).
    lambs : array_like
            Weights λᵢ.
    dofs : array_like
            Degrees of freedom, hᵢ.
    deltas : array_like
            Noncentrality parameters, 𝛿ᵢ.
    kurtosis : bool, optional
            ``True`` for using the modified approach proposed in [2]. ``False`` for using
            the original approach proposed in [1]. Defaults to ``False``.

    Returns
    -------
    q : float, ndarray
            Approximated survival function applied to 𝑡: Pr(𝑋 > 𝑡).
    dof : float
            Degrees of freedom of χ²(𝑙, 𝛿), 𝑙.
    ncen : float
            Noncentrality parameter of χ²(𝑙, 𝛿), 𝛿.
    info : dict
            Additional information: mu_q, sigma_q, mu_x, sigma_x, and t_star.

    References
    ----------
    [1] Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation to the
            distribution of non-negative definite quadratic forms in non-central normal
            variables. Computational Statistics & Data Analysis, 53(4), 853-856.
    [2] Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare variant
            effects in sequencing association studies." Biostatistics 13.4 (2012): 762-775.
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
