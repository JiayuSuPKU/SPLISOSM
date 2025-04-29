import unittest
import torch

from splisosm.likelihood import (
    log_prob_fastmvn,
    log_prob_fastmvn_batched,
    log_prob_fastmult,
    log_prob_fastmult_batched,
    log_prob_fastdm,
    log_prob_dm, # requires pyro
    log_prob_mvn,
    log_prob_mult,
)


class TestLikelihood(unittest.TestCase):
    def setUp(self):
        self.config = {
            "num_spots": 23,
            "num_genes": 10,
            "num_isos": 3,
            "mask_ratio": 0.3,
            "repeats": 50,
        }
        self.rng = torch.Generator().manual_seed(42)

    def generate_probs_counts_mask(self):
        num_isos, num_spots, mask_ratio = (
            self.config["num_isos"],
            self.config["num_spots"],
            self.config["mask_ratio"],
        )
        probs = torch.rand(num_isos, num_spots, generator=self.rng)
        probs = probs / probs.sum(0)
        counts = torch.zeros(probs.size()).int()
        _counts = torch.multinomial(
            probs, num_samples=1000, replacement=True, generator=self.rng
        )
        counts.scatter_add_(1, _counts, torch.ones_like(_counts).int())
        mask = (torch.rand(num_spots, generator=self.rng) > mask_ratio).int()
        return probs, counts, mask

    def generate_mvn_data(self):
        num_isos, num_spots = self.config["num_isos"], self.config["num_spots"]
        locs = torch.randn(num_isos, num_spots, generator=self.rng)
        gamma = torch.randn(num_isos, num_spots, generator=self.rng)
        covs = torch.randn(num_isos, num_spots, num_spots, generator=self.rng)
        covs = (
            covs.transpose(1, 2).double() @ covs.double()
            + torch.eye(num_spots).double() * 1e-5
        ).float()
        eig = torch.linalg.eigh(covs)
        eigenvalues, eigenvectors = eig.eigenvalues, eig.eigenvectors
        return locs, gamma, covs, eigenvalues, eigenvectors

    def generate_fastmult_data(self):
        num_genes, num_isos, num_spots = (
            self.config["num_genes"],
            self.config["num_isos"],
            self.config["num_spots"],
        )
        probs = torch.rand(num_genes, num_isos, num_spots, generator=self.rng)
        counts = torch.randint(1, 10, (num_genes, num_isos, num_spots), generator=self.rng)
        mask = torch.randint(0, 2, (num_genes, num_spots), generator=self.rng)
        return probs, counts, mask

    def generate_fastmvn_data(self):
        num_genes, num_isos, num_spots = (
            self.config["num_genes"],
            self.config["num_isos"],
            self.config["num_spots"],
        )
        locs = torch.randn(num_genes, num_isos, num_spots, generator=self.rng)
        cov_eigvals = torch.rand(num_genes, num_isos, num_spots, generator=self.rng)
        cov_eigvecs = torch.randn(num_genes, num_isos, num_spots, num_spots, generator=self.rng)
        data = torch.randn(num_genes, num_isos, num_spots, generator=self.rng)
        mask = torch.randint(0, 2, (num_genes, num_spots), generator=self.rng)
        return locs, cov_eigvals, cov_eigvecs, data, mask

    def perturb_masked_values(self, tensor, mask):
        mask = mask.bool()
        perturbation = torch.rand(tensor.size(), generator=self.rng)
        perturbed_tensor = tensor.clone()
        perturbed_tensor[:, mask] += perturbation[:, mask]
        return perturbed_tensor

    def test_mult_fast_return_same(self):
        probs, counts, _ = self.generate_probs_counts_mask()
        self.assertTrue(
            torch.allclose(
                log_prob_mult(probs, counts), log_prob_fastmult(probs, counts)
            ),
            "Multinomial log-likelihoods do not match",
        )

    def test_dm_fast_return_same(self):
        probs, counts, _ = self.generate_probs_counts_mask()
        self.assertTrue(
            torch.allclose(
                log_prob_dm(probs, counts), log_prob_fastdm(probs, counts)
            ),
            "Dirichlet-Multinomial log-likelihoods do not match",
        )

    def test_mvn_fast_return_same(self):
        locs, gamma, covs, eigenvalues, eigenvectors = self.generate_mvn_data()
        self.assertTrue(
            torch.allclose(
                log_prob_mvn(locs=locs, covs=covs, data=gamma),
                log_prob_fastmvn(
                    locs=locs, cov_eigvals=eigenvalues, cov_eigvecs=eigenvectors, data=gamma
                ),
                rtol=1e-3,
            ),
            "Multivariate normal log-likelihoods do not match",
        )

    def test_mvn_batch_return_same(self):
        locs, cov_eigvals, cov_eigvecs, data, mask = self.generate_fastmvn_data()
        ret_batched = log_prob_fastmvn_batched(
            locs, cov_eigvals, cov_eigvecs, data, mask=mask
        )
        ret = log_prob_fastmvn(
            locs[0], cov_eigvals[0], cov_eigvecs[0], data[0], mask=mask[0]
        )
        self.assertTrue(
            torch.allclose(ret, ret_batched[0]),
            "Batched and non-batched mvn log-likelihoods do not match",
        )

    def test_mult_batch_return_same(self):
        probs, counts, mask = self.generate_fastmult_data()
        ret_batched = log_prob_fastmult_batched(probs, counts, mask=mask)
        ret = log_prob_fastmult(probs[0], counts[0], mask=mask[0])
        self.assertTrue(
            torch.allclose(ret, ret_batched[0]),
            "Batched and non-batched multinomial log-likelihoods do not match",
        )

    def test_perturbation_mult_log_likelihood(self):
        probs, counts, mask = self.generate_probs_counts_mask()
        perturbed_probs = self.perturb_masked_values(probs, mask)
        log_prob_before = log_prob_fastmult(probs, counts, mask=mask)
        log_prob_after = log_prob_fastmult(perturbed_probs, counts, mask=mask)
        self.assertTrue(
            torch.allclose(log_prob_before, log_prob_after),
            "Multinomial log-likelihood changed after perturbation on masked values",
        )

    def test_masked_dm_log_likelihood(self):
        probs, counts, mask = self.generate_probs_counts_mask()
        perturbed_probs = self.perturb_masked_values(probs, mask)
        log_prob_before = log_prob_fastdm(probs, counts, mask=mask)
        log_prob_after = log_prob_fastdm(perturbed_probs, counts, mask=mask)
        self.assertTrue(
            torch.allclose(log_prob_before, log_prob_after),
            "Dirichlet-Multinomial log-likelihood changed after perturbation on masked values",
        )

    def test_masked_mvn_log_likelihood(self):
        locs, gamma, covs, eigenvalues, eigenvectors = self.generate_mvn_data()
        mask = (torch.rand(locs.size(1), generator=self.rng) > 0.3).int()
        perturbed_locs = self.perturb_masked_values(locs, mask)
        perturbed_gamma = self.perturb_masked_values(gamma, mask)
        log_prob_before = log_prob_fastmvn(
            locs, eigenvalues, eigenvectors, gamma, mask=mask
        )
        log_prob_after = log_prob_fastmvn(
            perturbed_locs, eigenvalues, eigenvectors, perturbed_gamma, mask=mask
        )
        self.assertTrue(
            torch.allclose(log_prob_before, log_prob_after),
            "Multivariate normal log-likelihood changed after perturbation on masked values",
        )


if __name__ == "__main__":
    unittest.main()
