import scipy
import numpy as np
import torch
from typing import Literal
from abc import ABC, abstractmethod
from smoother.weights import coordinate_to_weights_knn_sparse, sparse_weights_to_inv_cov

__all__ = ["SpatialCovKernel"]


class Kernel(ABC):
    """Abstract interface for kernel matrix representations.

    Implementations may store either dense kernels or low-rank factors, but
    must expose common operations required by SPLISOSM hypothesis tests.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize kernel-specific state."""

    @abstractmethod
    def realization(self) -> torch.Tensor:
        """Return the realized dense kernel matrix.

        Returns
        -------
        torch.Tensor
            Dense kernel matrix of shape ``(n_spots, n_spots)``.
        """

    @abstractmethod
    def xtKx(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the quadratic form ``x^T K x``.

        Parameters
        ----------
        x : torch.Tensor
            Input matrix of shape ``(n_spots, d)``.

        Returns
        -------
        torch.Tensor
            Quadratic form of shape ``(d, d)``.
        """

    @abstractmethod
    def eigenvalues(self, k: int | None = None) -> torch.Tensor:
        """Return leading eigenvalues of the kernel matrix.

        Parameters
        ----------
        k : int, optional
            Number of leading eigenvalues to return. If ``None``, returns all
            available eigenvalues.

        Returns
        -------
        torch.Tensor
            Eigenvalues sorted in descending order.
        """


class SpatialCovKernel(Kernel):
    """Graph-based spatial covariance kernel built from spot coordinates."""

    inv_cov: torch.Tensor
    """Sparse precision matrix (aka inverse spatial kernel) of shape (n_spots, n_spots)."""

    K_sp: torch.Tensor | None
    """
    Dense kernel matrix of shape (n_spots, n_spots) when full-rank storage is used.
    ``None`` when low-rank storage is used.
    """

    Q: torch.Tensor | None
    """
    Low-rank factor of shape (n_spots, rank) such that ``K_sp ~= Q @ Q.T``.
    ``None`` when full-rank storage is used.
    """

    def __init__(
        self,
        coords: np.ndarray | torch.Tensor,
        k_neighbors: int = 4,
        model: Literal["icar", "car", "isar", "sar"] = "icar",
        rho: float = 0.99,
        standardize_cov: bool = True,
        centering: bool = False,
        approx_rank: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        coords
            Spot coordinates of shape (n_spots, 2).
        k_neighbors
            Number of nearest neighbors used to build the graph.
        model
            Spatial process model for inverse covariance construction. Supported
            values are ``'icar'``, ``'car'``, ``'isar'``, and ``'sar'``.
        rho
            Spatial autocorrelation coefficient.
        standardize_cov
            If True, scales covariance to unit marginal variance.
        centering
            If True, applies centering so row/column sums are approximately
            zero.
        approx_rank
            If provided, computes and stores a rank-`approx_rank` factorization.
        """
        # store the configurations
        self._configs = {
            "k_neighbors": k_neighbors,
            "model": model,
            "rho": rho,
            "standardize_cov": standardize_cov,
            "centering": centering,
            "approx_rank": approx_rank,
        }

        # calculate the sparse inverse spatial covariance matrix from KNN graph
        swm = coordinate_to_weights_knn_sparse(
            coords, k=k_neighbors, symmetric=True, row_scale=False
        )
        inv_cov = sparse_weights_to_inv_cov(
            swm, model=model, rho=rho, standardize=False, return_sparse=True
        ).coalesce()  # (n_spots, n_spots), sparse
        self.inv_cov = inv_cov

        if approx_rank is None:
            # compute the (N, N) dense spatial covariance matrix
            # torch.cholesky() is faster than scipy.sparse.linalg.inv()
            cov_sp = torch.cholesky_inverse(torch.linalg.cholesky(inv_cov.to_dense()))

            if (
                standardize_cov
            ):  # standardize the variance of the covariance matrix to one
                inv_sds = torch.diagflat(torch.diagonal(cov_sp) ** (-0.5))
                cov_sp = inv_sds @ cov_sp @ inv_sds

            if centering:  # center the kernel matrix
                cov_sp = cov_sp - cov_sp.mean(dim=0, keepdim=True)
                cov_sp = cov_sp - cov_sp.mean(dim=1, keepdim=True)

            # store the dense spatial kernel matrix
            self.K_sp = cov_sp  # (n_spots, n_spots)
            self._rank = cov_sp.shape[0]  # full rank

            # compute eigenvalues and eigenvectors in later stages
            self.K_eigvals = None  # eigenvalues of the kernel matrix
            self.K_eigvecs = None  # eigenvectors of the kernel matrix
            self.Q = None  # low-rank approximation of the kernel matrix K_sp = Q @ Q^T
        else:
            # compute and store the low-rank approximation of the kernel matrix
            if approx_rank > inv_cov.shape[0]:
                raise ValueError("`approx_rank` must not exceed the number of spots.")

            # normalize the inverse covariance matrix by degree such that the diagonal entries are 1
            # inv_cov = D^(-1/2) @ inv_cov @ D^(-1/2)
            # this will update self.inv_cov
            degrees = swm.sum(dim=1).to_dense()  # (n_spots,)
            inv_cov.values()[:] = inv_cov.values() / degrees[inv_cov.indices()[0]].pow(
                0.5
            )  # (n_spots, n_spots)
            inv_cov.values()[:] = inv_cov.values() / degrees[inv_cov.indices()[1]].pow(
                0.5
            )  # (n_spots, n_spots)

            # first convert torch sparse tensor to scipy coo matrix
            indices = inv_cov.indices().numpy()
            values = inv_cov.values().numpy()
            shape = inv_cov.shape
            coo_mtx = scipy.sparse.coo_matrix(
                (values, (indices[0], indices[1])), shape=shape
            )

            # then compute the smallest k eigenvalues of the inverse covariance matrix
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(
                coo_mtx, k=approx_rank, which="SM"
            )

            # which is equivalent to the largest k eigenvalues of the covariance matrix
            eigvals = torch.from_numpy(eigvals + 1e-6).pow(
                -1
            )  # sorted in descending order
            eigvecs = torch.from_numpy(eigvecs)

            # store the eigenvalues and eigenvectors
            self.cov_eigvals = eigvals
            self.cov_eigvecs = eigvecs

            # the low-rank approximation of the kernel matrix is K_sp = Q @ Q^T
            Q = eigvecs @ torch.diag(eigvals.pow(0.5))  # (n_spots, rank)

            # scale K_sp to have unit variance (diagonal entries)
            if standardize_cov:
                # after scaling, cov_sp and Q @ Q.T would have different eigenvalues
                # due to the low-rank approximation
                _sd = (Q.pow(2).sum(dim=1) + 1e-6).pow(0.5)  # (n_spots,)
                Q = Q / _sd[:, None]

            # center K_sp to have zero row and column sums
            if centering:
                Q = Q - Q.mean(dim=0, keepdim=True)

            # compute kernel eigenvalues from Q
            K_eigvals = torch.linalg.eigvalsh(Q.T @ Q)  # (rank,), ascending order

            # sort the eigenvalues in descending order
            idx = K_eigvals.argsort(descending=True)
            self.K_eigvals = K_eigvals[idx]

            # store the low-rank approximation of the kernel matrix
            self.Q = Q
            self.K_sp = None  # K_sp = Q @ Q^T
            self._rank = self.Q.shape[1]

    def shape(self) -> tuple[int, int]:
        """Return kernel shape.

        Returns
        -------
        tuple of int
            Pair (n_spots, n_spots).
        """
        return self.inv_cov.shape

    def rank(self) -> int:
        """Return effective rank of the kernel representation.

        Returns
        -------
        int
            Kernel rank.
        """
        return self._rank

    def realization(self) -> torch.Tensor:
        """Return the realized dense kernel matrix.

        Returns
        -------
        torch.Tensor
            Dense matrix of shape (n_spots, n_spots).
        """
        if self.K_sp is not None:
            return self.K_sp
        else:
            return self.Q @ self.Q.t()

    def xtKx(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel quadratic form for multivariate inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input matrix of shape (n_spots, d).

        Returns
        -------
        torch.Tensor
            Quadratic form of shape (d, d).
        """
        if self.K_sp is not None:  # use the full rank kernel matrix
            return x.t() @ self.K_sp @ x
        else:  # use the low-rank approximation
            xtQ = x.t() @ self.Q  # (d, rank)
            return xtQ @ xtQ.t()  # (d, d)

    def eigenvalues(self, k: int | None = None) -> torch.Tensor:
        """Return leading eigenvalues of the kernel matrix.

        Parameters
        ----------
        k
            Number of leading eigenvalues to return. If None, all
            available eigenvalues are returned.

        Returns
        -------
        torch.Tensor
            Eigenvalues sorted in descending order.
        """
        if self.K_eigvals is None:
            # compute and store the eigendecomposition of self.K_sp
            self.eigendecomposition()

        # return the top k largest eigenvalues
        if k is None:
            return self.K_eigvals
        else:
            return self.K_eigvals[:k]

    def eigendecomposition(self) -> None:
        """Compute and store eigendecomposition of the dense kernel.

        Raises
        ------
        ValueError
            If only low-rank kernel factors are stored.
        """
        if self.K_sp is None:
            raise ValueError(
                "Dense kernel storage is required for eigendecomposition()."
            )

        eigvals, eigvecs = np.linalg.eigh(self.K_sp.numpy())

        # sort the eigenvalues and eigenvectors in descending order
        idx = eigvals.argsort()[::-1]
        self.K_eigvals = torch.from_numpy(eigvals[idx])
        self.K_eigvecs = torch.from_numpy(eigvecs[:, idx])
        # Q = eigvecs @ np.diag(eigvals ** 0.5) # (n_spots, n_spots)
        # self.Q = torch.from_numpy(Q)[:, idx]
