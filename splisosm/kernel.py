import scipy
import numpy as np
import torch
from abc import ABC, abstractmethod
from smoother.weights import coordinate_to_weights_knn_sparse, sparse_weights_to_inv_cov

class Kernel(ABC):
    """Abstract class for efficient computation and storage of the (N, N) kernel matrix."""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def realization(self):
        """Return the realized (N, N) kernel matrix."""
        pass

    @abstractmethod
    def xtKx(self, x):
        """Return the quadratic form x^T K x."""
        pass

    @abstractmethod
    def eigenvalues(self, k = None):
        """Return the top k eigenvalues of the kernel matrix."""
        pass

class SpatialCovKernel(Kernel):
    """Spatial kernel class that calculates the graph-based spatial covariance from coordinates.

    Uasge:
    ```
    K = SpatialCovKernel(coords, k_neighbors=4, model='icar', rho=0.99, approx_rank=1000, centering=True)
    K.realization() # return the realized dense (N, N) kernel matrix
    K.eigenvalues() # return the eigenvalues of the kernel matrix
    K.xtKx(x) # return the quadratic form x^T @ K @ x for HSIC computation
    """
    def __init__(self, coords, k_neighbors=4, model = 'icar', rho=0.99,
                 standardize_cov=True, centering=False, approx_rank=None):
        """Initialize the spatial covariance kernel.
        Args:
            coords: (n_spots, 2). Spatial coordinates of spots.
            k_neighbors: int. Number of neighbors per spot for KNN graph.
            model: str. Spatial process model for the covariance matrix. Options: 'icar', 'car', 'isar', 'sar'.
            rho: float. Spatial autocorrelation coefficient.
            standardize_cov: bool. Whether to scale the spatial covariance matrix so that the variance is 1.
                Will be ignored if approx_rank is not None.
            centering: bool. Whether to center the kernel matrix to have zero row and column sums.
            approx_rank: int. Approximate rank of the kernel matrix by keeping the top 'approx_rank'
                eigenvalues/vectors. If None, store the full rank.
        """
        # store the configurations
        self._configs = {
            'k_neighbors': k_neighbors,
            'model': model,
            'rho': rho,
            'standardize_cov': standardize_cov,
            'centering': centering,
            'approx_rank': approx_rank
        }

        # calculate the sparse inverse spatial covariance matrix from KNN graph
        swm = coordinate_to_weights_knn_sparse(coords, k=k_neighbors, symmetric = True, row_scale = False)
        inv_cov = sparse_weights_to_inv_cov(
            swm, model=model, rho=rho, standardize=False, return_sparse = True
        ).coalesce() # (n_spots, n_spots), sparse
        self.inv_cov = inv_cov

        if approx_rank is None:
            # compute the (N, N) dense spatial covariance matrix
            # torch.cholesky() is faster than scipy.sparse.linalg.inv()
            cov_sp = torch.cholesky_inverse(torch.linalg.cholesky(inv_cov.to_dense()))

            if standardize_cov: # standardize the variance of the covariance matrix to one
                inv_sds = torch.diagflat(torch.diagonal(cov_sp) ** (-0.5))
                cov_sp = inv_sds @ cov_sp @ inv_sds

            if centering: # center the kernel matrix
                cov_sp = cov_sp - cov_sp.mean(dim=0, keepdim=True)
                cov_sp = cov_sp - cov_sp.mean(dim=1, keepdim=True)

            # store the dense spatial kernel matrix
            self.K_sp = cov_sp # (n_spots, n_spots)
            self._rank = cov_sp.shape[0] # full rank

            # compute eigenvalues and eigenvectors in later stages
            self.K_eigvals = None # eigenvalues of the kernel matrix
            self.K_eigvecs = None # eigenvectors of the kernel matrix
            self.Q = None # low-rank approximation of the kernel matrix K_sp = Q @ Q^T
        else:
            # compute and store the low-rank approximation of the kernel matrix
            assert approx_rank <= inv_cov.shape[0], "approx_rank must be less than the number of spots."

            # normalize the inverse covariance matrix by degree such that the diagonal entries are 1
            # inv_cov = D^(-1/2) @ inv_cov @ D^(-1/2)
            # this will update self.inv_cov
            degrees = swm.sum(dim=1).to_dense() # (n_spots,)
            inv_cov.values()[:] = inv_cov.values() / degrees[inv_cov.indices()[0]].pow(0.5) # (n_spots, n_spots)
            inv_cov.values()[:] = inv_cov.values() / degrees[inv_cov.indices()[1]].pow(0.5) # (n_spots, n_spots)

            # first convert torch sparse tensor to scipy coo matrix
            indices = inv_cov.indices().numpy()
            values = inv_cov.values().numpy()
            shape = inv_cov.shape
            coo_mtx = scipy.sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape)

            # then compute the smallest k eigenvalues of the inverse covariance matrix
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(coo_mtx, k=approx_rank, which='SM')

            # which is equivalent to the largest k eigenvalues of the covariance matrix
            eigvals = torch.from_numpy(eigvals + 1e-6).pow(-1) # sorted in descending order
            eigvecs = torch.from_numpy(eigvecs)

            # store the eigenvalues and eigenvectors
            self.cov_eigvals = eigvals
            self.cov_eigvecs = eigvecs

            # the low-rank approximation of the kernel matrix is K_sp = Q @ Q^T
            Q = eigvecs @ torch.diag(eigvals.pow(0.5)) # (n_spots, rank)

            # scale K_sp to have unit variance (diagonal entries)
            if standardize_cov:
                # after scaling, cov_sp and Q @ Q.T would have different eigenvalues
                # due to the low-rank approximation
                _sd = (Q.pow(2).sum(dim=1) + 1e-6).pow(0.5) # (n_spots,)
                Q = Q / _sd[:, None]

            # center K_sp to have zero row and column sums
            if centering:
                Q = Q - Q.mean(dim=0, keepdim=True)

            # compute kernel eigenvalues from Q
            K_eigvals = torch.linalg.eigvalsh(Q.T @ Q) # (rank,), ascending order

            # sort the eigenvalues in descending order
            idx = K_eigvals.argsort(descending=True)
            self.K_eigvals = K_eigvals[idx]

            # store the low-rank approximation of the kernel matrix
            self.Q = Q
            self.K_sp = None # K_sp = Q @ Q^T
            self._rank = self.Q.shape[1]

    def shape(self):
        """Return the shape of the kernel matrix."""
        return self.inv_cov.shape

    def rank(self):
        """Return the rank of the kernel matrix."""
        return self._rank

    def realization(self):
        """Return the realized (N, N) kernel matrix.
        Args:
            centered_K: bool. Whether to return the centered kernel matrix.
        """
        if self.K_sp is not None:
            return self.K_sp
        else:
            return self.Q @ self.Q.t()

    def xtKx(self, x):
        """Given a vector x of shape (N, d), return the quadratic form x^T @ K_sp @ x.
        """
        if self.K_sp is not None: # use the full rank kernel matrix
            return x.t() @ self.K_sp @ x
        else: # use the low-rank approximation
            xtQ = x.t() @ self.Q # (d, rank)
            return xtQ @ xtQ.t() # (d, d)

    def eigenvalues(self, k = None):
        """Return the top k eigenvalues of the kernel matrix.
        Args:
            k: int. Number of top eigenvalues to return.
        """
        if self.K_eigvals is None:
            # compute and store the eigendecomposition of self.K_sp
            self.eigendecomposition()

        # return the top k largest eigenvalues
        if k is None:
            return self.K_eigvals
        else:
            return self.K_eigvals[:k]

    def eigendecomposition(self):
        """Compute and store the eigendecomposition of the kernel matrix."""
        assert self.K_sp is not None, "The kernel matrix is not stored in dense form."

        eigvals, eigvecs = np.linalg.eigh(self.K_sp.numpy())

        # sort the eigenvalues and eigenvectors in descending order
        idx = eigvals.argsort()[::-1]
        self.K_eigvals = torch.from_numpy(eigvals[idx])
        self.K_eigvecs = torch.from_numpy(eigvecs[:, idx])
        # Q = eigvecs @ np.diag(eigvals ** 0.5) # (n_spots, n_spots)
        # self.Q = torch.from_numpy(Q)[:, idx]