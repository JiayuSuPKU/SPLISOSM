"""Spatial kernel abstractions."""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
from abc import ABC, abstractmethod
from scipy.sparse.linalg import splu
from sklearn.neighbors import NearestNeighbors

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

    @abstractmethod
    def trace(self) -> torch.Tensor:
        """Return the trace of the kernel matrix tr(K).

        Returns
        -------
        torch.Tensor
            Scalar trace value.
        """

    @abstractmethod
    def square_trace(self) -> torch.Tensor:
        """Return tr(K²).

        Returns
        -------
        torch.Tensor
            Scalar value.
        """


def _build_adj_from_coords(
    coords: np.ndarray,
    k_neighbors: int,
    mutual_neighbors: bool = True,
) -> scipy.sparse.csc_matrix:
    """Build a symmetric k-NN adjacency matrix from spot coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spot coordinates of shape ``(n_spots, n_dim)``.
    k_neighbors : int
        Number of nearest neighbours (excluding self).
    mutual_neighbors : bool
        If ``True``, retain only edges where both spots list each other as a
        neighbour.  If ``False``, symmetrise by averaging.

    Returns
    -------
    scipy.sparse.csc_matrix
        Binary symmetric adjacency matrix of shape ``(n_spots, n_spots)``.
    """
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors + 1, algorithm="auto", metric="euclidean"
    ).fit(coords)
    W = nbrs.kneighbors_graph(coords, mode="connectivity").astype(float)

    if mutual_neighbors:
        W_sym = W + W.T
        W_sym.data = (W_sym.data > 1).astype(float)
    else:
        W_sym = 0.5 * (W + W.T)

    # Remove self-connections; if any rows become empty, add self-loop to avoid isolated nodes
    W_sym.setdiag(0)
    row_sums = np.asarray(W_sym.sum(axis=1)).ravel()
    isolated = row_sums == 0
    if isolated.any():
        W_sym.setdiag(isolated.astype(float))
    W_sym.eliminate_zeros()
    return W_sym.tocsc()


def _build_car_precision_from_adj(
    adj: scipy.sparse.spmatrix,
    rho: float,
) -> scipy.sparse.csc_matrix:
    """Build a sparse CAR precision matrix from an adjacency/weight matrix.

    Computes M = I - rho * D^{-1/2} W D^{-1/2}.

    Parameters
    ----------
    adj : scipy.sparse.spmatrix
        Symmetric adjacency or weight matrix of shape ``(n_spots, n_spots)``.
    rho : float
        Spatial autocorrelation coefficient (0 < rho < 1).

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse precision matrix of shape ``(n_spots, n_spots)``.
    """
    n = adj.shape[0]
    row_sums = np.asarray(adj.sum(axis=1)).ravel().clip(1e-12)
    inv_D_sqrt = scipy.sparse.diags(1.0 / np.sqrt(row_sums))
    W_norm = inv_D_sqrt @ adj @ inv_D_sqrt
    return (scipy.sparse.eye(n, format="csc") - rho * W_norm).tocsc()


class SpatialCovKernel(Kernel):
    """Graph-based spatial covariance kernel.

    Construct from spot coordinates (:meth:`from_coordinates` or the
    ``coords`` positional argument) or from a pre-built k-NN adjacency /
    weight matrix (:meth:`from_adjacency` or the ``adj_matrix`` keyword
    argument).  Exactly one of the two must be provided.

    The kernel is stored in one of two modes depending on dataset size:

    - **Dense mode** (``n ≤ DENSE_THRESHOLD``): the ``(n, n)`` covariance
      matrix ``K_sp`` is materialised, with optional variance standardisation
      and double-centring applied at construction time.
    - **Implicit mode** (``n > DENSE_THRESHOLD``): only the sparse precision
      ``inv_cov = K⁻¹`` is retained; ``K x`` is computed via a cached sparse
      LU factorisation.

    Eigenvalue decomposition and the low-rank Q factor are computed
    **lazily** on the first call to :meth:`eigenvalues`.
    """

    DENSE_THRESHOLD: int = 5000
    """Switch to implicit sparse mode above this number of spots."""

    inv_cov: scipy.sparse.csc_matrix
    """Sparse precision matrix M = K⁻¹ of shape ``(n_spots, n_spots)``."""

    K_sp: torch.Tensor | None
    """Dense centred/standardised kernel ``(n_spots, n_spots)``. ``None`` in
    implicit mode."""

    Q: torch.Tensor | None
    """Low-rank factor ``(n_spots, rank)`` such that K ≈ Q @ Q.T.
    Populated lazily after the first :meth:`eigenvalues` call."""

    K_eigvals: torch.Tensor | None
    """Cached eigenvalues in descending order. Populated lazily."""

    K_eigvecs: torch.Tensor | None
    """Cached eigenvectors (dense mode only). Populated lazily."""

    def __init__(
        self,
        coords: np.ndarray | torch.Tensor | None = None,
        *,
        adj_matrix: "scipy.sparse.spmatrix | np.ndarray | None" = None,
        k_neighbors: int = 4,
        rho: float = 0.99,
        standardize_cov: bool = True,
        centering: bool = False,
    ) -> None:
        """Construct from spot coordinates or a pre-built adjacency matrix.

        Exactly one of ``coords`` or ``adj_matrix`` must be supplied.
        Prefer the factory methods :meth:`from_coordinates` and
        :meth:`from_adjacency` for more descriptive call sites.

        Parameters
        ----------
        coords : array-like of shape ``(n_spots, n_dim)``, optional
            Spot coordinates.  A k-NN graph is built from these and used to
            construct the CAR precision matrix.
        adj_matrix : sparse or dense matrix of shape ``(n_spots, n_spots)``, optional
            Pre-built symmetric k-NN adjacency / weight matrix.  The CAR
            precision is built directly from this matrix.
        k_neighbors : int
            Number of nearest neighbours when building the graph from
            ``coords``.  Ignored when ``adj_matrix`` is provided.
        rho : float
            CAR spatial autocorrelation coefficient (0 < rho < 1).
        standardize_cov : bool
            Scale the covariance to unit marginal variance.
        centering : bool
            Apply double-centring H K H.
        """
        if (coords is None) == (adj_matrix is None):
            raise ValueError(
                "Provide exactly one of 'coords' or 'adj_matrix', not both or neither."
            )

        if coords is not None:
            if isinstance(coords, torch.Tensor):
                coords = coords.numpy()
            coords_np = np.asarray(coords, dtype=np.float64)
            adj = _build_adj_from_coords(coords_np, k_neighbors, mutual_neighbors=True)
        else:
            adj = (
                adj_matrix
                if scipy.sparse.issparse(adj_matrix)
                else scipy.sparse.csc_matrix(adj_matrix)
            )
            # Symmetrise if not already symmetric
            if not (adj != adj.T).nnz == 0:
                adj = (adj + adj.T) / 2
                warnings.warn(
                    "Provided adjacency matrix is not symmetric; symmetrising by averaging with its transpose.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # Through a warning if not sparse enough
            if adj.nnz > 0.1 * adj.shape[0] ** 2:
                warnings.warn(
                    f"Provided adjacency matrix has {adj.nnz} nonzeros, which may be inefficient for large n. "
                    "Consider using a sparse format or reducing k_neighbors.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        inv_cov = _build_car_precision_from_adj(adj, rho)
        self._init_from_precision(
            inv_cov, standardize_cov=standardize_cov, centering=centering
        )

    def _init_from_precision(
        self,
        inv_cov: scipy.sparse.csc_matrix,
        *,
        standardize_cov: bool,
        centering: bool,
    ) -> None:
        """Core initialisation from a sparse precision matrix."""
        n = inv_cov.shape[0]
        self._n = n
        self._standardize_cov = standardize_cov
        self._centering = centering
        self.inv_cov = inv_cov.tocsc()

        # Lazy state — not computed at construction
        self.Q = None
        self.K_eigvals = None
        self.K_eigvecs = None
        self._lu = None

        if n <= self.DENSE_THRESHOLD:
            # Dense path: invert M → K, apply standardise + centering
            M_dense = inv_cov.toarray().astype(np.float64)
            M_t = torch.from_numpy(M_dense)
            K = torch.cholesky_inverse(torch.linalg.cholesky(M_t)).numpy()

            if standardize_cov:
                d = K.diagonal().copy().clip(1e-12)
                s = 1.0 / np.sqrt(d)
                K = s[:, None] * K * s[None, :]

            if centering:
                K -= K.mean(axis=0, keepdims=True)
                K -= K.mean(axis=1, keepdims=True)

            self.K_sp = torch.from_numpy((0.5 * (K + K.T)).astype(np.float32))
            self._rank = n
        else:
            # Implicit path: keep sparse M; standardise if requested
            if standardize_cov:
                inv_cov = self._standardize_precision(inv_cov)
                self.inv_cov = inv_cov
            self.K_sp = None
            self._rank = n

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_coordinates(
        cls,
        coords: np.ndarray | torch.Tensor,
        k_neighbors: int = 4,
        rho: float = 0.99,
        standardize_cov: bool = True,
        centering: bool = False,
    ) -> "SpatialCovKernel":
        """Build a CAR spatial kernel from spot coordinates.

        Parameters
        ----------
        coords
            Spot coordinates of shape ``(n_spots, n_dim)``.
        k_neighbors
            Number of nearest neighbours for the spatial graph.
        rho
            CAR spatial autocorrelation coefficient (0 < rho < 1).
        standardize_cov
            Scale covariance to unit marginal variance.
        centering
            Apply double-centring H K H.

        Returns
        -------
        SpatialCovKernel
        """
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        coords_np = np.asarray(coords, dtype=np.float64)
        adj = _build_adj_from_coords(coords_np, k_neighbors, mutual_neighbors=True)
        inv_cov = _build_car_precision_from_adj(adj, rho)

        obj = cls.__new__(cls)
        obj._init_from_precision(
            inv_cov, standardize_cov=standardize_cov, centering=centering
        )
        return obj

    @classmethod
    def from_adjacency(
        cls,
        adj_matrix: "scipy.sparse.spmatrix | np.ndarray",
        rho: float = 0.99,
        standardize_cov: bool = True,
        centering: bool = False,
    ) -> "SpatialCovKernel":
        """Build a CAR kernel from a pre-built k-NN adjacency / weight matrix.

        Use this when the spatial graph has already been constructed (e.g.
        from external tools) and you want to skip the coordinate-based k-NN
        step.

        Parameters
        ----------
        adj_matrix
            Symmetric adjacency or weight matrix of shape
            ``(n_spots, n_spots)``.  Can be a dense ``np.ndarray`` or any
            ``scipy.sparse`` matrix.
        rho
            CAR spatial autocorrelation coefficient (0 < rho < 1).
        standardize_cov
            Scale covariance to unit marginal variance.
        centering
            Apply double-centring H K H.

        Returns
        -------
        SpatialCovKernel
        """
        adj = (
            adj_matrix
            if scipy.sparse.issparse(adj_matrix)
            else scipy.sparse.csc_matrix(adj_matrix)
        )
        inv_cov = _build_car_precision_from_adj(adj, rho)

        obj = cls.__new__(cls)
        obj._init_from_precision(
            inv_cov, standardize_cov=standardize_cov, centering=centering
        )
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _standardize_precision(
        self, M: scipy.sparse.csc_matrix
    ) -> scipy.sparse.csc_matrix:
        """Return D_s M D_s where D_s = diag(1/sqrt(diag(K))).

        Computes diag(K) = diag(M⁻¹) via batched column solves.
        """
        lu = splu(M)
        n = M.shape[0]
        diag_K = np.zeros(n)
        batch = 128
        for i in range(0, n, batch):
            end = min(i + batch, n)
            b = np.zeros((n, end - i))
            b[i:end] = np.eye(end - i)
            x = lu.solve(b)
            diag_K[i:end] = x[i:end].diagonal()
        diag_K = diag_K.clip(1e-12)
        s = 1.0 / np.sqrt(diag_K)
        S_inv = scipy.sparse.diags(1.0 / s, format="csc")
        return (S_inv @ M @ S_inv).tocsc()

    def _get_lu(self):
        """Return (and cache) the LU factorisation of ``inv_cov``."""
        if self._lu is None:
            self._lu = splu(self.inv_cov)
        return self._lu

    def _hutchinson_trace(self, squared: bool, n_vectors: int = 30) -> float:
        """Hutchinson stochastic estimator for tr(HKH) or tr((HKH)²).

        Centred ±1 probing vectors target the double-centred kernel K' = HKH.

        Parameters
        ----------
        squared
            If ``True`` estimate tr((HKH)²), else tr(HKH).
        n_vectors
            Number of probing vectors.
        """
        lu = self._get_lu()
        n, m = self._n, n_vectors
        rng = np.random.default_rng()
        rvs = rng.choice([-1.0, 1.0], size=(n, m))
        rvs_c = rvs - rvs.mean(axis=0)  # H @ rvs
        Kv = lu.solve(rvs_c)  # K (H rvs)
        Kv_c = Kv - Kv.mean(axis=0)  # H K H rvs
        if squared:
            # tr((HKH)²) ≈ (1/m) Σ ||HKH v_i||²
            return float((Kv_c**2).sum() / m)
        else:
            # tr(HKH) ≈ (1/m) Σ v_i^T HKH v_i
            return float((rvs_c * Kv_c).sum() / m)

    # ------------------------------------------------------------------
    # Kernel interface
    # ------------------------------------------------------------------

    def shape(self) -> tuple[int, int]:
        """Return kernel shape ``(n_spots, n_spots)``.

        Returns
        -------
        tuple of int
        """
        return (self._n, self._n)

    def rank(self) -> int:
        """Return effective rank of the stored representation.

        Returns
        -------
        int
        """
        return self._rank

    def realization(self) -> torch.Tensor:
        """Return the realised dense kernel matrix.

        For implicit kernels this forces dense inversion — prefer calling
        :meth:`eigenvalues` first to populate ``Q``, then use ``Q @ Q.T``.

        Returns
        -------
        torch.Tensor
            Dense matrix of shape ``(n_spots, n_spots)``.
        """
        if self.K_sp is not None:
            return self.K_sp  # already standardised + centred

        if self.Q is not None:
            return self.Q @ self.Q.t()

        # Fallback: dense inversion (expensive for large n)
        warnings.warn(
            "realization() on an implicit kernel forces dense inversion — "
            "call eigenvalues(k) first to populate Q, or avoid for large n.",
            RuntimeWarning,
            stacklevel=2,
        )
        lu = self._get_lu()
        K = lu.solve(np.eye(self._n, dtype=np.float64)).astype(np.float32)
        if self._centering:
            K -= K.mean(axis=0, keepdims=True)
            K -= K.mean(axis=1, keepdims=True)
        return torch.from_numpy(0.5 * (K + K.T))

    def xtKx(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x^T K x`` using the cached low-rank factor when available.

        When :attr:`Q` has been populated (by a prior :meth:`eigenvalues` call)
        the result is ``x^T Q Q^T x``, i.e. consistent with the approximation
        ``K ≈ Q Q^T``:

        * **Dense mode**: :meth:`eigendecomposition` produces a full-rank Q so
          ``Q Q^T = K`` exactly.
        * **Implicit mode**: :meth:`eigenvalues` produces a rank-k Q; the result
          is therefore a rank-k approximation of ``x^T K x``.

        Before :meth:`eigenvalues` is called (``Q`` is ``None``) this falls back
        to :meth:`xtKx_exact`.  Use :meth:`xtKx_exact` directly to always obtain
        the exact quadratic form regardless of whether Q has been computed.

        Parameters
        ----------
        x : torch.Tensor
            Input matrix of shape ``(n_spots, d)``.

        Returns
        -------
        torch.Tensor
            Matrix of shape ``(d, d)``.

        See Also
        --------
        xtKx_exact : Always uses the exact path (dense multiply or LU solve).
        """
        if self.Q is not None:
            xtQ = x.t() @ self.Q
            return xtQ @ xtQ.t()
        return self.xtKx_exact(x)

    def xtKx_exact(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x^T K x`` exactly, bypassing any cached low-rank factor.

        Always uses the full kernel:

        * **Dense mode** (``K_sp`` is not ``None``): direct matrix multiply
          ``x^T K_{sp} x``.
        * **Implicit mode** (``K_sp`` is ``None``): sparse LU solve
          ``M u = x`` where ``M = K^{-1}``, then returns ``x^T u = x^T K x``.

        Unlike :meth:`xtKx`, this method **ignores** :attr:`Q` and always
        returns the exact quadratic form under the full kernel K.  This is
        useful when you need the exact HSIC statistic for reporting while using
        a low-rank Q for the p-value null distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input matrix of shape ``(n_spots, d)``.

        Returns
        -------
        torch.Tensor
            Matrix of shape ``(d, d)``.

        See Also
        --------
        xtKx : Uses cached Q (low-rank approximation) when available.
        """
        # Dense: direct multiply (always exact regardless of Q).
        if self.K_sp is not None:
            return x.t() @ self.K_sp @ x

        # Implicit: sparse LU solve — M u = x  →  u = K x  →  x^T u = x^T K x.
        lu = self._get_lu()
        x_np = x.numpy().astype(np.float64)
        u = lu.solve(x_np)  # shape (n, d)
        return torch.from_numpy((x_np.T @ u).astype(np.float32))

    def eigenvalues(self, k: int | None = None) -> torch.Tensor:
        """Return leading eigenvalues in descending order.

        For dense kernels a full eigendecomposition is triggered once and
        cached.  For implicit kernels a partial eigsh via a
        ``LinearOperator`` is used; the low-rank factor ``Q`` is cached
        for subsequent :meth:`xtKx` calls.

        Parameters
        ----------
        k
            Number of leading eigenvalues.  ``None`` returns all cached values.

        Returns
        -------
        torch.Tensor
            Eigenvalues in descending order.
        """
        if self.K_eigvals is not None:
            cached_k = len(self.K_eigvals)
            # Dense mode always stores a full-rank decomposition — cache is always
            # sufficient (K_eigvals[:k] simply clips at n).
            # Implicit mode stores only the top-k_req eigenvalues from eigsh; if
            # a larger k is requested we must rerun eigsh with the new k.
            if self.K_sp is not None or k is None or k <= cached_k:
                return self.K_eigvals if k is None else self.K_eigvals[:k]
            # Fall through: implicit mode, k > cached_k — recompute below.

        if self.K_sp is not None:
            # Dense: full eigendecomposition (also populates Q)
            self.eigendecomposition()
            return self.K_eigvals if k is None else self.K_eigvals[:k]

        # Implicit: partial eigsh
        k_req = k if k is not None else self._n - 2
        lu = self._get_lu()

        if self._centering:

            def _matvec(v: np.ndarray) -> np.ndarray:
                v_c = v - v.mean()
                u = lu.solve(v_c)
                return u - u.mean()

        else:

            def _matvec(v: np.ndarray) -> np.ndarray:
                return lu.solve(v)

        op = scipy.sparse.linalg.LinearOperator(
            (self._n, self._n), matvec=_matvec, dtype=np.float64
        )
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(op, k=k_req, which="LM")

        # Sort descending
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx].astype(np.float32)
        eigvecs = eigvecs[:, idx].astype(np.float32)

        # Cache low-rank factor Q = V @ diag(sqrt(λ)) for xtKx reuse
        sqrt_eigvals = np.sqrt(eigvals.clip(0))
        self.Q = torch.from_numpy(eigvecs * sqrt_eigvals[None, :])
        self._rank = k_req

        # Always cache eigenvalues (even for partial k)
        self.K_eigvals = torch.from_numpy(eigvals.copy())
        return self.K_eigvals if k is None else self.K_eigvals[:k]

    def eigendecomposition(self) -> None:
        """Compute and cache the full eigendecomposition of the dense kernel.

        Populates :attr:`K_eigvals`, :attr:`K_eigvecs`, and :attr:`Q`.

        Raises
        ------
        ValueError
            If called on an implicit kernel (no ``K_sp``).
        """
        if self.K_sp is None:
            raise ValueError(
                "eigendecomposition() requires dense K_sp. "
                "Use eigenvalues(k) for implicit (large-n) kernels."
            )

        eigvals, eigvecs = np.linalg.eigh(self.K_sp.numpy())

        # Sort descending
        idx = eigvals.argsort()[::-1]
        self.K_eigvals = torch.from_numpy(eigvals[idx].astype(np.float32))
        self.K_eigvecs = torch.from_numpy(eigvecs[:, idx].astype(np.float32))

        # Build low-rank factor from positive eigenvalues
        pos = self.K_eigvals > 0
        self.Q = self.K_eigvecs[:, pos] * self.K_eigvals[pos].sqrt()[None, :]

    def trace(self) -> torch.Tensor:
        """Return tr(K) [or tr(HKH) when ``centering=True``].

        Uses exact arithmetic for the dense case (``K_sp`` is already centred).
        Falls back to a Hutchinson stochastic estimator for implicit kernels.

        Returns
        -------
        torch.Tensor
            Scalar trace value.
        """
        if self.K_sp is not None:
            return torch.trace(self.K_sp)
        return torch.tensor(self._hutchinson_trace(squared=False), dtype=torch.float32)

    def square_trace(self) -> torch.Tensor:
        """Return tr(K²) [or tr((HKH)²) when ``centering=True``].

        Uses ``||K||_F²`` for the dense case. Falls back to a Hutchinson
        stochastic estimator for implicit kernels.

        Returns
        -------
        torch.Tensor
            Scalar squared-trace value.
        """
        if self.K_sp is not None:
            return self.K_sp.pow(2).sum()
        return torch.tensor(self._hutchinson_trace(squared=True), dtype=torch.float32)
