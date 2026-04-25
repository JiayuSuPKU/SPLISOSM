"""Spatial kernel abstractions."""

from __future__ import annotations

import warnings
import numpy as np
import scipy.fft
import scipy.sparse
import scipy.sparse.linalg
import torch
from abc import ABC, abstractmethod
from scipy.sparse.linalg import splu
from sklearn.neighbors import NearestNeighbors

__all__ = ["IdentityKernel", "SpatialCovKernel", "FFTKernel"]


class Kernel(ABC):
    """Abstract interface for kernel matrix representations.

    Implementations may store either dense kernels or low-rank factors, but
    must expose common operations required by SPLISOSM hypothesis tests.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize kernel-specific state."""

    @property
    def n(self) -> int:
        """Number of observations represented by the kernel.

        Returns
        -------
        int
            Kernel dimension.
        """
        return int(self._n)

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

    @abstractmethod
    def Kx(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply the effective kernel operator to a vector or dense matrix.

        The effective operator is the same one used by :meth:`xtKx`; for
        kernels constructed with ``centering=True`` this is ``H K H``.

        Parameters
        ----------
        x
            Array of shape ``(n_spots,)`` or ``(n_spots, n_vectors)``.

        Returns
        -------
        np.ndarray or torch.Tensor
            Kernel product with the same shape and array backend as ``x``.
        """

    @abstractmethod
    def xtKx_approx(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Compute ``x^T K_k x`` using a rank-``k`` approximation.

        Implementations must use only the top-``k`` eigenvalues/vectors and
        must not expose :attr:`Q` to callers.  When ``k`` is ``None`` the
        full kernel approximation is used.

        Parameters
        ----------
        x : torch.Tensor
            Input matrix of shape ``(n_spots, d)``.
        k : int or None
            Number of leading eigenvalues to include.  ``None`` uses all
            available eigenvalues (full-rank approximation).

        Returns
        -------
        torch.Tensor
            Approximated quadratic form of shape ``(d, d)``.
        """


class IdentityKernel(Kernel):
    """Identity kernel K = I (placeholder when spatial kernel is skipped).

    All spots are treated as independent; no spatial smoothing is applied.
    This is used when ``setup_data(skip_spatial_kernel=True)`` is called
    and spatial variability testing is not needed.

    Parameters
    ----------
    n_spots : int
        Number of spatial locations.
    centering : bool, optional
        If ``True``, apply double-centring ``H K H`` with ``H = I - (1/n) 1 1^T``.
        For ``K = I`` this reduces to ``H`` itself (``H`` is idempotent), whose
        spectrum has :math:`n - 1` unit eigenvalues and one zero eigenvalue.
        Default ``False``.  HSIC-based SV/DU tests should set this to ``True``
        for consistency with the double-centred HSIC formulation.
    """

    def __init__(self, n_spots: int, centering: bool = False) -> None:
        self._n = n_spots
        self._centering = bool(centering)
        # Store Q as a sparse identity to avoid allocating a dense n×n matrix.
        # K = I = Q @ Q.T, but Q is never accessed externally — use xtKx_approx().
        _idx = torch.arange(n_spots).unsqueeze(0).repeat(2, 1)  # (2, n_spots)
        _val = torch.ones(n_spots)
        with torch.sparse.check_sparse_tensor_invariants(False):
            self.Q: torch.Tensor = torch.sparse_coo_tensor(
                _idx, _val, (n_spots, n_spots)
            )

    def Kx(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply ``I`` or ``H`` to ``x``."""
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            arr = x.detach().cpu().numpy()
            dtype = x.dtype
            device = x.device
        else:
            arr = np.asarray(x)
            dtype = None
            device = None

        out = np.asarray(arr, dtype=float)
        if out.shape[0] != self._n:
            raise ValueError(f"Expected first dimension {self._n}, got {out.shape[0]}.")
        if self._centering:
            axis = 0 if out.ndim > 1 else None
            out = out - out.mean(axis=axis, keepdims=out.ndim > 1)
        if is_torch:
            return torch.as_tensor(out, dtype=dtype, device=device)
        return out

    def realization(self) -> torch.Tensor:
        """Return ``I`` (or ``H = I - (1/n) 1 1^T`` when ``centering=True``)."""
        if self._centering:
            return torch.eye(self._n) - torch.full((self._n, self._n), 1.0 / self._n)
        return torch.eye(self._n)

    def xtKx(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x^T K x`` where ``K = I`` (or ``H I H = H`` when centred)."""
        xc = x - x.mean(dim=0, keepdim=True) if self._centering else x
        return xc.t() @ xc

    def xtKx_exact(self, x: torch.Tensor) -> torch.Tensor:
        """Exact quadratic form; identical to :meth:`xtKx` for this kernel."""
        return self.xtKx(x)

    def xtKx_approx(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Quadratic form; identical to :meth:`xtKx` (no approximation needed)."""
        return self.xtKx(x)

    def eigenvalues(self, k: int | None = None) -> torch.Tensor:
        """Return the leading eigenvalues of ``K``.

        For ``K = I`` all :math:`n` eigenvalues are ``1``.  For ``H I H = H``
        there are :math:`n - 1` unit eigenvalues and a single zero eigenvalue;
        only the nonzero ones are returned.
        """
        n_nonzero = max(self._n - 1, 0) if self._centering else self._n
        n = n_nonzero if k is None else min(int(k), n_nonzero)
        return torch.ones(n)

    def trace(self) -> torch.Tensor:
        """Return tr(K): ``n`` for ``I`` and ``n - 1`` for ``H``."""
        return torch.tensor(float(self._n - 1 if self._centering else self._n))

    def square_trace(self) -> torch.Tensor:
        """Return tr(K²): ``n`` for ``I`` and ``n - 1`` for ``H`` (idempotent)."""
        return torch.tensor(float(self._n - 1 if self._centering else self._n))


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

    @staticmethod
    def _as_numpy_2d(x: np.ndarray | torch.Tensor) -> tuple[np.ndarray, bool]:
        """Return a float64 ``(n, m)`` array and whether the input was 1-D."""
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("`x` must have shape (n_spots,) or (n_spots, n_vectors).")
        return np.asarray(arr, dtype=np.float64), squeeze

    @staticmethod
    def _restore_array_type(
        out: np.ndarray,
        template: np.ndarray | torch.Tensor,
        squeeze: bool,
    ) -> np.ndarray | torch.Tensor:
        """Restore shape/backend after an internal numpy calculation."""
        if squeeze:
            out = out[:, 0]
        if isinstance(template, torch.Tensor):
            return torch.as_tensor(out, dtype=template.dtype, device=template.device)
        return out

    def _hutchinson_trace(
        self, squared: bool, n_vectors: int = 30, seed: int = 0
    ) -> float:
        """Hutchinson stochastic estimator for tr(K') [or tr((K')²)].

        Here ``K'`` is the *effective* kernel: ``HKH`` when
        ``_centering=True`` and ``K`` otherwise.  With ``_centering=True``
        the ±1 probing vectors are double-centred on both sides of the LU
        solve; with ``_centering=False`` they are used directly.

        Parameters
        ----------
        squared
            If ``True`` estimate tr((K')²), else tr(K').
        n_vectors
            Number of probing vectors.
        seed
            Random seed for reproducibility.
        """
        lu = self._get_lu()
        n, m = self._n, n_vectors
        rng = np.random.default_rng(seed=seed)
        rvs = rng.choice([-1.0, 1.0], size=(n, m))

        if self._centering:
            v_left = rvs - rvs.mean(axis=0)  # H @ rvs
            Kv = lu.solve(v_left)  # K (H rvs)
            Kv_eff = Kv - Kv.mean(axis=0)  # H K H rvs
        else:
            v_left = rvs
            Kv_eff = lu.solve(rvs)  # K rvs

        if squared:
            # tr((K')²) ≈ (1/m) Σ ||K' v_i||²
            return float((Kv_eff**2).sum() / m)
        # tr(K') ≈ (1/m) Σ v_i^T K' v_i
        return float((v_left * Kv_eff).sum() / m)

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

    def Kx(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply ``K`` or ``H K H`` to ``x``.

        Parameters
        ----------
        x
            Dense vector or matrix with first dimension ``n_spots``.

        Returns
        -------
        np.ndarray or torch.Tensor
            Kernel product with the same shape/backend as ``x``.
        """
        arr, squeeze = self._as_numpy_2d(x)
        if arr.shape[0] != self._n:
            raise ValueError(f"Expected first dimension {self._n}, got {arr.shape[0]}.")

        if self._centering:
            arr = arr - arr.mean(axis=0, keepdims=True)

        if self.K_sp is not None:
            out = self.K_sp.numpy().astype(np.float64, copy=False) @ arr
        else:
            out = self._get_lu().solve(arr)

        if self._centering:
            out = out - out.mean(axis=0, keepdims=True)

        return self._restore_array_type(out, x, squeeze)

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

        # Implicit: sparse LU solve.  When ``_centering=True`` the effective
        # kernel is HKH, so column-centre x first: x_c = Hx, then
        # x^T (HKH) x = x_c^T K x_c  (H is symmetric idempotent).
        lu = self._get_lu()
        x_np = x.numpy().astype(np.float64)
        if self._centering:
            x_np = x_np - x_np.mean(axis=0, keepdims=True)
        u = lu.solve(x_np)  # shape (n, d)
        return torch.from_numpy((x_np.T @ u).astype(np.float32))

    def xtKx_approx(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Compute ``x^T K_k x`` via the top-``k`` low-rank factor.

        Calls :meth:`eigenvalues` to ensure :attr:`Q` is populated for at
        least ``k`` components, then computes ``(x^T Q_k)(Q_k^T x)`` where
        ``Q_k = Q[:, :k]``.  The result is consistent with the rank-``k``
        eigenvalue approximation used in the Liu null distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input matrix of shape ``(n_spots, d)``.
        k : int or None
            Number of leading eigenvalues to include.  ``None`` uses all
            available eigenvalues (populates the full Q first).

        Returns
        -------
        torch.Tensor
            Approximated quadratic form of shape ``(d, d)``.
        """
        self.eigenvalues(k=k)  # ensures Q is populated to at least rank k
        Q_k = self.Q if k is None else self.Q[:, :k]
        xtQ = x.t() @ Q_k
        return xtQ @ xtQ.t()

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

        # Cache eigenvectors (same convention as dense eigendecomposition())
        self.K_eigvecs = torch.from_numpy(eigvecs.copy())

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


class FFTKernel(Kernel):
    """FFT-based spatial kernel on a periodic 2D raster grid.

    This implementation supports a CAR-style spatial kernel equivalent to a
    periodic, neighbourhood-graph-based autoregressive model.  All operations
    are :math:`O(N \\log N)` via the 2-D Fast Fourier Transform.

    Parameters
    ----------
    shape
        Grid shape ``(ny, nx)``.
    spacing
        Physical spacing ``(dy, dx)`` between neighbouring raster cells.
    rho
        Spatial autocorrelation coefficient in CAR kernel.
    neighbor_degree
        Neighbour ring degree for graph construction.
        ``1`` uses nearest neighbours in the periodic metric.
    workers
        Number of workers used by ``scipy.fft.fft2``.
    centering
        If ``True``, apply double-centring ``H K H`` with ``H = I - (1/n) 1 1^T``.
        On a periodic grid the all-ones vector is the DFT DC eigenvector, so
        double-centring is equivalent to zeroing the ``(0, 0)`` entry of the
        spectrum; :meth:`xtKx`, :meth:`trace`, :meth:`square_trace`, and
        :meth:`eigenvalues` all reflect the centred kernel.  Default ``False``.
        HSIC-based SV/DU tests should set this to ``True`` for consistency with
        the double-centred HSIC formulation.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        spacing: tuple[float, float] = (1.0, 1.0),
        rho: float = 0.99,
        neighbor_degree: int = 1,
        workers: int | None = None,
        centering: bool = False,
    ) -> None:
        if len(shape) != 2:
            raise ValueError("`shape` must be a tuple of length 2: (ny, nx).")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("Grid dimensions must be positive.")
        if neighbor_degree < 1:
            raise ValueError("`neighbor_degree` must be >= 1.")

        self.ny, self.nx = int(shape[0]), int(shape[1])
        self.dy, self.dx = float(spacing[0]), float(spacing[1])
        self.n_grid = self.ny * self.nx
        self._n = self.n_grid
        self.neighbor_degree = int(neighbor_degree)
        self.rho = min(float(rho), 0.99)
        self.workers = workers
        self._centering = bool(centering)

        self._min_dist_sq = self._precompute_square_torus_distances()
        self._spectrum_2d = self._compute_car_spectrum()
        if self._centering:
            # Zero the DC component: on a periodic grid the (0, 0) DFT basis
            # vector is the constant mode, so dropping that eigenvalue is
            # equivalent to the double-centring H K H.
            self._spectrum_2d[0, 0] = 0.0
        self.spectrum = self._spectrum_2d.ravel()

    # ------------------------------------------------------------------
    # Internal construction helpers
    # ------------------------------------------------------------------

    def _precompute_square_torus_distances(self) -> np.ndarray:
        """Compute squared torus distances from origin for periodic grid."""
        y = np.arange(self.ny, dtype=float) * self.dy
        x = np.arange(self.nx, dtype=float) * self.dx
        y = np.minimum(y, (self.ny * self.dy) - y)
        x = np.minimum(x, (self.nx * self.dx) - x)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return yy**2 + xx**2

    def _compute_car_spectrum(self) -> np.ndarray:
        """Compute eigenvalues of the periodic CAR kernel via FFT."""
        unique_d2 = np.unique(self._min_dist_sq)
        if self.neighbor_degree < len(unique_d2):
            cutoff_sq = unique_d2[self.neighbor_degree]
        else:
            cutoff_sq = unique_d2[-1]

        w_img = (self._min_dist_sq <= cutoff_sq).astype(float)
        w_img[0, 0] = 0.0
        degree = float(np.sum(w_img))

        if degree <= 0.0:
            return np.ones((self.ny, self.nx), dtype=float)

        lam_w = np.real(scipy.fft.fft2(w_img, workers=self.workers)) / degree
        return 1.0 / (1.0 - self.rho * lam_w)

    # ------------------------------------------------------------------
    # Kernel ABC implementation
    # ------------------------------------------------------------------

    def realization(self) -> torch.Tensor:
        """Return the dense kernel matrix (not recommended for large grids)."""
        raise NotImplementedError(
            "FFTKernel does not support dense realisation; use xtKx() or eigenvalues()."
        )

    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """Compute ``x^T K x`` in ``O(N log N)`` via FFT.

        Parameters
        ----------
        x
            Input with shape ``(ny, nx)`` or ``(ny, nx, m)``.

        Returns
        -------
        float or np.ndarray
            Scalar for 2D input, or shape ``(m,)`` for 3D input.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            x = x[..., None]
        if x.ndim != 3:
            raise ValueError("`x` must have shape (ny, nx) or (ny, nx, m).")

        ny, nx, _ = x.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Input shape ({ny}, {nx}) does not match kernel shape "
                f"({self.ny}, {self.nx})."
            )

        x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=self.workers)
        power = np.abs(x_hat) ** 2
        weighted = np.sum(power * self._spectrum_2d[:, :, None], axis=(0, 1))
        q = weighted / (self.n_grid)
        return float(q[0]) if q.size == 1 else q

    def Kx(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply the FFT kernel to flat or grid-shaped input.

        Parameters
        ----------
        x
            Input with shape ``(n_grid,)``, ``(n_grid, m)``, ``(ny, nx)``,
            or ``(ny, nx, m)``.

        Returns
        -------
        np.ndarray or torch.Tensor
            Kernel product with the same shape/backend as ``x``.
        """
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            arr = x.detach().cpu().numpy()
            dtype = x.dtype
            device = x.device
        else:
            arr = np.asarray(x)
            dtype = None
            device = None

        original_shape = arr.shape
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            if arr.size != self.n_grid:
                raise ValueError(f"Expected length {self.n_grid}, got {arr.size}.")
            cube = arr.reshape(self.ny, self.nx, 1)
        elif arr.ndim == 2 and arr.shape == (self.ny, self.nx):
            cube = arr[:, :, None]
        elif arr.ndim == 2 and arr.shape[0] == self.n_grid:
            cube = arr.reshape(self.ny, self.nx, arr.shape[1])
        elif arr.ndim == 3 and arr.shape[:2] == (self.ny, self.nx):
            cube = arr
        else:
            raise ValueError(
                "`x` must have shape (n_grid,), (n_grid, m), (ny, nx), or (ny, nx, m)."
            )

        x_hat = scipy.fft.fft2(cube, axes=(0, 1), workers=self.workers)
        out = np.real(
            scipy.fft.ifft2(
                x_hat * self._spectrum_2d[:, :, None],
                axes=(0, 1),
                workers=self.workers,
            )
        )
        if len(original_shape) == 1:
            out = out.reshape(self.n_grid)
        elif len(original_shape) == 2 and original_shape == (self.ny, self.nx):
            out = out[:, :, 0]
        elif len(original_shape) == 2:
            out = out.reshape(self.n_grid, original_shape[1])

        if is_torch:
            return torch.as_tensor(out, dtype=dtype, device=device)
        return out

    def xtKx_approx(self, x, k: int | None = None):
        """Same as :meth:`xtKx` (FFT is already exact for periodic grids)."""
        return self.xtKx(x)

    def eigenvalues(self, k: int | None = None) -> np.ndarray:
        """Return kernel eigenvalues in descending order.

        Parameters
        ----------
        k
            Number of leading eigenvalues to return.  ``None`` returns all.

        Returns
        -------
        np.ndarray
            Eigenvalues in descending order.
        """
        evals = np.sort(self.spectrum)[::-1]
        if k is None:
            return evals
        return evals[:k]

    def trace(self) -> float:
        """Return ``tr(K)``."""
        return float(np.sum(self.spectrum))

    def square_trace(self) -> float:
        """Return ``tr(K^2)``."""
        return float(np.sum(self.spectrum**2))

    # ------------------------------------------------------------------
    # FFT-specific methods (not in Kernel ABC)
    # ------------------------------------------------------------------

    def power_spectral_density_1d(
        self, bins: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the 1D power spectral density (radial profile).

        Parameters
        ----------
        bins
            Number of bins for the 1D radial frequency.

        Returns
        -------
        freq_bins : np.ndarray
            The centre frequencies of the valid bins.
        psd_1d : np.ndarray
            The average power (eigenvalue) in each frequency bin.
        """
        if bins < 1:
            raise ValueError("`bins` must be >= 1.")

        fy = scipy.fft.fftfreq(self.ny, d=self.dy)
        fx = scipy.fft.fftfreq(self.nx, d=self.dx)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        F_r = np.sqrt(FY**2 + FX**2)
        f_r_flat = F_r.ravel()
        spectrum_flat = self._spectrum_2d.ravel()

        bin_edges = np.linspace(0, f_r_flat.max(), bins + 1)
        psd_sum, _ = np.histogram(f_r_flat, bins=bin_edges, weights=spectrum_flat)
        counts, _ = np.histogram(f_r_flat, bins=bin_edges)

        valid = counts > 0
        psd_1d = np.zeros(bins)
        psd_1d[valid] = psd_sum[valid] / counts[valid]

        freq_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        return freq_bins[valid], psd_1d[valid]

    def apply_residual_op(self, x: np.ndarray, epsilon: float) -> np.ndarray:
        """Apply the kernel regression residual operator.

        Computed in O(N log N) via FFT as::

            R @ v = IFFT2( epsilon / (lambda + epsilon) * FFT2(v) )

        Parameters
        ----------
        x
            Input with shape ``(ny, nx)`` or ``(ny, nx, m)``.
        epsilon
            Regularisation / noise level.

        Returns
        -------
        np.ndarray
            Residuals of the same shape as ``x``.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 2
        if scalar:
            x = x[..., None]
        if x.ndim != 3:
            raise ValueError("`x` must have shape (ny, nx) or (ny, nx, m).")

        ny, nx, _ = x.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Input shape ({ny}, {nx}) does not match kernel shape "
                f"({self.ny}, {self.nx})."
            )

        x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=self.workers)
        scale = epsilon / (self._spectrum_2d[:, :, None] + epsilon)
        result = np.real(
            scipy.fft.ifft2(scale * x_hat, axes=(0, 1), workers=self.workers)
        )
        return result[..., 0] if scalar else result
