"""Shared interfaces for spatial GP residualization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from splisosm.gpr.operators import SpatialKernelOp

__all__ = ["KernelGPR"]


class KernelGPR(ABC):
    """Abstract GP residualizer.

    Subclasses determine how hyperparameters are optimized (MLE, variational,
    fixed) and whether the kernel matrix is materialized (dense) or kept
    implicit.

    All subclasses expose the same ``fit_residuals`` interface so that the
    DU-test pipeline in ``SplisosmNP`` and ``SplisosmFFT`` is backend-agnostic.
    """

    @abstractmethod
    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit a spatial GP and return residuals.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Spatial coordinates, pre-normalized by
            the caller.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        residuals : torch.Tensor
            Shape ``(n_spots, m)``. Residuals
            ``Y - GP_smooth(Y | coords)``. NaN rows in ``Y`` are preserved
            as NaN.
        """

    def fit_residuals_batch(
        self,
        coords: torch.Tensor,
        Y_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Fit residuals for many targets sharing the same coordinates.

        The default implementation calls :meth:`fit_residuals` sequentially.
        Subclasses may override this to amortize work across targets
        (e.g. ``SklearnKernelGPR`` precomputes a shared eigendecomposition
        when signal bounds are fixed).

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Spatial coordinates.
        Y_list : list of torch.Tensor
            List of target matrices, each shape ``(n_spots, m_i)``.

        Returns
        -------
        list of torch.Tensor
            List of residual matrices, each shape ``(n_spots, m_i)``.
        """
        return [self.fit_residuals(coords, Y) for Y in Y_list]

    def get_kernel_op(self, coords: torch.Tensor) -> SpatialKernelOp:
        """Build the kernel operator for given coordinates.

        Returns a :class:`SpatialKernelOp` whose hyperparameters have been set
        (but not yet optimized for a specific target Y).  Useful when the
        caller needs the operator for its own purposes (e.g. computing
        eigenvalues for the HSIC null distribution).

        Not all backends support this method.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Spatial coordinates.

        Returns
        -------
        SpatialKernelOp
            Kernel operator for the given coordinates.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose an explicit SpatialKernelOp."
        )
