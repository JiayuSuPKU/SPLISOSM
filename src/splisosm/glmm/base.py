"""Shared GLM/GLMM model interfaces and tensor helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

__all__ = ["BaseModel", "_melt_tensor_along_first_dim", "update_at_idx"]


class BaseModel(ABC):
    """API for the GLM and GLMM model."""

    @abstractmethod
    def setup_data(
        self,
        counts: torch.Tensor,
        corr_sp: Optional[torch.Tensor],
        design_mtx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set up the data for the model.

        Parameters
        ----------
        counts : torch.Tensor
            Shape (n_genes, n_spots, n_isoforms), genes with the same number of isoforms.
        corr_sp : torch.Tensor
            Shape (n_spots, n_spots).
        design_mtx : torch.Tensor, optional
            Shape (n_spots, n_factors).
        """
        pass

    def forward(self) -> torch.Tensor:
        """Calculate the log-likelihood or log-marginal-likelihood of the model."""
        pass

    @abstractmethod
    def fit(self) -> None:
        """Fit the model using all data."""
        pass

    @abstractmethod
    def get_isoform_ratio(self) -> torch.Tensor:
        """Extract the fitted isoform ratio across space."""
        pass

    @abstractmethod
    def clone(self) -> "BaseModel":
        """Clone the model with data and model parameters."""
        pass


def _melt_tensor_along_first_dim(tensor_in):
    """Melt a 4D tensor into 3D and reorder entries by spots.

    tensor_in[:, i, j, k] -> matrix_out[:, i + j * n, i + k * n] where n = tensor_in.shape[1]

    Parameters
    ----------
    tensor_in : torch.Tensor
        Shape (n_genes, n_spots, n_isos - 1, n_isos - 1).

    Returns
    -------
    matrix_out : torch.Tensor
        Shape (n_genes, n_spots * (n_isos - 1), n_spots * (n_isos - 1)).
    """
    b, n, m = tensor_in.shape[:3]
    if tensor_in.shape != (b, n, m, m):
        raise ValueError(
            "`tensor_in` must have shape "
            f"(n_genes, n_spots, n_isos - 1, n_isos - 1); got {tuple(tensor_in.shape)}."
        )

    # example: at spot i, isoform j and k are connected via tensor[i, j, k]
    # tensor[i, j, k] -> out[i + j * n_spots, i + k * n_spots]
    matrix_out = torch.zeros(b, n * m, n * m, device=tensor_in.device)
    i, j, k = torch.meshgrid(
        torch.arange(n), torch.arange(m), torch.arange(m), indexing="ij"
    )
    row_indices = i + j * n
    col_indices = i + k * n
    matrix_out[:, row_indices.view(-1), col_indices.view(-1)] += tensor_in.flatten(1)

    return matrix_out


@torch.no_grad
def update_at_idx(
    params: torch.Tensor, new_params: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    idx = idx.view(-1, *([1] * (params.ndim - 1))).float()
    params = params * idx + new_params * (1 - idx)
    return params
