"""Dataset helpers for batched GLM and GLMM training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from typing import Any, Iterator, Optional

__all__ = ["IsoDataset"]


def sparse_collate(batch: list[Any]) -> Any:
    """Custom collate function to handle sparse tensors.

    Parameters
    ----------
    batch : list of Any
        Batch of data items to collate.

    Returns
    -------
    Any
        Collated batch.
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        if elem.is_sparse:
            return torch.stack(batch)
        return torch.stack(batch, 0, out=None)
    elif isinstance(elem, dict):
        return {key: sparse_collate([d[key] for d in batch]) for key in elem}
    return default_collate(batch)


class UngroupedIsoDataset(Dataset):
    """Per-gene dataset for training GLM and GLMM model.

    Genes are not grouped by the number of isoforms and are stored separatedly
    as a list of per-gene tensors.
    Each iteration returns a dictionary containing the following keys:

    - ``x``: torch.Tensor, isoform counts of the gene, shape (n_spots, n_isos).
    - ``n_isos``: int, number of isoforms for the gene.
    - ``gene_name``: str, name of the gene.
    """

    def __init__(self, data: list[torch.Tensor], gene_names: list[str]) -> None:
        """
        Parameters
        ----------
        data
            List of tensors each with shape (n_spots, n_isos).
        gene_names
            List of gene names.
        """
        self.n_genes = len(data)  # number of genes
        self.n_spots = len(data[0])  # number of spots
        self.n_isos_per_gene = [
            data_g.shape[1] for data_g in data
        ]  # number of isoforms for each gene
        self.gene_names = gene_names
        assert (
            len(self.gene_names) == self.n_genes
        ), "Gene names must match the number of genes."
        assert (
            min(self.n_isos_per_gene) > 1
        ), "At least two isoforms are required for each gene."

        self.data = data

    def __len__(self):
        return self.n_genes

    def __getitem__(self, idx):
        return {
            "n_isos": self.n_isos_per_gene[idx],
            "x": self.data[idx],
            "gene_name": self.gene_names[idx],
        }


class GroupedIsoDataset(Dataset):
    """Grouped dataset for training GLM and GLMM model.

    Genes with the same number of isoforms are grouped and stored together
    as a 3D tensor of shape (n_genes, n_spots, n_isos).
    Each iteration returns a dictionary containing the following keys:

    - ``x``: torch.Tensor, isoform counts of the gene, shape (n_spots, n_isos).
    - ``n_isos``: int, number of isoforms for the gene.
    - ``gene_name``: str, name of the gene.
    """

    def __init__(self, data: torch.Tensor, gene_names: list[str]) -> None:
        """
        Parameters
        ----------
        data
            Shape (n_genes, n_spots, n_isos), genes are grouped by number of isoforms.
        gene_names
            List of gene names.
        """
        self.n_genes, self.n_spots, self.n_isos = data.shape
        assert (
            len(gene_names) == self.n_genes
        ), "Gene names must match the number of genes."

        self.data = data
        self.gene_names = gene_names

    def __len__(self):
        return self.n_genes

    def __getitem__(self, idx):
        return {
            "n_isos": self.n_isos,
            "x": self.data[idx, :, :],
            "gene_name": self.gene_names[idx],
        }


def _iters_merger(*iters):
    for itr in iters:
        for v in itr:
            yield v


class IsoDataset:
    """Dataset for batched training of GLM and GLMM models.

    `IsoDataset.get_dataloader` returns a DataLoader that yields batches of genes for training.

    If `group_gene_by_n_iso` is True, genes with the same number of isoforms are grouped together
    and stored as a 3D tensor of shape (n_genes, n_spots, n_isos).
    Otherwise, genes are stored as a list of per-gene tensors of shape (n_spots, n_isos).

    Example
    -------
    >>> from splisosm.glmm.dataset import IsoDataset
    >>> import torch
    >>> # Simulate data for 10 genes with different number of isoforms
    >>> data_3_iso = [torch.randn(100, 3) for _ in range(5)]  # 5 genes with 3 isoforms
    >>> data_4_iso = [torch.randn(100, 4) for _ in range(5)]  # 5 genes with 4 isoforms
    >>> data = data_3_iso + data_4_iso
    >>> gene_names = [f"gene_{i}" for i in range(10)]
    >>> dataset = IsoDataset(data, gene_names, group_gene_by_n_iso=True)
    >>> # Get dataloader for batched training
    >>> dataloader = dataset.get_dataloader(batch_size=2)
    >>> batch = next(iter(dataloader))
    """

    data: list[torch.Tensor]
    """Input list of per-gene isoform count tensor."""

    group_by_n_iso: bool
    """Whether to group genes by the number of isoforms."""

    dataset: list[Dataset]
    """
    If `group_by_n_iso` is True, a list of ``GroupedIsoDataset`` where isoform counts are stored as 3D tensors.
    Otherwise, a list of ``UngroupedIsoDataset`` where isoform counts are stored as a list of 2D tensors.
    """

    gene_name: list[str]
    """List of gene names."""

    n_genes: int
    """Number of genes."""

    n_spots: int
    """Number of spots."""

    n_isos_per_gene: list[int]
    """List of numbers of isoforms per gene."""

    def __init__(
        self,
        data: list[torch.Tensor],
        gene_names: Optional[list[str]] = None,
        group_gene_by_n_iso: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        data
            List of tensors with shape (n_spots, n_isos).
        gene_names
            List of gene names. If None, auto-generated.
        group_gene_by_n_iso
            Whether to group genes by the number of isoforms.
        """
        self.n_genes = len(data)  # number of genes
        self.n_spots = len(data[0])  # number of spots
        self.n_isos_per_gene = [
            data_g.shape[1] for data_g in data
        ]  # number of isoforms for each gene
        self.gene_names = (
            gene_names
            if gene_names is not None
            else [f"gene_{str(i + 1).zfill(5)}" for i in range(self.n_genes)]
        )
        assert (
            len(self.gene_names) == self.n_genes
        ), "Gene names must match the number of genes."
        assert (
            min(self.n_isos_per_gene) > 1
        ), "At least two isoforms are required for each gene."

        # convert numpy.array to torch.tensor float if not already
        _data = [
            torch.from_numpy(arr).float() if isinstance(arr, np.ndarray) else arr
            for arr in data
        ]
        self.data = _data

        self.datasets = None

        # group and stack genes if they have the same number of isoforms
        self.group_gene_by_n_iso = group_gene_by_n_iso

        if group_gene_by_n_iso:
            self._group_and_stack_genes()
        else:
            self.datasets = [UngroupedIsoDataset(self.data, self.gene_names)]

    def _group_and_stack_genes(self):
        """Group and stack genes by the number of isoforms."""
        _datasets = []
        n_isos_per_gene = torch.tensor(self.n_isos_per_gene)
        for _n_iso in n_isos_per_gene.unique():
            _d = [self.data[i] for i in torch.where(n_isos_per_gene == _n_iso)[0]]
            _d = torch.stack(_d, dim=0)  # (n_genes, n_spots, n_isos)
            _gn = [
                self.gene_names[i] for i in torch.where(n_isos_per_gene == _n_iso)[0]
            ]

            # create a new dataset for the grouped genes with _n_iso isoforms
            _datasets.append(GroupedIsoDataset(_d, _gn))

        self.datasets = _datasets

    def get_dataloader(self, batch_size: int = 1) -> Iterator[Any]:
        """Get dataloader for the dataset.

        Parameters
        ----------
        batch_size
            Maximum number of genes in a batch.

        Returns
        -------
        Iterator[Any]
            DataLoader iterator.
        """
        if not self.group_gene_by_n_iso:
            return DataLoader(
                self.datasets[0], batch_size=1, shuffle=False, collate_fn=sparse_collate
            )
        else:
            dataloaders = [
                DataLoader(
                    ds, batch_size=batch_size, shuffle=False, collate_fn=sparse_collate
                )
                for ds in self.datasets
            ]
            return _iters_merger(*dataloaders)
