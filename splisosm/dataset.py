import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UngroupedIsoDataset(Dataset):
    """Dataset for spatial isoform expression."""
    def __init__(self, data, gene_names):
        """Initialize the dataset.

        Args:
            data: list of tensor(n_spots, n_isos)
            gene_names: list of str
        """
        self.n_genes = len(data) # number of genes
        self.n_spots = len(data[0]) # number of spots
        self.n_isos_per_gene = [data_g.shape[1] for data_g in data] # number of isoforms for each gene
        self.gene_names = gene_names
        assert len(self.gene_names) == self.n_genes, "Gene names must match the number of genes."
        assert min(self.n_isos_per_gene) > 1, "At least two isoforms are required for each gene."

        self.data = data

    def __len__(self):
        return self.n_genes

    def __getitem__(self, idx):
        return {
            'n_isos': self.n_isos_per_gene[idx],
            'x': self.data[idx],
            'gene_name': self.gene_names[idx],
        }


class GroupedIsoDataset(Dataset):
    """Dataset for spatial isoform expression per gene group."""
    def __init__(self, data, gene_names):
        """Initialize the dataset.

        Args:
            data: tensor(n_genes, n_spots, n_isos), genes are grouped by the number of isoforms
            gene_names: list of str
        """
        self.n_genes, self.n_spots, self.n_isos = data.shape
        assert len(gene_names) == self.n_genes, "Gene names must match the number of genes."

        self.data = data
        self.gene_names = gene_names

    def __len__(self):
        return self.n_genes

    def __getitem__(self, idx):
        return {
            'n_isos': self.n_isos,
            'x': self.data[idx, :, :],
            'gene_name': self.gene_names[idx],
        }

def _iters_merger(*iters):
    for itr in iters:
        for v in itr:
            yield v


class IsoDataset():
    """Dataset for spatial isoform expression."""
    def __init__(self, data, gene_names=None, group_gene_by_n_iso=False):
        """Initialize the dataset.

        Args:
            data: list of tensor(n_spots, n_isos)
            gene_names: list of str, gene names
            group_gene_by_n_iso: bool, whether to group genes by the number of isoforms
        """
        self.n_genes = len(data) # number of genes
        self.n_spots = len(data[0]) # number of spots
        self.n_isos_per_gene = [data_g.shape[1] for data_g in data] # number of isoforms for each gene
        self.gene_names = gene_names if gene_names is not None else [
            f"gene_{str(i + 1).zfill(5)}" for i in range(self.n_genes)
        ]
        assert len(self.gene_names) == self.n_genes, "Gene names must match the number of genes."
        assert min(self.n_isos_per_gene) > 1, "At least two isoforms are required for each gene."

        # convert numpy.array to torch.tensor float if not already
        _data = [torch.from_numpy(arr).float() if isinstance(arr, np.ndarray) else arr for arr in data]
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
            _d = torch.stack(_d, dim=0) # (n_genes, n_spots, n_isos)
            _gn = [self.gene_names[i] for i in torch.where(n_isos_per_gene == _n_iso)[0]]

            # create a new dataset for the grouped genes with _n_iso isoforms
            _datasets.append(GroupedIsoDataset(_d, _gn))

        self.datasets = _datasets

    def get_dataloader(self, batch_size = 1):
        """Get dataloader for the dataset.

        Args:
            batch_size: int, maximum number of genes in a batch

        Returns:
            iter: DataLoader iterator
        """
        if not self.group_gene_by_n_iso:
            return DataLoader(self.datasets[0], batch_size=1, shuffle=False)
        else:
            dataloaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in self.datasets]
            return _iters_merger(*dataloaders)

