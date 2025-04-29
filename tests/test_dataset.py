import unittest
import torch
import numpy as np
from splisosm.dataset import UngroupedIsoDataset, GroupedIsoDataset, IsoDataset

class TestDatasets(unittest.TestCase):

    def test_ungrouped_iso_dataset(self):
        data = [torch.rand(10, 3), torch.rand(10, 4)]
        gene_names = ["gene1", "gene2"]
        dataset = UngroupedIsoDataset(data, gene_names)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['n_isos'], 3)
        self.assertEqual(dataset[1]['n_isos'], 4)
        self.assertEqual(dataset[0]['gene_name'], "gene1")
        self.assertEqual(dataset[1]['gene_name'], "gene2")

    def test_grouped_iso_dataset(self):
        data = torch.rand(2, 10, 3)
        gene_names = ["gene1", "gene2"]
        dataset = GroupedIsoDataset(data, gene_names)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['n_isos'], 3)
        self.assertEqual(dataset[0]['gene_name'], "gene1")
        self.assertEqual(dataset[1]['gene_name'], "gene2")

    def test_iso_dataset_ungrouped(self):
        data = [np.random.rand(10, 3), np.random.rand(10, 4)]
        gene_names = ["gene1", "gene2"]
        dataset = IsoDataset(data, gene_names, group_gene_by_n_iso=False)

        self.assertEqual(len(dataset.datasets), 1)
        self.assertEqual(len(dataset.datasets[0]), 2)
        self.assertEqual(dataset.datasets[0][0]['n_isos'], 3)
        self.assertEqual(dataset.datasets[0][1]['n_isos'], 4)

    def test_iso_dataset_grouped(self):
        data = [np.random.rand(10, 3), np.random.rand(10, 3), np.random.rand(10, 4)]
        gene_names = ["gene1", "gene2", "gene3"]
        dataset = IsoDataset(data, gene_names, group_gene_by_n_iso=True)

        self.assertEqual(len(dataset.datasets), 2)
        self.assertEqual(len(dataset.datasets[0]), 2)  # Two genes with 3 isoforms
        self.assertEqual(len(dataset.datasets[1]), 1)  # One gene with 4 isoforms

    def test_iso_dataset_dataloader(self):
        data = [np.random.rand(10, 3), np.random.rand(10, 4)]
        gene_names = ["gene1", "gene2"]
        dataset = IsoDataset(data, gene_names, group_gene_by_n_iso=False)
        dataloader = dataset.get_dataloader(batch_size=1)

        batches = list(dataloader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]['n_isos'].item(), 3)
        self.assertEqual(batches[1]['n_isos'].item(), 4)

if __name__ == "__main__":
    unittest.main()