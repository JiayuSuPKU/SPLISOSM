
import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from anndata import AnnData
from splisosm.utils import extract_counts_n_ratios, extract_gene_level_statistics, run_hsic_gc, run_sparkx

class TestUtils(unittest.TestCase):
    def setUp(self):
        n_spots = 50
        n_genes = 3
        n_iso_per_gene = [2, 3, 1]
        n_total_isos = sum(n_iso_per_gene)

        # Create counts
        counts_dense = np.random.randint(0, 10, size=(n_spots, n_total_isos)).astype(np.float32)
        # Ensure some zeros for sparsity check
        counts_dense[counts_dense < 3] = 0

        # Create gene info
        gene_symbols = []
        isoform_names = []
        for i, n in enumerate(n_iso_per_gene):
            gene_name = f"Gene_{i}"
            for j in range(n):
                gene_symbols.append(gene_name)
                isoform_names.append(f"{gene_name}_Iso_{j}")

        var_df = pd.DataFrame({
            'gene_symbol': gene_symbols
        }, index=isoform_names)

        # Create AnnData
        self.adata = AnnData(X=counts_dense, var=var_df)
        self.adata.layers['counts'] = counts_dense
        # Store for comparison
        self.counts_dense = counts_dense
        self.gene_names = [f"Gene_{i}" for i in range(n_genes)]

    def test_extract_counts_n_ratios_dense(self):
        counts_list, ratios_list, gene_names, ratios_obs = extract_counts_n_ratios(
            self.adata, layer='counts', group_iso_by='gene_symbol', return_sparse=False
        )

        self.assertEqual(len(counts_list), 2)
        self.assertEqual(len(ratios_list), 2)
        self.assertIsInstance(counts_list[0], torch.Tensor)
        self.assertFalse(counts_list[0].is_sparse)
        self.assertIsInstance(ratios_list[0], torch.Tensor)
        self.assertIsNotNone(ratios_obs)

    def test_extract_counts_n_ratios_sparse_scipy(self):
        # Convert layer to sparse
        counts_sparse = scipy.sparse.csr_matrix(self.counts_dense)
        self.adata.layers['counts_sparse'] = counts_sparse

        counts_list, ratios_list, gene_names, ratios_obs = extract_counts_n_ratios(
            self.adata, layer='counts_sparse', group_iso_by='gene_symbol', return_sparse=True
        )

        self.assertEqual(len(counts_list), 2)
        self.assertEqual(len(ratios_list), 0)
        self.assertIsNone(ratios_obs)

        unique_genes = sorted(list(set(self.adata.var['gene_symbol'])))

        for i, counts in enumerate(counts_list):
            self.assertTrue(counts.is_sparse)
            # Convert to dense and compare with original
            gene_name = gene_names[i]
            iso_indices = np.where(self.adata.var['gene_symbol'] == gene_name)[0]
            expected = self.counts_dense[:, iso_indices]
            np.testing.assert_allclose(counts.to_dense().numpy(), expected, rtol=1e-5)

    def test_extract_gene_level_statistics_sparse_vs_dense(self):
        # Dense results
        stats_dense = extract_gene_level_statistics(self.adata, layer='counts')

        # Sparse results
        counts_sparse = scipy.sparse.csr_matrix(self.counts_dense)
        self.adata.layers['counts_sparse'] = counts_sparse
        stats_sparse = extract_gene_level_statistics(self.adata, layer='counts_sparse')

        # Compare columns
        cols_to_compare = ['n_iso', 'pct_spot_on', 'count_avg', 'count_std', 'perplexity', 'major_ratio_avg']

        # Sort by index to ensure alignment
        stats_dense = stats_dense.sort_index()
        stats_sparse = stats_sparse.sort_index()

        for col in cols_to_compare:
            np.testing.assert_allclose(
                stats_sparse[col].values,
                stats_dense[col].values,
                rtol=1e-5, err_msg=f"Mismatch in column {col}"
            )

    def test_run_sparkx_error_on_sparse(self):
        n_spots = 20
        n_genes = 5
        counts_dense = np.random.rand(n_spots, n_genes)
        coords = np.random.rand(n_spots, 2)

        # SciPy sparse
        counts_csr = scipy.sparse.csr_matrix(counts_dense)
        with self.assertRaisesRegex(ValueError, "does not support sparse input"):
            run_sparkx(counts_csr, coords)

        # Torch sparse
        counts_torch = torch.from_numpy(counts_dense).to_sparse()
        with self.assertRaisesRegex(ValueError, "does not support sparse input"):
            run_sparkx(counts_torch, coords)

    def test_run_hsic_gc_formats(self):
        n_spots = 50
        n_genes = 4

        counts_np = np.random.randint(0, 5, size=(n_spots, n_genes)).astype(np.float32)
        coords_np = np.random.rand(n_spots, 2).astype(np.float32)

        # 1. Test Dense Numpy
        res_np = run_hsic_gc(counts_np, coords_np)

        # 2. Test Dense Torch
        counts_torch = torch.from_numpy(counts_np)
        coords_torch = torch.from_numpy(coords_np)
        res_torch = run_hsic_gc(counts_torch, coords_torch)

        # 3. Test Scipy Sparse
        counts_csr = scipy.sparse.csr_matrix(counts_np)
        res_csr = run_hsic_gc(counts_csr, coords_np)

        # 4. Test Torch Sparse
        counts_torch_sparse = counts_torch.to_sparse()
        res_torch_sparse = run_hsic_gc(counts_torch_sparse, coords_np)

        # Check consistency

        np.testing.assert_allclose(res_np['statistic'], res_torch['statistic'], rtol=1e-5)
        np.testing.assert_allclose(res_np['statistic'], res_csr['statistic'], rtol=1e-5)
        np.testing.assert_allclose(res_np['statistic'], res_torch_sparse['statistic'], rtol=1e-5)

        # Check p-values roughly
        np.testing.assert_allclose(res_np['pvalue'], res_torch['pvalue'], rtol=1e-5)
        np.testing.assert_allclose(res_np['pvalue'], res_csr['pvalue'], rtol=1e-5)
        np.testing.assert_allclose(res_np['pvalue'], res_torch_sparse['pvalue'], rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
