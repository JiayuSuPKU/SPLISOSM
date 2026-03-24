import unittest

import numpy as np
from anndata import AnnData

import splisosm.hyptest_fft as hyptest_fft
from splisosm.hyptest_fft import FFTKernel, SplisosmFFT


class _SpatialDataStub:
    """Minimal SpatialData-like container for unit tests."""

    def __init__(self, table_name: str, adata: AnnData):
        self.tables = {table_name: adata}
        self._images = {}

    def __setitem__(self, key: str, value):
        self._images[key] = value

    def __getitem__(self, key: str):
        return self._images[key]


class _RasterLayerStub:
    """xarray-like raster layer stub with channel selection."""

    def __init__(self, data: np.ndarray, channels: list[str]):
        self._data = data
        self._channels = channels
        self.shape = data.shape

    def sel(self, c):
        if isinstance(c, (list, np.ndarray)):
            idx = [self._channels.index(str(name)) for name in c]
        else:
            idx = [self._channels.index(str(c))]
        return _RasterLayerStub(self._data[idx, :, :], [self._channels[i] for i in idx])

    @property
    def values(self):
        return self._data


class _SpatialDataModuleStub:
    """Stub module exposing rasterize_bins API used by SplisosmFFT."""

    @staticmethod
    def rasterize_bins(
        sdata,
        bins,
        table_name,
        col_key,
        row_key,
        value_key=None,
        return_region_as_labels=False,
    ):
        del bins, value_key, return_region_as_labels
        adata = sdata.tables[table_name]
        row = adata.obs[row_key].to_numpy(dtype=int)
        col = adata.obs[col_key].to_numpy(dtype=int)
        ny = int(row.max()) + 1
        nx = int(col.max()) + 1

        counts = np.asarray(adata.X, dtype=float)
        out = np.zeros((counts.shape[1], ny, nx), dtype=float)
        for i in range(counts.shape[0]):
            out[:, row[i], col[i]] += counts[i, :]

        return _RasterLayerStub(out, adata.var_names.astype(str).tolist())


class TestFFTKernel(unittest.TestCase):

    def test_power_spectral_density_1d_constant_spectrum(self):
        kernel = FFTKernel(shape=(6, 5), rho=0.0, neighbor_degree=1)

        freq_bins, psd_1d = kernel.power_spectral_density_1d(bins=8)

        self.assertGreater(len(freq_bins), 0)
        self.assertEqual(freq_bins.shape, psd_1d.shape)
        self.assertTrue(np.all(np.diff(freq_bins) > 0))
        np.testing.assert_allclose(psd_1d, np.ones_like(psd_1d))

    def test_power_spectral_density_1d_matches_manual_bin_average(self):
        kernel = FFTKernel(shape=(5, 4), rho=0.9, neighbor_degree=2)
        bins = 6

        freq_bins, psd_1d = kernel.power_spectral_density_1d(bins=bins)

        fy = np.fft.fftfreq(kernel.ny, d=kernel.dy)
        fx = np.fft.fftfreq(kernel.nx, d=kernel.dx)
        fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
        radial_freq = np.sqrt(fy_grid**2 + fx_grid**2).ravel()
        spectrum = kernel.spectrum
        bin_edges = np.linspace(0, radial_freq.max(), bins + 1)

        expected_bins = []
        expected_psd = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            if right == bin_edges[-1]:
                mask = (radial_freq >= left) & (radial_freq <= right)
            else:
                mask = (radial_freq >= left) & (radial_freq < right)
            if np.any(mask):
                expected_bins.append((left + right) / 2.0)
                expected_psd.append(float(np.mean(spectrum[mask])))

        np.testing.assert_allclose(freq_bins, np.asarray(expected_bins))
        np.testing.assert_allclose(psd_1d, np.asarray(expected_psd))

    def test_power_spectral_density_1d_rejects_invalid_bins(self):
        kernel = FFTKernel(shape=(4, 4), rho=0.9, neighbor_degree=1)

        with self.assertRaises(ValueError):
            kernel.power_spectral_density_1d(bins=0)


def _build_test_sdata(table_name: str = "isoform_table") -> _SpatialDataStub:
    rng = np.random.default_rng(0)

    ny, nx = 6, 5
    n_spots = ny * nx

    row_idx = np.repeat(np.arange(ny), nx)
    col_idx = np.tile(np.arange(nx), ny)

    # 3 genes total, one is single-isoform and should always be filtered.
    # g1 -> 2 isoforms, g2 -> 3 isoforms, g3 -> 1 isoform
    n_iso = 6
    gene_symbols = ["g1", "g1", "g2", "g2", "g2", "g3"]
    gene_ids = ["GID1", "GID1", "GID2", "GID2", "GID2", "GID3"]
    iso_names = [f"iso_{i}" for i in range(n_iso)]

    counts = rng.poisson(lam=2.0, size=(n_spots, n_iso)).astype(np.float32)

    # Add smooth spatial trend so tests are not completely null-like.
    counts[:, 0] += (row_idx >= 3).astype(np.float32) * 3.0
    counts[:, 3] += (col_idx >= 2).astype(np.float32) * 2.0

    adata = AnnData(X=counts)
    adata.layers["counts"] = counts
    adata.obs["array_row"] = row_idx
    adata.obs["array_col"] = col_idx
    adata.var_names = iso_names
    adata.var["gene_symbol"] = gene_symbols
    adata.var["gene_id"] = gene_ids

    return _SpatialDataStub(table_name=table_name, adata=adata)


class TestSplisosmFFT(unittest.TestCase):

    def setUp(self):
        self.table_name = "isoform_table"
        self.sdata = _build_test_sdata(table_name=self.table_name)
        self._old_sd = hyptest_fft.sd
        hyptest_fft.sd = _SpatialDataModuleStub()

    def tearDown(self):
        hyptest_fft.sd = self._old_sd

    def test_setup_data(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )

        # g3 (single-isoform gene) is always filtered.
        self.assertEqual(model.n_genes, 2)
        self.assertEqual(model.n_grid, 30)
        self.assertIn("FFT SPLISOSM", str(model))

    def test_setup_data_gene_names_column(self):
        """gene_names column in adata.var should be used as display names."""
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            group_iso_by="gene_symbol",
            gene_names="gene_id",
        )
        # Display names come from gene_id column, first value per group.
        self.assertEqual(sorted(model.gene_names), ["GID1", "GID2"])

    def test_docstring_example_workflow(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            sdata=self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            layer="counts",
            group_iso_by="gene_symbol",
            gene_names="gene_id",
            min_counts=10,
            min_bin_pct=0.0,
        )
        model.test_spatial_variability(
            method="hsic-ir",
            n_jobs=1,
            print_progress=False,
        )
        sv_results = model.get_formatted_test_results(test_type="sv")

        self.assertEqual(len(sv_results), model.n_genes)
        self.assertTrue(
            {"statistic", "pvalue", "pvalue_adj"}.issubset(sv_results.columns)
        )

    def test_setup_data_min_counts_filters_isoforms(self):
        """Isoforms below min_counts threshold should be excluded."""
        import numpy as np
        from anndata import AnnData

        rng = np.random.default_rng(42)
        ny, nx = 4, 4
        n_spots = ny * nx
        row_idx = np.repeat(np.arange(ny), nx)
        col_idx = np.tile(np.arange(nx), ny)

        # Gene A: iso_0 has many counts, iso_1 has very few (below threshold).
        # Gene B: both isoforms have many counts.
        counts = np.zeros((n_spots, 4), dtype=np.float32)
        counts[:, 0] = rng.poisson(50, size=n_spots)  # gene A iso 0 - high
        counts[:, 1] = 0.0  # gene A iso 1 - zero
        counts[:, 2] = rng.poisson(50, size=n_spots)  # gene B iso 0 - high
        counts[:, 3] = rng.poisson(50, size=n_spots)  # gene B iso 1 - high

        adata = AnnData(X=counts)
        adata.layers["counts"] = counts
        adata.obs["array_row"] = row_idx
        adata.obs["array_col"] = col_idx
        adata.var_names = ["isoA0", "isoA1", "isoB0", "isoB1"]
        adata.var["gene_symbol"] = ["gA", "gA", "gB", "gB"]

        from splisosm.hyptest_fft import SplisosmFFT

        sdata = _SpatialDataStub(table_name="tbl", adata=adata)
        model = SplisosmFFT()
        model.setup_data(
            sdata,
            bins="g",
            table_name="tbl",
            col_key="array_col",
            row_key="array_row",
            min_counts=10,
        )
        # Gene A has only 1 isoform above threshold — filtered out.
        # Only gene B remains.
        self.assertEqual(model.n_genes, 1)
        self.assertEqual(model.gene_names, ["gB"])

    def test_setup_data_min_bin_pct_filters_isoforms(self):
        """Isoforms below min_bin_pct threshold should be excluded."""
        ny, nx = 4, 4
        n_spots = ny * nx
        row_idx = np.repeat(np.arange(ny), nx)
        col_idx = np.tile(np.arange(nx), ny)

        counts = np.zeros((n_spots, 4), dtype=np.float32)
        counts[:, 0] = 5.0
        counts[[0, 1], 1] = 10.0
        counts[:, 2] = 6.0
        counts[:, 3] = 7.0

        adata = AnnData(X=counts)
        adata.layers["counts"] = counts
        adata.obs["array_row"] = row_idx
        adata.obs["array_col"] = col_idx
        adata.var_names = ["isoA0", "isoA1", "isoB0", "isoB1"]
        adata.var["gene_symbol"] = ["gA", "gA", "gB", "gB"]

        sdata = _SpatialDataStub(table_name="tbl", adata=adata)
        model = SplisosmFFT()
        model.setup_data(
            sdata,
            bins="g",
            table_name="tbl",
            col_key="array_col",
            row_key="array_row",
            min_counts=0,
            min_bin_pct=0.2,
        )

        # isoA1 is present in only 2/16 bins and is filtered, so gene A drops out.
        self.assertEqual(model.n_genes, 1)
        self.assertEqual(model.gene_names, ["gB"])

    def test_spatial_variability_methods(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )

        for method in ["hsic-ir", "hsic-ic", "hsic-gc"]:
            with self.subTest(method=method):
                model.test_spatial_variability(
                    method=method,
                    n_jobs=2,
                    print_progress=False,
                )
                res = model.get_formatted_test_results("sv")
                self.assertEqual(len(res), model.n_genes)
                self.assertTrue(np.isfinite(res["statistic"].to_numpy()).all())
                self.assertTrue(np.isfinite(res["pvalue"].to_numpy()).all())
                self.assertTrue(np.isfinite(res["pvalue_adj"].to_numpy()).all())

    def test_extract_feature_summary_gene(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )

        summary1 = model.extract_feature_summary(level="gene")
        summary2 = model.extract_feature_summary(level="gene")

        self.assertIs(summary1, summary2)
        self.assertEqual(len(summary1), model.n_genes)
        self.assertTrue(
            {"n_isos", "perplexity", "pct_bin_on", "count_avg", "count_std"}.issubset(
                summary1.columns
            )
        )
        self.assertTrue(np.isfinite(summary1["perplexity"].to_numpy()).all())
        self.assertTrue(np.isfinite(summary1["count_avg"].to_numpy()).all())

    def test_extract_feature_summary_isoform(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )

        summary1 = model.extract_feature_summary(level="isoform")
        summary2 = model.extract_feature_summary(level="isoform")

        self.assertIs(summary1, summary2)
        self.assertEqual(len(summary1), sum(model.n_isos))
        self.assertTrue(
            {
                "pct_bin_on",
                "count_total",
                "count_avg",
                "count_std",
                "ratio_total",
                "ratio_avg",
                "ratio_std",
            }.issubset(summary1.columns)
        )
        self.assertTrue(np.isfinite(summary1["ratio_total"].to_numpy()).all())
        self.assertTrue(np.isfinite(summary1["ratio_avg"].to_numpy()).all())
        self.assertTrue(np.isfinite(summary1["ratio_std"].to_numpy()).all())

    def test_extract_feature_summary_invalid_level(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )

        with self.assertRaises(ValueError):
            model.extract_feature_summary(level="bad-level")

    def _setup_model(self):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )
        return model

    def test_differential_usage(self):
        """hsic-gp (default): FFT GPR residualization + HSIC."""
        model = self._setup_model()
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(42)
        design = rng.standard_normal((adata.n_obs, 2))

        results = model.test_differential_usage(
            design_matrix=design,
            method="hsic-gp",
            n_jobs=1,
            return_results=True,
        )

        self.assertIn("statistic", results)
        self.assertIn("pvalue", results)
        self.assertIn("pvalue_adj", results)
        self.assertEqual(results["statistic"].shape, (model.n_genes, 2))
        self.assertEqual(results["pvalue"].shape, (model.n_genes, 2))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))

        # get_formatted_test_results should return a long-format DataFrame
        df = model.get_formatted_test_results("du")
        self.assertEqual(len(df), model.n_genes * 2)
        self.assertIn("gene", df.columns)
        self.assertIn("factor", df.columns)
        self.assertIn("pvalue", df.columns)

    def test_differential_usage_unconditional_hsic(self):
        """hsic (unconditional): no spatial residualization, spot-level."""
        model = self._setup_model()
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(1)
        design = rng.standard_normal((adata.n_obs, 2))

        results = model.test_differential_usage(
            design_matrix=design,
            method="hsic",
            n_jobs=1,
            return_results=True,
        )
        self.assertEqual(results["statistic"].shape, (model.n_genes, 2))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))
        self.assertEqual(results["method"], "hsic")

    def test_differential_usage_t_fisher(self):
        """t-fisher: per-isoform t-test with Fisher combination."""
        model = self._setup_model()
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(2)
        # Binary factor (0/1)
        binary = (rng.random(adata.n_obs) > 0.5).astype(float)[:, None]

        results = model.test_differential_usage(
            design_matrix=binary,
            method="t-fisher",
            n_jobs=1,
            return_results=True,
        )
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))
        self.assertEqual(results["method"], "t-fisher")

    def test_differential_usage_t_tippett(self):
        """t-tippett: per-isoform t-test with Tippett combination."""
        model = self._setup_model()
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(3)
        binary = (rng.random(adata.n_obs) > 0.5).astype(float)[:, None]

        results = model.test_differential_usage(
            design_matrix=binary,
            method="t-tippett",
            n_jobs=1,
            return_results=True,
        )
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))

    def test_differential_usage_dataframe_input(self):
        """Design matrix as pandas DataFrame preserves column names as factor names."""
        import pandas as pd

        model = self._setup_model()
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(7)
        design = pd.DataFrame(
            rng.standard_normal((adata.n_obs, 1)), columns=["condition"]
        )

        results = model.test_differential_usage(
            design_matrix=design, method="hsic", n_jobs=1, return_results=True
        )
        self.assertEqual(results["factor_names"], ["condition"])

    def test_differential_usage_invalid_inputs(self):
        model = self._setup_model()
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(0)

        # invalid method
        with self.assertRaises(ValueError):
            model.test_differential_usage(
                design_matrix=rng.standard_normal((adata.n_obs, 1)),
                method="invalid_method",
            )

        # wrong number of rows
        with self.assertRaises(ValueError):
            model.test_differential_usage(
                design_matrix=rng.standard_normal((adata.n_obs + 1, 1)),
            )

        # bad ratio transformation
        with self.assertRaises(ValueError):
            model.test_differential_usage(
                design_matrix=rng.standard_normal((adata.n_obs, 1)),
                ratio_transformation="bad_transform",
            )


if __name__ == "__main__":
    unittest.main()
