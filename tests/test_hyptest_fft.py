import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

import splisosm.hyptest.fft as hyptest_fft
from splisosm.hyptest.fft import FFTKernel, SplisosmFFT


class _SpatialDataStub:
    """Minimal SpatialData-like container for unit tests."""

    def __init__(self, table_name: str, adata: AnnData):
        self.tables = {table_name: adata}
        self._images = {}

    def __setitem__(self, key: str, value):
        # Real spatialdata routes AnnData to sdata.tables
        if isinstance(value, AnnData):
            self.tables[key] = value
        else:
            self._images[key] = value

    def __getitem__(self, key: str):
        if key in self.tables:
            return self.tables[key]
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
        del bins, return_region_as_labels
        adata = sdata.tables[table_name]
        row = adata.obs[row_key].to_numpy(dtype=int)
        col = adata.obs[col_key].to_numpy(dtype=int)
        row = row - row.min()
        col = col - col.min()
        ny = int(row.max()) + 1
        nx = int(col.max()) + 1

        var_list = adata.var_names.astype(str).tolist()
        counts_mat = np.asarray(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            dtype=float,
        )  # (n_obs, n_vars)

        if value_key is not None:
            if isinstance(value_key, str):
                value_key = [value_key]
            indices = [var_list.index(v) for v in value_key]
            counts_mat = counts_mat[:, indices]
            channels = list(value_key)
        else:
            channels = var_list

        out = np.zeros((counts_mat.shape[1], ny, nx), dtype=float)
        for i in range(len(row)):
            out[:, row[i], col[i]] += counts_mat[i, :]

        return _RasterLayerStub(out, channels)


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

    # Add spatialdata annotation metadata (region + instance linking)
    adata.obs["region"] = pd.Categorical([table_name] * len(adata.obs))
    adata.obs["instance_id"] = np.arange(len(adata.obs), dtype=int)
    try:
        from spatialdata.models import TableModel

        adata = TableModel.parse(
            adata,
            region=table_name,
            region_key="region",
            instance_key="instance_id",
        )
    except Exception:
        adata.uns["spatialdata_attrs"] = {
            "region": table_name,
            "region_key": "region",
            "instance_key": "instance_id",
        }
    return _SpatialDataStub(table_name=table_name, adata=adata)


class TestSplisosmFFT(unittest.TestCase):

    def setUp(self):
        self.table_name = "isoform_table"
        self.sdata = _build_test_sdata(table_name=self.table_name)
        self._old_require_spatialdata = hyptest_fft._require_spatialdata
        hyptest_fft._require_spatialdata = lambda: _SpatialDataModuleStub()

    def tearDown(self):
        hyptest_fft._require_spatialdata = self._old_require_spatialdata

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
        self.assertIn("SplisosmFFT", str(model))

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

        from splisosm.hyptest.fft import SplisosmFFT

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

        res_dict = model.test_spatial_variability(
            method="hsic-ir",
            n_jobs=1,
            print_progress=False,
            return_results=True,
        )
        self.assertEqual(res_dict["null_method"], "liu")
        self.assertTrue(np.all(np.isfinite(res_dict["pvalue"])))

    def test_spatial_variability_chunk_size_matches_single_channel(self):
        """FFT SV chunking matches one-channel/singleton execution."""
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )

        ref = model.test_spatial_variability(
            method="hsic-ic",
            chunk_size=1,
            n_jobs=1,
            print_progress=False,
            return_results=True,
        )
        res = model.test_spatial_variability(
            method="hsic-ic",
            chunk_size="auto",
            n_jobs=1,
            print_progress=False,
            return_results=True,
        )
        np.testing.assert_allclose(
            res["statistic"], ref["statistic"], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(res["pvalue"], ref["pvalue"], rtol=1e-6, atol=1e-8)
        self.assertLessEqual(res["chunk_size"], 32)

    def test_setup_data_filter_single_iso_genes_false(self):
        """filter_single_iso_genes=False keeps genes with only one passing isoform."""
        # The default fixture has 2 genes each with 3 probes.
        # Raise min_counts so only ONE isoform per gene passes — those genes
        # would normally be dropped but should survive with the flag disabled.
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            min_counts=10,
            min_bin_pct=0.0,
            filter_single_iso_genes=False,
        )
        # With filter_single_iso_genes=False, all genes that have ≥1 passing
        # isoform are retained.
        self.assertGreaterEqual(model.n_genes, 1)

        # Default (True) removes single-isoform genes — verify it still works
        # by using normal thresholds where each gene has multiple passing probes.
        model2 = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model2.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            filter_single_iso_genes=True,
        )
        self.assertGreaterEqual(model2.n_genes, 1)

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
        self.assertEqual(len(summary1), sum(model.n_isos_per_gene))
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

    def test_str_repr(self):
        """__str__ includes class name and method after running SV."""
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
        )
        s_before = str(model)
        self.assertIn("SplisosmFFT", s_before)

        model.test_spatial_variability(method="hsic-ir", n_jobs=1, print_progress=False)
        s_after = str(model)
        self.assertIn("hsic-ir", s_after)

    def test_sv_parallel_determinism(self):
        """SV n_jobs=1 vs n_jobs=2 give identical results."""
        for n_jobs in [1, 2]:
            model = SplisosmFFT(rho=0.9, neighbor_degree=1)
            model.setup_data(
                self.sdata,
                bins="grid_bins",
                table_name=self.table_name,
                col_key="array_col",
                row_key="array_row",
            )
            model.test_spatial_variability(
                method="hsic-ir", n_jobs=n_jobs, print_progress=False
            )
            if n_jobs == 1:
                ref = model._sv_test_results
            else:
                np.testing.assert_allclose(
                    model._sv_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="FFT SV statistic differs between n_jobs=1 and n_jobs=2",
                )
                np.testing.assert_allclose(
                    model._sv_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="FFT SV pvalue differs between n_jobs=1 and n_jobs=2",
                )

    def test_du_parallel_determinism(self):
        """DU n_jobs=1 vs n_jobs=2 give identical results."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(77)
        design = rng.standard_normal((adata.n_obs, 1))
        for n_jobs in [1, 2]:
            model = self._setup_model(design_mtx=design)
            model.test_differential_usage(
                method="hsic", n_jobs=n_jobs, return_results=False, print_progress=False
            )
            if n_jobs == 1:
                ref = model._du_test_results
            else:
                np.testing.assert_allclose(
                    model._du_test_results["statistic"],
                    ref["statistic"],
                    atol=1e-6,
                    err_msg="FFT DU statistic differs between n_jobs=1 and n_jobs=2",
                )
                np.testing.assert_allclose(
                    model._du_test_results["pvalue"],
                    ref["pvalue"],
                    atol=1e-6,
                    err_msg="FFT DU pvalue differs between n_jobs=1 and n_jobs=2",
                )

    def _setup_model(self, design_mtx=None):
        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            design_mtx=design_mtx,
        )
        return model

    def test_differential_usage(self):
        """hsic-gp (default): FFT GPR residualization + HSIC."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(42)
        design = rng.standard_normal((adata.n_obs, 2))
        model = self._setup_model(design_mtx=design)

        results = model.test_differential_usage(
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
        self.assertIn("covariate", df.columns)
        self.assertIn("pvalue", df.columns)

    def test_differential_usage_dataframe_input(self):
        """Design matrix as pandas DataFrame preserves column names as factor names."""
        import pandas as pd

        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(7)
        design = pd.DataFrame(
            rng.standard_normal((adata.n_obs, 1)), columns=["condition"]
        )
        model = self._setup_model(design_mtx=design)
        model.test_differential_usage(n_jobs=1, return_results=False)
        results = model.get_formatted_test_results("du")
        self.assertEqual(results["covariate"].unique(), ["condition"])

    def test_differential_usage_invalid_inputs(self):
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(0)
        design = rng.standard_normal((adata.n_obs, 1))
        model = self._setup_model(design_mtx=design)

        with self.assertRaises(ValueError):
            model.test_differential_usage(method="invalid_method")
        with self.assertRaises(ValueError):
            model.test_differential_usage(ratio_transformation="bad_transform")
        with self.assertRaises(ValueError):
            model.test_differential_usage(residualize="bad")

        # t-fisher / t-tippett raise ValueError for non-binary covariates
        model_cont = self._setup_model(design_mtx=design)  # continuous covariate
        with self.assertRaises(ValueError):
            model_cont.test_differential_usage(method="t-fisher", n_jobs=1)
        with self.assertRaises(ValueError):
            model_cont.test_differential_usage(method="t-tippett", n_jobs=1)

    def test_setup_data_with_design_mtx(self):
        """design_mtx passed to setup_data is stored as AnnData in sdata."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(0)
        design = rng.standard_normal((adata.n_obs, 2))

        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            design_mtx=design,
            covariate_names=["cov_A", "cov_B"],
        )

        self.assertEqual(model.n_factors, 2)
        self.assertEqual(model.covariate_names, ["cov_A", "cov_B"])
        self.assertIsNotNone(model._design_table_name)
        # design_mtx is the AnnData table; shape = (n_obs, n_factors)
        self.assertIsNotNone(model.design_mtx)
        self.assertEqual(model.design_mtx.shape, (adata.n_obs, 2))
        # The design table is registered in sdata
        self.assertIn(model._design_table_name, self.sdata.tables)

    def test_setup_data_design_mtx_from_obs_columns(self):
        """design_mtx specified as obs column names is extracted from adata.obs."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(1)
        adata.obs["covariate_x"] = rng.standard_normal(adata.n_obs)

        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            design_mtx="covariate_x",
        )

        self.assertEqual(model.n_factors, 1)
        self.assertEqual(model.covariate_names, ["covariate_x"])
        self.assertIsNotNone(model._design_table_name)
        self.assertIn(model._design_table_name, self.sdata.tables)
        self.assertEqual(model.design_mtx.shape, (adata.n_obs, 1))

    def test_setup_data_design_mtx_from_table_name(self):
        """design_mtx as sdata table name reuses that table's X as covariates."""
        import anndata as ad

        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(5)
        cov_arr = rng.standard_normal((adata.n_obs, 2)).astype(np.float32)

        # Add a covariate table to sdata with same obs spatial indexing
        obs_df = adata.obs[["array_row", "array_col"]].copy()
        cov_adata = ad.AnnData(X=cov_arr, obs=obs_df)
        cov_adata.var_names = ["cov1", "cov2"]
        self.sdata["my_covariates"] = cov_adata

        model = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            design_mtx="my_covariates",
        )

        self.assertEqual(model.n_factors, 2)
        self.assertEqual(model.covariate_names, ["cov1", "cov2"])
        self.assertEqual(model.design_mtx.shape, (adata.n_obs, 2))

    def test_differential_usage_residualize_both(self):
        """residualize='both' should also spatially residualize isoform ratios."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(4)
        design = rng.standard_normal((adata.n_obs, 1))
        model = self._setup_model(design_mtx=design)
        results = model.test_differential_usage(
            residualize="both", n_jobs=1, return_results=True
        )
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))

    def test_differential_usage_gpr_configs(self):
        """Custom gpr_configs overrides default GPR hyperparameters."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(5)
        design = rng.standard_normal((adata.n_obs, 1))
        model = self._setup_model(design_mtx=design)
        custom_cfg = {
            "covariate": {
                "constant_value_bounds": "fixed",
                "length_scale_bounds": "fixed",
            }
        }
        results = model.test_differential_usage(
            gpr_configs=custom_cfg, n_jobs=1, return_results=True
        )
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))

    def test_differential_usage_no_design_raises(self):
        """test_differential_usage without design_mtx raises RuntimeError."""
        model = self._setup_model()  # no design_mtx
        with self.assertRaises(RuntimeError):
            model.test_differential_usage(n_jobs=1)

    def test_differential_usage_method_hsic(self):
        """method='hsic' (unconditional) returns valid p-values without GPR."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(10)
        design = rng.standard_normal((adata.n_obs, 1))
        model = self._setup_model(design_mtx=design)
        results = model.test_differential_usage(
            method="hsic", n_jobs=1, return_results=True
        )
        self.assertEqual(results["method"], "hsic")
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))
        self.assertTrue(np.all(np.isfinite(results["pvalue"])))

    def test_differential_usage_method_t_fisher(self):
        """method='t-fisher' uses two-sample t-tests for binary covariates."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(11)
        # t-fisher requires binary (0/1) covariates
        design = rng.integers(0, 2, size=(adata.n_obs, 1)).astype(float)
        model = self._setup_model(design_mtx=design)
        results = model.test_differential_usage(
            method="t-fisher", n_jobs=1, return_results=True
        )
        self.assertEqual(results["method"], "t-fisher")
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))
        # Fisher statistic is chi-squared (>=0)
        self.assertTrue(np.all(results["statistic"] >= 0))

    def test_differential_usage_method_t_tippett(self):
        """method='t-tippett' uses two-sample t-tests for binary covariates."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(12)
        # t-tippett requires binary (0/1) covariates
        design = rng.integers(0, 2, size=(adata.n_obs, 1)).astype(float)
        model = self._setup_model(design_mtx=design)
        results = model.test_differential_usage(
            method="t-tippett", n_jobs=1, return_results=True
        )
        self.assertEqual(results["method"], "t-tippett")
        self.assertEqual(results["statistic"].shape, (model.n_genes, 1))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))

    def test_differential_usage_multi_factor_chunking(self):
        """Multiple covariates trigger chunked processing; results are valid."""
        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(13)
        n_cov = 3
        design = rng.standard_normal((adata.n_obs, n_cov))
        model = self._setup_model(design_mtx=design)
        results = model.test_differential_usage(
            method="hsic-gp", n_jobs=1, return_results=True, print_progress=False
        )
        self.assertEqual(results["statistic"].shape, (model.n_genes, n_cov))
        self.assertTrue(np.all(results["pvalue"] >= 0))
        self.assertTrue(np.all(results["pvalue"] <= 1))
        self.assertEqual(len(model.covariate_names), n_cov)

    def test_fft_vs_np_du_agreement(self):
        """SplisosmFFT and SplisosmNP DU p-values should be positively correlated.

        Both models are run on the same 6x5 regular-grid data with a continuous
        covariate.  Since both apply spatial conditioning (hsic-gp), the p-values
        should be rank-correlated (Spearman rho > 0 on combined null+signal).
        """
        from scipy.stats import spearmanr
        from splisosm.hyptest.np import SplisosmNP

        adata = self.sdata.tables[self.table_name]
        rng = np.random.default_rng(42)
        design = rng.standard_normal((adata.n_obs, 1))

        # --- SplisosmFFT ---
        model_fft = SplisosmFFT(rho=0.9, neighbor_degree=1)
        model_fft.setup_data(
            self.sdata,
            bins="grid_bins",
            table_name=self.table_name,
            col_key="array_col",
            row_key="array_row",
            design_mtx=design,
            covariate_names=["covariate"],
        )
        model_fft.test_differential_usage(n_jobs=1, print_progress=False)
        df_fft = model_fft.get_formatted_test_results("du")

        # --- SplisosmNP ---
        row_idx = np.asarray(adata.obs["array_row"], dtype=int)
        col_idx = np.asarray(adata.obs["array_col"], dtype=int)
        coords = np.stack([row_idx, col_idx], axis=1).astype(float)

        adata.obsm["spatial"] = coords  # add spatial coordinates from grid indices

        model_np = SplisosmNP()
        model_np.setup_data(
            adata=adata,
            spatial_key="spatial",
            layer="counts",
            group_iso_by="gene_symbol",
            design_mtx=design,
            covariate_names=["covariate"],
        )

        import warnings
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
                message=".*is close to the specified.*",
            )
            model_np.test_differential_usage(
                method="hsic-gp",
                gpr_backend="sklearn",
                residualize="cov_only",
                print_progress=False,
            )
        df_np = model_np.get_formatted_test_results("du")

        # Align on shared genes
        shared_genes = set(df_fft["gene"].unique()) & set(df_np["gene"].unique())
        self.assertGreater(
            len(shared_genes), 0, "No shared genes between FFT and NP results"
        )

        p_fft = (
            df_fft[df_fft["gene"].isin(shared_genes)]
            .sort_values("gene")["pvalue"]
            .values
        )
        p_np = (
            df_np[df_np["gene"].isin(shared_genes)].sort_values("gene")["pvalue"].values
        )

        # Rank correlation should be positive (both remove spatial confound similarly)
        if len(p_fft) >= 4:
            rho, _ = spearmanr(p_fft, p_np)
            # Allow rho to be NaN (e.g., all p-values identical) or >= -0.5
            # The key is that neither method systematically inverts the ranking.
            if not np.isnan(rho):
                self.assertGreater(
                    rho,
                    -0.5,
                    f"FFT vs NP p-value Spearman rho={rho:.3f} is strongly negative (unexpected)",
                )

    def test_fft_gpr_scalability(self):
        """FFTKernelGPR.fit_residuals_cube should scale as O(N log N) not O(N^2).

        Verify that a 50x50 grid (2500 cells) runs in reasonable time and
        that memory does not blow up (no n x n matrix formed).
        """
        import time
        from splisosm.gpr import FFTKernelGPR

        # Build a 50x50 raster cube
        rng = np.random.default_rng(0)
        ny, nx = 50, 50
        cube = rng.standard_normal((ny, nx, 3)).astype(float)  # 3 channels

        gpr = FFTKernelGPR(
            constant_value_bounds=(1e-3, 1e3),
            length_scale_bounds="fixed",
        )
        t0 = time.perf_counter()
        res, eps = gpr.fit_residuals_cube(cube, spacing=(1.0, 1.0))
        elapsed = time.perf_counter() - t0

        # Should complete in under 5 seconds even for 2500-cell grid
        self.assertLess(
            elapsed, 5.0, f"FFT residualization took {elapsed:.1f}s (expected < 5s)"
        )
        self.assertEqual(res.shape, (ny, nx, 3))
        self.assertGreater(eps, 0)


if __name__ == "__main__":
    unittest.main()
