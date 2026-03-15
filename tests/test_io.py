import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

_IO_PATH = Path(__file__).resolve().parents[1] / "splisosm" / "io.py"
_IO_SPEC = importlib.util.spec_from_file_location("splisosm_io_test_module", _IO_PATH)
if _IO_SPEC is None or _IO_SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec for {_IO_PATH}")
io_mod = importlib.util.module_from_spec(_IO_SPEC)
_IO_SPEC.loader.exec_module(io_mod)

load_visiumhd_probe = io_mod.load_visiumhd_probe
load_xenium_codeword = io_mod.load_xenium_codeword


class TestIO(unittest.TestCase):
    def test_visiumhd_bin_name_and_resolution_formatters(self):
        self.assertEqual(io_mod._normalize_visiumhd_bin_name(8), "square_008um")
        self.assertEqual(
            io_mod._normalize_visiumhd_bin_name("square_016um"), "square_016um"
        )
        self.assertEqual(io_mod._normalize_visiumhd_bin_name("2"), "square_002um")
        with self.assertRaisesRegex(ValueError, "Unsupported bin size format"):
            io_mod._normalize_visiumhd_bin_name("bad")

        self.assertEqual(io_mod._format_resolution_bin_name(8.0), "square_008um")
        self.assertEqual(io_mod._format_resolution_bin_name(1.5), "square_1p5um")
        self.assertEqual(io_mod._format_resolution_obs_token(2.0), "002um")
        self.assertEqual(io_mod._format_resolution_obs_token(2.5), "2p5um")
        self.assertEqual(
            io_mod._format_resolution_shape_name(16.0), "square_016um_bins"
        )

    def test_aggregate_visiumhd_probe_counts_no_valid_mapping(self):
        adata_2um = AnnData(
            X=scipy.sparse.csr_matrix(np.array([[1.0, 2.0]], dtype=np.float32)),
            obs=pd.DataFrame(index=["s_002um_00000_00000-1"]),
            var=pd.DataFrame(index=["p1", "p2"]),
        )
        mapping = pd.DataFrame(
            {
                "square_002um": ["missing_barcode"],
                "square_008um": ["s_008um_00000_00000-1"],
            }
        )
        out = io_mod._aggregate_visiumhd_probe_counts(
            adata_2um=adata_2um,
            barcode_mappings=mapping,
            source_col="square_002um",
            target_col="square_008um",
            target_obs_names=pd.Index(["s_008um_00000_00000-1"]),
        )
        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(out.nnz, 0)

    def test_xenium_locate_outs_path_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Case 1: transcripts exists at root
            (root / "transcripts.zarr.zip").touch()
            self.assertEqual(io_mod._xenium_locate_outs_path(root), root)

            # Case 2: transcripts exists under outs/
            (root / "transcripts.zarr.zip").unlink()
            (root / "outs").mkdir()
            (root / "outs" / "transcripts.zarr.zip").touch()
            self.assertEqual(io_mod._xenium_locate_outs_path(root), root / "outs")

            # Case 3: missing transcripts
            (root / "outs" / "transcripts.zarr.zip").unlink()
            with self.assertRaisesRegex(
                ValueError, "Could not locate transcripts.zarr.zip"
            ):
                io_mod._xenium_locate_outs_path(root)

        with self.assertRaisesRegex(ValueError, "Path does not exist"):
            io_mod._xenium_locate_outs_path(Path("/definitely/not/a/real/path"))

    def test_chunk_to_codeword_triplets_filters_and_counts(self):
        class _FakeArray:
            def __init__(self, arr):
                self._arr = np.asarray(arr)

            def __getitem__(self, item):
                return self._arr[item]

        chunk = {
            "quality_score": _FakeArray([30.0, 10.0, 40.0, 30.0]),
            "codeword_identity": _FakeArray([[0], [1], [2], [0]]),
            "location": _FakeArray(
                [
                    [0.1, 0.1],
                    [0.2, 0.2],
                    [0.8, 0.8],
                    [1.2, 1.2],
                ]
            ),
        }
        rows, cols, vals = io_mod._chunk_to_codeword_triplets(
            chunk=chunk,
            codeword_to_target_col=np.array([0, 1, -1], dtype=np.int64),
            quality_threshold=20.0,
            x_bins=np.array([0.0, 1.0]),
            y_bins=np.array([0.0, 1.0]),
            nrows=1,
            ncols=1,
        )
        np.testing.assert_array_equal(rows, np.array([0]))
        np.testing.assert_array_equal(cols, np.array([0]))
        np.testing.assert_array_equal(vals, np.array([1.0], dtype=np.float32))

        bad_chunk = {
            "quality_score": _FakeArray([30.0]),
            "codeword_identity": _FakeArray([[0]]),
            "location": _FakeArray([0.1, 0.1]),
        }
        rows, cols, vals = io_mod._chunk_to_codeword_triplets(
            chunk=bad_chunk,
            codeword_to_target_col=np.array([0], dtype=np.int64),
            quality_threshold=20.0,
            x_bins=np.array([0.0, 1.0]),
            y_bins=np.array([0.0, 1.0]),
            nrows=1,
            ncols=1,
        )
        self.assertEqual(rows.size, 0)
        self.assertEqual(cols.size, 0)
        self.assertEqual(vals.size, 0)

    def test_load_visiumhd_probe_early_validation_errors(self):
        scanpy_mod = types.ModuleType("scanpy")
        scanpy_mod.read_10x_h5 = lambda *args, **kwargs: AnnData(
            X=scipy.sparse.csr_matrix((1, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["s_002um_00000_00000-1"]),
            var=pd.DataFrame(index=["probe_1"]),
        )

        sdio_mod = types.ModuleType("spatialdata_io")
        sdio_readers_mod = types.ModuleType("spatialdata_io.readers")
        sdio_visium_hd_mod = types.ModuleType("spatialdata_io.readers.visium_hd")
        sdio_visium_hd_mod.visium_hd = lambda **kwargs: types.SimpleNamespace(tables={})

        with patch.dict(
            sys.modules,
            {
                "scanpy": scanpy_mod,
                "spatialdata_io": sdio_mod,
                "spatialdata_io.readers": sdio_readers_mod,
                "spatialdata_io.readers.visium_hd": sdio_visium_hd_mod,
            },
        ):
            with self.assertRaisesRegex(ValueError, "Path does not exist"):
                load_visiumhd_probe(Path("/definitely/not/a/real/path"))

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                with self.assertRaisesRegex(ValueError, "Cannot find binned_outputs"):
                    load_visiumhd_probe(root)

                (root / "binned_outputs").mkdir()
                with self.assertRaisesRegex(ValueError, r"No square_\*um bins found"):
                    load_visiumhd_probe(root)

                (root / "binned_outputs" / "square_008um").mkdir()
                with self.assertRaisesRegex(
                    ValueError, "square_002um is required but not found"
                ):
                    load_visiumhd_probe(root)

    def test_load_xenium_codeword_validation_errors(self):
        class _FakeZipStore:
            def __init__(self, *_args, **_kwargs):
                self.closed = False

            def close(self):
                self.closed = True

        fake_zarr = types.ModuleType("zarr")
        fake_zarr.open = lambda *_args, **_kwargs: None
        fake_zarr_storage = types.ModuleType("zarr.storage")
        fake_zarr_storage.ZipStore = _FakeZipStore

        sdio_mod = types.ModuleType("spatialdata_io")
        sdio_readers_mod = types.ModuleType("spatialdata_io.readers")
        sdio_xenium_mod = types.ModuleType("spatialdata_io.readers.xenium")
        sdio_xenium_mod.xenium = lambda **_kwargs: types.SimpleNamespace(
            tables={}, shapes={}
        )

        spatialdata_mod = types.ModuleType("spatialdata")
        spatialdata_models_mod = types.ModuleType("spatialdata.models")

        class _FakeTableModel:
            @staticmethod
            def parse(adata, **_kwargs):
                return adata

        spatialdata_models_mod.TableModel = _FakeTableModel

        with patch.dict(
            sys.modules,
            {
                "zarr": fake_zarr,
                "zarr.storage": fake_zarr_storage,
                "spatialdata_io": sdio_mod,
                "spatialdata_io.readers": sdio_readers_mod,
                "spatialdata_io.readers.xenium": sdio_xenium_mod,
                "spatialdata": spatialdata_mod,
                "spatialdata.models": spatialdata_models_mod,
            },
        ):
            with self.assertRaisesRegex(
                ValueError, "`spatial_resolutions` cannot be empty"
            ):
                load_xenium_codeword(
                    Path("."),
                    spatial_resolutions=[],
                    create_square_shapes=False,
                )

            with self.assertRaisesRegex(ValueError, "`chunk_batch_size` must be > 0"):
                load_xenium_codeword(
                    Path("."),
                    chunk_batch_size=0,
                    create_square_shapes=False,
                )

            with self.assertRaisesRegex(ValueError, "`quality_threshold` must be >= 0"):
                load_xenium_codeword(
                    Path("."),
                    quality_threshold=-1.0,
                    create_square_shapes=False,
                )

            with self.assertRaisesRegex(
                ValueError, "All `spatial_resolutions` values must be > 0"
            ):
                load_xenium_codeword(
                    Path("."),
                    spatial_resolutions=[0.0],
                    create_square_shapes=False,
                )

    def test_load_xenium_codeword_missing_density_codeword_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outs = root / "outs"
            outs.mkdir(parents=True)
            (outs / "transcripts.zarr.zip").touch()

            class _FakeDensityGroup:
                def __getitem__(self, key):
                    raise KeyError(key)

            class _FakeRoot:
                def __getitem__(self, key):
                    if key == "density":
                        return _FakeDensityGroup()
                    raise KeyError(key)

            class _FakeZipStore:
                def __init__(self, *_args, **_kwargs):
                    self.closed = False

                def close(self):
                    self.closed = True

            fake_zarr = types.ModuleType("zarr")
            fake_zarr.open = lambda *_args, **_kwargs: _FakeRoot()
            fake_zarr_storage = types.ModuleType("zarr.storage")
            fake_zarr_storage.ZipStore = _FakeZipStore

            sdio_mod = types.ModuleType("spatialdata_io")
            sdio_readers_mod = types.ModuleType("spatialdata_io.readers")
            sdio_xenium_mod = types.ModuleType("spatialdata_io.readers.xenium")
            sdio_xenium_mod.xenium = lambda **_kwargs: types.SimpleNamespace(
                tables={}, shapes={}
            )

            spatialdata_mod = types.ModuleType("spatialdata")
            spatialdata_models_mod = types.ModuleType("spatialdata.models")

            class _FakeTableModel:
                @staticmethod
                def parse(adata, **_kwargs):
                    return adata

            spatialdata_models_mod.TableModel = _FakeTableModel

            with patch.dict(
                sys.modules,
                {
                    "zarr": fake_zarr,
                    "zarr.storage": fake_zarr_storage,
                    "spatialdata_io": sdio_mod,
                    "spatialdata_io.readers": sdio_readers_mod,
                    "spatialdata_io.readers.xenium": sdio_xenium_mod,
                    "spatialdata": spatialdata_mod,
                    "spatialdata.models": spatialdata_models_mod,
                },
            ):
                with self.assertRaisesRegex(
                    ValueError, "Could not find `density/codeword` group"
                ):
                    load_xenium_codeword(
                        root,
                        spatial_resolutions=[8.0],
                        create_square_shapes=False,
                        show_progress=False,
                    )

    def test_load_visiumhd_probe_wrapper_probe_binning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "binned_outputs" / "square_002um").mkdir(parents=True)
            (root / "binned_outputs" / "square_008um").mkdir(parents=True)
            (root / "binned_outputs" / "square_016um").mkdir(parents=True)
            (
                root / "binned_outputs" / "square_002um" / "raw_probe_bc_matrix.h5"
            ).touch()
            (
                root
                / "binned_outputs"
                / "square_002um"
                / "filtered_feature_bc_matrix.h5"
            ).touch()
            (root / "barcode_mappings.parquet").touch()

            probe_obs = [
                "s_002um_00000_00000-1",
                "s_002um_00000_00001-1",
                "s_002um_00001_00000-1",
                "s_002um_00001_00001-1",
            ]
            probe_var = ["probe_1", "probe_2"]
            probe_x = scipy.sparse.csr_matrix(
                np.array(
                    [
                        [1.0, 0.0],
                        [2.0, 1.0],
                        [3.0, 0.0],
                        [4.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            )
            probe_adata = AnnData(
                X=probe_x,
                obs=pd.DataFrame(index=probe_obs),
                var=pd.DataFrame(index=probe_var),
            )

            class _FakeSData:
                def __init__(self):
                    self.tables = {
                        "square_002um": AnnData(
                            X=scipy.sparse.csr_matrix((3, 1), dtype=np.float32),
                            # Simulate in-tissue 2um barcodes loaded by visium_hd.
                            obs=pd.DataFrame(index=probe_obs[:3]),
                            var=pd.DataFrame(index=["gene_a"]),
                        ),
                        "square_008um": AnnData(
                            X=scipy.sparse.csr_matrix((2, 1), dtype=np.float32),
                            obs=pd.DataFrame(
                                index=["s_008um_00000_00000-1", "s_008um_00000_00001-1"]
                            ),
                            var=pd.DataFrame(index=["gene_a"]),
                        ),
                        "square_016um": AnnData(
                            X=scipy.sparse.csr_matrix((1, 1), dtype=np.float32),
                            obs=pd.DataFrame(index=["s_016um_00000_00000-1"]),
                            var=pd.DataFrame(index=["gene_a"]),
                        ),
                    }

            def _fake_visium_hd(**kwargs):
                return _FakeSData()

            mapping_df = pd.DataFrame(
                {
                    "square_002um": probe_obs,
                    "square_008um": [
                        "s_008um_00000_00000-1",
                        "s_008um_00000_00000-1",
                        "s_008um_00000_00001-1",
                        "s_008um_00000_00001-1",
                    ],
                    "square_016um": ["s_016um_00000_00000-1"] * 4,
                }
            )

            scanpy_mod = types.ModuleType("scanpy")
            scanpy_mod.read_10x_h5 = lambda *args, **kwargs: probe_adata

            sdio_mod = types.ModuleType("spatialdata_io")
            sdio_readers_mod = types.ModuleType("spatialdata_io.readers")
            sdio_visium_hd_mod = types.ModuleType("spatialdata_io.readers.visium_hd")
            sdio_visium_hd_mod.visium_hd = _fake_visium_hd

            with patch.dict(
                sys.modules,
                {
                    "scanpy": scanpy_mod,
                    "spatialdata_io": sdio_mod,
                    "spatialdata_io.readers": sdio_readers_mod,
                    "spatialdata_io.readers.visium_hd": sdio_visium_hd_mod,
                },
            ):
                with patch.object(io_mod.pd, "read_parquet", return_value=mapping_df):
                    out_filtered = load_visiumhd_probe(
                        root,
                        bin_sizes=[2, 8, 16],
                        counts_layer_name="counts",
                    )
                    out_unfiltered = load_visiumhd_probe(
                        root,
                        bin_sizes=[2, 8, 16],
                        filtered_counts_file=False,
                        counts_layer_name="counts",
                    )

            x_008_filtered = out_filtered.tables["square_008um"].X.toarray()
            x_016_filtered = out_filtered.tables["square_016um"].X.toarray()
            x_008_unfiltered = out_unfiltered.tables["square_008um"].X.toarray()
            x_016_unfiltered = out_unfiltered.tables["square_016um"].X.toarray()

            # Default filtered_counts_file=True keeps only first 3 source barcodes.
            np.testing.assert_allclose(
                x_008_filtered, np.array([[3.0, 1.0], [3.0, 0.0]])
            )
            np.testing.assert_allclose(x_016_filtered, np.array([[6.0, 1.0]]))

            # filtered_counts_file=False uses all 4 source barcodes.
            np.testing.assert_allclose(
                x_008_unfiltered, np.array([[3.0, 1.0], [7.0, 1.0]])
            )
            np.testing.assert_allclose(x_016_unfiltered, np.array([[10.0, 2.0]]))

            self.assertIn("counts", out_filtered.tables["square_008um"].layers)
            self.assertEqual(
                list(out_filtered.tables["square_002um"].var_names), probe_var
            )

            # 2um bin should be populated directly from adata_2um without aggregation.
            # filtered run: only first 3 source barcodes are in-tissue.
            x_002_filtered = out_filtered.tables["square_002um"].X.toarray()
            np.testing.assert_allclose(x_002_filtered, probe_x.toarray()[:3])

    def test_load_xenium_codeword_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outs = root / "outs"
            outs.mkdir(parents=True)
            (outs / "transcripts.zarr.zip").touch()

            class _FakeArray:
                def __init__(self, arr):
                    self._arr = np.asarray(arr)

                def __getitem__(self, item):
                    return self._arr[item]

            class _FakeChunk(dict):
                pass

            class _FakeGroup:
                def __init__(self, attrs):
                    self.attrs = attrs

            class _FakeDensityGene:
                attrs = {
                    "grid_size": [1.0, 1.0],
                    "origin": {"x": 0.0, "y": 0.0},
                    "rows": 1,
                    "cols": 1,
                }

            class _FakeDensityCodeword:
                attrs = {
                    "codeword_names": [
                        "GeneA",
                        "GeneB",
                        "UnassignedCodeword_0",
                    ]
                }

            class _FakeDensityGroup:
                def __getitem__(self, key):
                    if key == "gene":
                        return _FakeDensityGene()
                    if key == "codeword":
                        return _FakeDensityCodeword()
                    raise KeyError(key)

            class _FakeRoot:
                def __init__(self):
                    self._chunks = {
                        "grids/0/chunk_0": _FakeChunk(
                            {
                                "quality_score": _FakeArray([30.0, 25.0, 10.0]),
                                "codeword_identity": _FakeArray([[0], [1], [0]]),
                                "location": _FakeArray(
                                    [
                                        [0.2, 0.2],
                                        [0.8, 0.8],
                                        [0.4, 0.4],
                                    ]
                                ),
                            }
                        )
                    }

                def __contains__(self, key):
                    return key in self._chunks

                def __getitem__(self, key):
                    if key == "grids":
                        return _FakeGroup(attrs={"grid_keys": [["chunk_0"]]})
                    if key == "density":
                        return _FakeDensityGroup()
                    return self._chunks[key]

            class _FakeZipStore:
                def __init__(self, *_args, **_kwargs):
                    self.closed = False

                def close(self):
                    self.closed = True

            class _FakeSData:
                def __init__(self):
                    self.tables = {}
                    self.shapes = {}

            class _FakeTableModel:
                @staticmethod
                def parse(adata, **_kwargs):
                    return adata

            class _FakeShapesModel:
                @staticmethod
                def parse(geo_df, **_kwargs):
                    return geo_df

            class _FakeIdentity:
                pass

            def _fake_box(x0, y0, x1, y1):
                return (x0, y0, x1, y1)

            fake_zarr = types.ModuleType("zarr")
            fake_zarr.open = lambda *_args, **_kwargs: _FakeRoot()

            fake_zarr_storage = types.ModuleType("zarr.storage")
            fake_zarr_storage.ZipStore = _FakeZipStore

            sdio_mod = types.ModuleType("spatialdata_io")
            sdio_readers_mod = types.ModuleType("spatialdata_io.readers")
            sdio_xenium_mod = types.ModuleType("spatialdata_io.readers.xenium")
            sdio_xenium_mod.xenium = lambda **_kwargs: _FakeSData()

            spatialdata_mod = types.ModuleType("spatialdata")
            spatialdata_models_mod = types.ModuleType("spatialdata.models")
            spatialdata_models_mod.TableModel = _FakeTableModel
            spatialdata_models_mod.ShapesModel = _FakeShapesModel

            spatialdata_transformations_mod = types.ModuleType(
                "spatialdata.transformations"
            )
            spatialdata_transformations_transformations_mod = types.ModuleType(
                "spatialdata.transformations.transformations"
            )
            spatialdata_transformations_transformations_mod.Identity = _FakeIdentity

            geopandas_mod = types.ModuleType("geopandas")
            geopandas_mod.GeoDataFrame = lambda data, index=None: pd.DataFrame(
                data, index=index
            )

            shapely_mod = types.ModuleType("shapely")
            shapely_geometry_mod = types.ModuleType("shapely.geometry")
            shapely_geometry_mod.box = _fake_box

            with patch.dict(
                sys.modules,
                {
                    "zarr": fake_zarr,
                    "zarr.storage": fake_zarr_storage,
                    "spatialdata_io": sdio_mod,
                    "spatialdata_io.readers": sdio_readers_mod,
                    "spatialdata_io.readers.xenium": sdio_xenium_mod,
                    "spatialdata": spatialdata_mod,
                    "spatialdata.models": spatialdata_models_mod,
                    "spatialdata.transformations": spatialdata_transformations_mod,
                    "spatialdata.transformations.transformations": spatialdata_transformations_transformations_mod,
                    "geopandas": geopandas_mod,
                    "shapely": shapely_mod,
                    "shapely.geometry": shapely_geometry_mod,
                },
            ):
                out = load_xenium_codeword(
                    root,
                    spatial_resolutions=[1.0, 2.0],
                    quality_threshold=20.0,
                    n_jobs=1,
                    chunk_batch_size=1,
                    show_progress=False,
                )

            self.assertIn("square_001um", out.tables)
            self.assertIn("square_002um", out.tables)
            self.assertIn("square_001um_bins", out.shapes)
            self.assertIn("square_002um_bins", out.shapes)

            adata_1 = out.tables["square_001um"]
            adata_2 = out.tables["square_002um"]

            self.assertEqual(adata_1.shape, (1, 2))
            self.assertEqual(adata_2.shape, (1, 2))

            np.testing.assert_allclose(adata_1.X.toarray(), np.array([[1.0, 1.0]]))
            np.testing.assert_allclose(adata_2.X.toarray(), np.array([[1.0, 1.0]]))

            self.assertIn("counts", adata_1.layers)
            self.assertIn("counts", adata_2.layers)
            self.assertIn("array_row", adata_1.obs.columns)
            self.assertIn("array_col", adata_1.obs.columns)
            self.assertIn("gene_symbol", adata_1.var.columns)
            self.assertIn("spatial", adata_1.obsm)
            self.assertEqual(adata_1.obsm["spatial"].shape, (adata_1.n_obs, 2))


if __name__ == "__main__":
    unittest.main()
