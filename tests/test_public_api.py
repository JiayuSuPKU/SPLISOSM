"""Smoke tests for the curated public API modules."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest


def test_top_level_model_exports(monkeypatch):
    """Top-level model classes remain available through lazy exports."""
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")

    from splisosm import SplisosmFFT, SplisosmGLMM, SplisosmNP
    from splisosm.hyptest import (
        SplisosmFFT as HyptestFFT,
        SplisosmGLMM as HyptestGLMM,
        SplisosmNP as HyptestNP,
    )
    from splisosm.hyptest.fft import SplisosmFFT as ModuleFFT
    from splisosm.hyptest.glmm import SplisosmGLMM as ModuleGLMM
    from splisosm.hyptest.np import SplisosmNP as ModuleNP

    assert SplisosmNP.__name__ == "SplisosmNP"
    assert SplisosmFFT.__name__ == "SplisosmFFT"
    assert SplisosmGLMM.__name__ == "SplisosmGLMM"
    assert SplisosmNP is HyptestNP is ModuleNP
    assert SplisosmFFT is HyptestFFT is ModuleFFT
    assert SplisosmGLMM is HyptestGLMM is ModuleGLMM


def test_curated_helper_modules_import():
    """Focused helper modules expose the documented user-facing functions."""
    from splisosm.glmm import IsoDataset, MultinomGLM, MultinomGLMM
    from splisosm.gpr import FFTKernelGPR, NUFFTKernelGPR, make_kernel_gpr
    from splisosm.io import (
        load_visium_probe,
        load_visium_sp_meta,
        load_visiumhd_probe,
        load_xenium_codeword,
    )
    from splisosm.utils.preprocessing import (
        add_ratio_layer,
        auto_chunk_size,
        compute_feature_summaries,
        counts_to_ratios,
        prepare_inputs_from_anndata,
    )
    from splisosm.utils.stats import (
        false_discovery_control,
        linear_hsic_test,
        liu_sf_from_cumulants,
        run_hsic_gc,
        run_sparkx,
    )
    from splisosm.utils.simulation import simulate_isoform_counts

    assert callable(counts_to_ratios)
    assert callable(add_ratio_layer)
    assert callable(prepare_inputs_from_anndata)
    assert callable(compute_feature_summaries)
    assert callable(auto_chunk_size)
    assert callable(false_discovery_control)
    assert callable(run_hsic_gc)
    assert callable(run_sparkx)
    assert callable(linear_hsic_test)
    assert callable(liu_sf_from_cumulants)
    assert callable(simulate_isoform_counts)
    assert callable(load_visium_sp_meta)
    assert callable(load_visium_probe)
    assert callable(load_visiumhd_probe)
    assert callable(load_xenium_codeword)
    assert callable(make_kernel_gpr)
    assert IsoDataset.__name__ == "IsoDataset"
    assert MultinomGLM.__name__ == "MultinomGLM"
    assert MultinomGLMM.__name__ == "MultinomGLMM"
    assert FFTKernelGPR.__name__ == "FFTKernelGPR"
    assert NUFFTKernelGPR.__name__ == "NUFFTKernelGPR"


def test_utils_reexports_curated_helpers():
    """The legacy utility facade re-exports the focused helper APIs."""
    from splisosm.utils.preprocessing import counts_to_ratios
    from splisosm.utils.stats import run_hsic_gc
    from splisosm.utils import (
        counts_to_ratios as compat_counts_to_ratios,
        run_hsic_gc as compat_run_hsic_gc,
        simulate_isoform_counts,
    )

    assert compat_counts_to_ratios is counts_to_ratios
    assert compat_run_hsic_gc is run_hsic_gc
    assert callable(simulate_isoform_counts)


def test_removed_internal_module_paths_are_not_importable():
    """Moved private/internal modules are cleaned up instead of facaded."""
    removed_modules = [
        "splisosm.gpr.backends",
        "splisosm.gpr.statistics",
        "splisosm.utils.hsic_null",
        "splisosm.utils._hsic_null",
        "splisosm.glmm.model",
    ]
    for module_name in removed_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_optional_imports_stay_lazy_for_core_namespaces():
    """Core imports should not require optional FFT/GPR backends."""
    code = """
import builtins

blocked = {"spatialdata", "spatialdata_io", "gpytorch", "finufft"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise RuntimeError(f"unexpected optional import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import splisosm
from splisosm import SplisosmGLMM, SplisosmNP
import splisosm.gpr
from splisosm.gpr import GPyTorchKernelGPR, NUFFTKernelGPR, make_kernel_gpr

assert SplisosmNP.__name__ == "SplisosmNP"
assert SplisosmGLMM.__name__ == "SplisosmGLMM"
assert make_kernel_gpr.__name__ == "make_kernel_gpr"
assert GPyTorchKernelGPR.__name__ == "GPyTorchKernelGPR"
assert NUFFTKernelGPR.__name__ == "NUFFTKernelGPR"
"""
    env = {**os.environ, "PYTHONPATH": os.path.abspath("src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
