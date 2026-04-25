from splisosm.hyptest_np import SplisosmNP
from splisosm.hyptest_glmm import SplisosmGLMM
from splisosm.hyptest_fft import SplisosmFFT

__all__ = ["SplisosmNP", "SplisosmGLMM", "SplisosmFFT", "__version__"]

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("splisosm")
    except PackageNotFoundError:
        __version__ = "unknown"
