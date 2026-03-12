from splisosm.hyptest_np import SplisosmNP
from splisosm.hyptest_glmm import SplisosmGLMM
from splisosm.hyptest_fft import SplisosmFFT

__all__ = ["SplisosmNP", "SplisosmGLMM", "SplisosmFFT", "__version__"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("splisosm")
except PackageNotFoundError:
    __version__ = "unknown"
