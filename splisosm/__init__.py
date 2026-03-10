from splisosm.hyptest_np import SplisosmNP
from splisosm.hyptest_glmm import SplisosmGLMM

__all__ = ["SplisosmNP", "SplisosmGLMM", "__version__"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("splisosm")
except PackageNotFoundError:
    __version__ = "unknown"
