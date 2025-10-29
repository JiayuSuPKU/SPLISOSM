from splisosm.hyptest_np import SplisosmNP
from splisosm.hyptest_glmm import SplisosmGLMM

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore
try:
    __version__ = version("splisosm")
except PackageNotFoundError:
    __version__ = "unknown"