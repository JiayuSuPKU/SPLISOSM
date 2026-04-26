__all__ = ["SplisosmNP", "SplisosmGLMM", "SplisosmFFT", "__version__"]


def __getattr__(name: str) -> object:
    """Lazily import public model classes and optional backend dependencies."""
    if name == "SplisosmNP":
        from splisosm.hyptest_np import SplisosmNP

        return SplisosmNP
    if name == "SplisosmGLMM":
        from splisosm.hyptest_glmm import SplisosmGLMM

        return SplisosmGLMM
    if name == "SplisosmFFT":
        from splisosm.hyptest_fft import SplisosmFFT

        return SplisosmFFT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("splisosm")
    except PackageNotFoundError:
        __version__ = "unknown"
