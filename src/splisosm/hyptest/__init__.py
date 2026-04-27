"""Hypothesis-test model classes.

The top-level imports remain the primary user API:

>>> from splisosm import SplisosmNP, SplisosmFFT, SplisosmGLMM

Advanced users may import concrete implementations from
``splisosm.hyptest.np``, ``splisosm.hyptest.fft``, or
``splisosm.hyptest.glmm``. The package-level re-exports below are lazy so that
requesting one model class does not import optional dependencies for the
others.
"""

__all__ = ["SplisosmNP", "SplisosmFFT", "SplisosmGLMM"]


def __getattr__(name: str) -> object:
    """Lazily import hypothesis-test model classes."""
    if name == "SplisosmNP":
        from splisosm.hyptest.np import SplisosmNP

        return SplisosmNP
    if name == "SplisosmFFT":
        from splisosm.hyptest.fft import SplisosmFFT

        return SplisosmFFT
    if name == "SplisosmGLMM":
        from splisosm.hyptest.glmm import SplisosmGLMM

        return SplisosmGLMM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
