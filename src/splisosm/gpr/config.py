"""Default configuration and keyword validation sets for GPR backends."""

from __future__ import annotations

__all__ = ["_DEFAULT_GPR_CONFIGS"]


_DEFAULT_GP_PARAMS = {
    "constant_value": 1.0,
    "constant_value_bounds": (1e-3, 1e3),
    "length_scale": 1.0,
    "length_scale_bounds": "fixed",
}

_DEFAULT_GPR_SCALE_PARAMS = {
    # sklearn: subset-of-data size for hyperparameter fitting.
    # gpytorch: FITC inducing-point count.
    "n_inducing": 5_000,
}

_DEFAULT_NUFFT_PARAMS = {
    "epsilon_bounds": (1e-5, 1e1),
    "n_modes": None,
    "max_auto_modes": None,
    "nufft_eps": 1e-6,
    "nufft_opts": None,
    "lml_approx_rank": 256,
    "lml_exact_max_n": 512,
    "eigsh_tol": 1e-4,
    "period_margin": 0.5,
    "cg_rtol": 1e-5,
    "cg_maxiter": None,
    "workers": None,
}

_DEFAULT_GPR_CONFIGS = {
    # Covariate GPR: optimize signal amplitude via MLE; length scale fixed
    # at 1.0 because coordinates are z-score normalized before fitting.
    "covariate": {
        **_DEFAULT_GP_PARAMS,
        **_DEFAULT_GPR_SCALE_PARAMS,
        **_DEFAULT_NUFFT_PARAMS,
    },
    # Isoform GPR: used only when residualize='both'.
    "isoform": {
        **_DEFAULT_GP_PARAMS,
        **_DEFAULT_GPR_SCALE_PARAMS,
        **_DEFAULT_NUFFT_PARAMS,
    },
}

_COMMON_GPR_KWARGS = set(_DEFAULT_GP_PARAMS)
_SKLEARN_GPR_KWARGS = _COMMON_GPR_KWARGS | {"n_inducing"}
_GPYTORCH_GPR_KWARGS = _COMMON_GPR_KWARGS | {"n_inducing", "n_iter", "lr", "device"}
_FFT_GPR_KWARGS = _COMMON_GPR_KWARGS | {"epsilon_bounds", "workers"}
_NUFFT_GPR_KWARGS = _COMMON_GPR_KWARGS | set(_DEFAULT_NUFFT_PARAMS)
_KNOWN_GPR_KWARGS = (
    _SKLEARN_GPR_KWARGS | _GPYTORCH_GPR_KWARGS | _FFT_GPR_KWARGS | _NUFFT_GPR_KWARGS
)
