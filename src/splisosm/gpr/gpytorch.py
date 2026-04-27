"""GPyTorch GP residualization backend."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from splisosm.gpr.base import KernelGPR

__all__ = ["GPyTorchKernelGPR"]


class _GPyTorchExactGPModel:
    """Internal helper: wraps a gpytorch ExactGP with ScaleKernel(RBFKernel).

    Constructed lazily to avoid importing gpytorch at module load time.
    """

    @staticmethod
    def build(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Any,
        constant_value: float,
        constant_value_bounds: Union[tuple, str],
        length_scale: float,
        length_scale_bounds: Union[tuple, str],
    ) -> Any:
        """Build and return a gpytorch ExactGP model."""
        import gpytorch

        class _Model(gpytorch.models.ExactGP):
            def __init__(self) -> None:
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                # Set initial values
                covar.outputscale = constant_value
                covar.base_kernel.lengthscale = length_scale
                # Apply bounds / fix parameters
                if constant_value_bounds == "fixed":
                    covar.raw_outputscale.requires_grad_(False)
                else:
                    lo, hi = constant_value_bounds
                    covar.register_constraint(
                        "raw_outputscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    covar.outputscale = constant_value  # re-apply after constraint
                if length_scale_bounds == "fixed":
                    covar.base_kernel.raw_lengthscale.requires_grad_(False)
                else:
                    lo, hi = length_scale_bounds
                    covar.base_kernel.register_constraint(
                        "raw_lengthscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    covar.base_kernel.lengthscale = length_scale
                self.covar_module = covar

            def forward(self, x: torch.Tensor) -> Any:
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return _Model()

    @staticmethod
    def build_sparse(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Any,
        inducing_points: torch.Tensor,
        constant_value: float,
        constant_value_bounds: Union[tuple, str],
        length_scale: float,
        length_scale_bounds: Union[tuple, str],
    ) -> Any:
        """Build a GPyTorch SGPR model (FITC) using InducingPointKernel."""
        import gpytorch

        class _SparseModel(gpytorch.models.ExactGP):
            def __init__(self) -> None:
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                # Set initial values
                base.outputscale = constant_value
                base.base_kernel.lengthscale = length_scale
                # Apply bounds / fix parameters — mirror build() exactly
                if constant_value_bounds == "fixed":
                    base.raw_outputscale.requires_grad_(False)
                else:
                    lo, hi = constant_value_bounds
                    base.register_constraint(
                        "raw_outputscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    base.outputscale = constant_value  # re-apply after constraint
                if length_scale_bounds == "fixed":
                    base.base_kernel.raw_lengthscale.requires_grad_(False)
                else:
                    lo, hi = length_scale_bounds
                    base.base_kernel.register_constraint(
                        "raw_lengthscale",
                        gpytorch.constraints.Interval(lo, hi),
                    )
                    base.base_kernel.lengthscale = length_scale
                self.covar_module = gpytorch.kernels.InducingPointKernel(
                    base,
                    inducing_points=inducing_points,
                    likelihood=likelihood,
                )

            def forward(self, x: torch.Tensor) -> Any:
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return _SparseModel()


def _subsample_inducing(X: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """Uniformly sub-sample m rows from X as inducing points."""
    idx = np.random.default_rng(seed).choice(len(X), m, replace=False)
    return X[torch.from_numpy(idx)]


class GPyTorchKernelGPR(KernelGPR):
    """GPyTorch GP residualizer.

    Uses GPyTorch's lazy-tensor infrastructure so that all linear-system
    solves during training use Lanczos-based conjugate gradients rather
    than dense Cholesky.  Hyperparameters are optimized with Adam on the
    marginal log-likelihood.

    For moderate datasets, the exact-GP path is used. Passing
    ``n_inducing > 0`` enables the FITC sparse GP approximation via
    InducingPointKernel.

    Parameters
    ----------
    constant_value : float
        Initial signal amplitude (outputscale).
    constant_value_bounds : tuple or "fixed"
        Search bounds for the signal amplitude.  ``"fixed"`` disables
        optimization of this parameter.
    length_scale : float
        Initial RBF length scale.
    length_scale_bounds : tuple or "fixed"
        Search bounds for the length scale.
    n_inducing : int or None
        Number of inducing points for FITC sparse GP approximation. When set,
        memory scales as ``O(n_spots * n_inducing)`` rather than
        ``O(n_spots^2)``. Set to ``None`` to use exact GP.
    n_iter : int
        Adam optimizer iterations for hyperparameter learning.
    lr : float
        Adam learning rate.
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Notes
    -----
    Requires ``gpytorch`` (optional dependency)::

        pip install gpytorch

    Hyperparameter search uses the mean of all target columns as the
    training signal, then applies the fitted kernel to every column.
    This matches the sklearn backend behavior and amortizes GP fitting across
    output columns.

    The noise (epsilon) search is handled by GPyTorch's
    ``GaussianLikelihood``, which is always optimized regardless of the
    signal bounds setting.
    """

    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Union[tuple, str] = (1e-3, 1e3),
        length_scale: float = 1.0,
        length_scale_bounds: Union[tuple, str] = "fixed",
        n_inducing: Optional[int] = None,
        n_iter: int = 100,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        try:
            import gpytorch as _gpytorch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GPyTorchKernelGPR requires gpytorch. "
                "Install it with:  pip install gpytorch"
            ) from exc
        self._constant_value = constant_value
        self._constant_value_bounds = constant_value_bounds
        self._length_scale = length_scale
        self._length_scale_bounds = length_scale_bounds
        self.n_inducing = n_inducing
        self.n_iter = n_iter
        self.lr = lr
        self.device = device

    @property
    def signal_bounds_fixed(self) -> bool:
        """True when both constant_value and length_scale bounds are ``"fixed"``."""
        return (
            self._constant_value_bounds == "fixed"
            and self._length_scale_bounds == "fixed"
        )

    def fit_residuals(
        self,
        coords: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP residuals using GPyTorch's exact-GP backend.

        Hyperparameters are optimized on the mean of all target columns.
        The fitted kernel and noise level are then applied to every column
        of ``Y`` to compute residuals.

        NaN rows are excluded from fitting and reinserted as NaN in output.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_spots, d)``. Normalized spatial coordinates.
        Y : torch.Tensor
            Shape ``(n_spots, m)``. Target values; may contain NaN rows.

        Returns
        -------
        torch.Tensor
            Shape ``(n_spots, m)``. Spatial residuals
            ``epsilon * (K + epsilon * I)**(-1) @ Y``. NaN rows from ``Y``
            are preserved as NaN.
        """
        is_nan = torch.isnan(Y).any(1)
        if is_nan.any():
            coords_c = coords[~is_nan]
            Y_c = Y[~is_nan]
            res_clean = self._fit_no_nan(coords_c, Y_c)
            out = torch.full_like(Y, float("nan"))
            out[~is_nan] = res_clean
            return out
        return self._fit_no_nan(coords, Y)

    def _fit_no_nan(self, coords: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Inner residualization assuming no NaN rows."""
        import gpytorch

        Y2d = Y if Y.dim() == 2 else Y.unsqueeze(1)  # (n, m)
        n = Y2d.shape[0]
        dev = torch.device(self.device)

        train_x = coords.float().to(dev)
        # Optimize hyperparameters on the column mean
        train_y = Y2d.float().mean(dim=1).to(dev)  # (n,)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
        # Use FITC only when there are strictly more observations than the
        # requested inducing set (i.e. an actual sparse approximation).
        use_sparse = self.n_inducing is not None and n > self.n_inducing
        if use_sparse:
            inducing_pts = _subsample_inducing(train_x, self.n_inducing)
            model = _GPyTorchExactGPModel.build_sparse(
                train_x,
                train_y,
                likelihood,
                inducing_pts,
                self._constant_value,
                self._constant_value_bounds,
                self._length_scale,
                self._length_scale_bounds,
            ).to(dev)
        else:
            model = _GPyTorchExactGPModel.build(
                train_x,
                train_y,
                likelihood,
                self._constant_value,
                self._constant_value_bounds,
                self._length_scale,
                self._length_scale_bounds,
            ).to(dev)

        model.train()
        likelihood.train()
        # model.parameters() already includes likelihood params via ExactGP
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable and self.n_iter > 0:
            optimizer = torch.optim.Adam(trainable, lr=self.lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(self.n_iter):
                optimizer.zero_grad()
                loss = -mll(model(train_x), train_y)
                loss.backward()
                optimizer.step()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            eps = likelihood.noise.item()
            K_lazy = model.covar_module(train_x)  # lazy (n, n)
            Y_dev = Y2d.float().to(dev)  # (n, m)
            # Solve (K + eps*I) @ alpha = Y via CG (no explicit n x n matrix)
            K_noisy = K_lazy.add_diagonal(torch.full((n,), eps, device=dev))
            alpha = K_noisy.solve(Y_dev)  # (n, m)
            res = eps * alpha  # epsilon * (K + eps*I)^{-1} @ Y

        return res.cpu()
