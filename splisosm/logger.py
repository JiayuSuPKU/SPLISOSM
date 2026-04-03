"""Training logger utilities for GLM/GLMM optimization."""

from __future__ import annotations

import torch
from typing import Optional

__all__ = ["PatienceLogger"]


class PatienceLogger:
    """Logger for tracking training patience and convergence.

    For training MultinomGLM and MultinomGLMM.

    Loss history is always recorded (one scalar per gene per epoch).
    Per-epoch *parameter* snapshots are only stored when
    ``store_param_history=True``.
    """

    def __init__(
        self,
        batch_size: int,
        patience: int,
        min_delta: float = 1e-5,
        tol_relative: bool = False,
        diagnose: bool = False,
        store_param_history: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        batch_size
            Number of samples in the batch.
        patience
            Number of epochs to wait after the last significant improvement.
        min_delta
            Minimum change in the loss to qualify as an improvement.
            Interpreted as an absolute change when ``tol_relative=False``
            (default) or as a fraction of the current best loss when
            ``tol_relative=True``.
        tol_relative
            If ``True``, ``min_delta`` is treated as a *relative* threshold:
            an epoch counts as a "big improvement" only when
            ``(best - current) / max(|best|, 1) >= min_delta``.
            Relative mode is more robust when the loss scale varies widely
            across genes or datasets.  Default ``False`` preserves the
            original absolute behaviour.
        diagnose
            Deprecated alias for ``store_param_history``.  Will be removed in
            a future version.
        store_param_history
            Whether to store per-epoch parameter snapshots during training.
            This can consume significant memory (O(epochs × n_genes × params)).
            Loss history is always stored regardless of this flag.
        """
        self.batch_size = batch_size
        self.patience = torch.full((batch_size,), patience, dtype=int)
        self.min_delta = min_delta
        self.tol_relative = tol_relative

        # store_param_history: accept old `diagnose` kwarg for backward compat
        if diagnose:
            store_param_history = True
        self.store_param_history = store_param_history
        if store_param_history:
            self.params_iter = {"loss": [], "params": []}
        else:
            self.params_iter = None
        self.best_params = None

        self.best_loss = torch.full((batch_size,), float("inf"))
        self.best_epoch = torch.full((batch_size,), -1, dtype=torch.int)
        self.epochs_without_improvement = torch.zeros(batch_size, dtype=torch.int)
        self.convergence = torch.zeros(batch_size, dtype=torch.bool)
        self.epoch = 0

        # Loss history — always recorded (one (batch_size,) tensor per epoch).
        self._loss_history: list[torch.Tensor] = []

    @property
    def loss_history(self) -> torch.Tensor:
        """Loss history as a 2-D tensor of shape ``(n_epochs, batch_size)``.

        Each row is the per-gene negative log-likelihood at that epoch.
        Always available after at least one call to :meth:`log`.
        """
        if not self._loss_history:
            return torch.empty(0, self.batch_size)
        return torch.stack(self._loss_history, dim=0)

    def log(self, loss: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
        """Log loss for a given epoch and update best parameters if improved.

        Parameters
        ----------
        loss
            Loss for the current epoch, shape ``(batch_size,)``.
        params
            Current model parameters, keyed by parameter name.
        """
        # Always record loss history (small memory overhead).
        self._loss_history.append(loss.detach().cpu().clone())

        # Compute the improvement threshold — absolute or relative.
        if self.tol_relative:
            threshold = self.min_delta * self.best_loss.abs().clamp(min=1.0)
        else:
            threshold = self.min_delta

        big_improve = (self.best_loss - loss) >= threshold
        improve = (self.best_loss - loss) > 0

        self.epochs_without_improvement[~big_improve] += 1
        self.epochs_without_improvement[big_improve] = 0

        update = ~self.convergence & improve
        self.best_loss[update] = loss[update]
        self.best_epoch[update] = self.epoch

        if self.best_params is None:
            self.best_params = {k: v.clone() for k, v in params.items()}
        else:
            # Vectorized update: only copy improving genes.
            for k in params:
                self.best_params[k][update] = params[k][update]

        convergence = self.epochs_without_improvement >= self.patience
        self.convergence = self.convergence | convergence
        self.epoch += 1

        if self.store_param_history:
            self.params_iter["loss"].append(loss)
            self.params_iter["params"].append(params)

    def get_params_iter(self) -> Optional[dict]:
        """Return stored parameter history if ``store_param_history=True``.

        Returns
        -------
        dict or None
            Dictionary ``{"loss": list[Tensor], "params": list[dict]}`` where
            each entry corresponds to one epoch, if ``store_param_history=True``;
            otherwise ``None``.
        """
        if self.store_param_history:
            return self.params_iter
