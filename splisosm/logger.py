"""Training logger utilities for GLM/GLMM optimization."""

from __future__ import annotations

import torch
from torch.utils.data.dataloader import default_collate
from typing import Optional

__all__ = ["PatienceLogger"]


class PatienceLogger:
    """Logger for tracking training patience and convergence.

    For training MultinomGLM and MultinomGLMM.
    """

    def __init__(
        self,
        batch_size: int,
        patience: int,
        min_delta: float = 1e-5,
        diagnose: bool = False,
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
        diagnose
            Whether to store parameter changes during training.
        """
        self.batch_size = batch_size
        self.patience = torch.full((batch_size,), patience, dtype=int)
        self.min_delta = min_delta

        self.diagnose = diagnose
        if diagnose:
            self.params_iter = {"loss": [], "params": []}
        else:
            self.params_iter = None
        self.best_params = None

        self.best_loss = torch.full((batch_size,), float("inf"))
        # self.best_params = [None] * batch_size
        self.best_epoch = torch.full((batch_size,), -1, dtype=torch.int)
        self.epochs_without_improvement = torch.zeros(batch_size, dtype=torch.int)
        self.convergence = torch.zeros(batch_size, dtype=torch.bool)
        self.epoch = 0

    def log(self, loss: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
        """Log loss for a given epoch and update best parameters if improved.

        Parameters
        ----------
        loss
            Loss for the current epoch.
        params
            Parameters for the current epoch.
        """
        big_improve = (self.best_loss - loss) >= self.min_delta
        improve = (self.best_loss - loss) > 0

        # self.epochs_without_improvement[~improve] += 1
        self.epochs_without_improvement[~big_improve] += 1
        self.epochs_without_improvement[big_improve] = 0

        update = ~self.convergence & improve
        self.best_loss[update] = loss[update]
        self.best_epoch[update] = self.epoch
        # simply update all non-converged samples regardless of improvement
        # self.best_loss[~self.convergence] = loss[~self.convergence]
        # self.best_epoch[~self.convergence] = self.epoch

        if self.best_params is None:
            self.best_params = {k: v.clone() for k, v in params.items()}
        else:  # update best params only for newly converged samples
            for i in torch.where(update)[0]:
                for k in params:
                    self.best_params[k][i] = params[k][i]

        convergence = self.epochs_without_improvement >= self.patience
        self.convergence = self.convergence | convergence
        # if self.epoch > 800:
        #     print(self.epoch, self.convergence, loss, self.best_loss)
        self.epoch += 1
        if self.diagnose:
            self.params_iter["loss"].append(loss)
            self.params_iter["params"].append(params)

    def get_params_iter(self) -> Optional[list[dict]]:
        """Return stored parameters during training if diagnose is True.

        Returns
        -------
        list[dict] or None
            List of dictionaries containing loss and parameters for each sample,
            or None if diagnose is False.
        """
        if self.diagnose:
            return default_collate(self.params_iter)
