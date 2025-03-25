import torch
from torch.utils.data.dataloader import default_collate


class PatienceLogger:
    def __init__(
        self,
        batch_size: int,
        patience: int,
        min_delta: float = 1e-5,
        diagnose: bool = False,
    ):
        """
        Initializes the logger with a specified patience and minimum delta for improvement.

        Args:
            batch_size (int): Number of samples in the batch.
            patience (int): Number of epochs to wait after the last significant improvement.
            min_delta (float): Minimum change in the loss to qualify as an improvement.
            diagnose (bool): Whether to store parameter changes during training.
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

    def log(self, loss: torch.Tensor, params: dict[str, torch.tensor]) -> None:
        """
        Logs the loss for a given epoch and updates the best parameters if the loss improved significantly.

        Args:
            epoch (int): Current epoch number.
            loss (torch.Tensor): Loss for the current epoch.
            params (list): Parameters for the current epoch.
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

    def get_params_iter(self) -> list[dict] | None:
        """
        Returns the stored parameters during training if diagnose is True.

        Returns:
            list: A list of dictionaries containing loss and parameters for each sample.
        """
        if self.diagnose:
            return default_collate(self.params_iter)
