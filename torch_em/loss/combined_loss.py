from typing import List

import torch


class CombinedLoss(torch.nn.Module):
    """Combination of multiple losses.

    Args:
        losses: The loss functions to combine.
        loss_weights: The weights for the loss functions.
    """
    def __init__(self, *losses: torch.nn.Module, loss_weights: List[float] = None):
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)
        n_losses = len(self.losses)
        if loss_weights is None:
            try:
                self.loss_weights = [1.0 / n_losses] * n_losses
            except ZeroDivisionError:
                self.loss_weights = None
        else:
            assert len(loss_weights) == n_losses
            self.loss_weights = loss_weights

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            x: The prediction.
            y: The target.

        Returns:
            The loss value.
        """
        assert self.loss_weights is not None
        loss_value = sum([loss(x, y) * weight for loss, weight in zip(self.losses, self.loss_weights)])
        return loss_value
