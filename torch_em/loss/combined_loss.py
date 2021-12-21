import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, *losses, loss_weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        n_losses = len(self.losses)
        if loss_weights is None:
            try:
                self.loss_weights = [1.0 / n_losses] * n_losses
            except ZeroDivisionError:
                self.loss_weights = None
        else:
            assert len(loss_weights) == n_losses
            self.loss_weights = loss_weights

    def forward(self, x, y):
        assert self.loss_weights is not None
        loss_value = sum([loss(x, y) * weight for loss, weight in zip(self.losses, self.loss_weights)])
        return loss_value
