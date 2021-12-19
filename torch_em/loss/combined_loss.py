import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, *losses, loss_weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        n_losses = len(self.losses)
        if loss_weights is None:
            self.loss_weights = [1.0 / n_losses] * n_losses
        else:
            assert len(loss_weights) == n_losses
            self.loss_weights = loss_weights

    def forward(self, x, y):
        loss_value = sum([loss(x, y) * weight for loss, weight in zip(self.losses, self.loss_weights)])
        return loss_value
