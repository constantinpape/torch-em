import torch.nn as nn

from .dice import DiceLoss


# TODO
# - check how the loss can be extended to support training with BCE + Dice
# - test and document it
class DistanceLoss(nn.Module):
    def __init__(self, mask_distances_in_bg):
        super().__init__()

        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.mask_distances_in_bg = mask_distances_in_bg

    def forward(self, input_, target):
        assert input_.shape == target.shape, input_.shape
        assert input_.shape[1] == 3, input_.shape

        fg_input, fg_target = input_[:, 0, ...], target[:, 0, ...]
        fg_loss = self.dice_loss(fg_target, fg_input)

        cdist_input, cdist_target = input_[:, 1, ...], target[:, 1, ...]
        if self.mask_distances_in_bg:
            cdist_loss = self.mse_loss(cdist_target * fg_target, cdist_input * fg_target)
        else:
            cdist_loss = self.mse_loss(cdist_target, cdist_input)

        bdist_input, bdist_target = input_[:, 2, ...], target[:, 2, ...]
        if self.mask_distances_in_bg:
            bdist_loss = self.mse_loss(bdist_target * fg_target, bdist_input * fg_target)
        else:
            bdist_loss = self.mse_loss(bdist_target, bdist_input)

        overall_loss = fg_loss + cdist_loss + bdist_loss
        return overall_loss
