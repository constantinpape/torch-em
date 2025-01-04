import torch
import torch.nn as nn

from .dice import DiceLoss


class DistanceLoss(nn.Module):
    """Loss for distance based instance segmentation.

    Expects input and targets with three channels: foreground and two distance channels.
    Typically the distance channels are centroid and inverted boundary distance.

    Args:
        mask_distances_in_bg: whether to mask the loss for distance predictions in the background.
        foreground_loss: the loss for comparing foreground predictions and target.
        distance_loss: the loss for comparing distance predictions and target.
    """
    def __init__(
        self,
        mask_distances_in_bg: bool = True,
        foreground_loss: nn.Module = DiceLoss(),
        distance_loss: nn.Module = nn.MSELoss(reduction="mean")
    ) -> None:
        super().__init__()

        self.foreground_loss = foreground_loss
        self.distance_loss = distance_loss
        self.mask_distances_in_bg = mask_distances_in_bg

        self.init_kwargs = {"mask_distances_in_bg": mask_distances_in_bg}

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input_.shape == target.shape, input_.shape
        assert input_.shape[1] == 3, input_.shape

        # IMPORTANT: preserve the channels!
        # Otherwise the Dice Loss will do all kinds of shennanigans.
        # Because it always interprets the first axis as channel,
        # and treats it differently (sums over it independently).
        # This will lead to a very large dice loss that dominates over everything else.
        fg_input, fg_target = input_[:, 0:1], target[:, 0:1]
        fg_loss = self.foreground_loss(fg_input, fg_target)

        cdist_input, cdist_target = input_[:, 1:2], target[:, 1:2]
        if self.mask_distances_in_bg:
            mask = fg_target
            cdist_loss = self.distance_loss(cdist_input * mask, cdist_target * mask)
        else:
            cdist_loss = self.distance_loss(cdist_input, cdist_target)

        bdist_input, bdist_target = input_[:, 2:3], target[:, 2:3]
        if self.mask_distances_in_bg:
            mask = fg_target
            bdist_loss = self.distance_loss(bdist_input * mask, bdist_target * mask)
        else:
            bdist_loss = self.distance_loss(bdist_input, bdist_target)

        overall_loss = fg_loss + cdist_loss + bdist_loss
        return overall_loss


class DiceBasedDistanceLoss(DistanceLoss):
    """Similar to `DistanceLoss`, using the dice score for all losses.

    Args:
        mask_distances_in_bg: whether to mask the loss for distance predictions in the background.
    """
    def __init__(self, mask_distances_in_bg: bool) -> None:
        super().__init__(mask_distances_in_bg, foreground_loss=DiceLoss(), distance_loss=DiceLoss())
