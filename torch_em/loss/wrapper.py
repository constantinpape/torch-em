from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn


class LossWrapper(nn.Module):
    """A wrapper around a torch loss function.

    Applies transformations to prediction and/or target before passing it to the loss.

    Args:
        loss: The loss function.
        transform: The transformation applied to prediction and/or target.
            Must take both the prediction and target as arguments and then return them, potentially transformed.
    """
    def __init__(self, loss: nn.Module, transform: Callable):
        super().__init__()
        self.loss = loss

        if not callable(transform):
            raise ValueError("transform has to be callable.")
        self.transform = transform
        self.init_kwargs = {'loss': loss, 'transform': transform}

    def apply_transform(self, prediction, target, **kwargs):
        """@private
        """
        # Check if the prediction and target are lists.
        # If they are, apply the transform to each element individually.
        if isinstance(prediction, (list, tuple)):
            assert isinstance(target, (list, tuple))
            transformed_prediction, transformed_target = [], []
            for pred, targ in zip(prediction, target):
                tr_pred, tr_targ = self.transform(pred, targ, **kwargs)
                transformed_prediction.append(tr_pred)
                transformed_target.append(tr_targ)
            return transformed_prediction, transformed_target
        # Otherwise, we expect that prediction and target are both tensors.
        else:
            prediction, target = self.transform(prediction, target, **kwargs)
            return prediction, target

    def forward(
        self,
        prediction: Union[Sequence[torch.Tensor], torch.Tensor],
        target: Union[Sequence[torch.Tensor], torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Apply the tranformations to prediction and/or target before computing the loss.

        Args:
            prediction: The prediction.
            target: The target.
            kwargs: Additional keyword arguments for the transformation.

        Returns:
            The loss.
        """
        prediction, target = self.apply_transform(prediction, target, **kwargs)
        loss = self.loss(prediction, target)
        return loss


#
# Loss transformations
#
def _crop(prediction, target, mask, channel_dim):
    if mask.shape[channel_dim] != 1:
        raise ValueError(
            "_crop only supports a mask with a singleton channel axis. Please consider using masking_method=multiply."
        )
    mask = mask.type(torch.bool)
    # remove singleton axis
    mask = mask.squeeze(channel_dim)
    # move channel axis to end
    prediction = prediction.moveaxis(channel_dim, -1)
    target = target.moveaxis(channel_dim, -1)
    # output has shape N x C
    # correct for torch_em.loss.dice.flatten_samples
    return prediction[mask], target[mask]


def _multiply(prediction, target, mask, channel_dim):
    prediction = prediction * mask
    target = target * mask
    return prediction, target


class ApplyMask:
    """Apply a mask to prediction and target, so that only values in the mask are taken into account for the loss.

    Supports two different masking methods:
    - 'crop': Crop away the mask from the prediction and target. This only works if the mask just has a single channel,
        and if the loss function does not require spatial inputs.
    - 'multiply': Multiply the prediction and target with zeros outside of the mask.

    Args:
        masking_method: The masking method to use. Can be one of 'crop' or 'multiply'.
        channel_dim: The dimension of the channel axis.
    """
    MASKING_FUNCS = {"crop": _crop, "multiply": _multiply}

    def __init__(self, masking_method: str = "crop", channel_dim: int = 1):
        if masking_method not in self.MASKING_FUNCS.keys():
            raise ValueError(f"{masking_method} is not available, please use one of {list(self.MASKING_FUNCS.keys())}.")
        self.masking_func = self.MASKING_FUNCS[masking_method]
        self.channel_dim = channel_dim
        self.init_kwargs = {"masking_method": masking_method, "channel_dim": channel_dim}

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask predictions.

        Args:
            prediction: The prediction tensor.
            target: The target tensor.
            mask: The mask tensor.

        Returns:
            The masked prediction.
            The masked target.
        """
        mask.requires_grad = False
        return self.masking_func(prediction, target, mask, self.channel_dim)


class ApplyAndRemoveMask(ApplyMask):
    """Extract mask from extra channels from a target tensor and use it to mask the prediction.

    Supports the same masking methods as `ApplyMask`.
    """
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Remove masking channels from the target and then apply the mask.

        Args:
            prediction: The prediction tensor.
            target: The target tensor, with extra channels that contain the mask.

        Returns:
            The masked prediction.
            The masked target, with extra channels removed.
        """
        assert target.dim() == prediction.dim(), f"{target.dim()}, {prediction.dim()}"
        assert target.size(1) == 2 * prediction.size(1), f"{target.size(1)}, {prediction.size(1)}"
        assert target.shape[2:] == prediction.shape[2:], f"{str(target.shape)}, {str(prediction.shape)}"
        seperating_channel = target.size(1) // 2
        mask = target[:, seperating_channel:]
        target = target[:, :seperating_channel]
        prediction, target = super().__call__(prediction, target, mask)
        return prediction, target


class MaskIgnoreLabel(ApplyMask):
    """Mask ignore label from the target.

    Supports the same masking methods as `ApplyMask`.

    Args:
        ignore_label: The ignore label, which will be msaked.
        masking_method: The masking method to use. Can be one of 'crop' or 'multiply'.
        channel_dim: The dimension of the channel axis.
    """
    def __init__(self, ignore_label: int = -1, masking_method: str = "crop", channel_dim: int = 1):
        super().__init__(masking_method, channel_dim)
        self.ignore_label = ignore_label
        self.init_kwargs["ignore_label"] = ignore_label

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mask ignore label in the prediction and target.

        Args:
            prediction: The prediction tensor.
            target: The target tensor.

        Returns:
            The masked prediction.
            The masked target.
        """
        mask = (target != self.ignore_label)
        prediction, target = super().__call__(prediction, target, mask)
        return prediction, target
