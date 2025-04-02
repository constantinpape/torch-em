from typing import Optional

import torch
import torch.nn as nn


def flatten_samples(input_: torch.Tensor) -> torch.Tensor:
    """Flattens a tensor or a variable such that the channel axis is first and the sample (batch) axis is second.

    The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.

    Args:
        The input tensor.

    Returns:
        The transformed input tensor.
    """
    # Get number of channels
    num_channels = input_.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = input_.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened


def dice_score(
    input_: torch.Tensor,
    target: torch.Tensor,
    invert: bool = False,
    channelwise: bool = True,
    reduce_channel: Optional[str] = "sum",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the dice score between input and target.

    Args:
        input_: The input tensor.
        target: The target tensor.
        invert: Whether to invert the returned dice score to obtain the dice error instead of the dice score.
        channelwise: Whether to return the dice score independently per channel.
        reduce_channel: How to return the dice score over the channel axis.
        eps: The epsilon value added to the denominator for numerical stability.

    Returns:
        The dice score.
    """
    if input_.shape != target.shape:
        raise ValueError(f"Expect input and target of same shape, got: {input_.shape}, {target.shape}.")

    if channelwise:
        # Flatten input and target to have the shape (C, N),
        # where N is the number of samples
        input_ = flatten_samples(input_)
        target = flatten_samples(target)
        # Compute numerator and denominator (by summing over samples and
        # leaving the channels intact)
        numerator = (input_ * target).sum(-1)
        denominator = (input_ * input_).sum(-1) + (target * target).sum(-1)
        channelwise_score = 2 * (numerator / denominator.clamp(min=eps))
        if invert:
            channelwise_score = 1. - channelwise_score

        # Reduce the dice score over the channels to compute the overall dice score.
        # (default is to use the sum)
        if reduce_channel is None:
            score = channelwise_score
        elif reduce_channel == "sum":
            score = channelwise_score.sum()
        elif reduce_channel == "mean":
            score = channelwise_score.mean()
        elif reduce_channel == "max":
            score = channelwise_score.max()
        elif reduce_channel == "min":
            score = channelwise_score.min()
        else:
            raise ValueError(f"Unsupported channel reduction {reduce_channel}")

    else:
        numerator = (input_ * target).sum()
        denominator = (input_ * input_).sum() + (target * target).sum()
        score = 2. * (numerator / denominator.clamp(min=eps))
        if invert:
            score = 1. - score

    return score


class DiceLoss(nn.Module):
    """Loss computed based on the dice error between a binary input and binary target.

    Args:
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
    """
    def __init__(self, channelwise: bool = True, eps: float = 1e-7, reduce_channel: Optional[str] = "sum"):
        if reduce_channel not in ("sum", "mean", "max", "min", None):
            raise ValueError(f"Unsupported channel reduction {reduce_channel}")

        super().__init__()
        self.channelwise = channelwise
        self.eps = eps
        self.reduce_channel = reduce_channel

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {"channelwise": channelwise, "eps": self.eps, "reduce_channel": self.reduce_channel}

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_: The binary input.
            target: The binary target.

        Returns:
            The dice loss.
        """
        return dice_score(
            input_=input_,
            target=target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps,
            reduce_channel=self.reduce_channel
        )


class DiceLossWithLogits(nn.Module):
    """Loss computed based on the dice error between logits and binary target.

    Args:
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
    """
    def __init__(self, channelwise: bool = True, eps: float = 1e-7, reduce_channel: Optional[str] = "sum"):
        if reduce_channel not in ("sum", "mean", "max", "min", None):
            raise ValueError(f"Unsupported channel reduction {reduce_channel}")

        super().__init__()
        self.channelwise = channelwise
        self.eps = eps
        self.reduce_channel = reduce_channel

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {"channelwise": channelwise, "eps": self.eps, "reduce_channel": self.reduce_channel}

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_: The logits.
            target: The binary target.

        Returns:
            The dice loss.
        """
        return dice_score(
            input_=nn.functional.sigmoid(input_),
            target=target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps,
            reduce_channel=self.reduce_channel,
        )


class BCEDiceLoss(nn.Module):
    """Loss computed based on the binary cross entropy and the dice error between binary inputs and binary target.

    Args:
        alpha: The weight for combining the BCE and dice loss.
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, channelwise: bool = True, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.channelwise = channelwise
        self.eps = eps

        # All torch_em classes should store init kwargs to easily recreate the init call.
        self.init_kwargs = {"alpha": alpha, "beta": beta, "channelwise": channelwise, "eps": self.eps}

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_: The binary input.
            target: The binary target.

        Returns:
            The combined BCE and dice loss.
        """
        loss_dice = dice_score(
            input_=input_,
            target=target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps
        )
        loss_bce = nn.functional.binary_cross_entropy(input_, target)
        return self.alpha * loss_dice + self.beta * loss_bce


# TODO think about how to handle combined losses like this for mixed precision training
class BCEDiceLossWithLogits(nn.Module):
    """Loss computed based on the binary cross entropy and the dice error between logits and binary target.

    Args:
        alpha: The weight for combining the BCE and dice loss.
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, channelwise: bool = True, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.channelwise = channelwise
        self.eps = eps

        # All torch_em classes should store init kwargs to easily recreate the init call.
        self.init_kwargs = {"alpha": alpha, "beta": beta, "channelwise": channelwise, "eps": self.eps}

    def forward(self, input_, target):
        """Compute the loss.

        Args:
            input_: The logits.
            target: The binary target.

        Returns:
            The combined BCE and dice loss.
        """
        loss_dice = dice_score(
            input_=nn.functional.sigmoid(input_),
            target=target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps
        )

        loss_bce = nn.functional.binary_cross_entropy_with_logits(input_, target)

        return self.alpha * loss_dice + self.beta * loss_bce
