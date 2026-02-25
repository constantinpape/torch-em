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
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute (optionally masked) dice score between input and target.

    Args:
        input_: The input tensor.
        target: The target tensor.
        invert: Whether to invert the returned dice score to obtain the dice error instead of the dice score.
        channelwise: Whether to return the dice score independently per channel.
        reduce_channel: How to return the dice score over the channel axis.
        eps: The epsilon value added to the denominator for numerical stability.
        valid: Optional mask indicating voxels to include (True/1 = keep, False/0 = ignore).
               Shape [B,1,...] (broadcasted over C) or [B,C,...].

    Returns:
        Dice score (or dice error if invert=True).
    """
    if input_.shape != target.shape:
        raise ValueError(f"Expect input and target of same shape, got: {input_.shape}, {target.shape}.")
    if reduce_channel not in ("sum", "mean", "max", "min", None):
        raise ValueError(f"Unsupported channel reduction {reduce_channel}")

    # Convert valid mask once; keep it broadcastable ([B,1,...] is fine).
    valid_f = None
    if valid is not None:
        if valid.shape[0] != input_.shape[0] or valid.shape[2:] != input_.shape[2:]:
            raise ValueError(
                f"valid must have shape [B,1,...] or [B,C,...] matching input spatial dims. "
                f"Got valid={valid.shape}, input={input_.shape}."
            )
        if valid.dtype == torch.bool:
            valid_f = valid.to(dtype=input_.dtype)
        else:
            valid_f = valid.to(dtype=input_.dtype)

    if channelwise:
        # Reduce over spatial dims only -> results per batch and channel: [B,C]
        dims = tuple(range(2, input_.dim()))

        if valid_f is None:
            numerator = (input_ * target).sum(dims)
            den1 = (input_ * input_).sum(dims)
            den2 = (target * target).sum(dims)
        else:
            numerator = (input_ * target * valid_f).sum(dims)
            den1 = (input_ * input_ * valid_f).sum(dims)
            den2 = (target * target * valid_f).sum(dims)

        dice = 2.0 * numerator / (den1 + den2).clamp_min(eps)  # [B,C]
        dice = dice.mean(dim=0)  # average over batch -> [C]

        if invert:
            dice = 1.0 - dice

        if reduce_channel is None:
            return dice
        if reduce_channel == "sum":
            return dice.sum()
        if reduce_channel == "mean":
            return dice.mean()
        if reduce_channel == "max":
            return dice.max()
        if reduce_channel == "min":
            return dice.min()

    else:
        # Reduce over all dims -> scalar
        dims = tuple(range(0, input_.dim()))

        if valid_f is None:
            numerator = (input_ * target).sum(dims)
            den = (input_ * input_).sum(dims) + (target * target).sum(dims)
        else:
            numerator = (input_ * target * valid_f).sum(dims)
            den = (input_ * input_ * valid_f).sum(dims) + (target * target * valid_f).sum(dims)

        dice = 2.0 * numerator / den.clamp_min(eps)
        return (1.0 - dice) if invert else dice

    # unreachable due to checks above, but keeps mypy happy
    raise RuntimeError("Unexpected control flow in dice_score")


class DiceLoss(nn.Module):
    """Loss computed based on the dice error between a binary input and binary target.

    Args:
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
        ignore_label: Ignore label ID in target for the loss computation.
        ignore_state_value: Ignore state value ID in state_channel for the loss computation.
        state_channel: Channel to use for state in input.
    """
    def __init__(
        self,
        channelwise: bool = True,
        eps: float = 1e-7,
        reduce_channel: Optional[str] = "sum",
        ignore_state_value: Optional[int] = None,
        state_channel: Optional[int] = None,
    ):
        if reduce_channel not in ("sum", "mean", "max", "min", None):
            raise ValueError(f"Unsupported channel reduction {reduce_channel}")

        super().__init__()
        self.channelwise = channelwise
        self.eps = eps
        self.reduce_channel = reduce_channel
        self.ignore_state_value = ignore_state_value
        self.state_channel = state_channel

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {"channelwise": channelwise, "eps": self.eps, "reduce_channel": self.reduce_channel,
                            "ignore_state_value": ignore_state_value, "state_channel": state_channel
                            }

    def forward(self, input_: torch.Tensor, target: torch.Tensor, state: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_: The binary input.
            target: The binary target.

        Returns:
            The dice loss.
        """
        valid = None
        if state is not None and self.ignore_state_value is not None:
            state_ch = state[:, self.state_channel:self.state_channel + 1]
            valid = (state_ch != self.ignore_state_value)

        return dice_score(
            input_=input_,
            target=target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps,
            reduce_channel=self.reduce_channel,
            valid=valid
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
