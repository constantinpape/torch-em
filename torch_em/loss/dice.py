import torch.nn as nn


# TODO refactor
def flatten_samples(input_):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
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


def dice_score(input_, target, invert=False, channelwise=True, eps=1e-7):
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
        # Sum over the channels to compute the total score
        score = channelwise_score.sum()
    else:
        numerator = (input_ * target).sum()
        denominator = (input_ * input_).sum() + (target * target).sum()
        score = 2. * (numerator / denominator.clamp(min=eps))
        if invert:
            score = 1. - score
    return score


class DiceLoss(nn.Module):
    def __init__(self, channelwise=True, eps=1e-7):
        super().__init__()
        self.channelwise = channelwise
        self.eps = eps

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {'channelwise': channelwise, 'eps': self.eps}

    def forward(self, input_, target):
        return dice_score(input_, target,
                          invert=True, channelwise=self.channelwise,
                          eps=self.eps)


class DiceLossWithLogits(nn.Module):
    def __init__(self, channelwise=True, eps=1e-7):
        super().__init__()
        self.channelwise = channelwise
        self.eps = eps

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {'channelwise': channelwise, 'eps': self.eps}

    def forward(self, input_, target):
        return dice_score(
            nn.functional.sigmoid(input_),
            target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps
        )


# TODO think about how to handle combined losses like this for mixed precision training
class BCEDiceLossWithLogits(nn.Module):

    def __init__(self, alpha=1., beta=1., channelwise=True, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.channelwise = channelwise
        self.eps = eps

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {'alpha': alpha, 'beta': beta,
                            'channelwise': channelwise, 'eps': self.eps}

    def forward(self, input_, target):
        loss_dice = dice_score(
            nn.functional.sigmoid(input_),
            target,
            invert=True,
            channelwise=self.channelwise,
            eps=self.eps
        )
        loss_bce = nn.functional.binary_cross_entropy_with_logits(
            input_, target
        )
        return self.alpha * loss_dice + self.beta * loss_bce
