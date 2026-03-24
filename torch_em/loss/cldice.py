import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import dice_score

# From "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation":
# https://arxiv.org/abs/2003.07311

class SoftSkeletonize(torch.nn.Module):
    """`SoftSkeletonize` is a differentiable approximation for skeletonization,
        which applies iterative min- and max-pooling as a proxy for
        morphological erosion and dilation.

    Args:
        num_iter: Number of iterations for soft-skeletonization.
            Should be greater or equal to than the maximum observed radius.
    """
    def __init__(self, num_iter: int = 5):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, input_: torch.Tensor):

        if len(input_.shape) == 4:
            p1 = -F.max_pool2d(-input_, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-input_, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(input_.shape) == 5:
            p1 = -F.max_pool3d(-input_, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-input_, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-input_, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, input_: torch.Tensor):

        if len(input_.shape) == 4:
            return F.max_pool2d(input_, (3, 3), (1, 1), (1, 1))
        elif len(input_.shape) == 5:
            return F.max_pool3d(input_, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, input_: torch.Tensor):

        return self.soft_dilate(self.soft_erode(input_))

    def soft_skel(self, input_: torch.Tensor):

        input1 = self.soft_open(input_)
        skel = F.relu(input1)

        for j in range(self.num_iter):
            input_ = self.soft_erode(input_)
            input1 = self.soft_open(input_)
            delta = F.relu(input_-input1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, input_: torch.Tensor):
        """Skeletonize the input prediction.

        Args:
            input_: The input logits.

        Returns:
            The skeletonization.
        """
        return self.soft_skel(input_)

def cldice_score(
    input_: torch.Tensor,
    target: torch.Tensor,
    num_iter: int = 5,
    invert: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Adapted from .dice.py `dice_score`. Compute the soft clDice score between input and target.

    Args:
        input_: The input tensor.
        target: The target tensor.
        num_iter: Number of iterations for soft-skeletonization.
        invert: Whether to invert the returned dice score to obtain the cldice error instead of the cldice score.
        channelwise: Not implemented; whether to return the dice score independently per channel.
        reduce_channel: Not implemented; how to return the dice score over the channel axis.
        eps: The epsilon value added to the denominator for numerical stability.

    Returns:
        The clDice score.
    """
    if input_.shape != target.shape:
        raise ValueError(f"Expect input and target of same shape, got: {input_.shape}, {target.shape}.")

    soft_skeletonize = SoftSkeletonize(num_iter=num_iter)
    skel_input = soft_skeletonize(input_)
    skel_target = soft_skeletonize(target)

    t_prec = (skel_input * target).sum() / (skel_input.sum()).clamp(min=eps)
    t_sens = (skel_target * input_).sum() / (skel_target.sum()).clamp(min=eps)
    score = 2.*(t_prec*t_sens)/(t_prec+t_sens).clamp(min=eps)

    if invert:
        score = 1. - score

    return score


class SoftclDiceLoss(nn.Module):
    """Combined soft Dice and clDice loss for segmentation of tubular structures.

        The soft clDice loss computes topology-aware loss by computing the
        soft skeleton of both the prediction and target
        and measuring overlap of the two skeletons. Teaches the model to learn
        skeletons directly. In the clDice paper, the authors recommend using
        the combined soft-Dice and soft-clDice loss to learn topology-aware
        segmentations, which is implemented below as `CombinedclDiceLoss`.

    Args:
        num_iter: Number of iterations for soft-skeletonization.
        eps: The epsilon value added to the denominator for numerical
            stability.
        exclude_background: Whether to exclude background channel 0 from the
            loss computation.
            Useful for multi-class segmentation.
        channelwise: Not implemented; Whether to return the dice score
            independently per channel.
        reduce_channel: Not implemented; The epsilon value added to the 
            denominator for numerical stability.
    """
    def __init__(self, num_iter: int = 5, eps: float = 1e-7,
                 exclude_background: bool = False):
        super(SoftclDiceLoss, self).__init__()

        self.num_iter = num_iter
        self.eps = eps
        self.exclude_background = exclude_background
    
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft clDice score between the input logits and binary target.

        Args:
            input_: The input logits.
            target: The binary target.

        Returns:
            The soft clDice score.
        """
        if input_.shape != target.shape:
            raise ValueError(f"Expect input and target of same shape, got: {input_.shape}, {target.shape}.")
        
        if self.exclude_background:
            target = target[:, 1:, :, :]
            input_ = input_[:, 1:, :, :]
        
        cldice = cldice_score(input_, target, num_iter=self.num_iter, invert=True, eps=self.eps)

        return cldice


# TODO implement `channelwise` for multiclass segmentation
# TODO consider if `exclude_background` is needed for multiclass segmentation
class CombinedclDiceLoss(SoftclDiceLoss):
    """Combined soft-Dice and soft-clDice loss for segmentation of tubular structures.

        The soft-clDice loss computes topology-aware loss by computing the
        soft skeleton of both the prediction and target and measuring overlap
        of the two skeletons. This encourages the model to preserve the
        connectivity and topology of tubular structures. The final loss is a
        weighted combination of soft Dice and clDice, controlled by alpha.

    Args:
        num_iter: Number of iterations for soft-skeletonization.
        alpha: The weight for combining the soft Dice and soft clDice loss.
        eps: The epsilon value added to the denominator for numerical
            stability.
        exclude_background: Whether to exclude background channel 0 from the
            loss computation. Useful for multi-class segmentation.
        invert: Not implemented; Whether to invert the returned dice score to
            obtain the dice error instead of the dice score.
        channelwise: Not implemented; Whether to return the dice score
            independently per channel.
        reduce_chnanel: Not implemented; How to return the dice score over the
            channel axis.

    """
    def __init__(self, num_iter: int = 5, alpha: float = 0.5, eps: float = 1e-7,
                 exclude_background: bool = False):
        super(CombinedclDiceLoss, self).__init__(num_iter=num_iter, eps=eps, exclude_background=exclude_background)

        self.alpha = alpha
       
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined soft Dice and clDice loss.

        Args:
            input_: The input logits.
            target: The binary target.

        Returns:
            Combined clDice loss.
        """
        if input_.shape != target.shape:
            raise ValueError(f"Expect input and target of same shape, got: {input_.shape}, {target.shape}.")
  
        if self.exclude_background:
            target = target[:, 1:, :, :]
            input_ = input_[:, 1:, :, :]
        dice = dice_score(input_, target, invert=True, channelwise=False, eps=self.eps)
        cldice = cldice_score(input_, target, num_iter=self.num_iter, invert=True, eps=self.eps)

        return (1.0-self.alpha)*dice+self.alpha*cldice