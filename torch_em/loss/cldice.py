import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice import flatten_samples

# From "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation":
# https://arxiv.org/abs/2003.07311

class SoftSkeletonize(torch.nn.Module):
    """`SoftSkeletonize` is a differentiable approximation for skeletonization, which applies
        iterative min- and max-pooling as a proxy for morphological erosion and dilation. 

    Args:
        num_iter: Number of iterations for soft-skeletonization. 
            Should be greater or equal to than the maximum observed radius.
    """
    def __init__(self, num_iter: int = 5):
 
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, input_: torch.Tensor):

        if len(input_.shape)==4:
            p1 = -F.max_pool2d(-input_, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-input_, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(input_.shape)==5:
            p1 = -F.max_pool3d(-input_,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-input_,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-input_,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, input_: torch.Tensor):

        if len(input_.shape)==4:
            return F.max_pool2d(input_, (3,3), (1,1), (1,1))
        elif len(input_.shape)==5:
            return F.max_pool3d(input_,(3,3,3),(1,1,1),(1,1,1))

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


class SoftclDiceLoss(nn.Module):
    """Combined soft Dice and clDice loss for segmentation of tubular structures.

        The soft clDice loss computes topology-aware loss by computing the soft skeleton of both the prediction and target
        and measuring overlap of the two skeletons. Teaches the model to learn skeletons directly. 
        In the clDice paper, the authors recommend using the combined soft-Dice and soft-clDice loss to learn topology-aware 
        segmentations, which is implemented here as `SoftDiceclDiceLoss`. 

    Args:
        num_iter: Number of iterations for soft-skeletonization. 
        eps: The epsilon value added to the denominator for numerical stability.
        exclude_background: Whether to exclude background channel 0 from the loss computation. 
            Useful for multi-class segmentation.
        channelwise: Not implemented; Whether to return the dice score independently per channel.
        reduce_channel: Not implemented; The epsilon value added to the denominator for numerical stability.
    """
    def __init__(self, num_iter: int = 5, eps: float = 1.0, exclude_background: bool = False):
        super(SoftclDiceLoss, self).__init__()

        #TODO fix iter argument, soft skeletonize should get the self.num_iter instead of num_iter=10
        self.num_iter = num_iter
        self.eps = eps
        self.soft_skeletonize = SoftSkeletonize(num_iter=5)
        self.exclude_background = exclude_background

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft clDice score between the input logits and binary target.

        Args:
            input_: The input logits.
            target: The binary target.

        Returns:
            The soft clDice score.
        """
        if self.exclude_background:
            target = target[:, 1:, :, :]
            input_ = input_[:, 1:, :, :]
        skel_input = self.soft_skeletonize(input_)
        skel_target = self.soft_skeletonize(target)
        tprec = (torch.sum(torch.multiply(skel_input, target))+self.eps)/(torch.sum(skel_input)+self.eps)    
        tsens = (torch.sum(torch.multiply(skel_target, input_))+self.eps)/(torch.sum(skel_target)+self.eps)    
        cl_dice = 1.0 - 2.0*(tprec*tsens)/(tprec+tsens)

        return cl_dice

#TODO consider resuing dice_score from dice.py as the implementation is better
def soft_dice(input_: torch.Tensor, target: torch.Tensor, eps: float = 1.0):
    """Compute the soft dice score between the input logits and binary target.

    Args:
        input_: The input logits.
        target: The binary target.
        eps: The epsilon value added to the denominator for numerical stability.

    Returns:
        The soft dice score.
    """
    intersection = torch.sum((target * input_))
    coeff = (2.0 *  intersection + eps) / (torch.sum(target) + torch.sum(input_) + eps)
    return (1.0 - coeff)


#TODO implement `channelwise` for multiclass segmentation
#TODO consider if `exclude_background` is needed for multiclass segmentation
class SoftDiceclDiceLoss(nn.Module):
    """Combined soft-Dice and soft-clDice loss for segmentation of tubular structures.

        The soft-clDice loss computes topology-aware loss by computing the soft skeleton of both the prediction and target
        and measuring overlap of the two skeletons. This encourages the model to preserve the connectivity and topology 
        of tubular structures. The final loss is a weighted combination of soft Dice and clDice, controlled by alpha.

    Args:
        num_iter: Number of iterations for soft-skeletonization. 
        alpha: The weight for combining the soft Dice and soft clDice loss.
        eps: The epsilon value added to the denominator for numerical stability.
        exclude_background: Whether to exclude background channel 0 from the loss computation. Useful for multi-class segmentation.
        invert: Not implemented; Whether to invert the returned dice score to obtain the dice error instead of the dice score.
        channelwise: Not implemented; Whether to return the dice score independently per channel.
        reduce_chnanel: Not implemented; How to return the dice score over the channel axis.

    """
    def __init__(self, num_iter: int = 5, alpha: float = 0.5, eps: float = 1.0, exclude_background: bool = False):
        super(SoftDiceclDiceLoss, self).__init__()

        #TODO fix iter argument, soft skeletonize should get the self.num_iter instead of num_iter=10

        self.num_iter = num_iter
        self.eps = eps
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=5)
        self.exclude_background = exclude_background
        

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined soft Dice and clDice loss.

        Args:
            input_: The input logits.
            target: The binary target.
            

        Returns:
            Combined clDice loss. 
        """
        if self.exclude_background:
            target = target[:, 1:, :, :]
            input_ = input_[:, 1:, :, :]
        dice = soft_dice(target, input_)
        skel_input = self.soft_skeletonize(input_)
        skel_target = self.soft_skeletonize(target)
        tprec = (torch.sum(torch.multiply(skel_input, target))+self.eps)/(torch.sum(skel_input)+self.eps)    
        tsens = (torch.sum(torch.multiply(skel_target, input_))+self.eps)/(torch.sum(skel_target)+self.eps)    
        cl_dice = 1.0 - 2.0*(tprec*tsens)/(tprec+tsens)

        return (1.0-self.alpha)*dice+self.alpha*cl_dice


