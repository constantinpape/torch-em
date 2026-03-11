import torch
import torch.nn as nn
import torch.nn.functional as F

# From "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation":
# https://arxiv.org/abs/2003.07311

class SoftSkeletonize(torch.nn.Module):
    """`SoftSkeletonize` is a differentiable approximation for skeletonization, which applies
        iterative min- and max-pooling as a proxy for morphological erosion and dilation. 

    Args:
        num_iter: Number of iterations for soft-skeletonization. 
            Should be greater or equal to than the maximum observed radius.
    """
    def __init__(self, num_iter=10):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, input_):

        if len(input_.shape)==4:
            p1 = -F.max_pool2d(-input_, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-input_, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(input_.shape)==5:
            p1 = -F.max_pool3d(-input_,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-input_,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-input_,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, input_):

        if len(input_.shape)==4:
            return F.max_pool2d(input_, (3,3), (1,1), (1,1))
        elif len(input_.shape)==5:
            return F.max_pool3d(input_,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, input_):
        
        return self.soft_dilate(self.soft_erode(input_))

    def soft_skel(self, input_):

        input1 = self.soft_open(input_)
        skel = F.relu(input1)

        for j in range(self.num_iter):
            input_ = self.soft_erode(input_)
            input1 = self.soft_open(input_)
            delta = F.relu(input_-input1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, input_):
        """Skeletonize the input prediction. 

        Args:
            input_: The input logits.

        Returns:
            The skeletonization.
        """
        return self.soft_skel(input_)

#TODO update docstrings, args, and forward
class SoftclDice(nn.Module):
    """Placeholder.

    Args:
        iter:
        smooth:
        exclude_background:
    """
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(SoftclDice, self).__init__()

        #TODO fix iter argument, soft skeletonize should get the self.iter_ instead of num_iter=10
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
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
        tprec = (torch.sum(torch.multiply(skel_input, target))+self.smooth)/(torch.sum(skel_input)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_target, input_))+self.smooth)/(torch.sum(skel_target)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return cl_dice


def soft_dice(input_: torch.Tensor, target: torch.Tensor):
    """Compute the soft dice score between the input logits and binary target.

    Args:
        input_: The input logits.
        target: The binary target.

    Returns:
        The soft dice score.
    """
    smooth = 1
    intersection = torch.sum((target * input_))
    coeff = (2. *  intersection + smooth) / (torch.sum(target) + torch.sum(input_) + smooth)
    return (1. - coeff)


#TODO update docstrings, forward 
#TODO channelwise is default for DiceLoss, should we also implement that here?

class SoftDiceclDice(nn.Module):
    """Placeholder.

    Args:
        iter:
        alpha:
        smooth:
        exclude_background:

    """
    def __init__(self, iter_=5, alpha=0.5, smooth = 1., exclude_background=False):
        super(SoftDiceclDice, self).__init__()

        #TODO fix iter argument, soft skeletonize should get the self.iter_ instead of num_iter=10

        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=5)
        self.exclude_background = exclude_background

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined soft Dice and clDice loss.

        Args:
            input_: The input logits.
            target: The binary target.
            

        Returns:
            Combined dice loss. 
        """
        if self.exclude_background:
            target = target[:, 1:, :, :]
            input_ = input_[:, 1:, :, :]
        dice = soft_dice(target, input_)
        skel_input = self.soft_skeletonize(input_)
        skel_target = self.soft_skeletonize(target)
        tprec = (torch.sum(torch.multiply(skel_input, target))+self.smooth)/(torch.sum(skel_input)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_target, input_))+self.smooth)/(torch.sum(skel_target)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return (1.0-self.alpha)*dice+self.alpha*cl_dice


