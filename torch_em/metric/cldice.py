from skimage.morphology import skeletonize
import numpy as np

# From "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation":
# https://arxiv.org/abs/2003.07311

def cl_score(img, skel):
    """Compute the skeleton volume intersection. 

    Args:
        img: image
        skel: skeleton

    Returns:
        Skeleton volume intersection. 
    """
    return np.sum(img*skel)/np.sum(skel)


def clDice(input_, target):
    """Compute the clDice score between binary input and target. 

    Args:
        input_: The binary input.
        target: The binary target.

    Returns:
        clDice score.
    """
    tprec = cl_score(input_, skeletonize(target))
    tsens = cl_score(target, skeletonize(input_))
    
    return 2*tprec*tsens/(tprec+tsens)