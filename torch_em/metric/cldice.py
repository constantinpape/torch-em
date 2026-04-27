import numpy as np
import torch
from skimage.morphology import skeletonize

from ..loss.cldice import SoftSkeletonize

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
    return np.sum(img * skel) / np.sum(skel)


def clDice(input_, target, skeletonize_method="skimage", num_iter=5):
    """Compute the clDice score between binary input and target.

    Args:
        input_: The binary input.
        target: The binary target.
        skeletonize_method: The skeletonziation method. Either `skimage` for
            `skimage.morphology.skeletonize` or `soft` for `torch_em.loss.SoftSkeletonize`.
        num_iter: Number of iterations for soft skeletonization.
            Only used if skeletonize_method is `soft`

    Returns:
        The clDice score.
    """
    if input_.shape != target.shape:
        raise ValueError(f"Expect input and target of same shape, got: {input_.shape}, {target.shape}.")

    if skeletonize_method == "skimage":
        skel_input = skeletonize(input_)
        skel_target = skeletonize(target)

    elif skeletonize_method == "soft":
        soft_skeletonize = SoftSkeletonize(num_iter=num_iter)

        # add batch and channel dims for `SoftSkeletonize`
        input_tensor = torch.from_numpy(input_).float().unsqueeze(0).unsqueeze(0)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)

        # convert skeletons back to numpy
        skel_input = soft_skeletonize(input_tensor).squeeze().numpy()
        skel_target = soft_skeletonize(target_tensor).squeeze().numpy()
    else:
        raise ValueError("Unknown option for `skeletonize_method`. Valid options are `skimage` and `soft`.")

    # Tprec = |S_P ∩ V_L| / |S_P|
    # Tsens = |S_L ∩ V_P| / |S_L|
    t_prec = cl_score(target, skel_input)
    t_sens = cl_score(input_, skel_target)

    return 2.*(t_prec*t_sens) / max(t_prec+t_sens, 1e-7)
