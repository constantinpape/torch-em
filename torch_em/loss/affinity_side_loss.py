from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from .dice import dice_score


def shift_tensor(tensor: torch.Tensor, offset: List[int]) -> torch.Tensor:
    """Shift a tensor by the given spatial offset.

    Args:
        tensor: A 4D (2 spatial dims) or 5D (3 spatial dims) tensor. Needs to be of float type.
        offset: A 2d or 3d spatial offset used for shifting the tensor

    Returns:
        The shifted tensor.
    """
    ndim = len(offset)
    assert ndim in (2, 3)
    diff = tensor.dim() - ndim

    # don't pad for the first dimensions
    # (usually batch and/or channel dimension)
    slice_ = diff * [slice(None)]

    # torch padding behaviour is a bit weird.
    # we use nn.ReplicationPadND
    # (torch.nn.functional.pad is even weirder and ReflectionPad is not supported in 3d)
    # still, padding needs to be given in the inverse spatial order

    # add padding in inverse spatial order
    padding = []
    for off in offset[::-1]:
        # if we have a negative offset, we need to shift "to the left",
        # which means padding at the right border
        # if we have a positive offset, we need to shift "to the right",
        # which means padding to the left border
        padding.extend([max(0, off), max(0, -off)])

    # add slicing in the normal spatial order
    for off in offset:
        if off == 0:
            slice_.append(slice(None))
        elif off > 0:
            slice_.append(slice(None, -off))
        else:
            slice_.append(slice(-off, None))

    # pad the spatial part of the tensor with replication padding
    slice_ = tuple(slice_)
    padding = tuple(padding)
    padder = nn.ReplicationPad2d if ndim == 2 else nn.ReplicationPad3d
    padder = padder(padding)
    shifted = padder(tensor)

    # slice the oadded tensor to get the spatially shifted tensor
    shifted = shifted[slice_]
    assert shifted.shape == tensor.shape

    return shifted


def invert_offsets(offsets):
    """@private
    """
    return [[-off for off in offset] for offset in offsets]


def segmentation_to_affinities(segmentation: torch.Tensor, offsets: List[List[int]]) -> torch.Tensor:
    """Transform segmentation to affinities.

    Args:
        segmentation: A 4D (2 spatial dims) or 5D (3 spatial dims) segmentation tensor.
            The channel axis (= dimension 1) needs to be a singleton.
        offsets: List of offsets for which to compute the affinities.

    Returns:
        The affinities.
    """
    assert segmentation.shape[1] == 1, f"{segmentation.shape}"
    # Shift the segmentation and substract the shifted tensor from the segmentation.
    # We need to shift in the opposite direction of the offsets, so we invert them before applying the shift.
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(segmentation.float(), off) for off in offsets_], dim=1)
    affs = (segmentation - shifted)
    # The affinities are 1, where we had the same segment id (the difference is 0) and 0 otherwise.
    affs.eq_(0.)
    return affs


def embeddings_to_affinities(embeddings: torch.Tensor, offsets: List[List[int]], delta: float) -> torch.Tensor:
    """Transform embeddings to affinities.

    Args:
        embeddings: The pixel-wise embeddings.
        offsets: The offsets for computing affinities.
        delta: The push force hinge used for training the embedding prediction network.

    Returns:
        The affinities.
    """
    # Shift the embeddings by the offsets and stack them along a new axis.
    # We need to shift in the opposite direction of the offsets, so we invert them before applying the shift.
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(embeddings, off).unsqueeze(1) for off in offsets_], dim=1)
    # Substract the embeddings from the shifted embeddings, take the norm and
    # transform to affinities based on the delta distance.
    affs = (2 * delta - torch.norm(embeddings.unsqueeze(1) - shifted, dim=2)) / (2 * delta)
    affs = torch.clamp(affs, min=0) ** 2
    return affs


class AffinitySideLoss(nn.Module):
    """Loss computed between affinities derived from predicted embeddings and a target segmentation.

    The offsets for the affinities will be derived randomly from the given `offset_ranges`.

    Args:
        offset_ranges: Ranges for the offsets to sampled.
        n_samples: Number of offsets to sample per loss computation.
        delta: The push force hinge used for training the embedding prediction network.
    """
    def __init__(self, offset_ranges: List[Tuple[int, int]], n_samples: int, delta: float):
        assert all(len(orange) == 2 for orange in offset_ranges)
        super().__init__()
        self.ndim = len(offset_ranges)
        self.offset_ranges = offset_ranges
        self.n_samples = n_samples
        self.delta = delta

    def __call__(
        self,
        input_: torch.Tensor,
        target: torch.Tensor,
        ignore_labels: Optional[List[int]] = None,
        ignore_in_variance_term: Optional[List[int]] = None,
        ignore_in_distance_term: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute loss between affinities derived from predicted embeddings and a target segmentation.

        Note: Support for the ignore labels is currently not implemented.

        Args:
            input_: The predicted embeddings.
            target: The target segmentation.
            ignore_labels: Ignore labels for the loss computation.
            ignore_in_variance_term: Ignore labels for the variance term.
            ignore_in_distance_term: Ignore labels for the distance term.

        Returns:
            The affinity loss value.
        """
        assert input_.dim() == target.dim(), f"{input_.dim()}, {target.dim()}"
        assert input_.shape[2:] == target.shape[2:]

        # Sample the offsets.
        offsets = [[np.random.randint(orange[0], orange[1]) for orange in self.offset_ranges]
                   for _ in range(self.n_samples)]

        # We invert the affinities and the target affinities,
        # so that we get boundaries as foreground, which is benefitial for the dice loss.
        # Compute affinities from emebeddings.
        affs = 1. - embeddings_to_affinities(input_, offsets, self.delta)

        # Compute groundtruth affinities from the target segmentation.
        target_affs = 1. - segmentation_to_affinities(target, offsets)
        assert affs.shape == target_affs.shape, f"{affs.shape}, {target_affs.shape}"

        # TODO implement masking the ignore labels
        # Compute the dice score between affinities and target affinities.
        return dice_score(affs, target_affs, invert=True)
