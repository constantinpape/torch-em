import numpy as np
import torch
import torch.nn as nn
from .dice import dice_score


def shift_tensor(tensor, offset):
    """ Shift a tensor by the given (spatial) offset.
    Arguments:
        tensor [torch.Tensor] - 4D (=2 spatial dims) or 5D (=3 spatial dims) tensor.
            Needs to be of float type.
        offset (tuple) - 2d or 3d spatial offset used for shifting the tensor
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
    return [[-off for off in offset] for offset in offsets]


def segmentation_to_affinities(segmentation, offsets):
    """ Transform segmentation to affinities.
    Arguments:
        segmentation [torch.tensor] - 4D (2 spatial dims) or 5D (3 spatial dims) segmentation tensor.
            The channel axis (= dimension 1) needs to be a singleton.
        offsets [list[tuple]] - list of offsets for which to compute the affinities.
    """
    assert segmentation.shape[1] == 1, f"{segmentation.shape}"
    # shift the segmentation and substract the shifted tensor from the segmentation
    # we need to shift in the opposite direction of the offsets, so we invert them
    # before applying the shift
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(segmentation.float(), off) for off in offsets_], dim=1)
    affs = (segmentation - shifted)
    # the affinities are 1, where we had the same segment id (the difference is 0)
    # and 0 otherwise
    affs.eq_(0.)
    return affs


def embeddings_to_affinities(embeddings, offsets, delta):
    """ Transform embeddings to affinities.
    """
    # shift the embeddings by the offsets and stack them along a new axis
    # we need to shift in the opposite direction of the offsets, so we invert them
    # before applying the shift
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(embeddings, off).unsqueeze(1) for off in offsets_], dim=1)
    # substract the embeddings from the shifted embeddings, take the norm and
    # transform to affinities based on the delta distance
    affs = (2 * delta - torch.norm(embeddings.unsqueeze(1) - shifted, dim=2)) / (2 * delta)
    affs = torch.clamp(affs, min=0) ** 2
    return affs


class AffinitySideLoss(nn.Module):
    def __init__(self, offset_ranges, n_samples, delta):
        assert all(len(orange) == 2 for orange in offset_ranges)
        super().__init__()
        self.ndim = len(offset_ranges)
        self.offset_ranges = offset_ranges
        self.n_samples = n_samples
        self.delta = delta

    def __call__(
        self,
        input_,
        target,
        ignore_labels=None,
        ignore_in_variance_term=None,
        ignore_in_distance_term=None,
    ):
        assert input_.dim() == target.dim(), f"{input_.dim()}, {target.dim()}"
        assert input_.shape[2:] == target.shape[2:]

        # sample offsets
        offsets = [[np.random.randint(orange[0], orange[1]) for orange in self.offset_ranges]
                   for _ in range(self.n_samples)]

        # we invert the affinities and the target affinities
        # so that we get boundaries as foreground, which is benefitial for the dice loss.
        # compute affinities from emebeddings
        affs = 1. - embeddings_to_affinities(input_, offsets, self.delta)

        # compute groundtruth affinities from target
        target_affs = 1. - segmentation_to_affinities(target, offsets)
        assert affs.shape == target_affs.shape, f"{affs.shape}, {target_affs.shape}"

        # TODO implement masking the ignore labels
        # compute the dice score between affinities and target affinities
        loss = dice_score(affs, target_affs, invert=True)
        return loss
