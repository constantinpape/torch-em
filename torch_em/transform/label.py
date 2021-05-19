import numpy as np
import skimage.measure
import skimage.segmentation

from ..util import ensure_array, ensure_spatial_array

try:
    from affogato.affinities import compute_affinities
except ImportError:
    compute_affinities = None


def connected_components(labels, ndim=None, ensure_zero=False):
    labels = ensure_array(labels) if ndim is None else ensure_spatial_array(labels, ndim)
    labels = skimage.measure.label(labels)
    if ensure_zero and 0 not in labels:
        labels -= 1
    return labels


def labels_to_binary(labels, background_label=0):
    return (labels != background_label).astype(labels.dtype)


# TODO ignore label + mask, smoothing
class BoundaryTransform:
    def __init__(self, mode='thick', add_binary_target=False, ndim=None):
        self.mode = mode
        self.add_binary_target = add_binary_target
        self.ndim = ndim

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        boundaries = skimage.segmentation.find_boundaries(labels)[None]
        if self.add_binary_target:
            binary = labels_to_binary(labels)[None].astype(boundaries.dtype)
            target = np.concatenate([binary, boundaries], axis=0)
        else:
            target = boundaries
        return target


# TODO affinity smoothing
class AffinityTransform:
    def __init__(self, offsets,
                 ignore_label=None,
                 add_binary_target=False,
                 add_mask=False):
        assert compute_affinities is not None
        self.offsets = offsets
        self.ndim = len(self.offsets[0])
        assert self.ndim in (2, 3)

        self.ignore_label = ignore_label
        self.add_binary_target = add_binary_target
        self.add_mask = add_mask

    def __call__(self, labels):
        labels = ensure_spatial_array(labels, self.ndim, dtype='uint64')
        affs, mask = compute_affinities(labels, self.offsets,
                                        have_ignore_label=self.ignore_label is not None,
                                        ignore_label=0 if self.ignore_label is None else self.ignore_label)
        # we use the 'disaffinity' convention for training; i.e. 1 means repulsive, 0 attractive
        affs = 1. - affs

        if self.add_binary_target:
            binary = labels_to_binary(labels)[None].astype(affs.dtype)
            assert binary.ndim == affs.ndim
            affs = np.concatenate([binary, affs], axis=0)

        if self.add_mask:
            if self.add_binary_target:
                if self.ignore_label is None:
                    mask_for_bin = np.ones((1,) + labels.shape, dtype=mask.dtype)
                else:
                    mask_for_bin = (labels == self.ignore_label)[None].astype(mask.dtype)
                assert mask.ndim == mask_for_bin.ndim
                mask = np.concatenate([mask_for_bin, mask], axis=0)
            assert affs.shape == mask.shape
            affs = np.concatenate([affs, mask.astype(affs.dtype)], axis=0)

        return affs
