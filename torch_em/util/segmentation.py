import numpy as np
import elf.segmentation as elseg
import vigra

from elf.segmentation.utils import normalize_input
from skimage.measure import label
from skimage.segmentation import watershed


#
# segmentation functionality
#

# could also refactor this into elf
def size_filter(seg, min_size, hmap=None):
    if hmap is None:
        ids, sizes = np.unique(seg, return_counts=True)
        bg_ids = ids[sizes < min_size]
        seg[np.isin(seg, bg_ids)] = 0
        vigra.analysis.relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=True)
    else:
        raise NotImplementedError  # TODO
    return seg


def mutex_watershed(affinities, offsets, mask=None, strides=None):
    return elseg.mutex_watershed(
        affinities, offsets, mask=mask, strides=strides, randomize_strides=True
    ).astype("uint64")


def connected_components_with_boundaries(foreground, boundaries, threshold=0.5):
    input_ = np.clip(foreground - boundaries, 0, 1)
    seeds = label(input_ > threshold)
    mask = normalize_input(foreground > threshold)
    seg = watershed(boundaries, markers=seeds, mask=mask)
    return seg.astype("uint64")
