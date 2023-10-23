import numpy as np

import vigra
import elf.segmentation as elseg
from elf.segmentation.utils import normalize_input

from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt


#
# segmentation functionality
#


# could also refactor this into elf
def size_filter(seg, min_size, hmap=None, with_background=False):
    if hmap is None:
        ids, sizes = np.unique(seg, return_counts=True)
        bg_ids = ids[sizes < min_size]
        seg[np.isin(seg, bg_ids)] = 0
        seg, _, _ = vigra.analysis.relabelConsecutive(seg.astype(np.uint), start_label=1, keep_zeros=True)
    else:
        assert hmap.ndim in (seg.ndim, seg.ndim + 1)
        hmap_ = np.max(hmap[:seg.ndim], axis=0) if hmap.ndim > seg.ndim else hmap
        if with_background:
            seg, _ = elseg.watershed.apply_size_filter(seg + 1, hmap_, min_size, exclude=[1])
            seg[seg == 1] = 0
        else:
            seg, _ = elseg.watershed.apply_size_filter(seg, hmap_, min_size)
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


def watershed_from_components(boundaries, foreground, min_size, threshold1=0.5, threshold2=0.5):
    """The default approach:
    - Subtract the boundaries from the foreground to separate touching objects.
    - Use the connected components of this as seeds.
    - Use the thresholded foreground predictions as mask to grow back the pieces
      lost by subtracting the boundary prediction.

    Arguments:
        - boundaries: [np.ndarray] - The boundaries for objects
        - foreground: [np.ndarray] - The foregrounds for objects
        - min_size: [int] - The minimum pixels (below which) to filter objects
        - threshold1: [float] - To separate touching objects (by subtracting bd and fg) above threshold
        - threshold2: [float] - To threshold foreground predictions

    Returns:
        seg: [np.ndarray] - instance segmentation
    """
    seeds = label((foreground - boundaries) > threshold1)
    mask = foreground > threshold2
    seg = watershed(boundaries, seeds, mask=mask)
    seg = size_filter(seg, min_size)
    return seg


def watershed_from_maxima(boundaries, foreground, min_size, min_distance, sigma=1.0, threshold1=0.5):
    """Find objects via seeded watershed starting from the maxima of the distance transform instead.
    This has the advantage that objects can be better separated, but it may over-segment
    if the objects have complex shapes.

    The min_distance parameter controls the minimal distance between seeds, which
    corresponds to the minimal distance between object centers.

    Arguments:
        - boundaries: [np.ndarray] - The boundaries for objects
        - foreground: [np.ndarray] - The foreground for objects
        - min_size: [int] - min. pixels (below which) to filter objects
        - min_distance: [int] - min. distance of peaks (see `from skimage.feature import peak_local_max`)
        - sigma: [float] - standard deviation for gaussian kernel. (see `from skimage.filters import gaussian`)
        - threshold1: [float] - To threshold foreground predictions

    Returns
        seg: [np.ndarray] - instance segmentation
    """
    mask = foreground > threshold1
    boundary_distances = distance_transform_edt(boundaries < 0.1)
    boundary_distances[~mask] = 0  # type: ignore
    boundary_distances = gaussian(boundary_distances, sigma)  # type: ignore
    seed_points = peak_local_max(boundary_distances, min_distance=min_distance, exclude_border=False)
    seeds = np.zeros(mask.shape, dtype="uint32")
    seeds[seed_points[:, 0], seed_points[:, 1]] = np.arange(1, len(seed_points) + 1)
    seg = watershed(boundaries, markers=seeds, mask=foreground)
    seg = size_filter(seg, min_size)
    return seg
