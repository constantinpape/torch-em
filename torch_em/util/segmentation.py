from typing import Optional, List

import numpy as np

from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

import vigra

import elf.segmentation as elseg
from elf.segmentation.utils import normalize_input
from elf.segmentation.mutex_watershed import mutex_watershed


#
# segmentation functionality
#


# could also refactor this into elf
def size_filter(
    seg: np.ndarray, min_size: int, hmap: Optional[np.ndarray] = None, with_background: bool = False
) -> np.ndarray:
    if min_size == 0:
        return seg

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


def mutex_watershed_segmentation(
    foreground: np.ndarray,
    affinities: np.ndarray,
    offsets: List[List[int]],
    min_size: int = 50,
    threshold: float = 0.5,
    strides: Optional[List[int]] = None
) -> np.ndarray:
    """Computes the mutex watershed segmentation using the affinity maps for respective pixel offsets

    Arguments:
        - foreground: [np.ndarray] - The foreground background channel for the objects
        - affinities [np.ndarray] - The input affinity maps
        - offsets: [list[list[int]]] - The pixel offsets corresponding to the affinity channels
        - min_size: [int] - The minimum pixels (below which) to filter objects
        - threshold: [float] - To threshold foreground predictions
        - strides: [list[int]] - The strides used to sub-sample long range edges.
    """
    mask = (foreground >= threshold)
    if strides is None:
        strides = [2] * foreground.ndim

    seg = mutex_watershed(affinities, offsets=offsets, mask=mask, strides=strides, randomize_strides=True)
    seg = size_filter(seg.astype("uint32"), min_size=min_size, hmap=affinities, with_background=True)

    return seg


def connected_components_with_boundaries(
    foreground: np.ndarray, boundaries: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    input_ = np.clip(foreground - boundaries, 0, 1)
    seeds = label(input_ > threshold)
    mask = normalize_input(foreground > threshold)
    seg = watershed(boundaries, markers=seeds, mask=mask)
    return seg.astype("uint64")


def watershed_from_components(
    boundaries: np.ndarray,
    foreground: np.ndarray,
    min_size: int = 50,
    threshold1: float = 0.5,
    threshold2: float = 0.5,
) -> np.ndarray:
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


def watershed_from_maxima(
    boundaries: np.ndarray,
    foreground: np.ndarray,
    min_distance: int,
    min_size: int = 50,
    sigma: float = 1.0,
    threshold1: float = 0.5,
) -> np.ndarray:
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
    boundary_distances[~mask] = 0
    boundary_distances = gaussian(boundary_distances, sigma)
    seed_points = peak_local_max(boundary_distances, min_distance=min_distance, exclude_border=False)
    seeds = np.zeros(mask.shape, dtype="uint32")
    seeds[seed_points[:, 0], seed_points[:, 1]] = np.arange(1, len(seed_points) + 1)
    seg = watershed(boundaries, markers=seeds, mask=foreground)
    seg = size_filter(seg, min_size)
    return seg


def watershed_from_center_and_boundary_distances(
    center_distances: np.ndarray,
    boundary_distances: np.ndarray,
    foreground_map: np.ndarray,
    center_distance_threshold: float = 0.5,
    boundary_distance_threshold: float = 0.5,
    foreground_threshold: float = 0.5,
    distance_smoothing: float = 1.6,
    min_size: int = 0,
    debug: bool = False,
) -> np.ndarray:
    """Seeded watershed based on distance predictions to object center and boundaries.

    The seeds are computed by finding connected components where both distance predictions
    are smaller than the respective thresholds. Using both distances here should prevent merging
    narrow adjacent objects (if only using the center distance) or finding multiple seeds for non-convex
    cells (if only using the boundary distances).

    Args:
        center_distances [np.ndarray] - Distance prediction to the objcet center.
        boundary_distances [np.ndarray] - Inverted distance prediction to object boundaries.
        foreground_map [np.ndarray] - Predictio for foreground probabilities.
        center_distance_threshold [float] - Center distance predictions below this value will be
            used to find seeds (intersected with thresholded boundary distance predictions).
        boundary_distance_threshold [float] - Boundary distance predictions below this value will be
            used to find seeds (intersected with thresholded center distance predictions).
        foreground_threshold [float] - Foreground predictions above this value will be used as foreground mask.
        distance_smoothing [float] - Sigma value for smoothing the distance predictions.
        min_size [int] - Minimal object size in the segmentation result.
        debug [bool] - Return all intermediate results for debugging.

    Returns:
        np.ndarray - The instance segmentation.
    """
    center_distances = vigra.filters.gaussianSmoothing(center_distances, distance_smoothing)
    boundary_distances = vigra.filters.gaussianSmoothing(boundary_distances, distance_smoothing)

    fg_mask = foreground_map > foreground_threshold

    marker_map = np.logical_and(
        center_distances < center_distance_threshold, boundary_distances < boundary_distance_threshold
    )
    marker_map[~fg_mask] = 0
    markers = label(marker_map)

    seg = watershed(boundary_distances, markers=markers, mask=fg_mask)
    seg = size_filter(seg, min_size)

    if debug:
        debug_output = {
            "center_distances": center_distances,
            "boundary_distances": boundary_distances,
            "foreground_mask": fg_mask,
            "markers": markers,
        }
        return seg, debug_output

    return seg
