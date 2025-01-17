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
# Segmentation Functionality
#


def size_filter(
    seg: np.ndarray, min_size: int, hmap: Optional[np.ndarray] = None, with_background: bool = False
) -> np.ndarray:
    """Apply size filter to a segmentation to remove small segments.

    Args:
        seg: The input segmentation.
        min_size: The minimal segmentation size.
        hmap: A heightmap to use for watershed based segmentation filtering.
        with_background: Whether this is a segmentation problem with background.

    Returns:
        The filtered segmentation.
    """
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
    """Compute the mutex watershed segmentation using the affinity map for given pixel offsets.

    Args:
        foreground: The foreground/background probabilities.
        affinities: The input affinity maps.
        offsets: The pixel offsets corresponding to the affinity channels.
        min_size: The minimum pixel size for objects in the output segmentation.
        threshold: The threshold for the foreground predictions.
        strides: The strides used to subsample long range edges.

    Returns:
        The instance segmentation.
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
    """Compute instance segmentation based on foreground and boundary predictions.

    Args:
        foreground: The foreground probability predictions.
        boundaries: The boundary probability predictions.
        threshold: The threshold for finding connected components.

    Returns:
        The instance segmentation.
    """
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
    """Compute an instance segmentation based on boundary and foreground predictions

    The segmentation is computed as follows:
    - Subtract the boundaries from the foreground to separate touching objects.
    - Use the connected components of the result as seeds.
    - Use the thresholded foreground predictions as mask to grow back the pieces
      lost by subtracting the boundary prediction.

    Args:
        boundaries: The boundary probability predictions.
        foreground: The foreground probability predictions.
        min_size: The minimum pixel size for objects in the output segmentation.
        threshold1: The threshold for finding connected components.
        threshold2: The threshold for growing components via watershed on the boundary predictions.

    Returns:
        The instance segmentation.
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
    """Compute an instance segmentation based on a seeded watershed from distance maxima.

    This function thresholds the boundary probabilities, computes a distance transform,
    finds the distance maxima, and then applies a seeded watershed to obtain the instance segmentation.
    Compared to `watershed_from_components` this has the advantage that objects can be better separated,
    but it may over-segment objects with complex shapes.

    The min_distance parameter controls the minimal distance between seeds, which
    corresponds to the minimal distance between object centers.

    Args:
        boundaries: The boundary probability predictions.
        foreground: The foreground probability predictions.
        min_size: The minimum pixel size for objects in the output segmentation.
        min_distance: The minimum distance between peaks, see `from skimage.feature.peak_local_max`.
        sigma: The standard deviation for smoothing the distance map before computing maxima.
        threshold1: The threshold for foreground predictions.

    Returns:
        The instance segmentation.
    """
    mask = foreground > threshold1
    boundary_distances = distance_transform_edt(boundaries < 0.1)
    boundary_distances[~mask] = 0
    boundary_distances = gaussian(boundary_distances, sigma)
    seed_points = peak_local_max(boundary_distances, min_distance=min_distance, exclude_border=False)
    seeds = np.zeros(mask.shape, dtype="uint32")
    seeds[seed_points[:, 0], seed_points[:, 1]] = np.arange(1, len(seed_points) + 1)
    seg = watershed(boundaries, markers=seeds, mask=foreground)
    return size_filter(seg, min_size)


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
    """Compute an instance segmentation via a seeded watershed on distances to object centers and boundaries.

    The seeds are computed by finding connected components where both distance predictions
    are smaller than the respective thresholds. Using both distances is supposed to prevent merging
    narrow adjacent objects (if only using the center distance) or finding multiple seeds for non-convex
    cells (if only using the boundary distances).

    Args:
        center_distances: Distance prediction to the objcet centers.
        boundary_distances: Inverted distance prediction to object boundaries.
        foreground_map: Prediction for foreground probabilities.
        center_distance_threshold: Center distance predictions below this value will be
            used to find seeds (intersected with thresholded boundary distance predictions).
        boundary_distance_threshold: oundary distance predictions below this value will be
            used to find seeds (intersected with thresholded center distance predictions).
        foreground_threshold: Foreground predictions above this value will be used as foreground mask.
        distance_smoothing: Sigma value for smoothing the distance predictions.
        min_size: Minimal object size in the segmentation result.
        debug: Return all intermediate results in a dictionary for debugging.

    Returns:
        The instance segmentation.
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
