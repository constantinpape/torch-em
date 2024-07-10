from typing import Optional

import numpy as np
import skimage.measure
import skimage.segmentation
import vigra

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


def label_consecutive(labels, with_background=True):
    if with_background:
        seg = skimage.segmentation.relabel_sequential(labels)[0]
    else:
        if 0 in labels:
            labels += 1
        seg = skimage.segmentation.relabel_sequential(labels)[0]
        assert seg.min() == 1
        seg -= 1
    return seg


class MinSizeLabelTransform:
    def __init__(self, min_size=None, ndim=None, ensure_zero=False):
        self.min_size = min_size
        self.ndim = ndim
        self.ensure_zero = ensure_zero

    def __call__(self, labels):
        components = connected_components(labels, ndim=self.ndim, ensure_zero=self.ensure_zero)
        if self.min_size is not None:
            for component in np.unique(components)[1:]:  # Skip background (label 0)
                component_mask = (components == component)
                component_size = np.sum(component_mask)
                if component_size < self.min_size:
                    # make pixels to background
                    components[component_mask] = 0

        return components


# TODO smoothing
class BoundaryTransform:
    def __init__(self, mode="thick", add_binary_target=False, ndim=None):
        self.mode = mode
        self.add_binary_target = add_binary_target
        self.ndim = ndim

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        boundaries = skimage.segmentation.find_boundaries(labels, mode=self.mode)[None]
        if self.add_binary_target:
            binary = labels_to_binary(labels)[None].astype(boundaries.dtype)
            target = np.concatenate([binary, boundaries], axis=0)
        else:
            target = boundaries
        return target


# TODO smoothing
class NoToBackgroundBoundaryTransform:
    def __init__(self, bg_label=0, mask_label=-1, mode="thick", add_binary_target=False, ndim=None):
        self.bg_label = bg_label
        self.mask_label = mask_label
        self.mode = mode
        self.ndim = ndim
        self.add_binary_target = add_binary_target

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        # calc normal boundaries
        boundaries = skimage.segmentation.find_boundaries(labels, mode=self.mode)[None]

        # make label image binary and calculate to-background-boundaries
        labels_binary = (labels != self.bg_label)
        to_bg_boundaries = skimage.segmentation.find_boundaries(labels_binary, mode=self.mode)[None]

        # mask the to-background-boundaries
        boundaries = boundaries.astype(np.int8)
        boundaries[to_bg_boundaries] = self.mask_label

        if self.add_binary_target:
            binary = labels_to_binary(labels, self.bg_label).astype(boundaries.dtype)
            binary[labels == self.mask_label] = self.mask_label
            target = np.concatenate([binary[None], boundaries], axis=0)
        else:
            target = boundaries

        return target


# TODO smoothing
class BoundaryTransformWithIgnoreLabel:
    def __init__(self, ignore_label=-1, mode="thick", add_binary_target=False, ndim=None):
        self.ignore_label = ignore_label
        self.mode = mode
        self.ndim = ndim
        self.add_binary_target = add_binary_target

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        # calculate the normal boundaries
        boundaries = skimage.segmentation.find_boundaries(labels, mode=self.mode)[None]

        # calculate the boundaries for the ignore label
        labels_ignore = (labels == self.ignore_label)
        to_ignore_boundaries = skimage.segmentation.find_boundaries(labels_ignore, mode=self.mode)[None]

        # mask the to-background-boundaries
        boundaries = boundaries.astype(np.int8)
        boundaries[to_ignore_boundaries] = self.ignore_label

        if self.add_binary_target:
            binary = labels_to_binary(labels).astype(boundaries.dtype)
            binary[labels == self.ignore_label] = self.ignore_label
            target = np.concatenate([binary[None], boundaries], axis=0)
        else:
            target = boundaries

        return target


# TODO affinity smoothing
class AffinityTransform:
    def __init__(self, offsets,
                 ignore_label=None,
                 add_binary_target=False,
                 add_mask=False,
                 include_ignore_transitions=False):
        assert compute_affinities is not None
        self.offsets = offsets
        self.ndim = len(self.offsets[0])
        assert self.ndim in (2, 3)

        self.ignore_label = ignore_label
        self.add_binary_target = add_binary_target
        self.add_mask = add_mask
        self.include_ignore_transitions = include_ignore_transitions

    def add_ignore_transitions(self, affs, mask, labels):
        ignore_seg = (labels == self.ignore_label).astype(labels.dtype)
        ignore_transitions, invalid_mask = compute_affinities(ignore_seg, self.offsets)
        invalid_mask = np.logical_not(invalid_mask)
        # NOTE affinity convention returned by affogato: transitions are marked by 0
        ignore_transitions = ignore_transitions == 0
        ignore_transitions[invalid_mask] = 0
        affs[ignore_transitions] = 1
        mask[ignore_transitions] = 1
        return affs, mask

    def __call__(self, labels):
        dtype = "uint64"
        if np.dtype(labels.dtype) in (np.dtype("int16"), np.dtype("int32"), np.dtype("int64")):
            dtype = "int64"
        labels = ensure_spatial_array(labels, self.ndim, dtype=dtype)
        affs, mask = compute_affinities(labels, self.offsets,
                                        have_ignore_label=self.ignore_label is not None,
                                        ignore_label=0 if self.ignore_label is None else self.ignore_label)
        # we use the "disaffinity" convention for training; i.e. 1 means repulsive, 0 attractive
        affs = 1. - affs

        # remove transitions to the ignore label from the mask
        if self.ignore_label is not None and self.include_ignore_transitions:
            affs, mask = self.add_ignore_transitions(affs, mask, labels)

        if self.add_binary_target:
            binary = labels_to_binary(labels)[None].astype(affs.dtype)
            assert binary.ndim == affs.ndim
            affs = np.concatenate([binary, affs], axis=0)

        if self.add_mask:
            if self.add_binary_target:
                if self.ignore_label is None:
                    mask_for_bin = np.ones((1,) + labels.shape, dtype=mask.dtype)
                else:
                    mask_for_bin = (labels != self.ignore_label)[None].astype(mask.dtype)
                assert mask.ndim == mask_for_bin.ndim
                mask = np.concatenate([mask_for_bin, mask], axis=0)
            assert affs.shape == mask.shape
            affs = np.concatenate([affs, mask.astype(affs.dtype)], axis=0)

        return affs


class OneHotTransform:
    def __init__(self, class_ids=None):
        self.class_ids = list(range(class_ids)) if isinstance(class_ids, int) else class_ids

    def __call__(self, labels):
        class_ids = np.unique(labels).tolist() if self.class_ids is None else self.class_ids
        n_classes = len(class_ids)
        one_hot = np.zeros((n_classes,) + labels.shape, dtype="float32")
        for i, class_id in enumerate(class_ids):
            one_hot[i][labels == class_id] = 1.0
        return one_hot


class DistanceTransform:
    """Compute distances to foreground in the labels.

    Args:
        distances: Whether to compute the absolute distances.
        directed_distances: Whether to compute the directed distances (vector distances).
        normalize: Whether to normalize the computed distances.
        max_distance: Maximal distance at which to threshold the distances.
        foreground_id: Label id to which the distance is compute.
        invert Whether to invert the distances:
        func: Normalization function for the distances.
    """
    eps = 1e-7

    def __init__(
        self,
        distances: bool = True,
        directed_distances: bool = False,
        normalize: bool = True,
        max_distance: Optional[float] = None,
        foreground_id=1,
        invert=False,
        func=None
    ):
        if sum((distances, directed_distances)) == 0:
            raise ValueError("At least one of 'distances' or 'directed_distances' must be set to 'True'")
        self.directed_distances = directed_distances
        self.distances = distances
        self.normalize = normalize
        self.max_distance = max_distance
        self.foreground_id = foreground_id
        self.invert = invert
        self.func = func

    def _compute_distances(self, directed_distances):
        distances = np.linalg.norm(directed_distances, axis=0)
        if self.max_distance is not None:
            distances = np.clip(distances, 0, self.max_distance)
        if self.normalize:
            distances /= (distances.max() + self.eps)
        if self.invert:
            distances = distances.max() - distances
        if self.func is not None:
            distances = self.func(distances)
        return distances

    def _compute_directed_distances(self, directed_distances):
        if self.max_distance is not None:
            directed_distances = np.clip(directed_distances, -self.max_distance, self.max_distance)
        if self.normalize:
            directed_distances /= (np.abs(directed_distances).max(axis=(1, 2), keepdims=True) + self.eps)
        if self.invert:
            directed_distances = directed_distances.max(axis=(1, 2), keepdims=True) - directed_distances
        if self.func is not None:
            directed_distances = self.func(directed_distances)
        return directed_distances

    def _get_distances_for_empty_labels(self, labels):
        shape = labels.shape
        fill_value = 0.0 if self.invert else np.sqrt(np.linalg.norm(list(shape)) ** 2 / 2)
        data = np.full((labels.ndim,) + shape, fill_value)
        return data

    def __call__(self, labels):
        distance_mask = (labels == self.foreground_id).astype("uint32")
        # the distances are not computed corrected if they are all zero
        # so this case needs to be handled separately
        if distance_mask.sum() == 0:
            directed_distances = self._get_distances_for_empty_labels(labels)
        else:
            ndim = distance_mask.ndim
            to_channel_first = (ndim,) + tuple(range(ndim))
            directed_distances = vigra.filters.vectorDistanceTransform(distance_mask).transpose(to_channel_first)

        if self.distances:
            distances = self._compute_distances(directed_distances)

        if self.directed_distances:
            directed_distances = self._compute_directed_distances(directed_distances)

        if self.distances and self.directed_distances:
            return np.concatenate((distances[None], directed_distances), axis=0)
        if self.distances:
            return distances
        if self.directed_distances:
            return directed_distances


class PerObjectDistanceTransform:
    """Compute normalized distances per object in a segmentation.

    Args:
        distances: Whether to compute the undirected distances.
        boundary_distances: Whether to compute the distances to the object boundaries.
        directed_distances: Whether to compute the directed distances (vector distances).
        foreground: Whether to return a foreground channel.
        apply_label: Whether to apply connected components to the labels before computing distances.
        correct_centers: Whether to correct centers that are not in the objects.
        min_size: Minimal size of objects for distance calculdation.
        distance_fill_value: Fill value for the distances outside of objects.
    """
    eps = 1e-7

    def __init__(
        self,
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=False,
        apply_label=True,
        correct_centers=True,
        min_size=0,
        distance_fill_value=1.0,
    ):
        if sum([distances, directed_distances, boundary_distances]) == 0:
            raise ValueError("At least one of distances or directed distances has to be passed.")
        self.distances = distances
        self.boundary_distances = boundary_distances
        self.directed_distances = directed_distances
        self.foreground = foreground
        self.instances = instances

        self.apply_label = apply_label
        self.correct_centers = correct_centers
        self.min_size = min_size
        self.distance_fill_value = distance_fill_value

    def compute_normalized_object_distances(self, mask, boundaries, bb, center, distances):
        # Crop the mask and generate array with the correct center.
        cropped_mask = mask[bb]
        cropped_center = tuple(ce - b.start for ce, b in zip(center, bb))

        # The centroid might not be inside of the object.
        # In this case we correct the center by taking the maximum of the distance to the boundary.
        # Note: the centroid is still the best estimate for the center, as long as it's in the object.
        correct_center = not cropped_mask[cropped_center]

        # Compute the boundary distances if necessary.
        # (Either if we need to correct the center, or compute the boundary distances anyways.)
        if correct_center or self.boundary_distances:
            # Crop the boundary mask and compute the boundary distances.
            cropped_boundary_mask = boundaries[bb]
            boundary_distances = vigra.filters.distanceTransform(cropped_boundary_mask)
            boundary_distances[~cropped_mask] = 0
            max_dist_point = np.unravel_index(np.argmax(boundary_distances), boundary_distances.shape)

        # Set the crop center to the max dist point
        if correct_center:
            # Find the center (= maximal distance from the boundaries).
            cropped_center = max_dist_point

        cropped_center_mask = np.zeros_like(cropped_mask, dtype="uint32")
        cropped_center_mask[cropped_center] = 1

        # Compute the directed distances,
        if self.distances or self.directed_distances:
            this_distances = vigra.filters.vectorDistanceTransform(cropped_center_mask)
        else:
            this_distances = None

        # Keep only the specified distances:
        if self.distances and self.directed_distances:  # all distances
            # Compute the undirected ditacnes from directed distances and concatenate,
            undir = np.linalg.norm(this_distances, axis=-1, keepdims=True)
            this_distances = np.concatenate([undir, this_distances], axis=-1)

        elif self.distances:  # only undirected distances
            # Compute the undirected distances from directed distances and keep only them.
            this_distances = np.linalg.norm(this_distances, axis=-1, keepdims=True)

        elif self.directed_distances:  # only directed distances
            pass  # We don't have to do anything becasue the directed distances are already computed.

        # Add an extra channel for the boundary distances if specified.
        if self.boundary_distances:
            boundary_distances = (boundary_distances[max_dist_point] - boundary_distances)[..., None]
            if this_distances is None:
                this_distances = boundary_distances
            else:
                this_distances = np.concatenate([this_distances, boundary_distances], axis=-1)

        # Set distances outside of the mask to zero.
        this_distances[~cropped_mask] = 0

        # Normalize the distances.
        spatial_axes = tuple(range(mask.ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        # Set the distance values in the global result.
        distances[bb][cropped_mask] = this_distances[cropped_mask]

        return distances

    def __call__(self, labels):
        # Apply label (connected components) if specified.
        if self.apply_label:
            labels = skimage.measure.label(labels).astype("uint32")
        else:  # Otherwise just relabel the segmentation.
            labels = vigra.analysis.relabelConsecutive(labels)[0].astype("uint32")

        # Filter out small objects if min_size is specified.
        if self.min_size > 0:
            ids, sizes = np.unique(labels, return_counts=True)
            discard_ids = ids[sizes < self.min_size]
            labels[np.isin(labels, discard_ids)] = 0
            labels = vigra.analysis.relabelConsecutive(labels)[0].astype("uint32")

        # Compute the boundaries. They will be used to determine the most central point,
        # and if 'self.boundary_distances is True' to add the boundary distances.
        boundaries = skimage.segmentation.find_boundaries(labels, mode="inner").astype("uint32")

        # Compute region properties to derive bounding boxes and centers.
        ndim = labels.ndim
        props = skimage.measure.regionprops(labels)
        bounding_boxes = {
            prop.label: tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim))
            for prop in props
        }

        # Compute the object centers from centroids.
        centers = {prop.label: np.round(prop.centroid).astype("int") for prop in props}

        # Compute how many distance channels we have.
        n_channels = 0
        if self.distances:  # We need one channel for the overall distances.
            n_channels += 1
        if self.boundary_distances:  # We need one channel for the boundary distances.
            n_channels += 1
        if self.directed_distances:  # And ndim channels for directed distances.
            n_channels += ndim

        # Compute the per object distances.
        distances = np.full(labels.shape + (n_channels,), self.distance_fill_value, dtype="float32")
        for prop in props:
            label_id = prop.label
            mask = labels == label_id
            distances = self.compute_normalized_object_distances(
                mask, boundaries, bounding_boxes[label_id], centers[label_id], distances
            )

        # Bring the distance channel to the first dimension.
        to_channel_first = (ndim,) + tuple(range(ndim))
        distances = distances.transpose(to_channel_first)

        # Add the foreground mask as first channel if specified.
        if self.foreground:
            binary_labels = (labels > 0).astype("float32")
            distances = np.concatenate([binary_labels[None], distances], axis=0)

        if self.instances:
            distances = np.concatenate([labels[None], distances], axis=0)

        return distances
