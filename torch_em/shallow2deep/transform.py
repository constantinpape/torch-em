from typing import Optional

import numpy as np
import skimage.segmentation
from scipy.ndimage.morphology import distance_transform_edt
from torch_em.util import ensure_array, ensure_spatial_array


class ForegroundTransform:
    """Transformation to convert labels into a foreground mask.

    Args:
        label_id: The label id to use for extracting the foreground mask.
            If None, all label values larger than zero will be used to compute the foreground mask.
        ndim: The dimensionality of the data.
        ignore_radius: The radius around the foreground label to set to the ignore label.
    """
    def __init__(self, label_id: Optional[int] = None, ndim: Optional[int] = None, ignore_radius: int = 1):
        self.label_id = label_id
        self.ndim = ndim
        self.ignore_radius = ignore_radius

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Apply the transformation to the segmentation data.

        Args:
            labels: The segmentation data.

        Returns:
            The foreground mask.
        """
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        target = labels > 0 if self.label_id is None else labels == self.label_id
        if self.ignore_radius > 0:
            dist = distance_transform_edt(target == 0)
            ignore_mask = np.logical_and(dist <= self.ignore_radius, target == 0)
            target[ignore_mask] = -1
        return target[None]


class BoundaryTransform:
    """Transformation to convert labels into boundaries.

    Args:
        mode: The mode for computing the boundaries.
        ndim: The dimensionality of the data.
        ignore_radius: The radius around the foreground label to set to the ignore label.
        add_binary_target: Whether to add a binary mask as additional channel.
    """
    def __init__(
        self,
        mode: str = "thick",
        ndim: Optional[int] = None,
        ignore_radius: int = 2,
        add_binary_target: bool = False
    ):
        self.mode = mode
        self.ndim = ndim
        self.ignore_radius = ignore_radius
        self.foreground_trafo = ForegroundTransform(ndim=ndim, ignore_radius=0) if add_binary_target else None

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Apply the boundary transform to the data.

        Args:
            labels: The segmentation data.

        Returns:
            The transformed labels.
        """
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        target = skimage.segmentation.find_boundaries(labels, mode=self.mode).astype("int8")

        if self.ignore_radius > 0:
            dist = distance_transform_edt(target == 0)
            ignore_mask = np.logical_and(dist <= self.ignore_radius, target == 0)
            target[ignore_mask] = -1

        if self.foreground_trafo is not None:
            target[target == 1] = 2
            fg_target = self.foreground_trafo(labels)[0]
            assert fg_target.shape == target.shape, f"{fg_target}.shape, {target.shape}"
            fg_mask = np.logical_and(fg_target == 1, target == 0)
            target[fg_mask] = 1

        return target[None]
