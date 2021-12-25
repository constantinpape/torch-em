import numpy as np
import skimage.segmentation
from scipy.ndimage.morphology import distance_transform_edt
from torch_em.util import ensure_array, ensure_spatial_array


class ForegroundTransform:
    def __init__(self, ndim=None, ignore_radius=1):
        self.ndim = ndim
        self.ignore_radius = ignore_radius

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        target = labels > 0
        if self.ignore_radius > 0:
            dist = distance_transform_edt(target == 0)
            ignore_mask = np.logical_and(dist <= self.ignore_radius, target == 0)
            target[ignore_mask] = -1
        return target[None]


class BoundaryTransform:
    def __init__(self, mode="thick", ndim=None, ignore_radius=2):
        self.mode = mode
        self.ndim = ndim
        self.ignore_radius = ignore_radius

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        target = skimage.segmentation.find_boundaries(labels, mode=self.mode).astype("int8")
        if self.ignore_radius > 0:
            dist = distance_transform_edt(target == 0)
            ignore_mask = np.logical_and(dist <= self.ignore_radius, target == 0)
            target[ignore_mask] = -1
        return target[None]
