import numpy as np
import skimage.segmentation
from scipy.ndimage.morphology import distance_transform_edt
from torch_em.util import ensure_array, ensure_spatial_array


class ForegroundTransform:
    def __init__(self, label_id=None, ndim=None, ignore_radius=1):
        self.label_id = label_id
        self.ndim = ndim
        self.ignore_radius = ignore_radius

    def __call__(self, labels):
        labels = ensure_array(labels) if self.ndim is None else ensure_spatial_array(labels, self.ndim)
        target = labels > 0 if self.label_id is None else labels == self.label_id
        if self.ignore_radius > 0:
            dist = distance_transform_edt(target == 0)
            ignore_mask = np.logical_and(dist <= self.ignore_radius, target == 0)
            target[ignore_mask] = -1
        return target[None]


class BoundaryTransform:
    def __init__(self, mode="thick", ndim=None, ignore_radius=2, add_binary_target=False):
        self.mode = mode
        self.ndim = ndim
        self.ignore_radius = ignore_radius
        self.foreground_trafo = ForegroundTransform(ndim=ndim, ignore_radius=0) if add_binary_target else None

    def __call__(self, labels):
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
