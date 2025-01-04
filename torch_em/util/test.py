"""@private
"""
import os
import imageio
import h5py
import numpy as np
import torch

from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed


def make_gt(spatial_shape, n_batches=None, with_channels=False, with_background=False, dtype=None):
    def _make_gt():
        seeds = np.random.rand(*spatial_shape)
        seeds = label(seeds > 0.99)
        hmap = distance_transform_edt(seeds == 0)
        if with_background:
            mask = np.random.rand(*spatial_shape) > 0.5
            assert mask.shape == hmap.shape
        else:
            mask = None
        return watershed(hmap, markers=seeds, mask=mask)

    if n_batches is None and not with_channels:
        seg = _make_gt()
    elif n_batches is None and with_channels:
        seg = _make_gt[None]
    else:
        seg = []
        for _ in range(n_batches):
            batch_seg = _make_gt()
            if with_channels:
                batch_seg = batch_seg[None]
            seg.append(batch_seg[None])
        seg = np.concatenate(seg, axis=0)
    if dtype is not None:
        seg = seg.astype(dtype)
    return torch.from_numpy(seg)


def create_segmentation_test_data(data_path, raw_key, label_key, shape, chunks):
    with h5py.File(data_path, "a") as f:
        f.create_dataset(raw_key, data=np.random.rand(*shape), chunks=chunks)
        f.create_dataset(label_key, data=np.random.randint(0, 4, size=shape), chunks=chunks)


def create_image_collection_test_data(folder, n_images, min_shape, max_shape):
    im_folder = os.path.join(folder, "images")
    label_folder = os.path.join(folder, "labels")
    os.makedirs(im_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for i in range(n_images):
        shape = tuple(np.random.randint(mins, maxs) for mins, maxs in zip(min_shape, max_shape))
        raw = np.random.rand(*shape).astype("int16")
        label = np.random.randint(0, 4, size=shape)
        imageio.imwrite(os.path.join(im_folder, f"im_{i}.tif"), raw)
        imageio.imwrite(os.path.join(label_folder, f"im_{i}.tif"), label)
