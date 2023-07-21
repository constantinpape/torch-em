import os
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np
import torch_em
from skimage.measure import label

from . import util

URL = "https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip"
CHECKSUM = "f7bd0db03c86b64440a16b60360ad60c0a4411f89e2c021c7ee2c8d6af3d7e86"


def _create_volume(f, key, pattern, process=None):
    images = glob(pattern)
    images.sort()
    data = np.concatenate([imageio.imread(im)[None] for im in images], axis=0)
    if process is not None:
        data = process(data)
    f.create_dataset(key, data=data, compression="gzip")


def _get_vnc_data(path, download):
    train_path = os.path.join(path, "vnc_train.h5")
    test_path = os.path.join(path, "vnc_test.h5")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "vnc.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path, remove=True)

    root = os.path.join(path, "groundtruth-drosophila-vnc-master")
    assert os.path.exists(root)

    with h5py.File(train_path, "w") as f:
        _create_volume(f, "raw", os.path.join(root, "stack1", "raw", "*.tif"))
        _create_volume(f, "labels/mitochondria", os.path.join(root, "stack1", "mitochondria", "*.png"), process=label)
        _create_volume(f, "labels/synapses", os.path.join(root, "stack1", "synapses", "*.png"), process=label)
        # TODO find the post-processing to go from neuron labels to membrane labels
        # _create_volume(f, "labels/neurons", os.path.join(root, "stack1", "membranes", "*.png"))

    with h5py.File(test_path, "w") as f:
        _create_volume(f, "raw", os.path.join(root, "stack2", "raw", "*.tif"))

    rmtree(root)


def get_vnc_mito_dataset(
    path,
    patch_shape,
    offsets=None,
    boundaries=False,
    binary=False,
    download=False,
    **kwargs
):
    """Dataset for the segmentation of mitochondria in EM.

    This dataset is from https://doi.org/10.6084/m9.figshare.856713.v1.
    Please cite it if you use this dataset for a publication.
    """
    _get_vnc_data(path, download)
    data_path = os.path.join(path, "vnc_train.h5")

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )

    raw_key = "raw"
    label_key = "labels/mitochondria"
    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_vnc_mito_loader(
    path,
    patch_shape,
    batch_size,
    offsets=None,
    boundaries=False,
    binary=False,
    download=False,
    **kwargs
):
    """Dataloader for the segmentation of mitochondria in EM. See 'get_vnc_mito_loader'."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_vnc_mito_dataset(
        path, patch_shape, download=download, offsets=offsets, boundaries=boundaries, binary=binary, **kwargs
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


# TODO implement
# TODO extra kwargs for binary / boundaries / affinities
def get_vnc_neuron_loader(path, patch_shape, download=False, **kwargs):
    raise NotImplementedError
