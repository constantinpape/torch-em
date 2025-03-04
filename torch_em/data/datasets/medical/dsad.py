"""
"""

import os
from glob import glob
from natsort import natsorted

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/38494425"
CHECKSUM = ""  # TODO


def get_dsad_data(path, download):
    """
    """
    data_dir = os.path.join(path, "")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_dsad_paths(path, split, download):
    """
    """
    data_dir = get_dsad_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*")))

    return raw_paths, label_paths


def get_dsad_dataset(
    path, patch_shape, split, resize_inputs: bool = False,  download: bool = False, **kwargs
):
    """
    """
    raw_paths, label_paths = get_dsad_paths(path, split, download)

    ...

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_dsad_loader(
    path, batch_size, patch_shape, split, resize_inputs: bool = False,  download: bool = False, **kwargs
):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dsad_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
