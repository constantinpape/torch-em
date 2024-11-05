"""
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10969427/files/PSFHS.zip"
CHECKSUM = "3f4a8126c84640e4d1b8a4e296d0dfd599cea6529b64b9ee00e5489bfd17ea95"


def get_psfhs_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "PSFHS")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "PSFHS.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_psfhs_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[int], List[int]]:
    """
    """
    data_dir = get_psfhs_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "image_mha", "*.mha")))
    label_paths = natsorted(glob(os.path.join(data_dir, "label_mha", "*.mha")))

    return raw_paths, label_paths


def get_psfhs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_psfhs_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        ndim=2,
        is_seg_dataset=False,
        with_channels=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_psfhs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_psfhs_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
