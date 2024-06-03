import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip"
CHECKSUM = "582bf184af993541a4958a4d209a6a44e3bbe702a5daefaf9fb1733a4e7a6e39"


def get_mbh_seg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "label_192")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "label_192.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_mbh_seg_paths(path, download):
    data_dir = get_mbh_seg_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(data_dir, r"ground truths", "*.nii.gz")))

    return image_paths, gt_paths


def get_mbh_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    image_paths, gt_paths = _get_mbh_seg_paths(path=path, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_mbh_seg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mbh_seg_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
