import os
from glob import glob
from typing import Union, Tuple

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
CHECKSUM = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"


def get_kvasir_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "Kvasir-SEG")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "kvasir-seg.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_kvasir_paths(path, download):
    data_dir = get_kvasir_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, "masks", "*.jpg")))

    return image_paths, gt_paths


def get_kvasir_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    image_paths, gt_paths = _get_kvasir_paths(path=path, download=download)

    if resize_inputs:
        raw_trafo = ResizeInputs(target_shape=patch_shape, is_rgb=True)
        label_trafo = ResizeInputs(target_shape=patch_shape, is_label=True)
        patch_shape = None
    else:
        patch_shape = patch_shape
        raw_trafo, label_trafo = None, None

    dataset = ImageCollectionDataset(
        raw_image_paths=image_paths,
        label_image_paths=gt_paths,
        patch_shape=patch_shape,
        raw_transform=raw_trafo,
        label_transform=label_trafo,
        **kwargs
    )

    return dataset


def get_kvasir_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_kvasir_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
