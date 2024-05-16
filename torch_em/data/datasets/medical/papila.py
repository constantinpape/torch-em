import os
from glob import glob
from typing import Union, Tuple

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


URL = "https://figshare.com/ndownloader/files/28454352"
CHECKSUM = "f72ed286a63e40c13fb70802d4e600cc0c74110c01da1ec33201f0389d058c98"


def get_papila_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "papila.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_papila_paths(path, download):
    data_dir = get_papila_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "FundusImages", "*.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, "ExpertsSegmentations", "ImagesWithContours", "*.jpg")))

    return image_paths, gt_paths


def get_papila_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    image_paths, gt_paths = _get_papila_paths(path=path, download=download)

    if resize_inputs:
        raw_trafo = ResizeInputs(target_shape=patch_shape, is_label=False)
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


def get_papila_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_papila_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
