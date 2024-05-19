import os
from glob import glob
from typing import Union, Tuple

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


def get_m2caiseg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, r"m2caiSeg dataset")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(path=path, dataset_name="salmanmaq/m2caiseg", download=download)
    zip_path = os.path.join(path, "m2caiseg.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_m2caiseg_paths(path, split, download):
    data_dir = get_m2caiseg_data(path=path, download=download)

    if split == "val":
        split = "trainval"

    image_paths = sorted(glob(os.path.join(data_dir, split, "images", "*.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, split, "groundtruth", "*.png")))

    return image_paths, gt_paths


def get_m2caiseg_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    assert split in ["train", "val", "split"]

    image_paths, gt_paths = _get_m2caiseg_paths(path=path, split=split, download=download)

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


def get_m2caiseg_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_m2caiseg_dataset(
        path=path, split=split, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
