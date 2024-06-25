import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple

import imageio.v3 as imageio

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


URL = {
    "train": "https://www.dropbox.com/sh/z4hbbzqai0ilqht/AADp_8oefNFs2bjC2kzl2_Fqa/training.zip?dl=1",
    "test": "https://www.dropbox.com/sh/z4hbbzqai0ilqht/AABuUJQJ5yG5oCuziYzYu8jWa/test.zip?dl=1"
}

CHECKSUM = {
    "train": "7101e19598e2b7aacdbd5e6e7575057b9154a4aaec043e0f4e28902bf4e2e209",
    "test": "d76c95c98a0353487ffb63b3bb2663c00ed1fde7d8fdfd8c3282c6e310a02731"
}


def get_drive_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "training")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "training.zip")
    util.download_source_gdrive(
        path=zip_path, url=URL["train"], download=download, checksum=CHECKSUM["train"], download_type="zip",
    )
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_drive_ground_truth(data_dir):
    gt_paths = sorted(glob(os.path.join(data_dir, "1st_manual", "*.gif")))

    neu_gt_dir = os.path.join(data_dir, "gt")
    if os.path.exists(neu_gt_dir):
        return sorted(glob(os.path.join(neu_gt_dir, "*.tif")))
    else:
        os.makedirs(neu_gt_dir, exist_ok=True)

    neu_gt_paths = []
    for gt_path in gt_paths:
        gt = imageio.imread(gt_path).squeeze()
        neu_gt_path = os.path.join(
            neu_gt_dir, Path(os.path.split(gt_path)[-1]).with_suffix(".tif")
        )
        imageio.imwrite(neu_gt_path, (gt > 0).astype("uint8"))
        neu_gt_paths.append(neu_gt_path)

    return neu_gt_paths


def _get_drive_paths(path, download):
    data_dir = get_drive_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "images", "*.tif")))
    gt_paths = _get_drive_ground_truth(data_dir)

    return image_paths, gt_paths


def get_drive_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of retinal blood vessels in fundus images.

    This dataset is from the "DRIVE" challenge:
    - https://drive.grand-challenge.org/
    - https://doi.org/10.1109/TMI.2004.825627

    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_drive_paths(path=path, download=download)

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


def get_drive_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of retinal blood vessels in fundus images. See `get_drive_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_drive_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
