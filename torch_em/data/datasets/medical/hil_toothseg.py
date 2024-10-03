import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple

import cv2 as cv
import numpy as np
import imageio.v3 as imageio

import torch_em

from .. import util


URL = "https://hitl-public-datasets.s3.eu-central-1.amazonaws.com/Teeth+Segmentation.zip"
CHECKSUM = "3b628165a218a5e8d446d1313e6ecbe7cfc599a3d6418cd60b4fb78745becc2e"


def get_hil_toothseg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, r"Teeth Segmentation PNG")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "Teeth_Segmentation.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_hil_toothseg_paths(path, split, download):
    data_dir = get_hil_toothseg_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "d2", "img", "*")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "d2", "masks_machine", "*")))

    neu_gt_dir = os.path.join(data_dir, "preprocessed", "gt")
    os.makedirs(neu_gt_dir, exist_ok=True)

    neu_gt_paths = []
    for gt_path in tqdm(gt_paths):
        neu_gt_path = os.path.join(neu_gt_dir, f"{Path(gt_path).stem}.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        rgb_gt = cv.imread(gt_path)
        rgb_gt = cv.cvtColor(rgb_gt, cv.COLOR_BGR2RGB)
        incolors = np.unique(rgb_gt.reshape(-1, rgb_gt.shape[2]), axis=0)

        # the first id is always background, let's remove it
        if np.array_equal(incolors[0], np.array([0, 0, 0])):
            incolors = incolors[1:]

        instances = np.zeros(rgb_gt.shape[:2])

        color_to_id = {tuple(cvalue): i for i, cvalue in enumerate(incolors, start=1)}
        for cvalue, idx in color_to_id.items():
            binary_map = (rgb_gt == cvalue).all(axis=2)
            instances[binary_map] = idx

        imageio.imwrite(neu_gt_path, instances)

    if split == "train":
        image_paths, neu_gt_paths = image_paths[:450], neu_gt_paths[:450]
    elif split == "val":
        image_paths, neu_gt_paths = image_paths[425:475], neu_gt_paths[425:475]
    elif split == "test":
        image_paths, neu_gt_paths = image_paths[475:], neu_gt_paths[475:]
    else:
        raise ValueError(f"{split} is not a valid split.")

    return image_paths, neu_gt_paths


def get_hil_toothseg_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of teeth in panoramic dental radiographs.

    This dataset is from Lopez et al. - https://www.mdpi.com/1424-8220/21/9/3110.

    The database is located at https://humansintheloop.org/resources/datasets/teeth-segmentation-dataset/.

    Please cite it if you use this dataset in a publication.
    """
    image_paths, gt_paths = _get_hil_toothseg_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_hil_toothseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of teeth in panoramic dental radiographs. See `get_hil_toothseg_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hil_toothseg_dataset(
        path=path, split=split, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
