import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Tuple, Union, Literal

import numpy as np
import imageio.v3 as imageio

import torch_em

from .. import util


LABEL_MAPS = {
    (255, 255, 255): 0,  # small white frame around the image
    (50, 50, 50): 0,  # background
    (11, 11, 11): 1,  # abdominal wall
    (21, 21, 21): 2,  # liver
    (13, 13, 13): 3,  # gastrointestinal tract
    (12, 12, 12): 4,  # fat
    (31, 31, 31): 5,  # grasper
    (23, 23, 23): 6,  # connective tissue
    (24, 24, 24): 7,  # blood
    (25, 25, 25): 8,  # cystic dust
    (32, 32, 32): 9,  # l-hook electrocautery
    (22, 22, 22): 10,  # gallbladder
    (33, 33, 33): 11,  # hepatic vein
    (5, 5, 5): 12  # liver ligament
}


def get_cholecseg8k_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "cholecseg8k.zip")
    util.download_source_kaggle(path=zip_path, dataset_name="newslab/cholecseg8k", download=download)
    util.unzip(zip_path=zip_path, dst=data_dir)
    return data_dir


def _get_cholecseg8k_paths(path, split, download):
    data_dir = get_cholecseg8k_data(path=path, download=download)

    video_dirs = natsorted(glob(os.path.join(data_dir, "video*")))
    if split == "train":
        video_dirs = video_dirs[2:-2]
    elif split == "val":
        video_dirs = [video_dirs[1], video_dirs[-2]]
    elif split == "test":
        video_dirs = [video_dirs[0], video_dirs[-1]]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    ppdir = os.path.join(data_dir, "preprocessed", split)
    if os.path.exists(ppdir):
        _image_paths = natsorted(glob(os.path.join(ppdir, "images", "*")))
        _gt_paths = natsorted(glob(os.path.join(ppdir, "masks", "*")))
        return _image_paths, _gt_paths

    os.makedirs(os.path.join(ppdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(ppdir, "masks"), exist_ok=True)

    image_paths, gt_paths = [], []
    for video_dir in tqdm(video_dirs):
        org_image_paths = natsorted(glob(os.path.join(video_dir, "video*", "*_endo.png")))
        org_gt_paths = natsorted(glob(os.path.join(video_dir, "video*", "*_endo_watershed_mask.png")))

        for org_image_path, org_gt_path in zip(org_image_paths, org_gt_paths):
            image_id = os.path.split(org_image_path)[-1]

            image_path = os.path.join(ppdir, "images", image_id)
            gt_path = os.path.join(ppdir, "masks", Path(image_id).with_suffix(".tif"))

            image_paths.append(image_path)
            gt_paths.append(gt_path)

            if os.path.exists(image_path) and os.path.exists(gt_path):
                continue

            gt = imageio.imread(org_gt_path)
            assert gt.ndim == 3
            if gt.shape[-1] != 3:  # some labels have a 4th channel which has all values as 255
                print("Found a label with inconsistent format.")
                # let's verify the case
                assert np.unique(gt[..., -1]) == 255
                gt = gt[..., :3]

            instances = np.zeros(gt.shape[:2])
            for lmap in LABEL_MAPS:
                binary_map = (gt == lmap).all(axis=2)
                instances[binary_map > 0] = LABEL_MAPS[lmap]

            shutil.copy(src=org_image_path, dst=image_path)
            imageio.imwrite(gt_path, instances, compression="zlib")

    return image_paths, gt_paths


def get_cholecseg8k_dataset(
    path: Union[str, os.PathLike],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of organs and instruments in endoscopy.

    This dataset is from Twinanda et al. - https://doi.org/10.48550/arXiv.1602.03012

    This dataset is located at https://www.kaggle.com/datasets/newslab/cholecseg8k/data

    Please cite it if you use this data in a publication.
    """
    image_paths, gt_paths = _get_cholecseg8k_paths(path=path, split=split, download=download)

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


def get_cholecseg8k_loader(
    path: Union[str, os.PathLike],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of organs and instruments in endoscopy. See `get_cholecseg_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cholecseg8k_dataset(
        path=path, patch_shape=patch_shape, split=split, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
