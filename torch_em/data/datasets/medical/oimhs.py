import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple

import numpy as np
import imageio.v3 as imageio

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/42522673"
CHECKSUM = "d93ba18964614eb9b0ba4b8dfee269efbb94ff27142e4b5ecf7cc86f3a1f9d80"

LABEL_MAPS = {
    (255, 255, 0): 1,  # choroid
    (0, 255, 0): 2,  # retina
    (0, 0, 255): 3,  # intrarentinal cysts
    (255, 0, 0): 4  # macular hole
}


def get_oimhs_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "oimhs_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_oimhs_paths(path, download):
    data_dir = get_oimhs_data(path=path, download=download)

    image_dir = os.path.join(data_dir, "preprocessed", "images")
    gt_dir = os.path.join(data_dir, "preprocessed", "gt")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    eye_dirs = natsorted(glob(os.path.join(data_dir, "Images", "*")))
    for eye_dir in tqdm(eye_dirs):
        eye_id = os.path.split(eye_dir)[-1]
        all_oct_scan_paths = natsorted(glob(os.path.join(eye_dir, "*.png")))
        for per_scan_path in all_oct_scan_paths:
            scan_id = Path(per_scan_path).stem

            image_path = os.path.join(image_dir, f"{eye_id}_{scan_id}.tif")
            gt_path = os.path.join(gt_dir, f"{eye_id}_{scan_id}.tif")
            if os.path.exists(image_path) and os.path.exists(gt_path):
                image_paths.append(image_path)
                gt_paths.append(gt_path)
                continue

            scan = imageio.imread(per_scan_path)
            image, gt = scan[:, :512, :], scan[:, 512:, :]

            instances = np.zeros(image.shape[:2])
            for lmap in LABEL_MAPS:
                binary_map = (gt == lmap).all(axis=2)
                instances[binary_map > 0] = LABEL_MAPS[lmap]

            imageio.imwrite(image_path, image, compression="zlib")
            imageio.imwrite(gt_path, instances, compression="zlib")

            image_paths.append(image_path)
            gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_oimhs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of macular hole and retinal regions in OCT scans.

    The dataset is from Ye et al. - https://doi.org/10.1038/s41597-023-02675-1.

    Please cite it if you use this dataset for your publication.
    """
    image_paths, gt_paths = _get_oimhs_paths(path=path, download=download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )

    return dataset


def get_oimhs_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader  for segmentation of macular hole and retinal regions in OCT scans.
    See `get_oimhs_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oimhs_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
