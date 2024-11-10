"""The OIMHS dataset contains annotations for macular hole and retinal region segmentation in OCT images.

The dataset is from the publication https://doi.org/10.1038/s41597-023-02675-1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import json
import numpy as np
import imageio.v3 as imageio
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

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


def get_oimhs_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OIMHS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "oimhs_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _create_splits(data_dir, split_file, test_fraction=0.2):
    eye_dirs = natsorted(glob(os.path.join(data_dir, "Images", "*")))

    # let's split the data
    main_split, test_split = train_test_split(eye_dirs, test_size=test_fraction)
    train_split, val_split = train_test_split(main_split, test_size=0.1)

    decided_splits = {"train": train_split, "val": val_split, "test": test_split}

    with open(split_file, "w") as f:
        json.dump(decided_splits, f)


def _get_per_split_dirs(split_file, split):
    with open(split_file, "r") as f:
        data = json.load(f)

    return data[split]


def get_oimhs_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the OIMHS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_oimhs_data(path=path, download=download)

    image_dir = os.path.join(data_dir, "preprocessed", "images")
    gt_dir = os.path.join(data_dir, "preprocessed", "gt")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    split_file = os.path.join(path, "split_file.json")
    if not os.path.exists(split_file):
        _create_splits(data_dir, split_file)

    eye_dirs = _get_per_split_dirs(split_file=split_file, split=split)

    image_paths, gt_paths = [], []
    for eye_dir in tqdm(eye_dirs, desc="Preprocessing inputs"):
        eye_id = os.path.split(eye_dir)[-1]
        all_oct_scan_paths = natsorted(glob(os.path.join(eye_dir, "*.png")))
        for per_scan_path in all_oct_scan_paths:
            scan_id = Path(per_scan_path).stem

            image_path = os.path.join(image_dir, f"{eye_id}_{scan_id}.tif")
            gt_path = os.path.join(gt_dir, f"{eye_id}_{scan_id}.tif")

            image_paths.append(image_path)
            gt_paths.append(gt_path)

            if os.path.exists(image_path) and os.path.exists(gt_path):
                continue

            scan = imageio.imread(per_scan_path)
            image, gt = scan[:, :512, :], scan[:, 512:, :]

            instances = np.zeros(image.shape[:2])
            for lmap in LABEL_MAPS:
                binary_map = (gt == lmap).all(axis=2)
                instances[binary_map > 0] = LABEL_MAPS[lmap]

            imageio.imwrite(image_path, image, compression="zlib")
            imageio.imwrite(gt_path, instances, compression="zlib")

    return image_paths, gt_paths


def get_oimhs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OIMHS dataset for segmentation of macular hole and retinal regions in OCT scans.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_oimhs_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_oimhs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OIMHS dataloader for segmentation of macular hole and retinal regions in OCT scans.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oimhs_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
