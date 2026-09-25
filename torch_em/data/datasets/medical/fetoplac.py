"""The Fetoscopy Placenta Dataset contains annotations for placental vessel segmentation
in in-vivo fetoscopic videos of twin-to-twin transfusion syndrome surgery.

It consists of 483 frames with binary vessel masks, drawn from 6 annotated video clips
(6 further clips are provided unannotated, for mosaicking, and are not used by this module).

NOTE: The original host (weiss-develop.cs.ucl.ac.uk, UCL) has become unreachable; this module
downloads the same file from the Internet Archive's Wayback Machine, which mirrors it unchanged.

The dataset is located at https://www.ucl.ac.uk/interventional-surgical-sciences/fetoscopy-placenta-data.

This dataset is from the publication https://doi.org/10.1007/978-3-030-59716-0_73.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "http://web.archive.org/web/20220412011703/https://weiss-develop.cs.ucl.ac.uk/fetoscopy-data/fetoscopy-placenta-dataset/fetoscopy-placenta-dataset.zip"  # noqa
CHECKSUM = "a7beadf24f377d80c2305b10139697eee2ecce4f3ac8564dcec604c5f2dc55df"


def get_fetoplac_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Fetoscopy Placenta Dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the annotated vessel segmentation videos.
    """
    data_dir = os.path.join(path, "Fetoscopy Placenta Dataset", "Vessel_segmentation_annotations")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "fetoscopy-placenta-dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_fetoplac_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the Fetoscopy Placenta Dataset data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_fetoplac_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "video*", "images", "*.png")))
    gt_paths = sorted(glob(os.path.join(data_dir, "video*", "masks_gt", "*_mask.png")))
    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0, "Could not find matching image/mask pairs."

    neu_gt_paths = []
    for gt_path in tqdm(gt_paths, desc="Preprocessing Fetoscopy Placenta Dataset masks"):
        neu_gt_path = os.path.join(Path(gt_path).parent, f"{Path(gt_path).stem}_binary.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)
        gt = np.mean(gt, axis=-1)
        gt = (gt > 0).astype("uint8")
        imageio.imwrite(neu_gt_path, gt, compression="zlib")

    return image_paths, neu_gt_paths


def get_fetoplac_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Fetoscopy Placenta Dataset for placental vessel segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_fetoplac_paths(path, download)

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


def get_fetoplac_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Fetoscopy Placenta Dataset dataloader for placental vessel segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fetoplac_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
