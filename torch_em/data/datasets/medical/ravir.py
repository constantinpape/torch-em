"""The RAVIR dataset contains annotations for segmentation of retinal arteries and veins
in infrared reflectance imaging.

The dataset is from the RAVIR challenge: https://ravir.grand-challenge.org/RAVIR/.
This dataset is from the publication https://doi.org/10.1109/JBHI.2022.3163352.
Please cite them if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from typing import Union, Tuple, List

import torch_em

from .. import util


URL = "https://drive.google.com/uc?export=download&id=1ZlZoSStvE9VCRq3bJiGhQH931EF0h3hh"
CHECKSUM = "b9cc2e84660ab4ebeb583d510bd71057faf596a99ed6d1e27aee361e3a3f1381"


def get_ravir_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RAVIR dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "RAVIR_Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ravir.zip")
    util.download_source_gdrive(
        path=zip_path, url=URL, download=download, checksum=CHECKSUM, download_type="zip"
    )
    util.unzip(zip_path=zip_path, dst=path)

    # Updating the folder structure.
    tmp_dir = os.path.join(path, r"RAVIR Dataset")
    assert os.path.exists(tmp_dir), "Something went wrong with the data download"
    shutil.move(tmp_dir, data_dir)

    return data_dir


def get_ravir_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[int], List[int]]:
    """Get paths to the RAVIR dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ravir_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "train", "training_images", "*")))
    gt_paths = sorted(glob(os.path.join(data_dir, "train", "training_masks", "*")))

    return image_paths, gt_paths


def get_ravir_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the RAVIR dataset for segmentation of retinal arteries and veins.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ravir_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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


def get_ravir_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the RAVIR dataloader for segmentation of retinal arteries and veins.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the inputs to the patch shape.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ravir_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
