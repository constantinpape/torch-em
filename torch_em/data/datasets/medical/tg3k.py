"""The TG3K dataset contains annotations for thyroid gland segmentation in ultrasound images.

The dataset consists of 3585 thyroid ultrasound frames with pixel-level thyroid gland masks. The
masks are binary: 0 for background and 1 for thyroid gland. Unlike TN3K, which is bundled in the same
archive, TG3K labels the gland region instead of individual nodules and has no predefined train/test split.

The dataset is hosted at https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation, which
bundles it together with the TN3K (thyroid nodule) dataset and a copy of DDTI inside a single archive.
We download it from the Google Drive mirror linked in that repository's README.

This dataset is from the publications https://doi.org/10.1016/j.compbiomed.2022.106389 and
https://doi.org/10.1109/ISBI48211.2021.9434087. Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


GDRIVE_ID = "1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F"
CHECKSUM = "0ba1770076dab01b1f8fd661a227d80982168bc433de61c80fd266427b20cf60"


def get_tg3k_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TG3K dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Thyroid Dataset", "tg3k")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "tg3k.zip")
    util.download_source_gdrive(
        path=zip_path, url=f"https://drive.google.com/uc?id={GDRIVE_ID}", download=download, checksum=CHECKSUM,
    )
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_tg3k_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the TG3K data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_tg3k_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "thyroid-image", "*.jpg")))
    mask_paths = natsorted(glob(os.path.join(data_dir, "thyroid-mask", "*.jpg")))

    if len(image_paths) == 0 or len(image_paths) != len(mask_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    neu_gt_dir = os.path.join(data_dir, "thyroid-mask", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for mask_path in mask_paths:
        gt_path = os.path.join(neu_gt_dir, os.path.basename(mask_path).replace(".jpg", ".tif"))
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        # The masks are stored as jpegs, which introduces compression artifacts around otherwise binary values.
        mask = (imageio.imread(mask_path) > 127).astype("uint8")
        imageio.imwrite(gt_path, mask, compression="zlib")

    return image_paths, gt_paths


def get_tg3k_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TG3K dataset for thyroid gland segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_tg3k_paths(path, download)

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


def get_tg3k_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TG3K dataloader for thyroid gland segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tg3k_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
