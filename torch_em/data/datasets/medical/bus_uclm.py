"""The BUS-UCLM dataset contains annotations for breast lesion segmentation in ultrasound images.

The dataset comprises breast ultrasound images from 38 patients, acquired at the Ciudad Real
General University Hospital with a Siemens ACUSON S2000 Ultrasound System between 2022 and 2023.
The ground-truth is provided as RGB masks where green denotes benign lesions, red denotes
malignant lesions and black denotes background (including normal images without any lesion).
The label ids are: 0 = background, 1 = benign, 2 = malignant.

This dataset is located at https://data.mendeley.com/datasets/7fvgj4jsp7/3 (CC BY 4.0). We use the
mirror at https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset for the download,
as the Mendeley download links are unreliable.

This dataset is from the publication https://doi.org/10.1038/s41597-025-04562-3.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "orvile/bus-uclm-breast-ultrasound-dataset"

LABEL_COLORS = {
    0: (0, 0, 0),  # background / normal
    1: (0, 255, 0),  # benign
    2: (255, 0, 0),  # malignant
}


def get_bus_uclm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BUS-UCLM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dirs = glob(os.path.join(path, "**", "BUS-UCLM"), recursive=True)
    if data_dirs:
        return data_dirs[0]

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "bus-uclm-breast-ultrasound-dataset.zip")
    util.unzip(zip_path=zip_path, dst=path)

    data_dirs = glob(os.path.join(path, "**", "BUS-UCLM"), recursive=True)
    if not data_dirs:
        raise RuntimeError(f"The dataset could not be found at '{path}' after extraction.")

    return data_dirs[0]


def get_bus_uclm_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the BUS-UCLM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bus_uclm_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    mask_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.png")))

    if len(image_paths) == 0 or len(image_paths) != len(mask_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    neu_gt_dir = os.path.join(data_dir, "masks", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    reference_colors = np.array(list(LABEL_COLORS.values()))

    gt_paths = []
    for mask_path in tqdm(mask_paths, desc="Preprocessing labels"):
        gt_path = os.path.join(neu_gt_dir, f"{Path(mask_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        mask = imageio.imread(mask_path)[..., :3].astype("float32")
        distances = np.linalg.norm(mask[..., None, :] - reference_colors[None, None, :, :], axis=-1)
        semantic_gt = np.argmin(distances, axis=-1).astype("uint8")
        imageio.imwrite(gt_path, semantic_gt, compression="zlib")

    return image_paths, gt_paths


def get_bus_uclm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BUS-UCLM dataset for breast lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_bus_uclm_paths(path, download)

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


def get_bus_uclm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BUS-UCLM dataloader for breast lesion segmentation.

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
    dataset = get_bus_uclm_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
