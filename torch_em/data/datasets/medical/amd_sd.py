"""The AMD-SD dataset contains annotations for lesion segmentation.

This dataset is from the publication https://doi.org/10.1038/s41597-024-03844-6.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/48777037"
CHECKSUM = "16793aac36d814e2858362b4a3b9608e6f57120cf2227a81220407571b8fb359"

MAPPING_IDS = {
    (255, 0, 255): 1,  # Pink: Ellipsoid zone (IS/OS) junction disruption
    (0, 255, 0): 2,    # Green: intraretinal fluid (IRF)
    (255, 0, 0): 3,    # Red: subretinal fluid (SRF)
    (255, 255, 0): 4,  # Yellow: subretinal hyperreflective material (SHRM)
    (0, 0, 255): 5,    # Blue: pigment epithelial detachment (PED)
}


def _preprocess_data(data_dir):
    dirs = glob(os.path.join(data_dir, "images", "*"))
    for dir in tqdm(dirs, desc="Preprocessing inputs"):
        dname = os.path.basename(dir)

        image_dir = os.path.join(data_dir, "preprocessed", dname, "images")
        label_dir = os.path.join(data_dir, "preprocessed", dname, "labels")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for ipath in natsorted(glob(os.path.join(dir, "*.png"))):
            image = imageio.imread(ipath)
            image, label = image[:, :int(image.shape[1]/2), :], image[:, int(image.shape[1]/2):, :]

            # Normalize the label intensities
            label = (label / 255).round() * 255

            # Map all the labels to one channel
            segmentation = np.zeros(label.shape[:2], dtype=np.uint8)
            for rgb, label_id in MAPPING_IDS.items():
                mask = np.all(label == np.array(rgb), axis=-1)
                segmentation[mask] = label_id

            fname = Path(os.path.basename(ipath)).with_suffix(".tif")
            imageio.imwrite(os.path.join(image_dir, fname), image)
            imageio.imwrite(os.path.join(label_dir, fname), segmentation)


def get_amd_sd_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the AMD-SD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "AMD-SD")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "AMD-SD.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    _preprocess_data(data_dir)

    return data_dir


def get_amd_sd_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> List[str]:
    """Get paths to the AMD-SD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_amd_sd_data(path, download)

    patient_ids = natsorted(glob(os.path.join(data_dir, "preprocessed", "*")))
    if split == "train":
        patient_ids = patient_ids[:100]
    elif split == "val":
        patient_ids = patient_ids[100:115]
    elif split == "test":
        patient_ids = patient_ids[115:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    raw_paths, label_paths = [], []
    for id in patient_ids:
        raw_paths.extend(natsorted(glob(os.path.join(id, "images", "*.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(id, "labels", "*.tif"))))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_amd_sd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AMD-SD dataset for lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_amd_sd_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_amd_sd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AMD-SD dataloader for lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_amd_sd_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
