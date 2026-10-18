"""The EMcopilot dataset contains electron micrographs of catalyst materials with manual semantic annotations.

It contains two sub-datasets of (scanning transmission) electron micrographs with binary manual labels:
- PtSn_Al2O3_clusters: PtSn clusters on an Al2O3 support (23 labeled images).
- Au_ZSM5: Au on a ZSM-5 zeolite support (110 labeled images).

The dataset is located at https://doi.org/10.5281/zenodo.18929384 (version 4) and is distributed under the
CC BY 4.0 license. It belongs to the publication https://doi.org/10.1038/s41524-025-01790-y.
Please cite the publication if you use this dataset in your research.

NOTE: The record also contains a 'PtSn_Al2O3_nanoparticle' sub-dataset, which is not supported here because its labels
do not appear to be aligned with the images (some pairs differ in shape and the masks do not match the particles).
"""

import os
import zipfile
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/18929384/files/EMcopilot_dataset.zip?download=1"
CHECKSUM = "67629f015460ae59c01a24991a12f97f8853ab582cfd18010add5a1fbd8ccc12"

FOLDERS = {
    "PtSn_Al2O3_clusters": ("01_02_PtSn_Al2O3_clusters", "02_02_PtSn_Al2O3_clusters"),
    "Au_ZSM5": ("01_03_Au_ZSM5", "02_03_Au_ZSM5"),
}


def get_emcopilot_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the EMcopilot dataset.

    NOTE: The archive is 9.8 GB large, but only the experimental images and manual labels (~600 MB) are extracted.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    data_dir = os.path.join(path, "EMcopilot_dataset")
    if os.path.exists(os.path.join(data_dir, "02_manual_labels")):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "EMcopilot_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    with zipfile.ZipFile(zip_path, "r") as f:
        members = [m for m in f.namelist() if m.startswith(("EMcopilot_dataset/01_", "EMcopilot_dataset/02_"))]
        f.extractall(path, members=members)
    os.remove(zip_path)
    return data_dir


def get_emcopilot_paths(
    path: Union[os.PathLike, str],
    name: Literal["PtSn_Al2O3_clusters", "Au_ZSM5"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the EMcopilot data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the sub-dataset. One of 'PtSn_Al2O3_clusters' or 'Au_ZSM5'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert name in FOLDERS, f"'{name}' is not a valid sub-dataset name. Choose from {list(FOLDERS)}."

    data_dir = get_emcopilot_data(path, download)
    image_folder, label_folder = FOLDERS[name]
    image_paths = natsorted(glob(os.path.join(data_dir, "01_experimental_images", image_folder, "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "02_manual_labels", label_folder, "*.png")))

    assert len(image_paths) > 0 and len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_emcopilot_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    name: Literal["PtSn_Al2O3_clusters", "Au_ZSM5"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EMcopilot dataset for semantic segmentation of catalyst particles in electron micrographs.

    The labels are binary, where 1 is the foreground (particles) and 0 is the background.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        name: The name of the sub-dataset. One of 'PtSn_Al2O3_clusters' or 'Au_ZSM5'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_emcopilot_paths(path, name, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 2)
    kwargs = util.update_kwargs(kwargs, "label_transform", torch_em.transform.label.labels_to_binary)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_emcopilot_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    name: Literal["PtSn_Al2O3_clusters", "Au_ZSM5"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for semantic segmentation of catalyst particles in the EMcopilot dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        name: The name of the sub-dataset. One of 'PtSn_Al2O3_clusters' or 'Au_ZSM5'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_emcopilot_dataset(path, patch_shape, name, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
