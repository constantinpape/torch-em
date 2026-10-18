"""The U2OS FIB-SEM dataset contains training data for segmenting organelles in a FIB-SEM volume
of a human U2-OS (bone osteosarcoma) cell.

It contains four sub-datasets with binary semantic annotations of 2D slices:
- ne: Nuclear envelope (44 slices of 1235 x 530 pixels).
- mitochondria: Mitochondria (45 slices of 692 x 1697 pixels).
- er: Endoplasmic reticulum (45 slices of 1385 x 3394 pixels).
- golgi: Golgi apparatus (45 slices of 1385 x 3394 pixels).

For mitochondria, ER and Golgi the data is distributed as small stacks (9 or 5 slices), where only the central slice
is annotated. Only the annotated slices are used here.

The dataset is located at https://doi.org/10.5281/zenodo.10043461 and is distributed under the CC BY 4.0 license.
The full FIB-SEM volume is available at EMPIAR-11746. The annotations were created for training
the networks in DeepMIB (Belevich & Jokitalo, https://doi.org/10.1371/journal.pcbi.1008374).
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "ne": "https://zenodo.org/records/10043461/files/1_NE.zip?download=1",
    "mitochondria": "https://zenodo.org/records/10043461/files/1_Mitochondria.zip?download=1",
    "er": "https://zenodo.org/records/10043461/files/1_ER.zip?download=1",
    "golgi": "https://zenodo.org/records/10043461/files/1_Golgi.zip?download=1",
}
CHECKSUMS = {
    "ne": "4ae915ce64632d1f27328cf3a79e85806d6b899cc51e0e85be619f3c161233bc",
    "mitochondria": "6725fb41ed7c14061ee96ab6110209aa0a6f9bd9fdb3a9aca2adb82a408f5680",
    "er": "1c5df4101b5f3f376124fc0b87da03198158d039ce146536542bc90c943feffc",
    "golgi": "594bf6a51dc9c85b2af504063759e8e46b39e604e3eb506728f8ba9254e31102",
}
# The folders with the training images and labels inside of the extracted archives.
FOLDERS = {
    "ne": ("1_NE", "1_Train_NE", "Images", "Labels"),
    "mitochondria": ("1_Mitochondria", "1_Train_Mito", "Images", "Labels"),
    "er": ("1_ER", "1_Train_ER", "Images", "Labels"),
    "golgi": ("1_Golgi", "1_Train_Golgi", "Images_for_patches", "Labels_for_patches"),
}


def _read_amira(am_path):
    with open(am_path, "rb") as f:
        content = f.read()

    # The header is followed by the raw (uncompressed) voxel data, which starts after the '@1' marker.
    marker = content.find(b"\n@1\n")
    header = content[:marker].decode("latin1")
    assert "Lattice { byte Data } @1" in header, f"Unsupported Amira data type in {am_path}"
    x, y, z = map(int, re.search(r"define Lattice (\d+) (\d+) (\d+)", header).groups())

    start = marker + len(b"\n@1\n")
    return np.frombuffer(content[start:start + x * y * z], dtype="uint8").reshape(z, y, x)


def _convert_to_annotated_slice(am_path, label_path, image_tif_path, label_tif_path):
    import tifffile

    image = _read_amira(am_path)
    label = tifffile.imread(label_path)
    label = label[None] if label.ndim == 2 else label
    assert image.shape == label.shape, f"Image and label shapes differ for {am_path}: {image.shape}, {label.shape}"

    # Only a single slice per stack is annotated.
    z = int(np.argmax((label > 0).reshape(len(label), -1).sum(axis=1)))
    tifffile.imwrite(image_tif_path, image[z], compression="zlib")
    tifffile.imwrite(label_tif_path, label[z], compression="zlib")


def get_u2os_fibsem_data(
    path: Union[os.PathLike, str], name: Literal["ne", "mitochondria", "er", "golgi"], download: bool = False
) -> str:
    """Download the U2OS FIB-SEM dataset.

    The images are distributed in the Amira format and are converted to tif files after downloading.
    Only the annotated slice of each image / label stack is kept for training.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the sub-dataset. Either 'ne', 'mitochondria', 'er' or 'golgi'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    assert name in URLS, f"'{name}' is not a valid sub-dataset name. Choose from {list(URLS)}."
    root, train_dir, _, _ = FOLDERS[name]

    data_dir = os.path.join(path, root)
    if os.path.exists(os.path.join(data_dir, train_dir)):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{root}.zip")
    util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
    util.unzip(zip_path=zip_path, dst=path)

    _, _, image_dir, label_dir = FOLDERS[name]
    image_tif_dir = os.path.join(data_dir, train_dir, "images_tif")
    label_tif_dir = os.path.join(data_dir, train_dir, "labels_tif")
    os.makedirs(image_tif_dir, exist_ok=True)
    os.makedirs(label_tif_dir, exist_ok=True)

    am_paths = natsorted(glob(os.path.join(data_dir, train_dir, image_dir, "*.am")))
    label_paths = natsorted(glob(os.path.join(data_dir, train_dir, label_dir, "*.tif")))
    assert len(am_paths) > 0 and len(am_paths) == len(label_paths)
    for am_path, label_path in zip(am_paths, label_paths):
        fname = os.path.splitext(os.path.basename(am_path))[0]
        _convert_to_annotated_slice(
            am_path,
            label_path,
            os.path.join(image_tif_dir, f"{fname}.tif"),
            os.path.join(label_tif_dir, f"{fname}.tif"),
        )

    return data_dir


def get_u2os_fibsem_paths(
    path: Union[os.PathLike, str],
    name: Literal["ne", "mitochondria", "er", "golgi"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the U2OS FIB-SEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the sub-dataset. Either 'ne', 'mitochondria', 'er' or 'golgi'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_u2os_fibsem_data(path, name, download)
    train_dir = FOLDERS[name][1]

    image_paths = natsorted(glob(os.path.join(data_dir, train_dir, "images_tif", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, train_dir, "labels_tif", "*.tif")))
    assert len(image_paths) > 0 and len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_u2os_fibsem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    name: Literal["ne", "mitochondria", "er", "golgi"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the U2OS FIB-SEM dataset for organelle segmentation.

    The labels are binary (1 is the organelle).

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        name: The name of the sub-dataset. Either 'ne', 'mitochondria', 'er' or 'golgi'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_u2os_fibsem_paths(path, name, download)

    kwargs = util.update_kwargs(kwargs, "label_transform", torch_em.transform.label.labels_to_binary)
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_u2os_fibsem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    name: Literal["ne", "mitochondria", "er", "golgi"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in the U2OS FIB-SEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        name: The name of the sub-dataset. Either 'ne', 'mitochondria', 'er' or 'golgi'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_u2os_fibsem_dataset(path, patch_shape, name, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
