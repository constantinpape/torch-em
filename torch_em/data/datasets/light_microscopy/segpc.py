"""The SegPC dataset contains annotations for cytoplasm and nucleus segmentation in microscopy images
of multiple myeloma plasma cells.

This dataset is located at https://ieee-dataport.org/open-access/segpc-2021-segmentation-multiple-myeloma-plasma-cells-microscopic-images.  # noqa
The dataset is from the publication https://doi.org/10.1016/j.media.2022.102677.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_segpc_data(path: Union[os.PathLike, str], split: Literal['train', 'validation'], download: bool = False) -> str:
    """Instruction to download SegPC data.

    NOTE: Please download the dataset from https://ieee-dataport.org/open-access/segpc-2021-segmentation-multiple-myeloma-plasma-cells-microscopic-images.  # noqa

    Args:
        path: Filepath to a folder where the data should be manually downloaded for further processing.
        split: The data split to use. Either 'train' or 'validation'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data.
    """
    data_dir = os.path.join(path, "TCIA_SegPC_dataset", split)
    if os.path.exists(data_dir):
        return data_dir

    if download:
        raise NotImplementedError(
            "The dataset cannot be automatically downloaded. ",
            "Please see 'get_segpc_data' in 'torch_em/data/datasets/light_microscopy/segpc.py for details."
        )

    zip_path = os.path.join(path, "TCIA_SegPC_dataset.zip")
    os.path.exists(zip_path), f"The manually downloaded zip file should be placed at '{path}'."
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    # Unzip the split-wise zip files.
    if split not in ['train', 'validation']:
        if split == "test":
            raise ValueError("The 'test' split does not have labels.")
        raise ValueError(f"'{split}' is not a valid split.")

    util.unzip(zip_path=os.path.join(Path(data_dir).parent, f"{split}.zip"), dst=Path(data_dir).parent)

    return data_dir


def get_segpc_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'validation'], download: bool = False
) -> List[str]:
    """Get paths to the SegPC data.

    Args:
        path: Filepath to a folder where the data is stored.
        split: The data split to use. Either 'train' or 'validation'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_segpc_data(path, split, download)

    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    volume_paths = []
    raw_paths = natsorted(glob(os.path.join(data_dir, "x", "*.bmp")))
    for rpath in tqdm(raw_paths, desc=f"Preprocessing '{split}' inputs"):
        volume_path = os.path.join(preprocessed_dir, Path(os.path.basename(rpath)).with_suffix(".h5"))
        volume_paths.append(volume_path)
        if os.path.exists(volume_path):
            continue

        image = imageio.imread(rpath)

        label_paths = glob(rpath.replace("x", "y").replace(".bmp", "_*.bmp"))

        nuclei = np.zeros(image.shape[:2], dtype="uint32")
        cells = np.zeros(image.shape[:2], dtype="uint32")
        for i, lpath in enumerate(label_paths, start=1):
            label = imageio.imread(lpath)

            if label.ndim == 3:
                label = label[..., 0]

            nuclei[label == 40] = i
            cells[label > 0] = i

        import h5py
        with h5py.File(volume_path, "w") as f:
            f.create_dataset("raw", data=image.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels/nuclei", data=nuclei, compression="gzip")
            f.create_dataset("labels/cells", data=cells, compression="gzip")

    return volume_paths


def get_segpc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val'],
    label_choice: Literal['nuclei', 'cells'] = "cells",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegPC dataset for plasma cell (and nuclei) segmentation.

    Args:
        path: Filepath to a folder where the data is stored.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'validation'.
        label_choice: The choice of labels.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_segpc_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_segpc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val'],
    label_choice: Literal['nuclei', 'cells'] = "cells",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegPC dataloader for plasma cell (and nuclei) segmentation.

    Args:
        path: Filepath to a folder where the data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'validation'.
        label_choice: The choice of labels.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_segpc_dataset(path, patch_shape, split, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
