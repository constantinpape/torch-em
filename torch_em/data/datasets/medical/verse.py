"""The VerSe dataset contains annotations for vertebrae segmentation in CT scans.

This dataset is from the publication https://doi.org/10.1016/j.media.2021.102166.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "train": "https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa463786541a01e714d390/?zip=",
    "val": "https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa463686541a01eb15048c/?zip=",
    "test": "https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa4635ba010901f0891bd0/?zip="
}

# FIXME the checksums are not reliable (same behaviour spotted in PlantSeg downloads from osf)
CHECKSUM = {
    "train": None,
    "val": None,
    "test": None,
}


def get_verse_data(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> str:
    """Download the VerSe dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    assert split in ["train", "val", "test"], f"'{split}' is not a valid split."

    data_dir = os.path.join(path, "data", split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"verse2020_{split}.zip")
    util.download_source(path=zip_path, url=URL[split], download=download, checksum=CHECKSUM[split])
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_verse_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> str:
    """Get paths to the VerSe data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_verse_data(path, split, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "rawdata", "*", "*_ct.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "derivatives", "*", "*_msk.nii.gz")))

    return image_paths, gt_paths


def get_verse_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the VerSe dataset for vertebrae segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_verse_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths, raw_key="data", label_paths=gt_paths, label_key="data", patch_shape=patch_shape, **kwargs
    )


def get_verse_loader(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the VerSe dataloader for vertebrae segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_verse_dataset(path, split, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
