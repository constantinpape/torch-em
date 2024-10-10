"""
"""

import os
import shutil
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import tifffile
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _clean_redundant_files(path):
    # The "InDistribution" directory is redundant.
    shutil.rmtree(os.path.join(path, "valid", "InDistribution", "InDistribution"))


def get_emneuron_data(path: Union[os.PathLike, str], download: bool = False):
    """Get the EMNeuron data.

    Args:
        path:
        download:
    """
    if download:
        raise ValueError()

    os.makedirs(path, exist_ok=True)

    if os.path.exists(os.path.join(path, "labeled")) and os.path.exists(os.path.join(path, "valid")):
        return

    rar_path = os.path.join(path, "labeled.rar")
    util.unzip_rarfile(rar_path=rar_path, dst=path, remove=False, use_rarfile=False)

    rar_path = os.path.join(path, "valid.rar")
    util.unzip_rarfile(rar_path=rar_path, dst=path, remove=False, use_rarfile=False)

    _clean_redundant_files(path)


def _extract_array_from_tifs(input_path):
    """Reference:
    https://forum.image.sc/t/tiff-file-that-can-be-opened-in-fiji-but-not-python-matlab-due-to-offset-related-error
    """
    frames = []
    with tifffile.TiffFile(input_path) as tif:
        for page in tif.pages:
            assert page.is_contiguous
            frame = np.zeros(page.shape, page.dtype)
            tif.filehandle.seek(page.dataoffsets[0])
            tif.filehandle.readinto(frame)
            frames.append(frame)

    stack = np.stack(frames)
    stack = stack.squeeze()
    print(stack.shape)
    return stack


def _convert_inputs_to_h5(path, split):
    import h5py

    if split == "train":
        label_paths = natsorted(glob(os.path.join(path, "labeled", "*", "*_MaskIns.tif")))
        raw_paths = [os.path.join(os.path.dirname(p), os.path.basename(p).replace("_MaskIns", "")) for p in label_paths]

    elif split == "val":
        raw_paths = natsorted(glob(os.path.join(path, "valid", "*", "*", "raw.tif")))
        label_paths = [
            os.path.join(os.path.dirname(p), "label_0.tif")
            if os.path.exists(os.path.join(os.path.dirname(p), "label_0.tif"))
            else os.path.join(os.path.dirname(p), "label.tif") for p in raw_paths
        ]

    else:
        raise ValueError(f"'{split}' is not a valid split. Please choose either 'train' or 'val'.")

    assert len(raw_paths) == len(label_paths)

    volume_dir = os.path.join(path, "data", split)
    if os.path.exists(volume_dir):
        return volume_dir

    os.makedirs(volume_dir, exist_ok=True)

    for rpath, lpath in zip(raw_paths, label_paths):
        volume_path = os.path.join(volume_dir, f"{Path(rpath).parent.name}_{Path(rpath).stem}.h5")
        with h5py.File(volume_path, "w") as f:
            f.create_dataset("raw", data=_extract_array_from_tifs(rpath), compression="gzip")
            f.create_dataset("labels", data=_extract_array_from_tifs(lpath), compression="gzip")

    return volume_dir


def get_emneuron_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val'], download: bool = False
) -> List[str]:
    """Get paths to the EMNeuron data.

    Args:
        path:
        split:
        download:

    Returns:
        List of filepaths to the stored data.
    """
    get_emneuron_data(path, download)
    volume_dir = _convert_inputs_to_h5(path, split)
    data_paths = glob(os.path.join(volume_dir, "*.h5"))
    return data_paths


def get_emneuron_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for EMNeuron dataset.

    Args:
        path:
        patch_shape:
        split:
        download:
        kwargs:

    Returns:
        The segmentation dataset.
    """
    data_paths = get_emneuron_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_emneuron_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the dataloader for EMNeuron dataset.

    Args:
        path:
        batch_size:
        patch_shape:
        split:
        download:
        kwargs:

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_emneuron_dataset(path=path, patch_shape=patch_shape, split=split, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
