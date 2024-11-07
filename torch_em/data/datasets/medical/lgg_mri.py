"""The LGG MRI datasets contains annotations for low grade glioma segmentation
in FLAIR MRI scans.

The dataset is located at https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation.

This dataset is from the publication https://www.nejm.org/doi/full/10.1056/NEJMoa1402121.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
import warnings
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _merge_slices_to_volumes(path):
    volume_dir = os.path.join(path, "data")
    os.makedirs(volume_dir, exist_ok=True)

    patient_dirs = glob(os.path.join(path, "kaggle_3m", "TCGA_*"))
    for patient_dir in tqdm(patient_dirs, desc="Preprocessing inputs"):
        label_slice_paths = natsorted(glob(os.path.join(patient_dir, "*_mask.tif")))
        raw_slice_paths = [lpath.replace("_mask.tif", ".tif") for lpath in label_slice_paths]

        raw = [imageio.imread(rpath) for rpath in raw_slice_paths]
        labels = [imageio.imread(lpath) for lpath in label_slice_paths]

        raw, labels = np.stack(raw, axis=0), np.stack(labels, axis=0)

        volume_path = os.path.join(volume_dir, f"{os.path.basename(patient_dir)}.h5")

        import h5py
        with h5py.File(volume_path, "w") as f:
            f.create_dataset("raw/pre_contrast", data=raw[..., 0], compression="gzip")
            f.create_dataset("raw/flair", data=raw[..., 1], compression="gzip")
            f.create_dataset("raw/post_contrast", data=raw[..., 2], compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

    shutil.rmtree(os.path.join(path, "kaggle_3m"))


def get_lgg_mri_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the LGG MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="mateuszbuda/lgg-mri-segmentation", download=download)
    zip_path = os.path.join(path, "lgg-mri-segmentation.zip")
    util.unzip(zip_path=zip_path, dst=path)

    # Remove redundant volumes
    shutil.rmtree(os.path.join(path, "lgg-mri-segmentation"))

    _merge_slices_to_volumes(path)


def get_lgg_mri_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> List[str]:
    """Get paths to the LGG MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    get_lgg_mri_data(path, download)

    volume_paths = natsorted(glob(os.path.join(path, "data", "*.h5")))

    if split == "train":
        volume_paths = volume_paths[:70]
    elif split == "val":
        volume_paths = volume_paths[70:85]
    elif split == "test":
        volume_paths = volume_paths[85:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return volume_paths


def get_lgg_mri_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    channels: Optional[Literal['pre_contrast', 'flair', 'post_contrast']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LGG MRI dataset for glioma segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        channels: The choice of modality as input channel.
        resize_inputs:  Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_lgg_mri_paths(path, split, download)

    if resize_inputs and channels is not None:
        if channels is None:
            warnings.warn("The default for channels is set to 'None'. Choose one specific channel for resizing inputs.")

        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    available_channels = ["pre_contrast", "flair", "post_contrast"]
    if channels is not None and channels not in available_channels:
        raise ValueError(f"'{channels}' is not a valid channel.")

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=[f"raw/{chan}" for chan in available_channels] if channels is None else f"raw/{channels}",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True if channels is None else False,
        **kwargs
    )


def get_lgg_mri_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    channels: Optional[Literal['pre_contrast', 'flair', 'post_contrast']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LGG MRI dataloader for glioma segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        channels: The choice of modality as input channel.
        resize_inputs:  Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lgg_mri_dataset(path, patch_shape, split, channels, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
