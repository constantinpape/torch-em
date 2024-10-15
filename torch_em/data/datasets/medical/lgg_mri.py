"""The LGG MRI datasets contains annotations for low grade glioma segmentation
in FLAIR MRI scans.

The dataset is located at https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation.

This dataset is from the publication https://www.nejm.org/doi/full/10.1056/NEJMoa1402121.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _merge_slices_to_volumes(path):
    import nibabel as nib

    raw_dir = os.path.join(path, "data", "raw")
    label_dir = os.path.join(path, "data", "labels")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    patient_dirs = glob(os.path.join(path, "kaggle_3m", "TCGA_*"))
    for patient_dir in tqdm(patient_dirs, desc="Preprocessing inputs"):
        label_slice_paths = natsorted(glob(os.path.join(patient_dir, "*_mask.tif")))
        raw_slice_paths = [lpath.replace("_mask.tif", ".tif") for lpath in label_slice_paths]

        raw = [imageio.imread(rpath) for rpath in raw_slice_paths]
        labels = [imageio.imread(lpath) for lpath in label_slice_paths]

        raw, labels = np.stack(raw, axis=2), np.stack(labels, axis=2)

        raw_nifti = nib.Nifti2Image(raw, np.eye(4))
        label_nifti = nib.Nifti2Image(labels, np.eye(4))

        nib.save(raw_nifti, os.path.join(raw_dir, f"{os.path.basename(patient_dir)}.nii.gz"))
        nib.save(label_nifti, os.path.join(label_dir, f"{os.path.basename(patient_dir)}.nii.gz"))

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


def get_lgg_mri_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the LGG MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_lgg_mri_data(path, download)

    raw_paths = natsorted(glob(os.path.join(path, "data", "raw", "*.nii.gz")))
    label_paths = natsorted(glob(os.path.join(path, "data", "labels", "*.nii.gz")))

    return raw_paths, label_paths


def get_lgg_mri_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], download: bool, **kwargs
) -> Dataset:
    """Get the LGG MRI dataset for glioma segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lgg_mri_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True,
        **kwargs
    )


def get_lgg_mri_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], download: bool, **kwargs
) -> DataLoader:
    """Get the LGG MRI dataloader for glioma segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lgg_mri_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
