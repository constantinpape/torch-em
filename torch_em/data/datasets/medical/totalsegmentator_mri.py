"""The TotalSegmentator MRI dataset contains annotations for 50 anatomical structures in MRI scans.

The dataset (v3.0.0) consists of 1296 MRI volumes (whole body, pediatric and clinical routine scans with a wide
range of sequences, scanners and institutions) with an official train / test split (see 'meta.csv'; there is no
validation split).
Each anatomical structure is provided as a separate binary mask. `get_totalsegmentator_mri_data` merges these masks
into a single semantic label volume per case, where the label id of each structure is its (1-based) position in
`CLASS_NAMES` (see `CLASS_IDS` for the name -> id mapping). This is the class order of the 'total_mr' task in the
TotalSegmentator repository (https://github.com/wasserth/TotalSegmentator). The masks of a few structures may
overlap, in this case the structure with the higher label id takes precedence.

The dataset is located at https://doi.org/10.5281/zenodo.11367005.

This dataset is from the publication https://doi.org/10.1148/radiol.241613.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .totalsegmentator import merge_all_segmentations, read_split


URL = "https://zenodo.org/records/22688334/files/TotalsegmentatorMRI_dataset_v300.zip"
CHECKSUM = "791696df98dbbc5136c19d6b23335779a9bdcda7fcc3842e885869756ca17589"

CLASS_NAMES = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right",
    "adrenal_gland_left", "lung_left", "lung_right", "esophagus", "small_bowel", "duodenum", "colon",
    "urinary_bladder", "prostate", "sacrum", "vertebrae", "intervertebral_discs", "spinal_cord", "heart", "aorta",
    "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left", "scapula_right",
    "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right", "gluteus_maximus_left",
    "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left",
    "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain",
]
"""The anatomical structures of the TotalSegmentator MRI dataset. The label id of a structure is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the name of an anatomical structure to its label id in the merged label volumes."""


def get_totalsegmentator_mri_data(
    path: Union[os.PathLike, str], download: bool = False, n_workers: Optional[int] = None
) -> str:
    """Download the TotalSegmentator MRI dataset and merge the per-class masks into semantic label volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
        n_workers: The number of parallel workers for merging the per-class masks.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Totalsegmentator_dataset_v300")
    if not os.path.exists(os.path.join(data_dir, "meta.csv")):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "TotalsegmentatorMRI_dataset_v300.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path)

    case_dirs = sorted(glob(os.path.join(data_dir, "s*")))
    merge_all_segmentations(case_dirs, CLASS_NAMES, n_workers)

    return data_dir


def get_totalsegmentator_mri_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the TotalSegmentator MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_totalsegmentator_mri_data(path, download)
    case_ids = read_split(os.path.join(data_dir, "meta.csv"), split, valid_splits=("train", "test"))

    raw_paths = [os.path.join(data_dir, case_id, "mri.nii.gz") for case_id in case_ids]
    label_paths = [os.path.join(data_dir, case_id, "labels.nii.gz") for case_id in case_ids]
    assert all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_totalsegmentator_mri_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TotalSegmentator MRI dataset for segmentation of anatomical structures in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_totalsegmentator_mri_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_totalsegmentator_mri_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TotalSegmentator MRI dataloader for segmentation of anatomical structures in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_totalsegmentator_mri_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
