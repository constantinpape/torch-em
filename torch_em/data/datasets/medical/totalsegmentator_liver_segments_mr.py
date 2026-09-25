"""The TotalSegmentator liver segments MRI dataset contains annotations for the 8 Couinaud liver segments
in MRI scans.

This is the training dataset for the "liver_segments_mr" task of the TotalSegmentator repository
(https://github.com/wasserth/TotalSegmentator), which is distributed separately from the main
TotalSegmentator MRI dataset (see `torch_em.data.datasets.medical.totalsegmentator_mri`). It consists of
115 MRI volumes, of which about half are negative controls with an entirely empty (all-background) label
volume; `get_totalsegmentator_liver_segments_mr_paths` filters these out, so only volumes with manual
segmentations of the 8 Couinaud liver segments are returned.

The dataset is located at https://doi.org/10.5281/zenodo.20746820 and licensed under CC BY 4.0.

This dataset is part of the TotalSegmentator project, published at
https://doi.org/10.1007/s10278-025-01716-y. Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/20746820/files/Dataset576_liver_segments_mr.zip"
CHECKSUM = "621dc9cfe51348980cf1ef65eaf662cba2b18c499d27b186a236429aa954e438"

LABEL_IDS = {
    "background": 0,
    "liver_segment_1": 1,
    "liver_segment_2": 2,
    "liver_segment_3": 3,
    "liver_segment_4": 4,
    "liver_segment_5": 5,
    "liver_segment_6": 6,
    "liver_segment_7": 7,
    "liver_segment_8": 8,
}


def get_totalsegmentator_liver_segments_mr_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TotalSegmentator liver segments MRI dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the 'imagesTr' and 'labelsTr' folders.
    """
    data_dir = os.path.join(path, "Dataset576_liver_segments_mr")
    if os.path.exists(os.path.join(data_dir, "dataset.json")):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "Dataset576_liver_segments_mr.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_totalsegmentator_liver_segments_mr_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the TotalSegmentator liver segments MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    import nibabel as nib
    import numpy as np

    data_dir = get_totalsegmentator_liver_segments_mr_data(path, download)

    raw_paths, label_paths = [], []
    for raw_path in sorted(glob(os.path.join(data_dir, "imagesTr", "*_0000.nii.gz"))):
        case_id = os.path.basename(raw_path)[:-len("_0000.nii.gz")]
        label_path = os.path.join(data_dir, "labelsTr", f"{case_id}.nii.gz")
        assert os.path.exists(label_path), label_path

        # Skip the negative control cases, whose label volume is entirely background.
        if not np.any(nib.load(label_path).get_fdata()):
            continue

        raw_paths.append(raw_path)
        label_paths.append(label_path)

    assert len(raw_paths) > 0
    return raw_paths, label_paths


def get_totalsegmentator_liver_segments_mr_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TotalSegmentator liver segments MRI dataset for Couinaud liver segment segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_totalsegmentator_liver_segments_mr_paths(path, download)

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


def get_totalsegmentator_liver_segments_mr_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TotalSegmentator liver segments MRI dataloader for Couinaud liver segment segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_totalsegmentator_liver_segments_mr_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
