"""The UltraCortex dataset contains submillimeter ultra-high field 9.4T brain MR images with manual
cortical gray and white matter segmentations.

The dataset consists of 86 structural MR images (0.6-0.8mm resolution, MP-RAGE and MP2RAGE sequences),
of which 12 have manual cortical segmentations into gray and white matter, independently validated by
two expert neuroradiologists. This module only exposes the 12 volumes with manual segmentations.

The dataset is located at https://openneuro.org/datasets/ds005216, released under the CC0 1.0 license.

The dataset is from the publication https://doi.org/10.1038/s41597-025-04779-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds005216"

# The 12 subjects with manual cortical gray / white matter segmentations, out of the 86 total subjects.
SUBJECT_IDS = [3, 9, 20, 29, 37, 44, 45, 46, 57, 69, 70, 73]


def get_ultracortex_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the UltraCortex dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    for sub_id in SUBJECT_IDS:
        raw_path = os.path.join(path, f"sub-{sub_id}_ses-1_T1w.nii")
        raw_url = f"{BASE_URL}/sub-{sub_id}/ses-1/anat/sub-{sub_id}_ses-1_T1w.nii"
        util.download_source(path=raw_path, url=raw_url, download=download)

        label_path = os.path.join(path, f"sub-{sub_id}_ses-1_seg.nii")
        label_url = f"{BASE_URL}/derivatives/manual_segmentation/sub-{sub_id}_ses-1_seg.nii"
        util.download_source(path=label_path, url=label_url, download=download)

    return path


def get_ultracortex_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the UltraCortex data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_ultracortex_data(path, download)

    label_paths = natsorted(glob(os.path.join(path, "sub-*_ses-1_seg.nii")))
    raw_paths = [p.replace("_seg.nii", "_T1w.nii") for p in label_paths]
    assert all(os.path.exists(p) for p in raw_paths), "Some image volumes are missing."
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_ultracortex_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the UltraCortex dataset for cortical gray and white matter segmentation in 9.4T brain MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ultracortex_paths(path, download)

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


def get_ultracortex_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the UltraCortex dataloader for cortical gray and white matter segmentation in 9.4T brain MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ultracortex_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
