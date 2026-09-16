"""The Hippo-Subfields dataset contains annotations for hippocampal subfield segmentation in 7 Tesla
brain MRI.

The dataset consists of paired whole-brain T1-weighted and T2-weighted MRI scans acquired at 3 Tesla
and 7 Tesla from 20 healthy volunteers. The hippocampal formation subfields were manually delineated
bilaterally on every coronal section of the 7 Tesla T2-weighted scans (upsampled to 0.7 mm slice
thickness), which is the modality and annotation used by this module.

NOTE: The left and right hippocampus are distributed as two separate label volumes per subject, both
on the same whole-brain grid as the raw scan and with non-overlapping foreground regions. This module
merges them (via a voxel-wise maximum) into a single semantic label volume with the following 7
foreground classes, following `LABEL_IDS`:
0 = background, 1 = subiculum (SUB), 2 = CA2, 3 = CA1, 4 = CA4 and dentate gyrus (CA4&DG),
5 = entorhinal cortex (ERC), 6 = CA3, 7 = hippocampal tail.

The dataset is located at https://doi.org/10.25452/figshare.plus.26075713.v1 and is distributed under
the CC BY 4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-025-04586-9.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/50052801"
CHECKSUM = "2b2d32c6779cfdb79638f4de34934d62484f2eabbcbb17d43732f3b96c14d54c"

LABEL_IDS = {"background": 0, "SUB": 1, "CA2": 2, "CA1": 3, "CA4&DG": 4, "ERC": 5, "CA3": 6, "tail": 7}
"""The semantic label ids of the hippocampal subfield classes."""


def _preprocess_hippo_subfields(raw_dir, label_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)

    label_paths = natsorted(glob(os.path.join(label_dir, "sub-*-L.nii.gz")))
    subject_ids = [os.path.basename(p).split("-")[1] for p in label_paths]

    for subject_id in tqdm(subject_ids, desc="Preprocessing the Hippo-Subfields data"):
        out_path = os.path.join(preprocessed_dir, f"sub-{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        raw_path = os.path.join(raw_dir, f"t2_s{subject_id}_0.7.nii")
        left_path = os.path.join(label_dir, f"sub-{subject_id}-L.nii.gz")
        right_path = os.path.join(label_dir, f"sub-{subject_id}-R.nii.gz")

        raw = np.asarray(nib.load(raw_path).dataobj).astype("float32")
        left = np.asarray(nib.load(left_path).dataobj)
        right = np.asarray(nib.load(right_path).dataobj)
        labels = np.maximum(left, right).astype("uint8")
        assert raw.shape == labels.shape, f"Shape mismatch for {subject_id}: {raw.shape} vs. {labels.shape}."

        with h5py.File(f"{out_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        os.rename(f"{out_path}.tmp", out_path)

    return preprocessed_dir


def get_hippo_subfields_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Hippo-Subfields dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is preprocessed.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "hippo_subfield")
    if not os.path.exists(data_dir):
        zip_path = os.path.join(path, "hippo_subfield.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path)

    raw_dir = os.path.join(data_dir, "7T_T2w_0.7_for_subfield_delineation")
    label_dir = os.path.join(data_dir, "hippo_label")
    return _preprocess_hippo_subfields(raw_dir, label_dir, preprocessed_dir)


def get_hippo_subfields_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Hippo-Subfields data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    preprocessed_dir = get_hippo_subfields_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths, volume_paths


def get_hippo_subfields_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Hippo-Subfields dataset for hippocampal subfield segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hippo_subfields_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_hippo_subfields_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Hippo-Subfields dataloader for hippocampal subfield segmentation.

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
    dataset = get_hippo_subfields_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
