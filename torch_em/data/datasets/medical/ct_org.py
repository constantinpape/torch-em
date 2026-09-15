"""The CT-ORG dataset contains annotations for multiple organs in CT.

It consists of 140 CT volumes (abdominal and full-body, with and without contrast, including 9 PET-CT scans)
with semantic labels for: 1: liver, 2: bladder, 3: lungs, 4: kidneys, 5: bone, 6: brain (only in the minority
of scans which show the brain). Cases 0-20 form the official test split (all organs annotated manually), cases 21-139
form the training split (lungs and bones were segmented with morphological algorithms).

The dataset is located at https://www.cancerimagingarchive.net/collection/ct-org/ and is only offered via
IBM Aspera there. We download the data from a mirror of the original nifti files at
https://huggingface.co/datasets/MedOtter/ct-org instead.

This dataset is from the publication https://doi.org/10.1038/s41597-020-00715-8.
The data was released at https://doi.org/10.7937/tcia.2019.tt7f4v7o.
Please cite it if you use this dataset in your research.
"""

import os
from tqdm import tqdm
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/MedOtter/ct-org/resolve/main/"

# The data is downloaded as 280 individual files, which are not checksummed.
CHECKSUM = None

NUM_VOLUMES = 140
TEST_IDS = list(range(21))
TRAIN_IDS = list(range(21, NUM_VOLUMES))

LABEL_IDS = {"liver": 1, "bladder": 2, "lungs": 3, "kidneys": 4, "bone": 5, "brain": 6}


def get_ct_org_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CT-ORG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    os.makedirs(os.path.join(data_dir, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels"), exist_ok=True)

    for i in tqdm(range(NUM_VOLUMES), desc="Download CT-ORG"):
        for folder, name in [("volumes", "volume"), ("labels", "labels")]:
            fname = f"{name}-{i}.nii.gz"
            fpath = os.path.join(data_dir, folder, fname)
            util.download_source(path=fpath, url=f"{URL}{folder}/{fname}", download=download, checksum=CHECKSUM)

    return data_dir


def get_ct_org_paths(
    path: Union[os.PathLike, str], split: Optional[Literal["train", "test"]] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CT-ORG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'. If None, all volumes are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ct_org_data(path, download)

    if split is None:
        ids = TRAIN_IDS + TEST_IDS
    elif split == "train":
        ids = TRAIN_IDS
    elif split == "test":
        ids = TEST_IDS
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    raw_paths = [os.path.join(data_dir, "volumes", f"volume-{i}.nii.gz") for i in sorted(ids)]
    label_paths = [os.path.join(data_dir, "labels", f"labels-{i}.nii.gz") for i in sorted(ids)]
    return raw_paths, label_paths


def get_ct_org_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CT-ORG dataset for organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'. If None, all volumes are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ct_org_paths(path, split, download)

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


def get_ct_org_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CT-ORG dataloader for organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'. If None, all volumes are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ct_org_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
