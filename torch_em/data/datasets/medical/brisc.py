"""BRISC (BRain tumor Image Segmentation and Classification) is a curated, expert-annotated dataset
of contrast-enhanced T1-weighted brain MRI slices for brain tumor segmentation and classification.

The dataset consists of 6,000 slices (5,000 train / 1,000 test) collated from existing public MRI
collections, spanning axial, coronal and sagittal planes. It provides physician-reviewed pixel-wise
segmentation masks for three tumor types (glioma, meningioma, pituitary tumor), in addition to
image-level labels for a 'no tumor' class (which has no corresponding segmentation mask). While the
raw images originate from other public collections, the segmentation masks are a new, original
annotation contribution.

The dataset is located at https://doi.org/10.6084/m9.figshare.30533120 and is distributed under the
CC BY 4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-026-06753-y.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/59298329"
CHECKSUM = "3167e9bdfcb3b2a2502091f41c9bd3f1e2eeb64ec3ad9849680fa760e2e7633d"

TUMOR_TYPES = {"glioma": "gl", "meningioma": "me", "pituitary": "pi"}


def get_brisc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BRISC dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "brisc2025")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "brisc2025.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_brisc_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    tumor_type: Optional[Literal["glioma", "meningioma", "pituitary"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BRISC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        tumor_type: The choice of tumor type. One of 'glioma', 'meningioma', 'pituitary'. By default, all are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_brisc_data(path=path, download=download)

    if split not in ["train", "test"]:
        raise ValueError(f"'{split}' is not a valid split choice.")

    image_dir = os.path.join(data_dir, "segmentation_task", split, "images")
    label_dir = os.path.join(data_dir, "segmentation_task", split, "masks")

    if tumor_type is None:
        pattern = "*.jpg"
    elif tumor_type in TUMOR_TYPES:
        pattern = f"*_{TUMOR_TYPES[tumor_type]}_*.jpg"
    else:
        raise ValueError(f"'{tumor_type}' is not a valid tumor type. Choose from {list(TUMOR_TYPES.keys())}.")

    image_paths = natsorted(glob(os.path.join(image_dir, pattern)))
    label_paths = natsorted(
        os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0] + ".png") for p in image_paths
    )
    assert len(image_paths) > 0 and len(image_paths) == len(label_paths)

    return image_paths, label_paths


def get_brisc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    tumor_type: Optional[Literal["glioma", "meningioma", "pituitary"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BRISC dataset for brain tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        tumor_type: The choice of tumor type. One of 'glioma', 'meningioma', 'pituitary'. By default, all are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_brisc_paths(path, split, tumor_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        with_channels=True,
        **kwargs
    )


def get_brisc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    tumor_type: Optional[Literal["glioma", "meningioma", "pituitary"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BRISC dataloader for brain tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        tumor_type: The choice of tumor type. One of 'glioma', 'meningioma', 'pituitary'. By default, all are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brisc_dataset(path, patch_shape, split, tumor_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
