"""The UrinaryTract dataset contains annotations for urinary cells in
bright-field microscopy images of unstained and untreated urine.

The dataset holds 300 images of voided urine from patients with a symptomatic urinary tract
infection. Two experts labelled 3562 cells and assigned each one to one of seven clinically
significant classes. Every image comes with a foreground mask and a class mask.

NOTE: The dataset provides semantic labels only. The masks store class ids, not instance ids, so
this loader cannot return cell instances. Connected components do not recover the cells either,
because a single cell often breaks into several components in the masks.

NOTE: The three splits hold 100 images each, but they differ a lot in cell density.

The dataset is located at https://doi.org/10.14278/rodare.2473 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-024-02975-0.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://rodare.hzdr.de/record/2473/files/ds1.zip"
CHECKSUM = "ae66af80c2c0d589c8fc6be21327988cf3ab2c2ed1ccaeadc5783b6e6dd51f95"

SPLITS = ("train", "validation", "test")

LABEL_CHOICES = {"binary": "bin_mask", "semantic": "mult_mask"}

# The class ids of the multi class masks, see Table 1 of the publication.
CLASS_NAMES = {
    1: "rod",
    2: "rbc_wbc",
    3: "yeast",
    4: "miscellaneous",
    5: "single_epc",
    6: "small_epc_sheet",
    7: "large_epc_sheet",
}


def get_urinary_tract_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the UrinaryTract dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "ds1")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "ds1.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_urinary_tract_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "validation", "test"] = "train",
    label_choice: Literal["binary", "semantic"] = "semantic",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the UrinaryTract data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'validation' or 'test'.
        label_choice: The label to use. Either 'semantic' for the seven classes, or 'binary' for
            the foreground mask.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")
    if label_choice not in LABEL_CHOICES:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose from {list(LABEL_CHOICES)}.")

    data_dir = get_urinary_tract_data(path, download)
    label_dir = LABEL_CHOICES[label_choice]

    image_paths = natsorted(glob(os.path.join(data_dir, split, "img", "cls", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, split, label_dir, "cls", "*.tif")))

    if not image_paths:
        raise RuntimeError(f"Could not find any UrinaryTract images in {data_dir}.")
    if len(image_paths) != len(label_paths):
        raise RuntimeError(
            f"Found {len(image_paths)} images but {len(label_paths)} labels for the '{split}' split."
        )

    return image_paths, label_paths


def get_urinary_tract_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"] = "train",
    label_choice: Literal["binary", "semantic"] = "semantic",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the UrinaryTract dataset for urinary cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'validation' or 'test'.
        label_choice: The label to use. Either 'semantic' for the seven classes, or 'binary' for
            the foreground mask.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The UrinaryTract patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_urinary_tract_paths(path, split, label_choice, download)

    if label_choice == "binary":
        # The masks store 0 and 255, so map them to a background and a foreground id.
        kwargs["label_transform"] = torch_em.transform.label.labels_to_binary

    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs,
    )


def get_urinary_tract_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"] = "train",
    label_choice: Literal["binary", "semantic"] = "semantic",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the UrinaryTract dataloader for urinary cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'validation' or 'test'.
        label_choice: The label to use. Either 'semantic' for the seven classes, or 'binary' for
            the foreground mask.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_urinary_tract_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        label_choice=label_choice,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
