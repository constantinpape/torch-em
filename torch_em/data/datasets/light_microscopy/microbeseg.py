"""The microbeSEG dataset contains annotations for bacterial cell instance segmentation
in phase-contrast microscopy images of B. subtilis and E. coli.

The dataset is located at https://zenodo.org/records/6497715.
This dataset is from the publication https://doi.org/10.1371/journal.pone.0277601.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Literal, Tuple, Optional, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/6497715/files/microbeSEG_dataset.zip"
CHECKSUM = None

ANNOTATION_TYPES = ["30min-man", "30min-man_15min-pre"]
SPLITS = ["train", "val", "test", "complete"]


def get_microbeseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the microbeSEG dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "microbeSEG_dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "microbeSEG_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_microbeseg_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test", "complete"] = "train",
    annotation_type: Literal["30min-man", "30min-man_15min-pre"] = "30min-man_15min-pre",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the microbeSEG data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val', 'test' or 'complete'.
        annotation_type: The annotation type. Either '30min-man' (manual only)
            or '30min-man_15min-pre' (manual + pre-labeling correction, more data).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in SPLITS, f"'{split}' is not a valid split. Choose from {SPLITS}."
    assert annotation_type in ANNOTATION_TYPES, \
        f"'{annotation_type}' is not a valid annotation type. Choose from {ANNOTATION_TYPES}."

    data_dir = get_microbeseg_data(path, download)

    split_dir = os.path.join(data_dir, annotation_type, split)
    assert os.path.exists(split_dir), f"Split directory not found: {split_dir}"

    image_paths = natsorted(glob(os.path.join(split_dir, "img_*.tif")))
    seg_paths = natsorted(glob(os.path.join(split_dir, "mask_*.tif")))
    assert len(image_paths) == len(seg_paths) and len(image_paths) > 0

    return image_paths, seg_paths


def get_microbeseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test", "complete"] = "train",
    annotation_type: Literal["30min-man", "30min-man_15min-pre"] = "30min-man_15min-pre",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the microbeSEG dataset for bacterial cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'test' or 'complete'.
        annotation_type: The annotation type. Either '30min-man' (manual only)
            or '30min-man_15min-pre' (manual + pre-labeling correction, more data).
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, seg_paths = get_microbeseg_paths(path, split, annotation_type, download)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=seg_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs
    )


def get_microbeseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test", "complete"] = "train",
    annotation_type: Literal["30min-man", "30min-man_15min-pre"] = "30min-man_15min-pre",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the microbeSEG dataloader for bacterial cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'test' or 'complete'.
        annotation_type: The annotation type. Either '30min-man' (manual only)
            or '30min-man_15min-pre' (manual + pre-labeling correction, more data).
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_microbeseg_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        annotation_type=annotation_type,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
