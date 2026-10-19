"""This dataset contains nucleus instance segmentations in volume electron microscopy (EM) data.

It was generated for the user study on data annotation with microSAM
(https://github.com/computational-cell-analytics/micro-sam).
It has two 3D EM crops of size 32 x 512 x 512 voxels:
- 'train': The crop that was used to train the finetuned microSAM model, with manually annotated nucleus labels.
- 'annotation': The crop that was used for the user study, with nucleus annotations obtained via
  ilastik carving ('ilastik_carving'), microSAM with the default model ('default_sam')
  or microSAM with a finetuned model ('finetuned_sam').

The data is available at https://doi.org/10.5281/zenodo.14037020 (CC-BY-4.0).
The record does not list a publication. Please cite the Zenodo record if you use this dataset.
"""

import os
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/14037021/files/3d-user-study-data.zip/content"
CHECKSUM = "5eaa4650063a02becb46b6b11c1557362d3a9a9cb01fb88b3e6dc245819d4a5f"

ANNOTATIONS = ("ilastik_carving", "default_sam", "finetuned_sam")


def get_microsam_em_nuclei_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the microSAM EM nuclei data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "3d-user-study-data.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, data_dir, remove=True)
    return data_dir


def get_microsam_em_nuclei_paths(
    path: Union[os.PathLike, str], source: str = "train", download: bool = False
) -> Tuple[str, Optional[str], str, Optional[str]]:
    """Get paths to the microSAM EM nuclei data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The data to use. Either 'train' (the training crop) or the annotation method
            of the user study crop, one of 'ilastik_carving', 'default_sam' or 'finetuned_sam'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the raw data.
        The key for the raw data (None for tif files).
        The filepath to the labels.
        The key for the labels (None for tif files).
    """
    data_dir = get_microsam_em_nuclei_data(path, download)
    if source == "train":
        h5_path = os.path.join(data_dir, "train_data.h5")
        return h5_path, "raw", h5_path, "labels/nuclei"
    if source in ANNOTATIONS:
        raw_path = os.path.join(data_dir, "annotation_volume.tif")
        label_path = os.path.join(data_dir, "annotations", f"annotations_{source}.tif")
        return raw_path, None, label_path, None
    raise ValueError(f"'{source}' is not a valid source. Choose from 'train' or {ANNOTATIONS}.")


def get_microsam_em_nuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    source: Literal["train", "ilastik_carving", "default_sam", "finetuned_sam"] = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> Dataset:
    """Get the microSAM EM nuclei dataset for nucleus segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        source: The data to use. Either 'train' (the training crop) or the annotation method
            of the user study crop, one of 'ilastik_carving', 'default_sam' or 'finetuned_sam'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_path, raw_key, label_path, label_key = get_microsam_em_nuclei_paths(path, source, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_path, raw_key=raw_key, label_paths=label_path, label_key=label_key,
        patch_shape=patch_shape, **kwargs
    )


def get_microsam_em_nuclei_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    source: Literal["train", "ilastik_carving", "default_sam", "finetuned_sam"] = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> DataLoader:
    """Get the microSAM EM nuclei dataloader for nucleus segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The data to use. Either 'train' (the training crop) or the annotation method
            of the user study crop, one of 'ilastik_carving', 'default_sam' or 'finetuned_sam'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_microsam_em_nuclei_dataset(
        path=path, patch_shape=patch_shape, source=source, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
