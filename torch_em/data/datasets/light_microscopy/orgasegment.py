"""The OrgaSegment dataset contains annotations for organoid segmentation
of intestinal patient derived organoids in bright field images.

This dataset is from the publication https://doi.org/10.1038/s42003-024-05966-4.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from typing import Tuple, Union, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10278229/files/OrganoidBasic_v20211206.zip"
CHECKSUM = "d067124d734108e46e18f65daaf17c89cb0a40bdacc6f6031815a6839e472798"


def get_orgasegment_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval"],
    download: bool = False
) -> str:
    """Download the OrgaSegment dataset for organoid segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download. Either 'train', 'val or 'eval'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "OrganoidBasic_v20211206.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=True)

    shutil.move(os.path.join(path, "OrganoidBasic_v20211206", "train"), os.path.join(path, "train"))
    shutil.move(os.path.join(path, "OrganoidBasic_v20211206", "val"), os.path.join(path, "val"))
    shutil.move(os.path.join(path, "OrganoidBasic_v20211206", "eval"), os.path.join(path, "eval"))
    shutil.rmtree(os.path.join(path, "OrganoidBasic_v20211206"))

    return data_dir


def get_orgasegment_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval"],
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths for the OrgaSegment data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download. Either 'train', 'val or 'eval'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_orgasegment_data(path=path, split=split, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "*_img.jpg")))
    label_paths = sorted(glob(os.path.join(data_dir, "*_masks_organoid.png")))

    return image_paths, label_paths


def get_orgasegment_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval"],
    patch_shape: Tuple[int, int],
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OrgaSegment dataset for organoid segmentation

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download. Either 'train', 'val or 'eval'.
        patch_shape: The patch shape to use for training.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in ["train", "val", "eval"]

    image_paths, label_paths = get_orgasegment_paths(path=path, split=split, download=download)

    kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True, binary=binary, boundaries=boundaries)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_orgasegment_loader(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval"],
    patch_shape: Tuple[int, int],
    batch_size: int,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OrgaSegment dataloader for organoid segmentation

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download. Either 'train', 'val or 'eval'.
        patch_shape: The patch shape to use for training.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orgasegment_dataset(
        path=path,
        split=split,
        patch_shape=patch_shape,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
