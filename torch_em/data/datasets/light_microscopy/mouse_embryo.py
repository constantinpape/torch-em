"""This dataset contains confocal microscopy stacks of a mouse embryo
with annotations for cell and nucleus segmentation.

This dataset is part of the publication https://doi.org/10.15252/embj.2022113280.
Please cite it if you use this data in your research.
"""

import os
from glob import glob
from typing import List, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/record/6546550/files/MouseEmbryos.zip?download=1"
CHECKSUM = "bf24df25e5f919489ce9e674876ff27e06af84445c48cf2900f1ab590a042622"


def get_mouse_embryo_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the mouse embryo dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    if os.path.exists(path):
        return path
    os.makedirs(path, exist_ok=True)
    tmp_path = os.path.join(path, "mouse_embryo.zip")
    util.download_source(tmp_path, URL, download, CHECKSUM)
    util.unzip(tmp_path, path, remove=True)
    # Remove empty volume.
    os.remove(os.path.join(path, "Membrane", "train", "fused_paral_stack0_chan2_tp00073_raw_crop_bg_noise.h5"))
    return path


def get_mouse_embryo_paths(path: Union[os.PathLike, str], name: str, split: str, download: bool = False) -> List[str]:
    """Get paths to the Mouse Embryo data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the segmentation task. Either 'membrane' or 'nuclei'.
        split: The split to use for the dataset. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    get_mouse_embryo_data(path, download)

    # the naming of the data is inconsistent: membrane has val, nuclei has test;
    # we treat nuclei:test as val
    split_ = "test" if name == "nuclei" and split == "val" else split
    file_paths = glob(os.path.join(path, name.capitalize(), split_, "*.h5"))
    file_paths.sort()

    return file_paths


def get_mouse_embryo_dataset(
    path: Union[os.PathLike, str],
    name: str,
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the mouse embryo dataset for cell or nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the segmentation task. Either 'membrane' or 'nuclei'.
        split: The split to use for the dataset. Either 'train' or 'val'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert name in ("membrane", "nuclei")
    assert split in ("train", "val")
    assert len(patch_shape) == 3

    file_paths = get_mouse_embryo_paths(path, name, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs,
        add_binary_target=binary,
        binary=binary,
        boundaries=boundaries,
        offsets=offsets,
        binary_is_exclusive=False
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=file_paths,
        raw_key="raw",
        label_paths=file_paths,
        label_key="label",
        patch_shape=patch_shape,
        **kwargs
    )


def get_mouse_embryo_loader(
    path: Union[os.PathLike, str],
    name: str,
    split: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the mouse embryo dataloader for cell or nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the segmentation task. Either 'membrane' or 'nuclei'.
        split: The split to use for the dataset. Either 'train' or 'val'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mouse_embryo_dataset(
        path, name, split, patch_shape, download=download, offsets=offsets,
        boundaries=boundaries, binary=binary, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
