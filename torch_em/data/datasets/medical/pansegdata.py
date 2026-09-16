"""PanSegData contains annotations for pancreas segmentation in T1-weighted and T2-weighted abdominal MRI.

The dataset consists of 385 T1W and 382 T2W MRI volumes from five institutions, with binary pancreas labels.

NOTE: The dataset is distributed under the CC BY-NC 4.0 license. It is located at https://osf.io/kysnj/.

This dataset is from the publication https://doi.org/10.1016/j.media.2024.103382.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "t1": "https://osf.io/download/ch8ay/",
    "t2": "https://osf.io/download/bre8p/",
}

CHECKSUMS = {
    "t1": "f94c319f0ac8c627d1d871c542b4f8b2defcd206cb22959fd75965910f108b7f",
    "t2": "c1b6ab676f92e27a3743a1bb9da7a3794a6c70075d2d4b724cc522bd6e6e1351",
}


def get_pansegdata_data(path: Union[os.PathLike, str], modality: Literal["t1", "t2"], download: bool = False) -> str:
    """Download the PanSegData dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of MRI modality. Either 't1' or 't2'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the data is stored.
    """
    if modality not in URLS:
        raise ValueError(f"'{modality}' is not a valid modality. Choose either 't1' or 't2'.")

    data_dir = os.path.join(path, modality)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{modality}.zip")
    util.download_source(path=zip_path, url=URLS[modality], download=download, checksum=CHECKSUMS[modality])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_pansegdata_paths(
    path: Union[os.PathLike, str], modality: Literal["t1", "t2"] = "t2", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PanSegData data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of MRI modality. Either 't1' or 't2'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pansegdata_data(path, modality, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_pansegdata_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["t1", "t2"] = "t2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PanSegData dataset for pancreas segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of MRI modality. Either 't1' or 't2'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_pansegdata_paths(path, modality, download)

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


def get_pansegdata_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["t1", "t2"] = "t2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PanSegData dataloader for pancreas segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of MRI modality. Either 't1' or 't2'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pansegdata_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
