"""The BBBC050 dataset contains 3D time-lapse fluorescence microscopy images of early mouse embryos
(from fertilization to blastocyst formation) with nuclei labeled via histone H2B, and ground truth
annotations for nucleus segmentation.

The dataset consists of 11 training embryos (121 volumes, imaged with an IX71 microscope,
voxel size 0.8 x 0.8 x 1.75 um) and 4 test embryos (44 volumes, imaged with a CV1000 microscope,
voxel size 0.8 x 0.8 x 2.0 um). Every embryo is sampled at 11 time points.

Three types of ground truth are available for the training split (the test split only has 'QCANet'):
- 'QCANet': instance segmentation, every nucleus has its own id.
- 'NSN': semantic segmentation of the whole nuclear regions (foreground = 255).
- 'NDN': semantic segmentation of the nuclear center regions (foreground = 255).

The dataset is located at https://bbbc.broadinstitute.org/BBBC050.

This dataset is from the publication https://doi.org/10.1038/s41540-020-00152-8.
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
    "images": "https://data.broadinstitute.org/bbbc/BBBC050/Images.zip",
    "labels": "https://data.broadinstitute.org/bbbc/BBBC050/GroundTruth.zip",
}

CHECKSUMS = {
    "images": "29f100abbfebfb1986b8e87eac091e86d8ec27cd8194f9a1c02c805e76b6dcd8",
    "labels": "1f19b308730dccf217c4d4dcf5745ad0fcde4eeb9f9c9306b2c8abd1fe73e5d1",
}


def get_bbbc050_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BBBC050 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    image_dir, label_dir = os.path.join(path, "Images"), os.path.join(path, "GroundTruth")
    if os.path.exists(image_dir) and os.path.exists(label_dir):
        return path

    os.makedirs(path, exist_ok=True)

    for name, url in URLS.items():
        zip_path = os.path.join(path, os.path.basename(url))
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=path)

    return path


def get_bbbc050_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    label_type: Literal["QCANet", "NSN", "NDN"] = "QCANet",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC050 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        label_type: The choice of ground truth. Either 'QCANet' (instances), 'NSN' (nuclear regions)
            or 'NDN' (nuclear center regions). The 'test' split only provides 'QCANet' labels.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose either 'train' or 'test'.")
    if label_type not in ("QCANet", "NSN", "NDN"):
        raise ValueError(f"'{label_type}' is not a valid label type. Choose one of 'QCANet', 'NSN' or 'NDN'.")
    if split == "test" and label_type != "QCANet":
        raise ValueError("The 'test' split only provides 'QCANet' labels.")

    data_dir = get_bbbc050_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "Images", split, "Images", "*.tif")))
    label_paths = [
        os.path.join(data_dir, "GroundTruth", split, f"GroundTruth_{label_type}", os.path.basename(p))
        for p in raw_paths
    ]

    assert len(raw_paths) > 0, f"No volumes found for the '{split}' split at '{data_dir}'."
    assert all(os.path.exists(p) for p in label_paths), "Some label volumes are missing."

    return raw_paths, label_paths


def get_bbbc050_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    label_type: Literal["QCANet", "NSN", "NDN"] = "QCANet",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC050 dataset for nucleus segmentation in 3D time-lapse images of mouse embryos.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        label_type: The choice of ground truth. Either 'QCANet' (instances), 'NSN' (nuclear regions)
            or 'NDN' (nuclear center regions). The 'test' split only provides 'QCANet' labels.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bbbc050_paths(path, split, label_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_bbbc050_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    label_type: Literal["QCANet", "NSN", "NDN"] = "QCANet",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC050 dataloader for nucleus segmentation in 3D time-lapse images of mouse embryos.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        label_type: The choice of ground truth. Either 'QCANet' (instances), 'NSN' (nuclear regions)
            or 'NDN' (nuclear center regions). The 'test' split only provides 'QCANet' labels.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc050_dataset(path, patch_shape, split, label_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
