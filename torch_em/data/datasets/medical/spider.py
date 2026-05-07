"""The SPIDER dataset contains annotations for segmentation of vertebrae,
intervertebral discs and spinal canal in T1 and T2 MRI series.

This dataset is from the following publication:
- https://zenodo.org/records/10159290
- https://www.nature.com/articles/s41597-024-03090-w

Please cite it if you use this data in a publication.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Tuple, List, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "images": "https://zenodo.org/records/10159290/files/images.zip?download=1",
    "masks": "https://zenodo.org/records/10159290/files/masks.zip?download=1"
}

CHECKSUMS = {
    "images": "a54cba2905284ff6cc9999f1dd0e4d871c8487187db7cd4b068484eac2f50f17",
    "masks": "13a6e25a8c0d74f507e16ebb2edafc277ceeaf2598474f1fed24fdf59cb7f18f"
}


def get_spider_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SPIDER dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "images.zip")
    util.download_source(path=zip_path, url=URL["images"], download=download, checksum=CHECKSUMS["images"])
    util.unzip(zip_path=zip_path, dst=data_dir)

    zip_path = os.path.join(path, "masks.zip")
    util.download_source(path=zip_path, url=URL["masks"], download=download, checksum=CHECKSUMS["masks"])
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_spider_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the SPIDER data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_spider_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.mha")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.mha")))

    return image_paths, gt_paths


def get_spider_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SPIDER dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    # TODO: expose the choice to choose specific MRI modality, for now this works for our interests.
    image_paths, gt_paths = get_spider_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_spider_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SPIDER dataloader.

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
    dataset = get_spider_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
