"""The US Nerve dataset contains annotations for segmentation of the Brachial Plexus (BP)
nerve structure in ultrasound images of the neck.

NOTE: This dataset requires the Kaggle API. You need to install it via 'pip install kaggle'
and set up an API token, see https://www.kaggle.com/docs/api. You also need to accept the
competition rules on the Kaggle website (https://www.kaggle.com/c/ultrasound-nerve-segmentation/rules)
before the download will succeed.

NOTE: Not all training images contain the nerve structure. For images where the Brachial
Plexus is not visible, the corresponding mask is empty (all-background).

The dataset is located at https://www.kaggle.com/c/ultrasound-nerve-segmentation.
This dataset is from the "Ultrasound Nerve Segmentation" Kaggle competition, hosted by Kensho.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_us_nerve_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the US Nerve dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(
        path=path, dataset_name="ultrasound-nerve-segmentation", download=download, competition=True
    )

    zip_path = os.path.join(path, "ultrasound-nerve-segmentation.zip")
    util.unzip(zip_path=zip_path, dst=path)

    # The competition bundle ships 'train' and 'test' as nested zip archives.
    for name in ["train", "test"]:
        nested_zip = os.path.join(path, f"{name}.zip")
        if os.path.exists(nested_zip):
            util.unzip(zip_path=nested_zip, dst=path)

    return path


def get_us_nerve_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the US Nerve data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_us_nerve_data(path=path, download=download)

    image_paths = natsorted([
        p for p in glob(os.path.join(data_dir, "train", "*.tif")) if not p.endswith("_mask.tif")
    ])
    gt_paths = natsorted(glob(os.path.join(data_dir, "train", "*_mask.tif")))

    assert len(image_paths) == len(gt_paths), f"{len(image_paths)} != {len(gt_paths)}"

    return image_paths, gt_paths


def get_us_nerve_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the US Nerve dataset for brachial plexus nerve segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_us_nerve_paths(path, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_us_nerve_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the US Nerve dataloader for brachial plexus nerve segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_us_nerve_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
