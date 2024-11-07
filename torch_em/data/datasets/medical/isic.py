"""The ISIC dataset contains annotations for lesion segmentation in dermoscopy images.

This dataset is located at https://challenge.isic-archive.com/data/#2018
The dataset is related to the following publication(s):
- https://doi.org/10.1038/sdata.2018.161
- https://doi.org/10.48550/arXiv.1710.05006
- https://doi.org/10.48550/arXiv.1902.03368

Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


URL = {
    "images": {
        "train": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "val": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "test": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
    },
    "gt": {
        "train": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        "val": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "test": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip",
    },
}

CHECKSUM = {
    "images": {
        "train": "80f98572347a2d7a376227fa9eb2e4f7459d317cb619865b8b9910c81446675f",
        "val": "0ea920fcfe512d12a6e620b50b50233c059f67b10146e1479c82be58ff15a797",
        "test": "e59ae1f69f4ed16f09db2cb1d76c2a828487b63d28f6ab85997f5616869b127d",
    },
    "gt": {
        "train": "99f8b2bb3c4d6af483362010715f7e7d5d122d9f6c02cac0e0d15bef77c7604c",
        "val": "f6911e9c0a64e6d687dd3ca466ca927dd5e82145cb2163b7a1e5b37d7a716285",
        "test": "2e8f6edce454a5bdee52485e39f92bd6eddf357e81f39018d05512175238ef82",
    }
}


def get_isic_data(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> Tuple[str, str]:
    """Download the ISIC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the image data is downloaded.
        Filepath where the label data is downloaded.
    """
    assert split in list(URL["images"].keys()), f"{split} is not a valid split."

    im_url = URL["images"][split]
    im_checksum = CHECKSUM["images"][split]

    gt_url = URL["gt"][split]
    gt_checksum = CHECKSUM["gt"][split]

    im_zipfile = os.path.split(im_url)[-1]
    gt_zipfile = os.path.split(gt_url)[-1]

    imdir = os.path.join(path, Path(im_zipfile).stem)
    gtdir = os.path.join(path, Path(gt_zipfile).stem)

    if os.path.exists(imdir) and os.path.exists(gtdir):
        return imdir, gtdir

    os.makedirs(path, exist_ok=True)

    im_zip_path = os.path.join(path, im_zipfile)
    gt_zip_path = os.path.join(path, gt_zipfile)

    # download the images
    util.download_source(path=im_zip_path, url=im_url, download=download, checksum=im_checksum)
    util.unzip(zip_path=im_zip_path, dst=path, remove=False)
    # download the ground-truth
    util.download_source(path=gt_zip_path, url=gt_url, download=download, checksum=gt_checksum)
    util.unzip(zip_path=gt_zip_path, dst=path, remove=False)

    return imdir, gtdir


def get_isic_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ISIC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, gt_dir = get_isic_data(path=path, split=split, download=download)

    image_paths = natsorted(glob(os.path.join(image_dir, "*.jpg")))
    gt_paths = natsorted(glob(os.path.join(gt_dir, "*.png")))

    return image_paths, gt_paths


def get_isic_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    make_rgb: bool = True,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ISIC dataset for skin lesion segmentation in dermoscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        make_rgb: Ensure all inputs are in RGB-format.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_isic_paths(path=path, split=split, download=download)

    if make_rgb:
        kwargs["raw_transform"] = to_rgb

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
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


def get_isic_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    make_rgb: bool = True,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ISIC dataloader for skin lesion segmentation in dermoscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        make_rgb: Ensure all inputs are in RGB-format.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isic_dataset(path, patch_shape, split, make_rgb, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
