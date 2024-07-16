"""This is a dataset for counting HeLA cells in phasecontrast microscopy.

It is described in the publication https://www.robots.ox.ac.uk/~vgg/publications/2012/Arteta12/.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from shutil import rmtree
from typing import Tuple, Union

import imageio.v3 as imageio
import numpy as np
import torch_em
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from .. import util

URL = "https://www.robots.ox.ac.uk/~vgg/software/cell_detection/downloads/CellDetect_v1.0.tar.gz"
CHECKSUM = "09825d6a8e287ddf2c4b1ef3d2f62585ec6876e3bfcd4b9bbcd3dd300e4be282"


def get_vgg_hela_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the HeLA VGG dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    os.makedirs(path, exist_ok=True)
    url = URL
    checksum = CHECKSUM

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return path

    dl_path = os.path.join(path, "cell_detect.tar.gz")
    util.download_source(dl_path, url, download, checksum)
    util.unzip_tarfile(dl_path, path, True)

    extracted_path = os.path.join(path, "CellDetect_v1.0")
    assert os.path.exists(extracted_path), extracted_path

    splits_in = ["trainPhasecontrast", "testPhasecontrast"]
    splits_out = [train_path, test_path]

    for split_in, out_folder in zip(splits_in, splits_out):
        out_im_folder = os.path.join(out_folder, "images")
        os.makedirs(out_im_folder, exist_ok=True)

        out_label_folder = os.path.join(out_folder, "labels")
        os.makedirs(out_label_folder, exist_ok=True)

        split_root = os.path.join(extracted_path, "phasecontrast", split_in)
        image_files = sorted(glob(os.path.join(split_root, "*.pgm")))
        mat_files = sorted(glob(os.path.join(split_root, "*.mat")))

        for ii, (im, mat) in enumerate(zip(image_files, mat_files), 1):
            im = imageio.imread(im)
            coordinates = loadmat(mat)["gt"] - 1
            coordinates = (coordinates[:, 1], coordinates[:, 0])

            out_im = os.path.join(out_im_folder, f"im{ii:02}.tif")
            imageio.imwrite(out_im, im, compression="zlib")

            labels = np.zeros(im.shape, dtype="uint8")
            labels[coordinates] = 1
            out_labels = os.path.join(out_label_folder, f"im{ii:02}.tif")
            imageio.imwrite(out_labels, labels, compression="zlib")

    rmtree(extracted_path)
    return path


def get_vgg_hela_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HeLA VGG dataset for cell counting.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert split in ("test", "train"), split
    get_vgg_hela_data(path, download)

    image_path = os.path.join(path, split, "images")
    label_path = os.path.join(path, split, "labels")

    kwargs = util.update_kwargs(kwargs, "ndim", 2)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    return torch_em.default_segmentation_dataset(
        image_path, "*.tif", label_path, "*.tif", patch_shape, **kwargs
    )


def get_vgg_hela_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HeLA VGG dataloader for cell counting.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_vgg_hela_dataset(
        path, split, patch_shape, download=download, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
