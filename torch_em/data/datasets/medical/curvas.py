"""The CURVAS dataset contains annotations for pancreas, kidney and liver
in abdominal CT scans.

This dataset is from the challenge: https://curvas.grand-challenge.org.
The dataset is located at: https://zenodo.org/records/12687192.
Please cite tem if you use this dataset for your research.
"""

import os
import subprocess
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/12687192/files/training_set.zip"
CHECKSUM = "1126a2205553ae1d4fe5fbaee7ea732aacc4f5a92b96504ed521c23e5a0e3f89"


def get_curvas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CURVAS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "training_set")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "training_set.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    # HACK: The zip file is broken. We fix it using the following script.
    fixed_zip_path = os.path.join(path, "training_set_fixed.zip")
    subprocess.run(["zip", "-FF", zip_path, "--out", fixed_zip_path])
    subprocess.run(["unzip", fixed_zip_path, "-d", path])

    return data_dir


def get_curvas_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    rater: Literal["1"] = "1",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CURVAS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        rater: The choice of rater providing the annotations.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_curvas_data(path, download)

    if not isinstance(rater, list):
        rater = [rater]

    assert len(rater) == 1, "The segmentations for multiple raters is not supported at the moment."

    image_paths = natsorted(glob(os.path.join(data_dir, "*", "image.nii.gz")))
    gt_paths = []
    for _rater in rater:
        gt_paths.extend(natsorted(glob(os.path.join(data_dir, "*", f"annotation_{_rater}.nii.gz"))))

    assert len(image_paths) == len(gt_paths)

    if split == "train":
        image_paths, gt_paths = image_paths[:10], gt_paths[:10]
    elif split == "val":
        image_paths, gt_paths = image_paths[10:13], gt_paths[10:13]
    elif split == "test":
        image_paths, gt_paths = image_paths[13:], gt_paths[13:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_curvas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    rater: Literal["1"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CURVAS dataset for pancreas, kidney and liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of rater providing the annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_curvas_paths(path, split, rater, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )


def get_curvas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    rater: Literal["1"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CURVAS dataloader for pancreas, kidney and liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of rater providing the annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_curvas_dataset(path, patch_shape, split, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
