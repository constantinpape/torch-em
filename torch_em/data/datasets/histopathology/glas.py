"""
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import imageio.v3 as imageio

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


def _extract_images(split, path):
    import h5py

    data_folder = os.path.join(path, "Warwick_QU_Dataset")

    label_paths = natsorted(glob(os.path.join(data_folder, f"{split}*anno.bmp")))
    image_paths = [
        image_path for image_path in natsorted(glob(os.path.join(data_folder, f"{split}*.bmp")))
        if image_path not in label_paths
    ]
    assert image_paths and len(image_paths) == len(label_paths)

    os.makedirs(os.path.join(path, split), exist_ok=True)

    for image_path, label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths),
        desc=f"Extract images from {os.path.abspath(data_folder)}"
    ):
        fname = os.path.basename(image_path).split(".")[0]

        image = imageio.imread(image_path)
        segmentation = imageio.imread(label_path)
        image = image.transpose(2, 0, 1)

        with h5py.File(os.path.join(path, split, f"{fname}.h5"), "a") as f:
            f.create_dataset("raw", data=image, compression="gzip")
            f.create_dataset("labels", data=segmentation, compression="gzip")


def get_glas_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> str:
    """Download the GlaS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    # Download the files.
    util.download_source_kaggle(path=path, dataset_name="sani84/glasmiccai2015-gland-segmentation", download=download)
    util.unzip(zip_path=os.path.join(path, "glasmiccai2015-gland-segmentation.zip"), dst=path)

    # Preprocess inputs per split.
    splits = ["train", "test"]
    if split not in splits:
        raise ValueError(f"'{split}' is not a valid split choice.")

    for _split in splits:
        _extract_images(_split, path)

    # Remove original data
    shutil.rmtree(os.path.join(path, "Warwick_QU_Dataset"))

    return data_dir


def get_glas_paths(path: Union[os.PathLike], split: Literal["train", "test"], download: bool = False) -> List[str]:
    """Get paths to the GlaS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    data_dir = get_glas_data(path, split, download)
    data_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return data_paths


def get_glas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the GlaS dataset for gland segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the input images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_glas_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_glas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the GlaS dataloader for gland segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_glas_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
