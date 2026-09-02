"""The Khoshdeli dataset contains annotations for nucleus segmentation in H&E stained
histopathology images of brain tumor (TCGA) and breast cancer tissue.

The dataset is located at https://doi.org/10.6084/m9.figshare.6944522.
This dataset is from the publication https://doi.org/10.1186/s12859-018-2285-0.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Tuple, Union, List

import imageio.v3 as imageio
from bioimage_cpp.segmentation import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/12737267"
CHECKSUM = "929db05b0fff9139d25c8119daebe27baf6696bb6cedfc935e6e4d4be75a7620"


def get_khoshdeli_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Khoshdeli dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Nuclear-Segmentation-Data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "khoshdeli_supplement.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    tar_path = os.path.join(path, "12859_2018_2285_MOESM1_ESM", "Nuclear-Segmentation-Data.tar")
    util.unzip_tarfile(tar_path=tar_path, dst=path)

    return data_dir


def get_khoshdeli_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the Khoshdeli data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_khoshdeli_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "Images", "*.bmp")))
    mask_paths = natsorted(glob(os.path.join(data_dir, "Masks", "*_MASK.bmp")))
    assert len(raw_paths) == len(mask_paths) and len(raw_paths) > 0

    label_paths = []
    for mpath in tqdm(mask_paths, desc="Preprocessing 'khoshdeli' labels"):
        label_path = mpath.replace("_MASK.bmp", "_instances.tif")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue

        mask = imageio.imread(mpath) > 0
        label = connected_components(mask)  # run connected components to derive nucleus instances.
        imageio.imwrite(label_path, label, compression="zlib")

    return raw_paths, label_paths


def get_khoshdeli_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Khoshdeli dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_khoshdeli_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_khoshdeli_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Khoshdeli dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_khoshdeli_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
