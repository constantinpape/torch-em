"""The OrganoIDNet dataset contains annotations of panceratic organoids.

This dataset is from the publication https://doi.org/10.1007/s13402-024-00958-2.
Please cite it if you use this dataset for a publication.
"""


import os
import shutil
import zipfile
from glob import glob
from typing import Tuple, Union, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10643410/files/OrganoIDNetData.zip?download=1"
CHECKSUM = "3cd9239bf74bda096ecb5b7bdb95f800c7fa30b9937f9aba6ddf98d754cbfa3d"


def get_organoidnet_data(path: Union[os.PathLike, str], split: str, download: bool = False) -> str:
    """Download the OrganoIDNet dataset.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        split: The data split to use.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath where the data is downloaded.
    """
    splits = ["Training", "Validation", "Test"]
    assert split in splits

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    # Download and extraction.
    zip_path = os.path.join(path, "OrganoIDNetData.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    # Only "Training", "Test", "Validation" from the zip are relevant and need to be extracted.
    # They are in "/OrganoIDNetData/Dataset/"
    prefix = "OrganoIDNetData/Dataset/"
    for dl_split in splits:

        dl_prefix = prefix + dl_split

        with zipfile.ZipFile(zip_path) as archive:
            for ff in archive.namelist():
                if ff.startswith(dl_prefix):
                    archive.extract(ff, path)

    for dl_split in splits:
        shutil.move(
            os.path.join(path, "OrganoIDNetData/Dataset", dl_split),
            os.path.join(path, dl_split)
        )

    assert os.path.exists(data_dir)

    os.remove(zip_path)
    return data_dir


def get_organoidnet_paths(
    path: Union[os.PathLike, str], split: str, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the OrganoIDNet data.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        split: The data split to use.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_organoidnet_data(path=path, split=split, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "Images", "*.tif")))
    label_paths = sorted(glob(os.path.join(data_dir, "Masks", "*.tif")))

    return image_paths, label_paths


def get_organoidnet_dataset(
    path: Union[os.PathLike, str], split: str, patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the OrganoIDNet dataset for organoid segmentation in microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_organoidnet_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_organoidnet_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OrganoIDNet dataset for organoid segmentation in microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_organoidnet_dataset(
        path=path, split=split, patch_shape=patch_shape, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
