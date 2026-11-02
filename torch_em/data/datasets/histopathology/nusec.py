"""The NuSeC dataset contains nucleus annotations for H&E stained images of invasive breast carcinoma.

The dataset holds 100 images of 1024 x 1024 pixels at 40x magnification from 25 patients of Ankara University,
75 for training and 25 for testing. The nuclei were outlined by hand in QuPath and checked by pathologists.
NOTE: The annotations are not exhaustive. Many nuclei, most of all the stromal nuclei, carry no annotation
and become background in the label images. Take this into account when you train on this data, and when you
measure the recall of a model on it.

The data is located at https://doi.org/10.6084/m9.figshare.29525111 (CC BY 4.0), together with the
MiDeSeC mitosis dataset of the same group, which this module does not use.
This dataset is from the publication https://doi.org/10.48550/arXiv.2507.14272.
Please cite it if you use this dataset for your research.
"""

import os
import zipfile
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/56111276"
CHECKSUM = "71102cec12a74f643d727b80e0a24ff12b1446a5fe3e0a94905617b28432ddf3"
ZIP_ROOT = "Ankara university datasets/NuSeC"


def _create_h5_files(path, zip_path):
    import h5py

    with zipfile.ZipFile(zip_path) as archive:
        for split in ("train", "test"):
            out_dir = os.path.join(path, "preprocessed", split)
            os.makedirs(out_dir, exist_ok=True)
            prefix = f"{ZIP_ROOT}/{split} nuclei/"
            names = [n for n in archive.namelist() if n.startswith(prefix) and not n.endswith("/")]
            for name in tqdm(names, desc=f"Preprocess NuSeC {split} images"):
                fname = os.path.basename(name)
                image = imageio.imread(archive.read(name), extension=".tif")[..., :3]
                labels = imageio.imread(archive.read(f"{ZIP_ROOT}/mask of {split} nuclei/{fname}"), extension=".tif")
                # The masks are stored as float, but hold integer instance ids.
                assert np.array_equal(labels, np.round(labels)), fname
                labels = labels.astype("uint32")
                assert image.shape[:2] == labels.shape, (fname, image.shape, labels.shape)

                with h5py.File(os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.h5"), "w") as f:
                    f.create_dataset("raw", data=image.transpose(2, 0, 1), compression="gzip")
                    f.create_dataset("labels/instances", data=labels, compression="gzip")


def get_nusec_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NuSeC dataset.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "nusec_midesec.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    _create_h5_files(path, zip_path)
    return data_dir


def get_nusec_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> List[str]:
    """Get paths to the NuSeC data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    assert split in ("train", "test"), split
    data_dir = get_nusec_data(path, download)
    return natsorted(glob(os.path.join(data_dir, split, "*.h5")))


def get_nusec_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NuSeC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_nusec_paths(path, split, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels/instances",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_nusec_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NuSeC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nusec_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
