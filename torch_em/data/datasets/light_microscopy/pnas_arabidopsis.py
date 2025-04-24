"""The PNAS Arabidopsis dataset contains cell segmentation in confocal microscopy images of
arabidopsis plantlets.

NOTE: There is tracking information available for this data.

This dataset is from the publication https://doi.org/10.1073/pnas.1616768113.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.repository.cam.ac.uk/bitstream/handle/1810/262530/PNAS.zip?sequence=4&isAllowed=y"
CHECKSUM = "39341398389baf6d93c3f652b7e2e8aedc5579c29dfaf2b82b41ebfc3caa05c4"


def get_pnas_arabidopsis_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PNAS Arabidopsis dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and pre-processed.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir)

    zip_path = os.path.join(path, "PNAS.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    # Convert the data to h5 (It's hard to keep a track of filenames as they are not completely consistent)
    import h5py

    raw_paths = natsorted(glob(os.path.join(data_dir, "plant*", "processed_tiffs", "*trim-acylYFP.tif")))
    for rpath in tqdm(raw_paths, desc="Preprocessing images"):
        # Let's find the label.
        label_path = rpath.replace("processed_tiffs", "segmentation_tiffs")
        label_path = glob(label_path.replace(".tif", "*.tif"))
        assert len(label_path) == 1, "It should not be possible to find more than one labels from the search above."

        raw = imageio.imread(rpath)
        labels = imageio.imread(label_path)

        # Store both image and corresponding labels in a h5 file.
        vol_path = os.path.join(data_dir, Path(os.path.basename(rpath)).with_suffix(".h5"))
        with h5py.File(vol_path, "w") as f:
            f.create_dataset("raw", data=raw, dtype=raw.dtype, compression="gzip")
            f.create_dataset("labels", data=labels, dtype=labels.dtype, compression="gzip")

    # Remove old data folder
    shutil.rmtree(os.path.join(path, "PNAS"))

    return data_dir


def get_pnas_arabidopsis_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the PNAS Arabidopsis data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data.
    """
    data_dir = get_pnas_arabidopsis_data(path, download)
    volume_paths = glob(os.path.join(data_dir, ".h5"))
    return volume_paths


def get_pnas_arabidopsis_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> Dataset:
    """Get the PNAS Arabidopsis dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_pnas_arabidopsis_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_pnas_arabidopsis_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> DataLoader:
    """Get the PNAS Arabidopsis dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pnas_arabidopsis_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
