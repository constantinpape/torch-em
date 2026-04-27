"""The Vibrio Cholerae dataset contains 3D confocal fluorescence microscopy images
of Vibrio cholerae biofilms with instance segmentation annotations for single-cell
segmentation.

The dataset provides two annotation types for 5 biofilm volumes:
- semi-manual-annotation: all 5 volumes labeled via automated segmentation + manual correction.
- fully-manual-annotation: 1 cropped volume (biofilm_1) with fully manual annotations —
  intended as a held-out evaluation set.

NOTE: The semi-manual labels are used by default for training. Whether all cells in each
volume are annotated should be verified against the paper before assuming dense coverage.

The dataset is located at https://zenodo.org/records/7704410.
This dataset is from the publication https://doi.org/10.1111/mmi.15064.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7704410/files/ZENODO.zip"
CHECKSUM = "31edb3edbbd308261ead96fa6ec201aff4daf6a0fa8624462c0384e61d67d4c8"


def get_vibrio_cholerae_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Vibrio Cholerae dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data directory.
    """
    data_dir = os.path.join(path, "training-data-from-experimentally-acquired-images")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "ZENODO.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    return data_dir


def get_vibrio_cholerae_paths(
    path: Union[os.PathLike, str], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Vibrio Cholerae data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_vibrio_cholerae_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "raw-data", "*_raw.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "semi-manual-annotation", "*_labels.tif")))

    if len(raw_paths) == 0:
        raise RuntimeError(
            f"No image files found in {os.path.join(data_dir, 'raw-data')}. "
            "Please check the dataset structure."
        )
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Number of images ({len(raw_paths)}) and labels ({len(label_paths)}) do not match."
        )

    return raw_paths, label_paths


def get_vibrio_cholerae_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Vibrio Cholerae dataset for 3D cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_vibrio_cholerae_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_vibrio_cholerae_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Vibrio Cholerae dataloader for 3D cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_vibrio_cholerae_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
