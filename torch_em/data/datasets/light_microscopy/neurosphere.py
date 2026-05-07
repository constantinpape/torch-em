"""The Neurosphere dataset contains a 3D fluorescence light-sheet microscopy image
of a cancer cell neurosphere with ground truth instance segmentation from the
OpenSegSPIM analysis pipeline.

The dataset consists of a single volume of approximately 115 x 150 x 150 voxels.

NOTE: The segmentations are pixelated at the boundaries and don't exactly match the segmentation.

The dataset is located at https://sourceforge.net/projects/opensegspim/.
This dataset is from the publication https://doi.org/10.1093/bioinformatics/btw093.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Tuple, Union

import numpy as np
from scipy.ndimage import binary_fill_holes

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


RAW_URL = "https://downloads.sourceforge.net/project/opensegspim/Sample%20Data/Neurosphere_Dataset.zip"
LABEL_URL = "https://downloads.sourceforge.net/project/opensegspim/Sample%20Data/Neurosphere_OpenSegSPIM.zip"
RAW_CHECKSUM = None
LABEL_CHECKSUM = None


def get_neurosphere_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Neurosphere dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "neurosphere")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    raw_zip = os.path.join(path, "Neurosphere_Dataset.zip")
    label_zip = os.path.join(path, "Neurosphere_OpenSegSPIM.zip")

    util.download_source(raw_zip, RAW_URL, download, checksum=RAW_CHECKSUM)
    util.download_source(label_zip, LABEL_URL, download, checksum=LABEL_CHECKSUM)

    util.unzip(raw_zip, data_dir)
    util.unzip(label_zip, data_dir)

    return data_dir


def _fill_labels(data_dir: str) -> str:
    """Convert thin-shell contour labels to filled 3D instance segmentations.

    Loads Nucleisegmented2.tif, applies binary_fill_holes per instance,
    renumbers to sequential IDs (1, 2, 3 ...), and saves as filled_labels.tif.

    Args:
        data_dir: The neurosphere data directory.

    Returns:
        Path to the filled label file.
    """
    import imageio.v3 as imageio

    filled_path = os.path.join(data_dir, "filled_labels.tif")
    if os.path.exists(filled_path):
        return filled_path

    label_paths = natsorted(glob(os.path.join(data_dir, "**", "Nucleisegmented2.tif"), recursive=True))
    if len(label_paths) == 0:
        raise RuntimeError(f"Label file 'Nucleisegmented2.tif' not found in {data_dir}.")

    raw_labels = imageio.imread(label_paths[0])
    instance_ids = np.unique(raw_labels)
    instance_ids = instance_ids[instance_ids != 0]

    filled = np.zeros(raw_labels.shape, dtype=np.int32)
    for new_id, val in enumerate(instance_ids, start=1):
        mask = binary_fill_holes(raw_labels == val)
        filled[mask] = new_id

    imageio.imwrite(filled_path, filled)
    return filled_path


def get_neurosphere_paths(
    path: Union[os.PathLike, str], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Neurosphere data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_neurosphere_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "OriginalStack.tif")))
    if len(raw_paths) == 0:
        raise RuntimeError(
            f"Raw image 'OriginalStack.tif' not found in {data_dir}. "
            "Please check the dataset structure after downloading."
        )

    filled_label_path = _fill_labels(data_dir)
    label_paths = [filled_label_path]

    return raw_paths, label_paths


def get_neurosphere_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Neurosphere dataset for 3D cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_neurosphere_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_neurosphere_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Neurosphere dataloader for 3D cell instance segmentation.

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
    dataset = get_neurosphere_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
