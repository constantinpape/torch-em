"""This dataset contains annotation for nucleus segmentation in 3d fluorescence microscopy from mesoSPIM microscopy.

This dataset is from the publication https://doi.org/10.1101/2024.05.17.594691 .
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util

URL = "https://zenodo.org/records/11095111/files/DATASET_WITH_GT.zip?download=1"
CHECKSUM = "6d8e8d778e479000161fdfea70201a6ded95b3958a703f69def63e69bbddf9d6"


def get_cellseg_3d_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the CellSeg3d training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    url = URL
    checksum = CHECKSUM

    data_path = os.path.join(path, "DATASET_WITH_GT")
    if os.path.exists(data_path):
        return data_path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "cellseg3d.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)

    return data_path


def get_cellseg_3d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    sample_ids: Optional[Tuple[int, ...]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CellSeg3d dataset for segmenting nuclei in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The volume ids to load.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    data_root = get_cellseg_3d_data(path, download)

    raw_paths = sorted(glob(os.path.join(data_root, "*.tif")))
    label_paths = sorted(glob(os.path.join(data_root, "labels", "*.tif")))
    assert len(raw_paths) == len(label_paths)
    if sample_ids is not None:
        assert all(sid < len(raw_paths) for sid in sample_ids)
        raw_paths = [raw_paths[i] for i in sample_ids]
        label_paths = [label_paths[i] for i in sample_ids]

    raw_key, label_key = None, None

    return torch_em.default_segmentation_dataset(
        raw_paths, raw_key, label_paths, label_key, patch_shape, **kwargs
    )


def get_cellseg_3d_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    sample_ids: Optional[Tuple[int, ...]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CellSeg3d dataloder for segmenting nuclei in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The volume ids to load.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellseg_3d_dataset(
        path, patch_shape, sample_ids=sample_ids, download=download, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
    return loader
