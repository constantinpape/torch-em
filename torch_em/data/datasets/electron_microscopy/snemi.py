"""SNEMI is a dataset for neuron segmentation in EM.

It contains an annotated volumes from the mouse brain.
The data is part of the publication https://doi.org/10.1016/j.cell.2015.06.054.
Please cite it if you use this dataset for a publication.
"""

import os
from typing import List, Optional, Union, Tuple

import torch_em
from torch.utils.data import Dataset, DataLoader
from .. import util

SNEMI_URLS = {
    "train": "https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download",
    "test": "https://oc.embl.de/index.php/s/aRhphk35H23De2s/download"
}
CHECKSUMS = {
    "train": "5b130a24d9eb23d972fede0f1a403bc05f6808b361cfa22eff23b930b12f0615",
    "test": "3df3920a0ddec6897105845f842b2665d37a47c2d1b96d4f4565682e315a59fa"
}


def get_snemi_data(path: Union[os.PathLike, str], sample: str, download: bool) -> str:
    """Download the SNEMI training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: The sample to download, either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
    """
    os.makedirs(path, exist_ok=True)
    data_path = os.path.join(path, f"snemi_{sample}.h5")
    util.download_source(data_path, SNEMI_URLS[sample], download, CHECKSUMS[sample])
    assert os.path.exists(data_path), data_path
    return data_path


def get_snemi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sample: str = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SNEMI dataset for the segmentation of neurons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample: The sample to download, either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3
    data_path = get_snemi_data(path, sample, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    raw_key = "volumes/raw"
    label_key = "volumes/labels/neuron_ids"
    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_snemi_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample: str = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for EM neuron segmentation in the SNEMI dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample: The sample to download, either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_snemi_dataset(
        path=path,
        patch_shape=patch_shape,
        sample=sample,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
