"""The ISBI2012 dataset was the first neuron segmentation challenge, held at the ISBI 2012 competition.
It contains a small annotated EM volume from the fruit-fly brain.

If you use this dataset in your research please cite the following publication:
https://doi.org/10.3389/fnana.2015.00142.
"""

import os
from typing import List, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em
from .. import util


ISBI_URL = "https://oc.embl.de/index.php/s/h0TkwqxU0PJDdMd/download"
CHECKSUM = "0e10fe909a1243084d91773470856993b7d40126a12e85f0f1345a7a9e512f29"


def get_isbi_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the ISBI2012 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    os.makedirs(path, exist_ok=True)
    util.download_source(os.path.join(path, "isbi.h5"), ISBI_URL, download, CHECKSUM)


def get_isbi_paths(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Get path to ISBI data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the stored data.
    """
    get_isbi_data(path, download)
    volume_path = os.path.join(path, "isbi.h5")
    return volume_path


def get_isbi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    use_original_labels: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for EM neuron segmentation in ISBI 2012.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        use_original_labels: Whether to use the original annotations or postprocessed 3d annotations.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3

    volume_path = get_isbi_paths(path, download)

    ndim = 2 if patch_shape[0] == 1 else 3
    kwargs = util.update_kwargs(kwargs, "ndim", ndim)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_path,
        raw_key="raw",
        label_paths=volume_path,
        label_key="labels/membranes" if use_original_labels else "labels/gt_segmentation",
        patch_shape=patch_shape,
        **kwargs
    )


def get_isbi_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    use_original_labels: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for EM neuron segmentation in ISBI 2012.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        use_original_labels: Whether to use the original annotations or postprocessed 3d annotations.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isbi_dataset(
        path, patch_shape, download=download, offsets=offsets,
        boundaries=boundaries, use_original_labels=use_original_labels, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
