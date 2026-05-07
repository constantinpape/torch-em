"""The PhMamm dataset contains 3D light-sheet microscopy volumes of Phallusia mammillata
embryos with cell membrane segmentations.

The dataset is located at https://figshare.com/articles/dataset/3D_Mask_R-CNN_data/26973085.
The original data is from the publication https://doi.org/10.1126/science.aar5663.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "inputs": "https://ndownloader.figshare.com/files/51130115",
    "ground_truth": "https://ndownloader.figshare.com/files/51130100",
}
CHECKSUMS = {
    "inputs": None,
    "ground_truth": None,
}


def get_phmamm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PhMamm dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is stored.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    inputs_zip = os.path.join(path, "Inputs.zip")
    util.download_source(inputs_zip, URLS["inputs"], download, checksum=CHECKSUMS["inputs"])
    util.unzip(inputs_zip, data_dir, remove=True)

    gt_zip = os.path.join(path, "ASTEC_Ground_truth.zip")
    util.download_source(gt_zip, URLS["ground_truth"], download, checksum=CHECKSUMS["ground_truth"])
    util.unzip(gt_zip, data_dir, remove=True)

    return data_dir


def get_phmamm_paths(
    path: Union[os.PathLike, str], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PhMamm data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_phmamm_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "Inputs", "*.tiff")))
    label_paths = natsorted(glob(os.path.join(data_dir, "ASTEC_Ground_truth", "*.tiff")))
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_phmamm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PhMamm dataset for cell segmentation in light-sheet microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_phmamm_paths(path, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_phmamm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PhMamm dataloader for cell segmentation in light-sheet microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_phmamm_dataset(
        path=path,
        patch_shape=patch_shape,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
