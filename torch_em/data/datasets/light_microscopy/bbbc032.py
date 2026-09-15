"""The BBBC032 dataset contains a 3D fluorescence microscopy image of a mouse embryo blastocyst
with instance segmentation ground truth for the nuclei.

The volume was acquired with a spinning disk confocal microscope and has a shape of (172, 1344, 1024) voxels (ZYX)
with a voxel size of 0.5 x 0.101 x 0.101 micrometer. It contains four channels, which are stored as separate volumes:
- channel 0: BMP4 transcripts (647 nm)
- channel 1: GAPDH transcripts (568 nm)
- channel 2: WGA, wheat germ agglutinin membrane stain (488 nm)
- channel 3: Hoechst nuclear stain (405 nm)
The ground truth contains 56 manually annotated nuclei as a labeled 16-bit volume (one id per nucleus, 0 background).
NOTE: The annotations are sparse, only a subset of the nuclei visible in the volume is annotated.

The dataset is located at https://bbbc.broadinstitute.org/BBBC032.
This dataset is from the publication https://doi.org/10.1038/s41586-018-0051-0.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from typing import List, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_dataset.zip"
CHECKSUM = "02df5ca7cdc9afb751c63161cabc0e7967310ed1911edaa8572764fb44455321"

GT_URL = "https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_DatasetGroundTruth.tif"
GT_CHECKSUM = "7ef577da64e1f95038d7eb40c03b3bce56ca2d5cd358cc9ea0aeba2be4526f3b"


def get_bbbc032_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BBBC032 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "BBBC032")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "BBBC032_v1_dataset.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, data_dir)
    # The zip file contains macOS metadata files, which we remove.
    shutil.rmtree(os.path.join(data_dir, "__MACOSX"), ignore_errors=True)

    gt_path = os.path.join(data_dir, "BBBC032_v1_DatasetGroundTruth.tif")
    util.download_source(gt_path, GT_URL, download, GT_CHECKSUM)

    return data_dir


def get_bbbc032_paths(
    path: Union[os.PathLike, str], channel: int = 3, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC032 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        channel: The channel to use as raw input. 0: BMP4, 1: GAPDH, 2: WGA (membranes), 3: Hoechst (nuclei).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if channel not in (0, 1, 2, 3):
        raise ValueError(f"'{channel}' is not a valid channel. Choose from 0, 1, 2 or 3.")

    data_dir = get_bbbc032_data(path, download)
    raw_path = os.path.join(data_dir, f"BMP4blastocystC{channel}.tif")
    label_path = os.path.join(data_dir, "BBBC032_v1_DatasetGroundTruth.tif")
    assert os.path.exists(raw_path) and os.path.exists(label_path)

    return [raw_path], [label_path]


def get_bbbc032_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    channel: int = 3,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC032 dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        channel: The channel to use as raw input. 0: BMP4, 1: GAPDH, 2: WGA (membranes), 3: Hoechst (nuclei).
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bbbc032_paths(path, channel, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_bbbc032_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    channel: int = 3,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC032 dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        channel: The channel to use as raw input. 0: BMP4, 1: GAPDH, 2: WGA (membranes), 3: Hoechst (nuclei).
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc032_dataset(path, patch_shape, channel, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
