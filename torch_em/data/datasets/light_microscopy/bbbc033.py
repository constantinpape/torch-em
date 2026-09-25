"""The BBBC033 dataset contains a 3D fluorescence microscopy image of a clustered monolayer of
mouse trophoblast stem cells with instance segmentation ground truth for the nuclei.

The volume was acquired with a spinning disk confocal microscope and has a shape of (32, 1344, 1024) voxels (ZYX)
with a z-spacing of 0.5 micrometer. It contains two channels, which are distributed as separate 8-bit RGB volumes
(named 'C0' and 'C2'). The BBBC page does not document the stains. From the image content, 'C0' is a membrane /
cytoplasm stain (stored as grayscale RGB) and 'C2' is the nuclear stain (stored as a blue-tinted RGB rendering).
Both are converted to single-channel 8-bit volumes and stored, together with the labels, in a HDF5 file
(keys 'raw/c0', 'raw/c2' and 'labels') when the data is prepared.
The ground truth contains 15 manually annotated nuclei as a labeled 16-bit volume (one id per nucleus, 0 background).

The dataset is located at https://bbbc.broadinstitute.org/BBBC033.
This dataset is from the publication https://doi.org/10.1038/s41586-018-0051-0.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from typing import Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.broadinstitute.org/bbbc/BBBC033/BBBC033_v1_dataset.zip"
CHECKSUM = "232e73c781f8658cef2abe2ea41ac1e352f8032e9292ec9618e2eb2aa5dc760b"

GT_URL = "https://data.broadinstitute.org/bbbc/BBBC033/BBBC033_v1_DatasetGroundTruth.tif"
GT_CHECKSUM = "c9bef6906ee450fac0d8fbcf588dad13d941ed71620010cf92a137984368c21f"


def _convert_to_h5(tmp_dir, gt_path, volume_path):
    import h5py
    import tifffile

    with h5py.File(volume_path, "w") as f:
        for channel in ("C0", "C2"):
            # The channels are stored as RGB volumes, we reduce them to a single intensity channel.
            raw = tifffile.imread(os.path.join(tmp_dir, f"{channel}.tif"))
            assert raw.ndim == 4 and raw.shape[-1] == 3, f"Unexpected shape for {channel}: {raw.shape}"
            f.create_dataset(f"raw/{channel.lower()}", data=raw.max(axis=-1), compression="gzip")

        labels = tifffile.imread(gt_path)
        f.create_dataset("labels", data=labels, compression="gzip")


def get_bbbc033_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BBBC033 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the HDF5 file with the data.
    """
    data_dir = os.path.join(path, "BBBC033")
    volume_path = os.path.join(data_dir, "BBBC033.h5")
    if os.path.exists(volume_path):
        return volume_path

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(path, "BBBC033_v1_dataset.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    tmp_dir = os.path.join(data_dir, "tif")
    util.unzip(zip_path, tmp_dir)

    gt_path = os.path.join(path, "BBBC033_v1_DatasetGroundTruth.tif")
    util.download_source(gt_path, GT_URL, download, GT_CHECKSUM)

    _convert_to_h5(tmp_dir, gt_path, volume_path)
    shutil.rmtree(tmp_dir)
    os.remove(gt_path)

    return volume_path


def get_bbbc033_paths(
    path: Union[os.PathLike, str], channel: Literal[0, 2] = 2, download: bool = False
) -> Tuple[str, str]:
    """Get paths to the BBBC033 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        channel: The channel to use as raw input. 0: membrane / cytoplasm stain, 2: nuclear stain.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the HDF5 file with the data.
        The key of the raw data for the chosen channel within the HDF5 file.
    """
    if channel not in (0, 2):
        raise ValueError(f"'{channel}' is not a valid channel. Choose from 0 or 2.")
    volume_path = get_bbbc033_data(path, download)
    return volume_path, f"raw/c{channel}"


def get_bbbc033_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    channel: Literal[0, 2] = 2,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC033 dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        channel: The channel to use as raw input. 0: membrane / cytoplasm stain, 2: nuclear stain.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_path, raw_key = get_bbbc033_paths(path, channel, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_path,
        raw_key=raw_key,
        label_paths=volume_path,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_bbbc033_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    channel: Literal[0, 2] = 2,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC033 dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        channel: The channel to use as raw input. 0: membrane / cytoplasm stain, 2: nuclear stain.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc033_dataset(path, patch_shape, channel, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
