"""The SNEMI Mito dataset contains annotations for mitochondria segmentation in electron microscopy (EM)
images of mouse neocortex.

It provides mitochondria labels for the sections of the SNEMI3D challenge volume, which comes from the publication
https://doi.org/10.1016/j.cell.2015.06.054. The data consists of 100 images (1024 x 1024 pixels, 8-bit, CLAHE contrast
enhanced) that are split into 50 train, 20 validation and 30 test images.
The images are consecutive sections of the volume (6 x 6 x 30 nm), so they can also be used as one 3D volume.
The volume is stored as `snemi_mito.h5` with 'raw' and binary 'labels', and the splits are its ROIs in `VOLUME_ROIS`.

The data is available at https://github.com/sumiya-kuroda/EMsegmentation-mito/tree/master/dataset/SNEMI3D_mito.
The repository does not state a license. The mitochondria labels were prepared by members of the Kubota lab
(see the repository for details). Please cite the SNEMI3D / Kasthuri et al. publication and the repository
if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from typing import List, Tuple, Union

import h5py
import numpy as np
import imageio.v3 as imageio
from natsort import natsorted

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# The archive is pinned to a commit, so that the checksum does not change with new commits to the repository.
URL = "https://github.com/sumiya-kuroda/EMsegmentation-mito/archive/44134f6dc27b1ad7f04293398e482fec854bb658.zip"
CHECKSUM = "1d560ad05d0faab8e6fedd9347131c2a0f395dcdba079f8699a66c3e39d80a25"
SPLITS = {"train": "train", "val": "valid", "test": "test"}
# The splits are contiguous ranges of sections in the volume.
VOLUME_ROIS = {"train": np.s_[0:50, :, :], "val": np.s_[50:70, :, :], "test": np.s_[70:100, :, :]}


def _convert_rgba_labels(data_dir):
    """Convert the label images that are stored as RGBA, with the gray values in every channel, to one channel."""
    from PIL import Image

    for label_path in glob(os.path.join(data_dir, "*", "labels", "*.png")):
        with Image.open(label_path) as image:
            if image.mode != "RGBA":
                continue
            labels = image.getchannel(0)
        labels.save(label_path)


def _create_volume(data_dir, volume_path):
    raw_paths, label_paths = [], []
    for split in SPLITS.values():
        raw_paths.extend(natsorted(glob(os.path.join(data_dir, split, "images", "*.png"))))
        label_paths.extend(natsorted(glob(os.path.join(data_dir, split, "labels", "*.png"))))
    names = [os.path.basename(raw_path) for raw_path in raw_paths]
    assert names == [os.path.basename(label_path) for label_path in label_paths]
    assert names == [f"{i:04}.png" for i in range(len(names))], "The sections are not consecutive."

    raw = np.stack([imageio.imread(raw_path) for raw_path in raw_paths])
    labels = np.stack([imageio.imread(label_path) > 0 for label_path in label_paths]).astype("uint8")

    tmp_path = f"{volume_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(8, 256, 256))
        f.create_dataset("labels", data=labels, compression="gzip", chunks=(8, 256, 256))
        f.attrs["voxel_size_nm"] = [30, 6, 6]
    os.replace(tmp_path, volume_path)


def _download_data(path, data_dir, download):
    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "snemi_mito.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)

    # The archive contains the whole repository, we only keep the SNEMI3D_mito folder.
    tmp_dir = os.path.join(path, "tmp")
    util.unzip(zip_path, tmp_dir, remove=True)
    extracted = glob(os.path.join(tmp_dir, "*", "dataset", "SNEMI3D_mito"))
    assert len(extracted) == 1, "Unexpected structure of the downloaded archive."
    shutil.move(extracted[0], data_dir)
    shutil.rmtree(tmp_dir)


def get_snemi_mito_data(path: Union[os.PathLike, str], download: bool = False, as_volume: bool = False) -> str:
    """Download the SNEMI Mito dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
        as_volume: Whether to stack the sections into one 3D volume.

    Returns:
        The filepath to the folder with the data, or the filepath to the volume if `as_volume` is set.
    """
    data_dir = os.path.join(path, "SNEMI3D_mito")
    volume_path = os.path.join(path, "snemi_mito.h5")
    if as_volume and os.path.exists(volume_path):
        return volume_path

    if not os.path.exists(data_dir):
        _download_data(path, data_dir, download)
    _convert_rgba_labels(data_dir)

    if as_volume:
        _create_volume(data_dir, volume_path)
        return volume_path
    return data_dir


def get_snemi_mito_paths(
    path: Union[os.PathLike, str], split: str, download: bool = False, as_volume: bool = False
) -> Union[Tuple[List[str], List[str]], Tuple[str, Tuple[slice, ...]]]:
    """Get paths to the SNEMI Mito data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        as_volume: Whether to return the 3D volume and the ROI of the split instead of the 2D images.

    Returns:
        List of filepaths for the image data, or the filepath to the volume if `as_volume` is set.
        List of filepaths for the label data, or the ROI of the split in the volume if `as_volume` is set.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose from {list(SPLITS.keys())}.")

    if as_volume:
        return get_snemi_mito_data(path, download, as_volume=True), VOLUME_ROIS[split]

    data_dir = get_snemi_mito_data(path, download)
    raw_paths = natsorted(glob(os.path.join(data_dir, SPLITS[split], "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, SPLITS[split], "labels", "*.png")))
    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def _binary_labels(labels):
    return (labels > 0).astype("uint8")


def get_snemi_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    binary: bool = True,
    download: bool = False,
    as_volume: bool = False,
    **kwargs
) -> Dataset:
    """Get the SNEMI Mito dataset for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        binary: Whether to return a binary foreground mask. Otherwise the mask values are returned as they are.
            The labels of the volume are always binary.
        download: Whether to download the data if it is not present.
        as_volume: Whether to use the split of the 3D volume instead of the 2D images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if as_volume:
        volume_path, roi = get_snemi_mito_paths(path, split, download, as_volume=True)
        return torch_em.default_segmentation_dataset(
            raw_paths=volume_path,
            raw_key="raw",
            label_paths=volume_path,
            label_key="labels",
            patch_shape=patch_shape,
            rois=roi,
            is_seg_dataset=True,
            **kwargs
        )

    raw_paths, label_paths = get_snemi_mito_paths(path, split, download)

    if binary:
        kwargs = util.update_kwargs(kwargs, "label_transform", _binary_labels)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_snemi_mito_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: str,
    binary: bool = True,
    download: bool = False,
    as_volume: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SNEMI Mito dataloader for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        binary: Whether to return a binary foreground mask. Otherwise the mask values are returned as they are.
            The labels of the volume are always binary.
        download: Whether to download the data if it is not present.
        as_volume: Whether to use the split of the 3D volume instead of the 2D images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_snemi_mito_dataset(path, patch_shape, split, binary, download, as_volume, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
