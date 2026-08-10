"""The DeMemSeg dataset contains annotations for prospore membrane segmentation
in fluorescence microscopy images of sporulating budding yeast.

The dataset contains 2133 single cell crops of Saccharomyces cerevisiae with 7593 expert annotated
prospore membranes (PSMs). Each crop comes from a 2d maximum intensity projection of a 3d z-stack.
The original annotations overlap, because several prospore membranes can cover the same pixel. This
loader flattens them into one instance label image. A pixel that belongs to several membranes keeps
the label of the last membrane, so about 13 percent of the foreground pixels get an arbitrary label.

The dataset is located at https://ssbd.riken.jp/repository/443/ under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1247/csf.25032.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://ssbd.riken.jp/data/ssbd-000443/zip/OriginalData_MMdetDataset.zip"
CHECKSUM = "b9772e343956358cf5e89459a28b02e6e3f0e05f403f640e8705957a0185a4c6"

SPLITS = ("train", "val", "test")


def _create_instance_labels(data_dir: str, split: str) -> str:
    """Merge the per-instance masks of each crop into one instance label image."""
    import h5py
    from tqdm import tqdm

    image_dir = os.path.join(data_dir, "images", split)
    mask_dir = os.path.join(data_dir, "masks", split)

    preprocessed_dir = os.path.join(data_dir, "preprocessed", split)
    os.makedirs(preprocessed_dir, exist_ok=True)

    image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
    for image_path in tqdm(image_paths, desc=f"Preprocess the '{split}' split"):
        stem = Path(image_path).stem
        output_path = os.path.join(preprocessed_dir, f"{stem}.h5")
        if os.path.exists(output_path):
            continue

        mask_paths = natsorted(glob(os.path.join(mask_dir, f"{stem}_RoiRegion_*.png")))
        if not mask_paths:
            raise RuntimeError(f"Could not find any mask for the DeMemSeg image {image_path}.")

        raw = imageio.imread(image_path)
        if raw.ndim == 3:
            raw = raw[..., 0]  # Only the first channel holds the membrane signal.

        masks = [imageio.imread(p) > 0 for p in mask_paths]
        shape = masks[0].shape

        # A crop at the border of the field is clipped, but its masks keep the full size.
        # The clipped crop sits in the center of the mask canvas, so pad it on both sides.
        if raw.shape != shape:
            raw = np.pad(raw, [((s - r) // 2, s - r - (s - r) // 2) for r, s in zip(raw.shape, shape)])

        # Paint the large membranes first, so that a membrane inside a larger one keeps its label.
        labels = np.zeros(shape, dtype="uint16")
        for instance_id, mask in enumerate(sorted(masks, key=lambda mask: -mask.sum()), start=1):
            labels[mask] = instance_id

        with h5py.File(output_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

    return preprocessed_dir


def get_dememseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DeMemSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "OriginalData_MMdetDataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "OriginalData_MMdetDataset.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_dememseg_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> List[str]:
    """Get paths to the DeMemSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the preprocessed h5 data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    data_dir = get_dememseg_data(path, download)
    preprocessed_dir = _create_instance_labels(data_dir, split)

    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    if not volume_paths:
        raise RuntimeError(f"Could not find any preprocessed DeMemSeg data in {preprocessed_dir}.")

    return volume_paths


def get_dememseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the DeMemSeg dataset for prospore membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The DeMemSeg patch shape must be two-dimensional, got {patch_shape}.")

    volume_paths = get_dememseg_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs,
    )


def get_dememseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DeMemSeg dataloader for prospore membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dememseg_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
