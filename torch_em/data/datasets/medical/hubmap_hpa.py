"""The HuBMAP + HPA dataset contains annotations for functional tissue unit (FTU) segmentation
in histopathology images of five human organs: kidney, large intestine, spleen, lung and prostate.

This is the dataset for the "HuBMAP + HPA - Hacking the Human Body" Kaggle competition, located at
https://www.kaggle.com/competitions/hubmap-organ-segmentation. It combines tissue images from the
Human BioMolecular Atlas Program (HuBMAP) and the Human Protein Atlas (HPA), prepared with different
staining protocols and imaged at different resolutions.

NOTE: Only the 'train' split has public annotations. The 'test' split labels are held out by Kaggle
for competition scoring, so this loader only exposes the labeled training images.

NOTE: Downloading this dataset requires a Kaggle account that has accepted the competition rules at
https://www.kaggle.com/competitions/hubmap-organ-segmentation/rules. Without that, the Kaggle API
download fails with an HTTP 403 error, even with valid API credentials.

This dataset is from the publication https://doi.org/10.1038/s42003-023-04848-5.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from natsort import natsorted
from typing import List, Optional, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util

# The RLE-encoded masks in 'train.csv' exceed the default field size limit for large images.
csv.field_size_limit(10**8)


ORGANS = ["kidney", "large_intestine", "spleen", "lung", "prostate"]


def _decode_rle(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode a Kaggle-style run-length-encoded mask into a binary array.

    The encoding lists 1-indexed (start, length) pairs over a column-major (Fortran order)
    flattening of the image, i.e. pixels are numbered from top to bottom, then left to right.
    """
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)

    values = [int(v) for v in rle.split()]
    starts = values[0::2]
    lengths = values[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1

    return mask.reshape((width, height)).T


def get_hubmap_hpa_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HuBMAP + HPA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "train_images")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "hubmap-organ-segmentation.zip")
    util.download_source_kaggle(
        path=path, dataset_name="hubmap-organ-segmentation", download=download, competition=True
    )
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_hubmap_hpa_paths(
    path: Union[os.PathLike, str],
    organ: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HuBMAP + HPA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        organ: The choice of organ(s) to subselect the data for. Refer to `ORGANS` for the supported organs.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_hubmap_hpa_data(path, download)

    if organ is None:
        organs = ORGANS
    else:
        organs = [organ] if isinstance(organ, str) else organ
        for _organ in organs:
            if _organ not in ORGANS:
                raise ValueError(f"'{_organ}' is not a valid organ. Choose from {ORGANS}.")

    label_dir = os.path.join(data_dir, "masks")
    os.makedirs(label_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Could not find 'train.csv' at {csv_path}.")

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    image_paths, label_paths = [], []
    for row in rows:
        organ_name = row["organ"].replace(" ", "_")
        if organ_name not in organs:
            continue

        image_id = row["id"]
        image_path = os.path.join(data_dir, "train_images", f"{image_id}.tiff")
        if not os.path.exists(image_path):
            continue

        label_path = os.path.join(label_dir, f"{image_id}.tif")
        if not os.path.exists(label_path):
            shape = (int(row["img_height"]), int(row["img_width"]))
            mask = _decode_rle(row["rle"], shape)
            imageio.imwrite(label_path, mask, compression="zlib")

        image_paths.append(image_path)
        label_paths.append(label_path)

    return natsorted(image_paths), natsorted(label_paths)


def get_hubmap_hpa_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    organ: Optional[Union[str, List[str]]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HuBMAP + HPA dataset for functional tissue unit segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        organ: The choice of organ(s) to subselect the data for. Refer to `ORGANS` for the supported organs.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_hubmap_hpa_paths(path, organ, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_hubmap_hpa_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    organ: Optional[Union[str, List[str]]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HuBMAP + HPA dataloader for functional tissue unit segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organ: The choice of organ(s) to subselect the data for. Refer to `ORGANS` for the supported organs.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hubmap_hpa_dataset(path, patch_shape, organ, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
