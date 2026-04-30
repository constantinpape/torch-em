"""BCData contains point annotations for Ki-67 positive and negative tumor cells in breast cancer IHC.

Note that BCData does not provide true semantic or instance segmentation masks. The original annotations are
cell-center coordinates only. This loader rasterizes these point annotations into single-pixel labels or small
disks, controlled by `cell_radius`, so they can be used with the torch-em segmentation dataset API.

The dataset is located at https://sites.google.com/view/bcdataset.
It is from the MICCAI 2020 publication "BCData: A Large-Scale Dataset and Benchmark for Cell Detection
and Counting". Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import List, Literal, Tuple, Union

import imageio.v3 as imageio
import numpy as np
from natsort import natsorted

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://drive.google.com/uc?export=download&id=16W04QOR1E-G3ifc4061Be4eGpjRYDlkA"

SPLITS = ("train", "validation", "test")


def _load_coordinates(path):
    import h5py

    with h5py.File(path, "r") as f:
        coordinates = np.asarray(f["coordinates"])
    return coordinates


def _draw_disk(labels, x, y, value, radius):
    h, w = labels.shape
    if radius <= 0:
        if 0 <= y < h and 0 <= x < w:
            labels[y, x] = value
        return

    y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
    x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    labels[y_min:y_max, x_min:x_max][mask] = value


def _points_to_mask(shape, positive_coordinates, negative_coordinates, cell_radius, binary):
    labels = np.zeros(shape, dtype="uint8")
    annotations = ((positive_coordinates, 1), (negative_coordinates, 1 if binary else 2))

    for coordinates, value in annotations:
        for coordinate in coordinates:
            if len(coordinate) < 2:
                continue
            # BCData stores coordinates as (x, y) pixel positions.
            x, y = int(round(coordinate[0])), int(round(coordinate[1]))
            _draw_disk(labels, x, y, value, cell_radius)

    return labels


def _label_subdir(cell_radius, binary):
    return f"radius_{cell_radius}_{'binary' if binary else 'classes'}"


def _check_data(path):
    data_root = os.path.join(path, "BCData")
    return all(os.path.exists(os.path.join(data_root, "images", split)) for split in SPLITS)


def _process_bcdata(path, cell_radius, binary):
    data_root = os.path.join(path, "BCData")
    label_root = os.path.join(data_root, "labels")

    for split in SPLITS:
        image_paths = natsorted(glob(os.path.join(data_root, "images", split, "*.png")))
        split_label_root = os.path.join(label_root, split, _label_subdir(cell_radius, binary))
        os.makedirs(split_label_root, exist_ok=True)

        for image_path in image_paths:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(split_label_root, f"{image_id}.tif")
            if os.path.exists(label_path):
                continue

            shape = imageio.improps(image_path).shape[:2]
            positive_path = os.path.join(data_root, "annotations", split, "positive", f"{image_id}.h5")
            negative_path = os.path.join(data_root, "annotations", split, "negative", f"{image_id}.h5")
            positive_coordinates = _load_coordinates(positive_path)
            negative_coordinates = _load_coordinates(negative_path)

            labels = _points_to_mask(
                shape, positive_coordinates, negative_coordinates, cell_radius=cell_radius, binary=binary
            )
            imageio.imwrite(label_path, labels, compression="zlib")


def get_bcdata_data(
    path: Union[os.PathLike, str],
    download: bool = False,
    cell_radius: int = 0,
    binary: bool = False,
) -> str:
    """Download and preprocess the BCData dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
        cell_radius: Radius for rasterizing point annotations. The default keeps single-pixel labels.
        binary: Whether to merge positive and negative cells into a single foreground class.

    Returns:
        Filepath to the BCData folder.
    """
    if cell_radius < 0:
        raise ValueError(f"Invalid cell radius: {cell_radius}.")

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "BCData.zip")

    if not _check_data(path):
        util.download_source_gdrive(zip_path, URL, download=download)
        util.unzip(zip_path, path, remove=True)

    _process_bcdata(path, cell_radius=cell_radius, binary=binary)
    return os.path.join(path, "BCData")


def get_bcdata_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "validation", "test"],
    download: bool = False,
    cell_radius: int = 0,
    binary: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to BCData images and rasterized point annotations.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. Either "train", "validation" or "test".
        download: Whether to download the data if it is not present.
        cell_radius: Radius for rasterizing point annotations. The default keeps single-pixel labels.
        binary: Whether to merge positive and negative cells into a single foreground class.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the rasterized point annotation labels.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split choice. Choose from {SPLITS}.")

    data_root = get_bcdata_data(path, download=download, cell_radius=cell_radius, binary=binary)
    image_paths = natsorted(glob(os.path.join(data_root, "images", split, "*.png")))
    label_paths = [
        os.path.join(
            data_root, "labels", split, _label_subdir(cell_radius, binary),
            f"{os.path.splitext(os.path.basename(image_path))[0]}.tif"
        )
        for image_path in image_paths
    ]

    if not image_paths or not all(os.path.exists(label_path) for label_path in label_paths):
        raise RuntimeError("Could not find BCData images and labels for the requested settings.")

    return image_paths, label_paths


def get_bcdata_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"],
    download: bool = False,
    cell_radius: int = 0,
    binary: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the BCData dataset for Ki-67 cell detection/counting.

    The original BCData annotations are cell-center coordinates, not segmentation masks. They are rasterized into
    point/disk labels for compatibility with the segmentation dataset API. By default positive cells are labeled as
    1 and negative cells as 2. Pass `binary=True` to use a single foreground class.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use. Either "train", "validation" or "test".
        download: Whether to download the data if it is not present.
        cell_radius: Radius for rasterizing point annotations. The default keeps single-pixel labels.
        binary: Whether to merge positive and negative cells into a single foreground class.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_bcdata_paths(path, split, download, cell_radius, binary)

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
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_bcdata_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"],
    download: bool = False,
    cell_radius: int = 0,
    binary: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BCData dataloader for Ki-67 cell detection/counting.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split to use. Either "train", "validation" or "test".
        download: Whether to download the data if it is not present.
        cell_radius: Radius for rasterizing point annotations. The default keeps single-pixel labels.
        binary: Whether to merge positive and negative cells into a single foreground class.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bcdata_dataset(
        path, patch_shape, split, download=download, cell_radius=cell_radius,
        binary=binary, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
