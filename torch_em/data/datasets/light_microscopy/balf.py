"""The BALF dataset contains annotations for cell instance segmentation
in bronchoalveolar lavage fluid microscopy images.

The dataset is located at https://zenodo.org/records/14871206.
The dataset is from the publication https://doi.org/10.1038/s41597-025-05452-4.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Literal, Tuple, Optional, List

import numpy as np
import imageio.v3 as imageio

from skimage.draw import polygon as draw_polygon

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://zenodo.org/records/14871206/files/Images.rar",
    "labels": "https://zenodo.org/records/14871206/files/Labels.rar",
}
CHECKSUMS = {
    "images": None,
    "labels": None,
}

CELL_TYPES = [
    "erythrocyte",
    "ciliated_columnar_epithelial",
    "squamous_epithelial",
    "macrophage",
    "lymphocyte",
    "neutrophil",
    "eosinophil",
]

SPLITS = ["train", "val"]


def _create_data_from_yolo(image_dir, label_dir, data_dir):
    """Convert YOLO polygon annotations to HDF5 files with image, instance and semantic masks.

    Each HDF5 file contains:
        - 'raw': RGB image in (C, H, W) format.
        - 'labels/instances': Instance segmentation mask with unique IDs per cell.
        - 'labels/semantic': Semantic segmentation mask with the following class mapping:
            0: background
            1: erythrocyte
            2: ciliated columnar epithelial
            3: squamous epithelial
            4: macrophage
            5: lymphocyte
            6: neutrophil
            7: eosinophil
    """
    import h5py

    os.makedirs(data_dir, exist_ok=True)

    label_paths = natsorted(glob(os.path.join(label_dir, "*.txt")))
    assert len(label_paths) > 0, f"No label files found in {label_dir}"

    data_paths = []
    for label_path in tqdm(label_paths, desc="Creating BALF data"):
        stem = os.path.splitext(os.path.basename(label_path))[0]

        image_path = os.path.join(image_dir, f"{stem}.jpg")
        assert os.path.exists(image_path), f"Image not found: {image_path}"

        data_path = os.path.join(data_dir, f"{stem}.h5")
        data_paths.append(data_path)

        if os.path.exists(data_path):
            continue

        image = imageio.imread(image_path)
        h, w = image.shape[:2]

        with open(label_path) as f:
            lines = f.readlines()

        # Parse YOLO polygon annotations and compute areas for sorting.
        polygons = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            xs = [coords[i] * w for i in range(0, len(coords), 2)]
            ys = [coords[i] * h for i in range(1, len(coords), 2)]
            rr, cc = draw_polygon(ys, xs, shape=(h, w))
            area = len(rr)
            polygons.append((rr, cc, area, class_id))

        # Sort by area (largest first so smaller objects are not occluded).
        sorting = np.argsort([p[2] for p in polygons])[::-1]

        instances = np.zeros((h, w), dtype="uint16")
        semantic = np.zeros((h, w), dtype="uint16")
        for seg_id, idx in enumerate(sorting, 1):
            rr, cc, _, class_id = polygons[idx]
            instances[rr, cc] = seg_id
            semantic[rr, cc] = class_id + 1  # 0 = background, 1-7 = cell types

        # Store image as channels-first (C, H, W).
        raw = image.transpose(2, 0, 1)

        with h5py.File(data_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")

    return natsorted(data_paths)


def get_balf_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BALF dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path where the data is stored.
    """
    for key in URLS:
        fname = URLS[key].rsplit("/", 1)[-1]
        dirname = os.path.splitext(fname)[0].lower()

        if os.path.exists(os.path.join(path, dirname)):
            continue

        os.makedirs(path, exist_ok=True)
        rar_path = os.path.join(path, fname)
        util.download_source(path=rar_path, url=URLS[key], download=download, checksum=CHECKSUMS[key])
        util.unzip_rarfile(rar_path=rar_path, dst=path)

    return path


def get_balf_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the BALF data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    assert split in SPLITS, f"'{split}' is not a valid split. Choose from {SPLITS}."

    get_balf_data(path, download)

    image_dir = os.path.join(path, "images", split)
    label_dir = os.path.join(path, "labels", split)
    data_dir = os.path.join(path, "data", split)

    if not os.path.exists(data_dir) or len(glob(os.path.join(data_dir, "*.h5"))) == 0:
        data_paths = _create_data_from_yolo(image_dir, label_dir, data_dir)
    else:
        data_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))

    assert len(data_paths) > 0
    return data_paths


def get_balf_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"] = "train",
    segmentation_type: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BALF dataset for cell segmentation in bronchoalveolar lavage fluid microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        segmentation_type: The segmentation target. Either 'instances' or 'semantic'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_balf_paths(path, split, download)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key=f"labels/{segmentation_type}",
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_balf_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"] = "train",
    segmentation_type: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BALF dataloader for cell segmentation in bronchoalveolar lavage fluid microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        segmentation_type: The segmentation target. Either 'instances' or 'semantic'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_balf_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        segmentation_type=segmentation_type,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
