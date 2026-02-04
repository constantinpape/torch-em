"""The YeastCellSeg dataset contains annotations for yeast cell segmentation
in 2D bright field microscopy images.

The dataset provides 15 images of 1024x1024 pixels with binary cell body annotations.
Instance segmentation labels are derived via connected components.

The dataset is from the publication https://doi.org/10.1109/ISBI.2014.6868107.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/records/344879/files"
_FILENAMES = [f"DS01_{i:02d}" for i in range(1, 16)]


def _create_h5_data(path, raw_dir, gt_dir):
    """Create h5 files with raw images, semantic masks and instance labels.

    Each h5 file contains:
        - 'raw': (H, W) uint8 grayscale bright field image.
        - 'labels/semantic': (H, W) uint8 binary mask (0=background, 1=cell).
        - 'labels/instances': (H, W) int64 connected component labels.
    """
    import h5py
    from skimage.measure import label

    h5_dir = os.path.join(path, "h5_data")
    os.makedirs(h5_dir, exist_ok=True)

    for name in _FILENAMES:
        h5_path = os.path.join(h5_dir, f"{name}.h5")
        if os.path.exists(h5_path):
            continue

        img = imageio.imread(os.path.join(raw_dir, f"{name}.tif"))
        gt = imageio.imread(os.path.join(gt_dir, f"{name}_gt.tif"))

        semantic = (gt > 0).astype("uint8")
        instances = label(semantic).astype("int64")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=img, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")

    return h5_dir


def get_yeastcellseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the YeastCellSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the h5 data.
    """
    h5_dir = os.path.join(path, "h5_data")
    if os.path.exists(h5_dir) and len(glob(os.path.join(h5_dir, "*.h5"))) == len(_FILENAMES):
        return h5_dir

    raw_dir = os.path.join(path, "raw")
    gt_dir = os.path.join(path, "gt")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for name in _FILENAMES:
        raw_path = os.path.join(raw_dir, f"{name}.tif")
        if not os.path.exists(raw_path):
            util.download_source(path=raw_path, url=f"{BASE_URL}/{name}.tif", download=download, checksum=None)

        gt_path = os.path.join(gt_dir, f"{name}_gt.tif")
        if not os.path.exists(gt_path):
            util.download_source(path=gt_path, url=f"{BASE_URL}/{name}_gt.tif", download=download, checksum=None)

    return _create_h5_data(path, raw_dir, gt_dir)


def get_yeastcellseg_paths(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> List[str]:
    """Get paths to the YeastCellSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    h5_dir = get_yeastcellseg_data(path, download)
    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) == len(_FILENAMES), f"Expected {len(_FILENAMES)} h5 files, found {len(h5_paths)}"
    return h5_paths


def get_yeastcellseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    segmentation_type: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the YeastCellSeg dataset for yeast cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation_type: The type of segmentation labels to use.
            One of 'instances' (connected component instance labels) or 'semantic' (binary cell mask).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert segmentation_type in ("instances", "semantic"), \
        f"'{segmentation_type}' is not valid. Choose from 'instances' or 'semantic'."

    h5_paths = get_yeastcellseg_paths(path, download)

    label_key = f"labels/{segmentation_type}"

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, label_dtype=np.int64,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_yeastcellseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    segmentation_type: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the YeastCellSeg dataloader for yeast cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        segmentation_type: The type of segmentation labels to use.
            One of 'instances' (connected component instance labels) or 'semantic' (binary cell mask).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_yeastcellseg_dataset(
        path=path,
        patch_shape=patch_shape,
        segmentation_type=segmentation_type,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
