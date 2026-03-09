"""The BCCD dataset contains annotations for blood cell segmentation
in microscopy images of blood smears.

The dataset provides 1,328 images with corresponding segmentation masks.
Instance segmentation labels are derived via connected components from the semantic masks.

The dataset is located at https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask
(https://doi.org/10.34740/kaggle/dsv/6107556)
Please cite it (the respective doi above) if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _create_h5_data(path, split):
    """Create h5 files with raw images, semantic masks and instance labels."""
    import h5py
    from skimage.measure import label
    from tqdm import tqdm

    data_dir = os.path.join(path, "data", "BCCD Dataset with mask")
    h5_dir = os.path.join(path, "h5_data", split)
    os.makedirs(h5_dir, exist_ok=True)

    raw_dir = os.path.join(data_dir, split, "original")
    mask_dir = os.path.join(data_dir, split, "mask")

    raw_paths = sorted(glob(os.path.join(raw_dir, "*.png")))

    for raw_path in tqdm(raw_paths, desc=f"Creating h5 files for {split}"):
        fname = os.path.basename(raw_path)
        h5_path = os.path.join(h5_dir, fname.replace(".png", ".h5"))

        if os.path.exists(h5_path):
            continue

        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            continue

        raw = imageio.imread(raw_path)
        mask = imageio.imread(mask_path)

        # Convert mask to binary semantic segmentation
        if mask.ndim == 3:
            mask = mask[..., 0]  # Take first channel if RGB
        semantic = (mask > 0).astype("uint8")

        # Create instance labels via connected components
        instances = label(semantic).astype("int64")

        with h5py.File(h5_path, "w") as f:
            # Store raw as (C, H, W) if RGB
            if raw.ndim == 3:
                raw = raw.transpose(2, 0, 1)
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")

    return h5_dir


def get_bccd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BCCD dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    data_dir = os.path.join(path, "data", r"BCCD Dataset with mask")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="jeetblahiri/bccd-dataset-with-mask", download=download)
    util.unzip(zip_path=os.path.join(path, "bccd-dataset-with-mask.zip"), dst=os.path.join(path, "data"))

    return data_dir


def get_bccd_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the BCCD data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    assert split in ("train", "test"), f"'{split}' is not a valid split."

    get_bccd_data(path, download)

    h5_dir = os.path.join(path, "h5_data", split)
    if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
        _create_h5_data(path, split)

    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) > 0, f"No data found for split '{split}'"

    return h5_paths


def get_bccd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    segmentation_type: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BCCD dataset for blood cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train' or 'test'.
        segmentation_type: The type of segmentation labels to use.
            One of 'instances' (connected component instance labels) or 'semantic' (binary cell mask).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert segmentation_type in ("instances", "semantic"), \
        f"'{segmentation_type}' is not valid. Choose from 'instances' or 'semantic'."

    h5_paths = get_bccd_paths(path, split, download)

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
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_bccd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    segmentation_type: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BCCD dataloader for blood cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train' or 'test'.
        segmentation_type: The type of segmentation labels to use.
            One of 'instances' (connected component instance labels) or 'semantic' (binary cell mask).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bccd_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        segmentation_type=segmentation_type,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
