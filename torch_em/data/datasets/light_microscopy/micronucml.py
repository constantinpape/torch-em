"""The MicroNucML dataset contains annotations for micronucleus segmentation
in live-cell fluorescence microscopy images of MCF10A and RPE-1 cells.

The dataset is located at https://data.mendeley.com/datasets/hrjn4dy6z9/1.
This dataset is from the publication https://doi.org/10.1016/j.crmeth.2026.101573.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "h2b_gfp": "https://data.mendeley.com/public-files/datasets/hrjn4dy6z9/files/622ca122-7e2f-4203-81d9-ce01ad7bf590/file_downloaded",  # noqa
    "h2b_mcherry": "https://data.mendeley.com/public-files/datasets/hrjn4dy6z9/files/2b8206d4-aa79-4879-b638-895d4ededf3c/file_downloaded",  # noqa
}
CHECKSUMS = {
    "h2b_gfp": "13028b103f40c70fe8868f9c85d84bc2e3fa333dc01418914653993cad5122bd",
    "h2b_mcherry": "ca89f6b9fa8d9912da721e8dda6397e6675c6e353e5480f32b6658a18c36c1ad",
}

# The archives are named after the fluorescent reporter, "H2B-GFP" and "H2B-mCherry" in the data record.
# 'train' and 'test' are the official split of the H2B-GFP images, the mCherry images are evaluation sets
# that capture a different colour and image quality.
SPLITS = {
    "train": ("h2b_gfp", "H2B-GFP", "train_image", "train_mask"),
    "test": ("h2b_gfp", "H2B-GFP", "test_image", "test_mask"),
    "mcherry_red": ("h2b_mcherry", "H2B-mCherry", "red_image", "red_mask"),
    "mcherry_grey": ("h2b_mcherry", "H2B-mCherry", "grey_image", "grey_mask"),
}


def _preprocess_labels(data_dir, mask_dir):
    import numpy as np
    import imageio.v3 as imageio

    label_dir = os.path.join(data_dir, f"{mask_dir}_instances")
    mask_paths = natsorted(glob(os.path.join(data_dir, mask_dir, "*.npy")))
    if os.path.exists(label_dir) and len(glob(os.path.join(label_dir, "*.tif"))) == len(mask_paths):
        return label_dir

    os.makedirs(label_dir, exist_ok=True)
    for mask_path in tqdm(mask_paths, desc=f"Preprocessing '{mask_dir}'"):
        masks = np.load(mask_path)
        # The masks are stored as one binary plane per micronucleus, in some files with an extra singleton axis.
        masks = masks.reshape(masks.shape[0], *masks.shape[-2:])

        labels = np.zeros(masks.shape[-2:], dtype="uint16")
        for instance_id, mask in enumerate(masks, start=1):
            labels[mask > 0] = instance_id

        fname = os.path.splitext(os.path.basename(mask_path))[0]
        imageio.imwrite(os.path.join(label_dir, f"{fname}.tif"), labels, compression="zlib")

    return label_dir


def get_micronucml_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test", "mcherry_red", "mcherry_grey"],
    download: bool = False,
) -> str:
    """Download the MicroNucML dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split choice.")

    source, dname = SPLITS[split][:2]
    data_dir = os.path.join(path, dname)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{source}.zip")
    util.download_source(path=zip_path, url=URLS[source], download=download, checksum=CHECKSUMS[source])
    util.unzip(zip_path=zip_path, dst=path)
    shutil.rmtree(os.path.join(path, "__MACOSX"), ignore_errors=True)

    return data_dir


def get_micronucml_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test", "mcherry_red", "mcherry_grey"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the MicroNucML data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_micronucml_data(path, split, download)
    image_dir, mask_dir = SPLITS[split][2:]
    label_dir = _preprocess_labels(data_dir, mask_dir)

    raw_paths = natsorted(glob(os.path.join(data_dir, image_dir, "*.png")))
    label_paths = [
        os.path.join(label_dir, f"{os.path.splitext(os.path.basename(p))[0]}.tif") for p in raw_paths
    ]

    assert raw_paths and len(raw_paths) == len(label_paths)
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_micronucml_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test", "mcherry_red", "mcherry_grey"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MicroNucML dataset for micronucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_micronucml_paths(path, split, download)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        with_channels=split != "mcherry_grey",
        ndim=2,
        **kwargs
    )


def get_micronucml_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test", "mcherry_red", "mcherry_grey"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MicroNucML dataloader for micronucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_micronucml_dataset(path, patch_shape, split, offsets, boundaries, binary, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
