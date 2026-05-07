"""The SynthMT dataset contains synthetic interference reflection microscopy (IRM) images
of microtubules with instance segmentation annotations.

The dataset provides 6,600 synthetically generated 512x512 RGB images with per-instance
binary masks for microtubule segmentation. It was designed to train foundation models
(e.g. SAM) for automated in vitro microtubule analysis.

The dataset is located at https://huggingface.co/datasets/HTW-KI-Werkstatt/SynthMT.
This dataset is from the publication https://doi.org/10.64898/2026.01.09.698597.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/HTW-KI-Werkstatt/SynthMT/resolve/main/data/{FILENAME}"
NUM_PARQUET_FILES = 8


def _download_parquets(path, download):
    """Download all parquet files for the dataset."""
    parquet_dir = os.path.join(path, "parquets")
    os.makedirs(parquet_dir, exist_ok=True)

    for i in range(NUM_PARQUET_FILES):
        fname = f"train-{i:05d}-of-{NUM_PARQUET_FILES:05d}.parquet"
        fpath = os.path.join(parquet_dir, fname)
        if not os.path.exists(fpath):
            url = URL.format(FILENAME=fname)
            util.download_source(path=fpath, url=url, download=download, checksum=None)

    return parquet_dir


def _create_images_from_parquets(path):
    """Extract images and instance labels from parquet files and save as TIF."""
    import imageio.v3 as imageio
    import pandas as pd
    from io import BytesIO
    from PIL import Image
    from tqdm import tqdm

    image_dir = os.path.join(path, "images")
    label_dir = os.path.join(path, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    parquet_dir = os.path.join(path, "parquets")
    parquet_files = sorted(glob(os.path.join(parquet_dir, "*.parquet")))

    for pfile in tqdm(parquet_files, desc="Processing parquet files"):
        df = pd.read_parquet(pfile)
        for _, row in df.iterrows():
            sample_id = row["id"]
            img_path = os.path.join(image_dir, f"{sample_id}.tif")
            lbl_path = os.path.join(label_dir, f"{sample_id}.tif")

            if os.path.exists(img_path) and os.path.exists(lbl_path):
                continue

            # Decode the image.
            img = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
            img_arr = np.array(img)

            # Decode instance masks and merge into a single label map.
            masks = row["mask"]
            instances = np.zeros(img_arr.shape[:2], dtype="uint32")
            for i, mask_entry in enumerate(masks, start=1):
                mask = np.array(Image.open(BytesIO(mask_entry["bytes"])).convert("L"))
                instances[mask > 0] = i

            imageio.imwrite(img_path, img_arr, compression="zlib")
            imageio.imwrite(lbl_path, instances, compression="zlib")


def get_synthmt_data(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> str:
    """Download the SynthMT dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    _download_parquets(path, download)

    image_dir = os.path.join(path, "images")
    label_dir = os.path.join(path, "labels")
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        _create_images_from_parquets(path)

    return path


def get_synthmt_paths(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SynthMT data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    from natsort import natsorted

    get_synthmt_data(path, download)

    image_paths = natsorted(glob(os.path.join(path, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(path, "labels", "*.tif")))

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_synthmt_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SynthMT dataset for microtubule instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_synthmt_paths(path, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_synthmt_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the SynthMT dataloader for microtubule instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_synthmt_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
