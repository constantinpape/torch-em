"""The Ar-PlaqSegm1 dataset contains annotations for atherosclerotic plaque segmentation
in B-mode vascular ultrasound.

The dataset consists of 541 pairs of B-mode ultrasound images (800 x 800 pixels) acquired in
Cordoba, Argentina, each paired with a binary segmentation mask of the atherosclerotic plaque.
201 pairs contain no visible plaque (an empty mask) and 340 pairs show one or more plaques. The
ground truth was manually delineated by two experienced physicians and released as a monochrome
raw image and its corresponding binary mask (foreground: plaque).

The dataset is located at https://data.mendeley.com/datasets/8srkpz52dy/1 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-026-06952-7.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


MANIFEST_URL = "https://data.mendeley.com/public-api/datasets/8srkpz52dy?folder_id=&dataset_version=1"

N_IMAGES = 541


def _get_manifest(path, download):
    manifest_path = os.path.join(path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    if not download:
        raise RuntimeError(f"Cannot find the data at '{path}', but download was set to False.")

    import requests

    response = requests.get(MANIFEST_URL)
    response.raise_for_status()
    manifest = response.json()["files"]

    os.makedirs(path, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest


def get_plaque_us_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Ar-PlaqSegm1 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the images and masks are stored.
    """
    data_dir = os.path.join(path, "images")
    os.makedirs(data_dir, exist_ok=True)

    manifest = _get_manifest(path, download)
    for entry in tqdm(manifest, desc="Downloading Ar-PlaqSegm1"):
        fpath = os.path.join(data_dir, entry["filename"])
        content = entry["content_details"]
        util.download_source(
            path=fpath, url=content["download_url"], download=download, checksum=content["sha256_hash"]
        )

    return data_dir


def _preprocess_labels(label_paths, data_dir):
    # Most masks are single-channel, but a subset are stored as an RGB image with the same binary
    # mask duplicated across all three channels, which `default_segmentation_dataset` cannot use
    # directly, so all masks are normalized to a single-channel (0, 1) label map.
    neu_dir = os.path.join(data_dir, "preprocessed_masks")
    os.makedirs(neu_dir, exist_ok=True)

    neu_label_paths = []
    for label_path in label_paths:
        neu_path = os.path.join(neu_dir, os.path.basename(label_path))
        if not os.path.exists(neu_path):
            mask = imageio.imread(label_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            imageio.imwrite(neu_path, (mask > 0).astype("uint8"))
        neu_label_paths.append(neu_path)

    return neu_label_paths


def get_plaque_us_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the Ar-PlaqSegm1 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_plaque_us_data(path, download)

    all_paths = natsorted(glob(os.path.join(data_dir, "*.png")))
    image_paths = [p for p in all_paths if not p.endswith("_labeled.png")]
    # One image in the original release is named with a stray trailing space before the extension
    # ("544 .png"), while its mask is not ("544_labeled.png"), so the image id is stripped of
    # whitespace before deriving the mask filename.
    label_paths = [
        os.path.join(os.path.dirname(p), f"{os.path.basename(p)[:-len('.png')].strip()}_labeled.png")
        for p in image_paths
    ]

    assert len(image_paths) == N_IMAGES, f"Expected {N_IMAGES} images, found {len(image_paths)}."
    assert all(os.path.exists(p) for p in label_paths)

    label_paths = _preprocess_labels(label_paths, os.path.dirname(data_dir))

    return image_paths, label_paths


def get_plaque_us_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Ar-PlaqSegm1 dataset for atherosclerotic plaque segmentation in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_plaque_us_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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


def get_plaque_us_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Ar-PlaqSegm1 dataloader for atherosclerotic plaque segmentation in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_plaque_us_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
