"""The CISD dataset contains 3,911 samples of touching or overlapping urothelial cells
from digital cytology, with manually annotated instance segmentation masks.

The data comes from 30 cytology slides prepared from healthy patient urine samples
and digitized with 21 focal planes. Two 2D image modes are supported:
- center_slice: Single best-focus 2D plane (JPG)
- edf: Extended Depth of Field — 21 planes merged into one focused 2D image (JPG)

NOTE: The raw dataset also provides a "stack" mode (all 21 focal planes per sample),
but it is not supported here because the annotations are always 2D instance masks.
A 3D stack with 2D-only labels cannot form a valid segmentation dataset.

Annotations are 2D instance masks stored in RLE format in CISD.json.
Cell categories: RED_BLOOD_CELL, NEUTROPHIL, SUPERFICIAL, UROTHELIAL.

The dataset is located at https://zenodo.org/records/5938893.
This dataset is from the publication https://doi.org/10.1109/ISBI52829.2022.9761495.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/5938893/files/CISD.zip"
CHECKSUM = None


def get_cisd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CISD dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "CISD")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "CISD.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    return data_dir


def _decode_rle(rle_counts, height, width):
    """Decode an uncompressed RLE mask (row-major order) to a 2D array."""
    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(rle_counts):
        if i % 2 == 1:
            flat[pos:pos + count] = 1
        pos += count
    return flat.reshape((height, width), order="C")


def _convert_annotations(data_dir: str, mode: str) -> str:
    """Convert CISD.json RLE masks to per-sample 2D TIFF label images.

    Reads image dimensions from the mask 'size' field — no raw images are loaded.
    Runs once; subsequent calls return the cached label directory immediately.

    Args:
        data_dir: The root CISD data directory (contains CISD.json).
        mode: One of "center_slice" or "edf".

    Returns:
        Path to the directory containing the generated label TIFFs.
    """
    import imageio.v3 as imageio
    from tqdm import tqdm

    label_dir = os.path.join(data_dir, f"{mode}_labels")
    if os.path.exists(label_dir) and len(glob(os.path.join(label_dir, "*.tif"))) > 0:
        return label_dir

    os.makedirs(label_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "CISD.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Annotation file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    assets = data["assets"]  # list of {"asset_id", "file_name", "annotations": [...]}

    for asset in tqdm(assets, desc=f"Converting CISD {mode} labels"):
        file_name = asset["file_name"]               # e.g. "0241_BB_01471.jpg"
        base_name = os.path.splitext(file_name)[0]   # e.g. "0241_BB_01471"
        anns = asset.get("annotations", [])

        # Get (H, W) from the first RLE size field — no image loading needed
        h, w = None, None
        for ann in anns:
            for item in ann.get("data", []):
                mask_info = item.get("mask", {})
                if "size" in mask_info:
                    h, w = mask_info["size"]
                    break
            if h is not None:
                break

        if h is None or w is None:
            continue

        label = np.zeros((h, w), dtype=np.int32)
        inst_id = 1
        for ann in anns:
            for item in ann.get("data", []):
                mask_info = item.get("mask", {})
                counts = mask_info.get("counts", [])
                size = mask_info.get("size", [h, w])
                if not counts:
                    continue
                mask = _decode_rle(counts, size[0], size[1])
                label[mask > 0] = inst_id
                inst_id += 1

        out_path = os.path.join(label_dir, f"{base_name}.tif")
        imageio.imwrite(out_path, label)

    return label_dir


def _convert_raw_to_grayscale(data_dir: str, mode: str) -> str:
    """Convert RGB JPG images to grayscale TIFFs so shapes match the 2D label masks.

    Runs once; subsequent calls return the cached directory immediately.

    Args:
        data_dir: The root CISD data directory.
        mode: One of "center_slice" or "edf".

    Returns:
        Path to the directory containing the grayscale TIFFs.
    """
    import imageio.v3 as imageio
    from tqdm import tqdm

    gray_dir = os.path.join(data_dir, f"{mode}_gray")
    if os.path.exists(gray_dir) and len(glob(os.path.join(gray_dir, "*.tif"))) > 0:
        return gray_dir

    os.makedirs(gray_dir, exist_ok=True)

    src_dir = os.path.join(data_dir, mode)
    for jpg_path in tqdm(natsorted(glob(os.path.join(src_dir, "*.jpg"))), desc=f"Converting CISD {mode} to grayscale"):
        img = imageio.imread(jpg_path)
        if img.ndim == 3:
            img = (img[..., :3] @ np.array([0.2989, 0.5870, 0.1140])).astype(np.uint8)
        stem = os.path.splitext(os.path.basename(jpg_path))[0]
        imageio.imwrite(os.path.join(gray_dir, f"{stem}.tif"), img)

    return gray_dir


def get_cisd_paths(
    path: Union[os.PathLike, str],
    mode: Literal["center_slice", "edf"] = "center_slice",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CISD data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        mode: The image format to use. One of "center_slice" (single best-focus 2D plane)
            or "edf" (Extended Depth of Field 2D composite).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if mode not in ("center_slice", "edf"):
        raise ValueError(f"Invalid mode '{mode}'. Choose 'center_slice' or 'edf'.")

    data_dir = get_cisd_data(path, download)

    img_dir = os.path.join(data_dir, mode)
    if not os.path.exists(img_dir):
        raise RuntimeError(
            f"Image directory for mode '{mode}' not found: {img_dir}. "
            "Expected modes: 'center_slice', 'edf'."
        )

    label_dir = _convert_annotations(data_dir, mode)
    raw_dir = _convert_raw_to_grayscale(data_dir, mode)
    raw_paths = natsorted(glob(os.path.join(raw_dir, "*.tif")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    if len(raw_paths) == 0:
        raise RuntimeError(f"No image files found in {img_dir}.")
    if len(label_paths) == 0:
        raise RuntimeError(f"No label files found in {label_dir}.")

    # Match by stem name
    raw_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in raw_paths}
    label_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}
    common = natsorted(set(raw_stems) & set(label_stems))

    raw_paths = [raw_stems[s] for s in common]
    label_paths = [label_stems[s] for s in common]

    return raw_paths, label_paths


def get_cisd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    mode: Literal["center_slice", "edf"] = "center_slice",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CISD dataset for urothelial cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        mode: The image format to use. One of "center_slice" or "edf".
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cisd_paths(path, mode, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_cisd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    mode: Literal["center_slice", "edf"] = "center_slice",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CISD dataloader for urothelial cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        mode: The image format to use. One of "center_slice" or "edf".
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cisd_dataset(path, patch_shape, mode, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
