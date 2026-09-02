"""The MorphoSeg dataset contains bright-field microscopy images of NTERA-2 (NT2)
human preneuronal embryonic cells at day 11 and day 12 of all-trans-retinoic acid
differentiation, annotated for cell instance segmentation.

Images were acquired with a Leica DM IRB bright-field microscope (10x and 20x) and
a Google Pixel 4 mobile phone camera. The dataset has 36 annotated training images
and an unannotated test set.

Note: annotations are sparse - only a subset of the visible cells in each image
are labeled (~10% pixel coverage despite ~20% cell-like content).

The dataset is located at https://doi.org/10.15131/shef.data.25604421.
This dataset is from the following publication:
- Zhang et al. (2025): https://doi.org/10.1016/j.neucom.2025.130511
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import List, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://ndownloader.figshare.com/files/45654198",
    "test": "https://ndownloader.figshare.com/files/45654201",
    "rois": "https://ndownloader.figshare.com/files/45654207",
}
CHECKSUMS = {
    "train": None,
    "test": None,
    "rois": None,
}


def _rois_to_masks(data_dir: str) -> None:
    """Convert polygon ROI JSON files to per-image instance segmentation TIF masks."""
    import imageio.v3 as imageio
    from skimage.draw import polygon as draw_polygon

    roi_dir = os.path.join(data_dir, "roi_jsons_combined")
    mask_dir = os.path.join(data_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    img_dir = os.path.join(data_dir, "training_dataset")
    for json_path in natsorted(glob(os.path.join(roi_dir, "*_ROI.json"))):
        stem = os.path.basename(json_path).replace("_ROI.json", "")
        img_path = os.path.join(img_dir, stem + ".tif")
        if not os.path.exists(img_path):
            # Try .MP.tif variant.
            img_path = os.path.join(img_dir, stem + ".MP.tif")
            if not os.path.exists(img_path):
                continue

        img = imageio.imread(img_path)
        h, w = img.shape[:2]

        with open(json_path) as f:
            rois = json.load(f)

        mask = np.zeros((h, w), dtype=np.int32)
        for instance_id, roi in enumerate(rois, start=1):
            pts = np.array(roi["points"])  # [[x, y], ...]
            rr, cc = draw_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
            mask[rr, cc] = instance_id

        imageio.imwrite(os.path.join(mask_dir, stem + "_mask.tif"), mask)


def get_morphoseg_data(path: Union[os.PathLike, str], split: str, download: bool = False) -> str:
    """Download the MorphoSeg (NTERA-2) dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    assert split in ("train", "test"), f"'{split}' is not a valid split. Choose 'train' or 'test'."

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{split}_dataset.zip")
    util.download_source(zip_path, URLS[split], download, checksum=CHECKSUMS[split])
    util.unzip(zip_path, data_dir)

    if split == "train":
        roi_zip = os.path.join(path, "Training_ROIs_json.zip")
        util.download_source(roi_zip, URLS["rois"], download, checksum=CHECKSUMS["rois"])
        util.unzip(roi_zip, data_dir)
        _rois_to_masks(data_dir)

    return data_dir


def get_morphoseg_paths(
    path: Union[os.PathLike, str],
    split: str,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the MorphoSeg (NTERA-2) data.

    NOTE: Only the training split has segmentation masks (36 annotated images).
    The test split contains images without annotations.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split == "test":
        raise RuntimeError(
            "The MorphoSeg test split does not contain segmentation masks - only images are available."
        )

    data_dir = get_morphoseg_data(path, split, download)
    mask_dir = os.path.join(data_dir, "masks")

    if not os.path.isdir(mask_dir) or len(glob(os.path.join(mask_dir, "*_mask.tif"))) == 0:
        raise RuntimeError(
            f"No mask files found in {mask_dir}. Check the dataset structure after downloading."
        )

    label_paths = natsorted(glob(os.path.join(mask_dir, "*_mask.tif")))
    img_dir = os.path.join(data_dir, "training_dataset")

    raw_paths = []
    for lp in label_paths:
        stem = os.path.basename(lp).replace("_mask.tif", "")
        candidate = os.path.join(img_dir, stem + ".tif")
        if not os.path.exists(candidate):
            candidate = os.path.join(img_dir, stem + ".MP.tif")
        raw_paths.append(candidate)

    missing = [r for r in raw_paths if not os.path.exists(r)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} image file(s) not found for their masks. First missing: {missing[0]}"
        )

    return raw_paths, label_paths


def get_morphoseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str = "train",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MorphoSeg dataset for bright-field NTERA-2 cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_morphoseg_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_morphoseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: str = "train",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the MorphoSeg dataloader for bright-field NTERA-2 cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_morphoseg_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
