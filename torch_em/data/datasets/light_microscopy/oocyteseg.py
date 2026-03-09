"""The OocyteSeg dataset contains annotations for binary membrane segmentation
in transmitted light microscopy images of oocytes from multiple species.

NOTE: The dataset only has semantic (binary) segmentation.

The dataset is from the publication https://doi.org/10.1242/jcs.260281.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Literal, Optional, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/6502830/files/SegmentationCortex.tar.gz"
CHECKSUM = "1da5d4fd102d8e903744db424f6114c6"

SPECIES = ["mouse", "human", "sea_urchin"]

_SUBDIRS = {
    "mouse": {
        "train": ["exp1", "exp2"],
        "test": ["exp1_test", "exp2_test"],
    },
    "human": {
        "train": ["clin1", "clin2"],
        "test": ["clin1_test", "clin2_test"],
    },
    "sea_urchin": {
        "train": ["train"],
        "test": ["test"],
    },
}


def _preprocess_data(data_dir, processed_dir, species, split):
    """Preprocess images and masks to ensure consistent format.

    Some sea urchin images are stored as RGB instead of grayscale.
    Masks are stored as 0/255 and need to be normalized to 0/1.
    This function converts all data to a consistent single-channel uint8 format.
    """
    img_out_dir = os.path.join(processed_dir, "images")
    mask_out_dir = os.path.join(processed_dir, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    subdirs = _SUBDIRS[species][split]

    for subdir in subdirs:
        input_dir = os.path.join(data_dir, species, subdir, "input")
        mask_dir = os.path.join(data_dir, species, subdir, "mask")

        input_names = {os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".png")}
        mask_names = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith(".png")}
        matched = sorted(input_names & mask_names)

        for name in matched:
            img_out = os.path.join(img_out_dir, f"{subdir}_{name}.tif")
            mask_out = os.path.join(mask_out_dir, f"{subdir}_{name}.tif")

            if os.path.exists(img_out) and os.path.exists(mask_out):
                continue

            img = imageio.imread(os.path.join(input_dir, f"{name}.png"))
            if img.ndim == 3:
                img = np.mean(img[..., :3], axis=-1).astype("uint8")
            imageio.imwrite(img_out, img, compression="zlib")

            mask = imageio.imread(os.path.join(mask_dir, f"{name}.png"))
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).astype("uint8")
            imageio.imwrite(mask_out, mask, compression="zlib")


def get_oocyteseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OocyteSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "SegmentationCortex")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    tar_path = os.path.join(path, "SegmentationCortex.tar.gz")
    util.download_source(path=tar_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=tar_path, dst=path)

    return data_dir


def get_oocyteseg_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    species: Optional[str] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OocyteSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train' or 'test'.
        species: The species to select. One of 'mouse', 'human' or 'sea_urchin'.
            If None, data from all species is returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in ("train", "test"), f"'{split}' is not a valid split. Choose from 'train' or 'test'."

    if species is None:
        species_list = SPECIES
    else:
        assert species in SPECIES, f"'{species}' is not a valid species. Choose from {SPECIES}."
        species_list = [species]

    data_dir = get_oocyteseg_data(path, download)

    all_image_paths = []
    all_seg_paths = []

    from natsort import natsorted

    for sp in species_list:
        processed_dir = os.path.join(path, "processed", sp, split)
        img_out_dir = os.path.join(processed_dir, "images")
        mask_out_dir = os.path.join(processed_dir, "masks")

        if not os.path.exists(img_out_dir) or len(glob(os.path.join(img_out_dir, "*.tif"))) == 0:
            _preprocess_data(data_dir, processed_dir, sp, split)

        image_paths = natsorted(glob(os.path.join(img_out_dir, "*.tif")))
        seg_paths = natsorted(glob(os.path.join(mask_out_dir, "*.tif")))

        assert len(image_paths) == len(seg_paths), \
            f"Mismatch: {len(image_paths)} images vs {len(seg_paths)} masks for {sp}/{split}"
        assert len(image_paths) > 0, f"No images found for {sp}/{split}"

        all_image_paths.extend(image_paths)
        all_seg_paths.extend(seg_paths)

    return all_image_paths, all_seg_paths


def get_oocyteseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    species: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OocyteSeg dataset for binary membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train' or 'test'.
        species: The species to select. One of 'mouse', 'human' or 'sea_urchin'.
            If None, data from all species is returned.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, seg_paths = get_oocyteseg_paths(path, split, species, download)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=seg_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs
    )


def get_oocyteseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    species: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OocyteSeg dataloader for binary membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train' or 'test'.
        species: The species to select. One of 'mouse', 'human' or 'sea_urchin'.
            If None, data from all species is returned.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oocyteseg_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        species=species,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
