"""The HeLaCytoNuc dataset contains fluorescence images of HeLa cells with instance annotations.

The red image channel shows the cytoplasm, the blue channel shows the nuclei and the green channel is unused.
The dataset is available at https://doi.org/10.14278/rodare.3001 under the CC BY 4.0 license.
Please cite the dataset record if you use this dataset in your research.
"""

import os
from glob import glob
from operator import itemgetter
from typing import List, Literal, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://rodare.hzdr.de/api/files/fae71336-c6d2-45b2-ae65-416c8f57b5a0"
URLS = {
    "train": f"{BASE_URL}/HeLaCytoNuc_train.zip",
    "val": f"{BASE_URL}/HeLaCytoNuc_validation.zip",
    "test": f"{BASE_URL}/HeLaCytoNuc_test.zip",
}
CHECKSUMS = {
    "train": "9241233246977df5b177f038257855773996ed4cdb5682f86d3a9d3f127970f3",
    "val": "37c4c0904db2146620d5f9b4df4c116768712dafead8c8008b38f249a034bc8b",
    "test": "9f50e9b5df435cafb95d3434571df5cf70e0af387f5cdeb06081fb521f5ded0e",
}
EXPECTED_SAMPLES = {"train": 1873, "val": 535, "test": 268}
ARCHIVE_NAMES = {"train": "train", "val": "validation", "test": "test"}
RAW_CHANNELS = {"cytoplasm": 0, "nuclei": 2}


def get_hela_cytonuc_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> str:
    """Download the HeLaCytoNuc data for one split.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. One of 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data for the requested split.
    """
    if split not in URLS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(URLS)}.")

    split_path = os.path.join(path, split)
    data_folders = [os.path.join(split_path, name) for name in ("images", "nuclei_masks", "cytoplasm_masks")]
    if all(os.path.exists(folder) for folder in data_folders):
        return split_path

    os.makedirs(split_path, exist_ok=True)
    archive_name = ARCHIVE_NAMES[split]
    zip_path = os.path.join(path, f"HeLaCytoNuc_{archive_name}.zip")
    util.download_source(zip_path, URLS[split], download, CHECKSUMS[split])
    util.unzip(zip_path, split_path)

    if not all(os.path.exists(folder) for folder in data_folders):
        raise RuntimeError(f"The downloaded archive for split '{split}' has an unexpected structure.")
    return split_path


def get_hela_cytonuc_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    label_choice: Literal["nuclei", "cytoplasm"] = "nuclei",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HeLaCytoNuc images and instance labels.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. One of 'train', 'val', or 'test'.
        label_choice: The instance annotations to load. Either 'nuclei' or 'cytoplasm'.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding label paths.
    """
    if label_choice not in ("nuclei", "cytoplasm"):
        raise ValueError("The label choice must be either 'nuclei' or 'cytoplasm'.")

    split_path = get_hela_cytonuc_data(path, split, download)
    image_paths = sorted(glob(os.path.join(split_path, "images", "*.tif")))
    expected_samples = EXPECTED_SAMPLES[split]
    if len(image_paths) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} images for split '{split}', but found {len(image_paths)}."
        )

    label_folder = os.path.join(split_path, f"{label_choice}_masks")
    label_paths = [os.path.join(label_folder, os.path.basename(image_path)) for image_path in image_paths]
    missing_labels = [label_path for label_path in label_paths if not os.path.exists(label_path)]
    if missing_labels:
        raise RuntimeError(f"Could not find labels for {len(missing_labels)} images in '{label_folder}'.")

    return image_paths, label_paths


def get_hela_cytonuc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    raw_channel: Literal["rgb", "nuclei", "cytoplasm"] = "rgb",
    label_choice: Literal["nuclei", "cytoplasm"] = "nuclei",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the HeLaCytoNuc dataset for nucleus or cytoplasm instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split. One of 'train', 'val', or 'test'.
        raw_channel: The image channels to load. Either 'rgb', 'nuclei', or 'cytoplasm'.
        label_choice: The instance annotations to load. Either 'nuclei' or 'cytoplasm'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if raw_channel not in ("rgb", "nuclei", "cytoplasm"):
        raise ValueError("The raw channel must be 'rgb', 'nuclei', or 'cytoplasm'.")

    image_paths, label_paths = get_hela_cytonuc_paths(path, split, label_choice, download)

    if raw_channel != "rgb":
        raw_transform = kwargs.pop("raw_transform", None)
        if raw_transform is None:
            raw_transform = torch_em.transform.get_raw_transform()
        kwargs["raw_transform"] = torch_em.transform.Compose(
            itemgetter(RAW_CHANNELS[raw_channel]), raw_transform, is_multi_tensor=False,
        )

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", False)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_hela_cytonuc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    raw_channel: Literal["rgb", "nuclei", "cytoplasm"] = "rgb",
    label_choice: Literal["nuclei", "cytoplasm"] = "nuclei",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the HeLaCytoNuc dataloader for nucleus or cytoplasm instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split. One of 'train', 'val', or 'test'.
        raw_channel: The image channels to load. Either 'rgb', 'nuclei', or 'cytoplasm'.
        label_choice: The instance annotations to load. Either 'nuclei' or 'cytoplasm'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hela_cytonuc_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        raw_channel=raw_channel,
        label_choice=label_choice,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
