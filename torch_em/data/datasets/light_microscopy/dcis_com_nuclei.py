"""The DCIS.COM nuclei dataset contains SiR-DNA fluorescence images with instance annotations.

The dataset contains images of DCIS.COM cells acquired with spinning disk confocal microscopy.
It is located at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD895.
The original dataset is available at https://doi.org/10.5281/zenodo.3715492 under the CC BY 4.0 license.
Please cite the dataset record if you use this dataset in your research.
"""

import os
from glob import glob
from shutil import rmtree
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://zenodo.org/records/3715492/files/Stardist_v2.zip?download=1"
CHECKSUM = "aec767afae76942b7c97e31c500284f8b5862150d8e81b57f513e66d7258c05e"

SPLIT_FOLDERS = {
    "train": ("Training - Images", "Training - Masks"),
    "test": ("Test - Images", "Test - Masks"),
}
EXPECTED_SAMPLES = {"train": 45, "test": 2}
INVALID_TRAIN_IMAGES = {
    "cell migration R1 - Position 0_XY1562686096_Z0_T00_C1-1-image7.tif",
    "cell migration R1 - Position 0_XY1562686096_Z0_T00_C1-1-image14.tif",
}


def get_dcis_com_nuclei_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DCIS.COM nuclei dataset (S-BIAD895).

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    data_dir = os.path.join(path, "Stardist")
    expected_folders = [os.path.join(data_dir, folder) for folders in SPLIT_FOLDERS.values() for folder in folders]
    if all(os.path.exists(folder) for folder in expected_folders):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "Stardist_v2.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path)

    macos_dir = os.path.join(path, "__MACOSX")
    if os.path.exists(macos_dir):
        rmtree(macos_dir)

    if not all(os.path.exists(folder) for folder in expected_folders):
        raise RuntimeError("The downloaded S-BIAD895 archive has an unexpected structure.")
    return data_dir


def get_dcis_com_nuclei_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DCIS.COM nuclei images and instance labels.

    Two training images are excluded because their source masks are exact copies of the mask for
    ``image1.tif`` and do not align with the corresponding images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding label paths.
    """
    if split not in SPLIT_FOLDERS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLIT_FOLDERS)}.")

    data_dir = get_dcis_com_nuclei_data(path, download)
    image_folder, label_folder = SPLIT_FOLDERS[split]
    image_paths = sorted(glob(os.path.join(data_dir, image_folder, "*.tif")))

    expected_samples = EXPECTED_SAMPLES[split]
    if len(image_paths) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} images for split '{split}', but found {len(image_paths)}."
        )

    if split == "train":
        image_paths = [
            image_path for image_path in image_paths if os.path.basename(image_path) not in INVALID_TRAIN_IMAGES
        ]

    label_folder = os.path.join(data_dir, label_folder)
    label_paths = [os.path.join(label_folder, os.path.basename(image_path)) for image_path in image_paths]
    missing_labels = [label_path for label_path in label_paths if not os.path.exists(label_path)]
    if missing_labels:
        raise RuntimeError(f"Could not find labels for {len(missing_labels)} images in '{label_folder}'.")

    return image_paths, label_paths


def get_dcis_com_nuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the DCIS.COM nuclei dataset for instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_dcis_com_nuclei_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
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


def get_dcis_com_nuclei_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DCIS.COM nuclei dataloader for instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dcis_com_nuclei_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
