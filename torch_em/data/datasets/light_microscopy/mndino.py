"""The mnDINO dataset contains annotated micronuclei for training deep learning segmentation models.

The dataset comprises 232 fluorescence microscopy images of HeLa, U2OS, and RPE1 cell lines
with 5,685 annotated micronuclei. Each image comes with two types of instance segmentation masks:
nuclei masks (main nucleus bodies) and micronuclei masks (small nuclear fragments).
Images were acquired on four different microscopy platforms.

The dataset is located at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2809.
This dataset is from the publication https://doi.org/10.7554/elife.101579.
Please cite it if you use this dataset for your research.
"""

import os
import tarfile
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

from natsort import natsorted
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD2809/annotated_mn_datasets.tar.gz"
CHECKSUM = None

# The archive folder is "validation" but we expose it as "val" to callers.
_SPLIT_DIR = {"train": "train", "val": "validation", "test": "test"}


def _preprocess_data(path: str) -> None:
    import h5py
    import imageio.v3 as imageio

    extracted_root = os.path.join(path, "annotated_mn_datasets")

    for split_key, split_dir in _SPLIT_DIR.items():
        h5_dir = os.path.join(path, "h5_data", split_key)
        os.makedirs(h5_dir, exist_ok=True)

        image_paths = natsorted(glob(os.path.join(extracted_root, split_dir, "images", "*.tif")))
        if not image_paths:
            continue

        for img_path in tqdm(image_paths, desc=f"Preprocessing '{split_key}'"):
            fname = os.path.splitext(os.path.basename(img_path))[0]
            h5_path = os.path.join(h5_dir, f"{fname}.h5")
            if os.path.exists(h5_path):
                continue

            nuclei_path = os.path.join(extracted_root, split_dir, "nuclei_masks", f"{fname}.tif")
            mn_path = os.path.join(extracted_root, split_dir, "mn_masks", f"{fname}.png")

            raw = imageio.imread(img_path)
            nuclei_labels = imageio.imread(nuclei_path) if os.path.exists(nuclei_path) else None
            mn_labels = imageio.imread(mn_path) if os.path.exists(mn_path) else None

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                if nuclei_labels is not None:
                    f.create_dataset("labels/nuclei", data=nuclei_labels.astype("int64"), compression="gzip")
                if mn_labels is not None:
                    f.create_dataset("labels/micronuclei", data=mn_labels.astype("int64"), compression="gzip")


def get_mndino_data(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> str:
    """Download the mnDINO dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the downloaded data.
    """
    path = str(path)
    os.makedirs(path, exist_ok=True)

    extracted_root = os.path.join(path, "annotated_mn_datasets")
    if not os.path.exists(extracted_root):
        tar_path = os.path.join(path, "annotated_mn_datasets.tar.gz")
        util.download_source(path=tar_path, url=URL, download=download, checksum=CHECKSUM)

        # The file is a plain tar archive despite the .tar.gz extension.
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path)
        os.remove(tar_path)

    h5_root = os.path.join(path, "h5_data")
    if not os.path.exists(h5_root):
        _preprocess_data(path)

    return path


def get_mndino_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    download: bool = False,
) -> List[str]:
    """Get paths to the mnDINO HDF5 files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. One of 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the HDF5 files for the given split.
    """
    if split not in _SPLIT_DIR:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(_SPLIT_DIR)}.")

    get_mndino_data(path, download)

    h5_dir = os.path.join(path, "h5_data", split)
    if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
        _preprocess_data(str(path))

    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) > 0, f"No data found for split '{split}' at '{h5_dir}'."
    return h5_paths


def get_mndino_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Literal["nuclei", "micronuclei"] = "micronuclei",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the mnDINO dataset for nucleus / micronucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (height, width) to use for training.
        split: The data split. One of 'train', 'val', or 'test'.
        label_choice: Which segmentation target to use. Either 'nuclei' (main nucleus
            instance masks) or 'micronuclei' (micronucleus instance masks).
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_choice == "nuclei":
        label_key = "labels/nuclei"
    elif label_choice == "micronuclei":
        label_key = "labels/micronuclei"
    else:
        raise ValueError(f"'{label_choice}' is not a valid label_choice. Choose 'nuclei' or 'micronuclei'.")

    h5_paths = get_mndino_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=binary, boundaries=boundaries, offsets=offsets,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs,
    )


def get_mndino_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Literal["nuclei", "micronuclei"] = "micronuclei",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for the mnDINO dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (height, width) to use for training.
        split: The data split. One of 'train', 'val', or 'test'.
        label_choice: Which segmentation target to use. Either 'nuclei' (main nucleus
            instance masks) or 'micronuclei' (micronucleus instance masks).
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mndino_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        label_choice=label_choice,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
