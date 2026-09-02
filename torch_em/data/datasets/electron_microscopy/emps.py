"""The EMPS dataset contains electron microscopy images of nanoparticles with
pixel-level instance segmentation annotations.

It contains 465 TEM/SEM images of nanoparticles sourced from scientific publications,
each paired with a 32-bit integer instance segmentation map where each unique value
identifies an individual particle (0 = background).

The dataset is available at https://github.com/by256/emps.
The dataset was published in https://doi.org/10.1021/acs.jcim.0c01455.
Please cite this publication if you use the dataset in your research.
"""

import os
from glob import glob
from shutil import rmtree
from typing import List, Literal, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://github.com/by256/emps/archive/refs/heads/main.zip"
CHECKSUM = None


def _create_h5_files(data_root, split, out_dir):
    """Convert PNG image/segmap pairs for the given split into HDF5 files."""
    import h5py
    import imageio.v3 as imageio

    split_csv = os.path.join(data_root, f"{split}.csv")
    with open(split_csv) as f:
        filenames = [line.strip() for line in f if line.strip()]

    # The CSV may or may not include the .png extension.
    filenames = [fn if fn.endswith(".png") else f"{fn}.png" for fn in filenames]

    os.makedirs(out_dir, exist_ok=True)

    for fname in filenames:
        img_path = os.path.join(data_root, "images", fname)
        seg_path = os.path.join(data_root, "segmaps", fname)

        assert os.path.exists(img_path), f"Image not found: {img_path}"
        assert os.path.exists(seg_path), f"Segmap not found: {seg_path}"

        raw = imageio.imread(img_path)
        if raw.ndim == 3:
            raw = raw[..., 0]

        labels = imageio.imread(seg_path)
        if labels.ndim == 3:
            labels = labels[..., 0]

        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{stem}.h5")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw.astype("uint8"), compression="gzip")
            f.create_dataset("labels", data=labels.astype("int32"), compression="gzip")


def get_emps_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    download: bool = False,
) -> str:
    """Download and preprocess the EMPS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split, either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the directory containing the HDF5 files for the given split.
    """
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"

    out_dir = os.path.join(path, split)
    if os.path.exists(out_dir) and len(glob(os.path.join(out_dir, "*.h5"))) > 0:
        return out_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "emps.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)

    extract_dir = os.path.join(path, "_extracted")
    util.unzip(zip_path, extract_dir, remove=True)

    # The zip extracts to a single root folder (e.g. "emps-main/").
    subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    data_root = os.path.join(extract_dir, subdirs[0]) if subdirs else extract_dir

    for s in ("train", "test"):
        _create_h5_files(data_root, s, os.path.join(path, s))

    rmtree(extract_dir)

    return out_dir


def get_emps_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    download: bool = False,
) -> List[str]:
    """Get paths to the EMPS HDF5 files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split, either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the HDF5 files.
    """
    data_dir = get_emps_data(path, split, download)
    paths = sorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(paths) > 0, f"No HDF5 files found in '{data_dir}'"
    return paths


def get_emps_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the EMPS dataset for nanoparticle instance segmentation in electron microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split, either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_emps_paths(path, split, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_emps_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for nanoparticle instance segmentation in the EMPS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split, either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_emps_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
