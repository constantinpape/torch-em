"""MitoEM v2 is a benchmark collection for 3D mitochondria instance segmentation in electron microscopy.

It contains eight challenging datasets with expert-verified labels, covering biologically complex
scenarios such as dense mitochondrial packing, hyperfused networks, thin-necked morphologies,
and ultrastructurally ambiguous boundaries.

The dataset is from the publication https://doi.org/10.5281/zenodo.17635006.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Literal, Optional, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/records/17635006/files"

DATASETS = {
    "beta": "Dataset001_ME2-Beta",
    "jurkat": "Dataset002_ME2-Jurkat",
    "macro": "Dataset003_ME2-Macro",
    "mossy": "Dataset004_ME2-Mossy",
    "podo": "Dataset005_ME2-Podo",
    "pyra": "Dataset006_ME2-Pyra",
    "sperm": "Dataset007_ME2-Sperm",
    "stem": "Dataset008_ME2-Stem",
}

DATASET_NAMES = list(DATASETS.keys())


def _convert_nifti_to_n5(nifti_path, n5_path):
    """Convert NIfTI file to n5 format for efficient access."""
    import nibabel as nib
    import z5py

    if os.path.exists(n5_path):
        return

    nii = nib.load(nifti_path)
    data = np.asarray(nii.dataobj)

    # NIfTI stores as (X, Y, Z), we want (Z, Y, X)
    data = np.transpose(data, (2, 1, 0))

    chunks = (32, 256, 256)
    with z5py.File(n5_path, "a") as f:
        f.create_dataset("data", data=data, chunks=chunks, compression="gzip")


def _preprocess_dataset(path, dataset_name, dataset_dir):
    """Preprocess a single dataset: convert NIfTI to n5."""
    import json

    n5_dir = os.path.join(path, "n5_data", dataset_name)
    os.makedirs(n5_dir, exist_ok=True)

    # Read split info
    with open(os.path.join(dataset_dir, "split.json")) as f:
        split_info = json.load(f)[0]

    processed = {}
    for split_name, split_key in [("train", "train"), ("val", "val"), ("test", "test")]:
        samples = split_info.get(split_key, [])
        if not samples:
            continue

        for sample in samples:
            # Determine source directories based on split
            if split_name == "test":
                img_dir = "imagesTs"
                lbl_dir = "labelsTs"
            else:
                img_dir = "imagesTr"
                lbl_dir = "labelsTr"

            img_nifti = os.path.join(dataset_dir, img_dir, f"{sample}_0000.nii.gz")
            lbl_nifti = os.path.join(dataset_dir, lbl_dir, f"{sample}.nii.gz")

            if not os.path.exists(img_nifti) or not os.path.exists(lbl_nifti):
                continue

            n5_path = os.path.join(n5_dir, f"{sample}.n5")

            if not os.path.exists(n5_path):
                print(f"Converting {sample} to n5...")
                _convert_nifti_to_n5(img_nifti, os.path.join(n5_dir, f"{sample}_raw.n5"))
                _convert_nifti_to_n5(lbl_nifti, os.path.join(n5_dir, f"{sample}_labels.n5"))

                # Combine into single n5 file
                import z5py
                with z5py.File(os.path.join(n5_dir, f"{sample}_raw.n5"), "r") as f_raw:
                    raw = f_raw["data"][:]
                with z5py.File(os.path.join(n5_dir, f"{sample}_labels.n5"), "r") as f_lbl:
                    labels = f_lbl["data"][:]

                with z5py.File(n5_path, "a") as f:
                    f.create_dataset("raw", data=raw, chunks=(32, 256, 256), compression="gzip")
                    f.create_dataset("labels", data=labels.astype("uint64"), chunks=(32, 256, 256), compression="gzip")

                # Clean up temp files
                import shutil
                shutil.rmtree(os.path.join(n5_dir, f"{sample}_raw.n5"))
                shutil.rmtree(os.path.join(n5_dir, f"{sample}_labels.n5"))

            if split_name not in processed:
                processed[split_name] = []
            processed[split_name].append(n5_path)

    return processed


def get_mitoemv2_data(
    path: Union[os.PathLike, str],
    dataset: str,
    download: bool = False,
) -> str:
    """Download and preprocess a MitoEM v2 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset: The dataset to download. One of 'beta', 'jurkat', 'macro', 'mossy',
            'podo', 'pyra', 'sperm', or 'stem'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed n5 data directory.
    """
    assert dataset in DATASETS, f"'{dataset}' is not valid. Choose from {DATASET_NAMES}."

    dataset_folder = DATASETS[dataset]
    n5_dir = os.path.join(path, "n5_data", dataset)

    # Check if already preprocessed
    if os.path.exists(n5_dir) and len(glob(os.path.join(n5_dir, "*.n5"))) > 0:
        return n5_dir

    # Download if needed
    zip_path = os.path.join(path, f"{dataset_folder}.zip")
    dataset_dir = os.path.join(path, dataset_folder)

    if not os.path.exists(dataset_dir):
        os.makedirs(path, exist_ok=True)
        url = f"{BASE_URL}/{dataset_folder}.zip"
        util.download_source(path=zip_path, url=url, download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=path)

    # Preprocess
    _preprocess_dataset(path, dataset, dataset_dir)

    return n5_dir


def get_mitoemv2_paths(
    path: Union[os.PathLike, str],
    dataset: Optional[Union[str, List[str]]] = None,
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the MitoEM v2 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset: The dataset(s) to use. One of 'beta', 'jurkat', 'macro', 'mossy',
            'podo', 'pyra', 'sperm', or 'stem'. Can also be a list of dataset names.
            If None, all datasets will be used.
        split: The data split to use. One of 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the n5 data.
    """
    import json
    from natsort import natsorted

    assert split in ("train", "val", "test"), f"'{split}' is not a valid split."

    if dataset is None:
        dataset = DATASET_NAMES
    elif isinstance(dataset, str):
        dataset = [dataset]

    all_n5_paths = []
    for ds in dataset:
        n5_dir = get_mitoemv2_data(path, ds, download)

        # Read split info to get correct samples
        dataset_folder = DATASETS[ds]
        dataset_dir = os.path.join(path, dataset_folder)
        with open(os.path.join(dataset_dir, "split.json")) as f:
            split_info = json.load(f)[0]

        samples = split_info.get(split, [])
        n5_paths = [os.path.join(n5_dir, f"{sample}.n5") for sample in samples]
        n5_paths = [p for p in n5_paths if os.path.exists(p)]
        all_n5_paths.extend(n5_paths)

    assert len(all_n5_paths) > 0, f"No data found for {dataset}/{split}"

    return natsorted(all_n5_paths)


def get_mitoemv2_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    dataset: Optional[Union[str, List[str]]] = None,
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> Dataset:
    """Get the MitoEM v2 dataset for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        dataset: The dataset(s) to use. One of 'beta', 'jurkat', 'macro', 'mossy',
            'podo', 'pyra', 'sperm', or 'stem'. Can also be a list of dataset names.
            If None, all datasets will be used.
        split: The data split to use. One of 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    n5_paths = get_mitoemv2_paths(path, dataset, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=n5_paths,
        raw_key="raw",
        label_paths=n5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_mitoemv2_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    dataset: Optional[Union[str, List[str]]] = None,
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MitoEM v2 dataloader for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        dataset: The dataset(s) to use. One of 'beta', 'jurkat', 'macro', 'mossy',
            'podo', 'pyra', 'sperm', or 'stem'. Can also be a list of dataset names.
            If None, all datasets will be used.
        split: The data split to use. One of 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset_obj = get_mitoemv2_dataset(
        path=path,
        patch_shape=patch_shape,
        dataset=dataset,
        split=split,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset_obj, batch_size=batch_size, **loader_kwargs)
