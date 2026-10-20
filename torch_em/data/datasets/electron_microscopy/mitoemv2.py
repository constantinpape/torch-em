"""MitoEM v2 is a benchmark collection for 3D mitochondria instance segmentation in electron microscopy.

It contains eight challenging datasets with expert-verified labels, covering biologically complex
scenarios such as dense mitochondrial packing, hyperfused networks, thin-necked morphologies,
and ultrastructurally ambiguous boundaries.

The data is located at https://doi.org/10.5281/zenodo.17635006.
The dataset is from the publication https://doi.org/10.1101/2025.11.12.687478.
Please cite it if you use this dataset in your research.
"""

import os
import json
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


CHUNKS = (32, 256, 256)


def _read_split_info(dataset_dir):
    # The split file is called 'split.json' in most datasets, but 'splits.json' in the pyra dataset.
    for name in ("split.json", "splits.json"):
        split_path = os.path.join(dataset_dir, name)
        if os.path.exists(split_path):
            with open(split_path) as f:
                return json.load(f)[0]
    raise FileNotFoundError(f"Cannot find the split file in {dataset_dir}.")


def _write_nifti(f, key, nifti_path, transpose):
    """Write a NIfTI volume into the n5 file as (z, y, x).

    NIfTI stores (x, y, z) in Fortran order, so a slab along the last axis is contiguous on disk and the volume can
    be converted slab by slab, without loading it into memory as a whole.
    """
    import nibabel as nib

    nii = nib.load(nifti_path)
    dtype = nii.get_data_dtype()
    if not transpose:
        f.create_dataset(key, data=np.asarray(nii.dataobj).astype(dtype), chunks=CHUNKS, compression="gzip")
        return

    shape = nii.shape[::-1]
    ds = f.create_dataset(key, shape=shape, dtype=dtype, chunks=CHUNKS, compression="gzip")
    for z0 in range(0, shape[0], CHUNKS[0]):
        z1 = min(z0 + CHUNKS[0], shape[0])
        ds[z0:z1] = np.asarray(nii.dataobj[:, :, z0:z1]).astype(dtype).transpose(2, 1, 0)


def _preprocess_dataset(path, dataset_name, dataset_dir):
    """Preprocess a single dataset: convert NIfTI to n5."""
    import shutil
    import z5py

    n5_dir = os.path.join(path, "n5_data", dataset_name)
    os.makedirs(n5_dir, exist_ok=True)
    split_info = _read_split_info(dataset_dir)

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
                # The volume is written under a temporary name and moved in place once it is complete,
                # so that an interrupted conversion is not mistaken for a complete one.
                tmp_path = os.path.join(n5_dir, f"{sample}_incomplete.n5")
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)

                # The labels of this sample are stored in (z, y, x) order in the NIfTI file already.
                # NOTE: Even then they do not match the raw data (the mitochondria are drawn in empty resin).
                transpose_labels = sample != "me2-jurkat_train02"
                with z5py.File(tmp_path, "a") as f:
                    _write_nifti(f, "raw", img_nifti, transpose=True)
                    _write_nifti(f, "labels", lbl_nifti, transpose=transpose_labels)
                    if f["raw"].shape != f["labels"].shape:
                        raise RuntimeError(f"There is a shape mismatch between raw and labels for {sample}.")
                os.rename(tmp_path, n5_path)

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
    zip_path = os.path.join(path, f"{dataset_folder}.zip")
    dataset_dir = os.path.join(path, dataset_folder)

    # Without the NIfTI data there is nothing left to convert, so the converted data is used as it is.
    if not os.path.exists(dataset_dir) and len(glob(os.path.join(n5_dir, "*.n5"))) > 0:
        return n5_dir

    # Download if needed
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
        split_info = _read_split_info(dataset_dir)

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
