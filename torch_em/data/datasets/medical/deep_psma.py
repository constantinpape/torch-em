"""The DEEP-PSMA dataset contains annotations for total tumour burden segmentation in whole-body
PSMA and FDG PET/CT scans of the same patients.

The dataset consists of 100 patients with metastatic prostate cancer, each imaged with both a PSMA
PET/CT scan (Ga-68 PSMA-617 or F-18 DCFPyL) and an FDG PET/CT scan, acquired prior to staging for
Lu-177 PSMA therapy. This is a different dataset from `torch_em.data.datasets.medical.autopet`
(single-tracer FDG PET/CT) and `torch_em.data.datasets.medical.psma_pet_ct` (single-tracer PSMA
PET/CT): here every patient has a *paired* FDG and PSMA scan with independent total tumour burden
(TTB) annotations for each tracer, curated for the DEEP-PSMA challenge (MICCAI 2026).
Link: https://deep-psma.grand-challenge.org/

The data is distributed under the CC BY-NC 4.0 license and is hosted on Zenodo at
https://doi.org/10.5281/zenodo.15281784.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "0001-0020": "https://zenodo.org/records/15281784/files/0001-0020.zip",
    "0021-0040": "https://zenodo.org/records/15281784/files/0021-0040.zip",
    "0041-0060": "https://zenodo.org/records/15281784/files/0041-0060.zip",
    "0061-0080": "https://zenodo.org/records/15281784/files/0061-0080.zip",
    "0081-0100": "https://zenodo.org/records/15281784/files/0081-0100.zip",
}

CHECKSUMS = {
    "0001-0020": "ae8198db5e8fc975b192fcb955ac6aa8",
    "0021-0040": "66e985d54ca3b6d32c139c2e3ccb530a",
    "0041-0060": "2668fe23ac06e12455c49009502a9496",
    "0061-0080": "e95924233c1811d728983514aa4627f8",
    "0081-0100": "f77e122734e77b90ab41d0570a1e9b3f",
}


def get_deep_psma_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DEEP-PSMA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded and extracted dataset.
    """
    data_dir = os.path.join(path, "DeepPSMA")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    for shard, url in URLS.items():
        zip_path = os.path.join(path, f"{shard}.zip")
        util.download_source(path=zip_path, url=url, download=download, checksum=None)
        _verify_md5_checksum(zip_path, CHECKSUMS[shard])
        util.unzip(zip_path=zip_path, dst=data_dir, remove=True)

    return data_dir


def _verify_md5_checksum(path, checksum):
    import hashlib

    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024 * 1024), b""):
            hasher.update(chunk)

    this_checksum = hasher.hexdigest()
    if this_checksum != checksum:
        raise RuntimeError(
            f"The checksum of '{path}' does not match the expected checksum. "
            f"Expected: {checksum}, got: {this_checksum}"
        )


def get_deep_psma_paths(
    path: Union[os.PathLike, str],
    tracer: Literal["psma", "fdg"],
    modality: Literal["CT", "PET"] = "PET",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DEEP-PSMA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        tracer: The choice of PET tracer. Either 'psma' or 'fdg'.
        modality: The choice of imaging modality. Either 'CT' or 'PET'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if tracer not in ("psma", "fdg"):
        raise ValueError(f"'{tracer}' is not a valid tracer. Please choose one of ['psma', 'fdg'].")
    if modality not in ("CT", "PET"):
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of ['CT', 'PET'].")

    data_dir = get_deep_psma_data(path, download)

    tracer_dir = tracer.upper()
    raw_paths = natsorted(glob(os.path.join(data_dir, "*", tracer_dir, f"{modality}.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*", tracer_dir, "TTB.nii.gz")))

    assert len(raw_paths) > 0, f"Could not find any volumes in '{data_dir}'."
    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_deep_psma_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    tracer: Literal["psma", "fdg"],
    modality: Literal["CT", "PET"] = "PET",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DEEP-PSMA dataset for total tumour burden segmentation in whole-body PET/CT scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        tracer: The choice of PET tracer. Either 'psma' or 'fdg'.
        modality: The choice of imaging modality. Either 'CT' or 'PET'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_deep_psma_paths(path, tracer, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )


def get_deep_psma_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    tracer: Literal["psma", "fdg"],
    modality: Literal["CT", "PET"] = "PET",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DEEP-PSMA dataloader for total tumour burden segmentation in whole-body PET/CT scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        tracer: The choice of PET tracer. Either 'psma' or 'fdg'.
        modality: The choice of imaging modality. Either 'CT' or 'PET'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deep_psma_dataset(path, patch_shape, tracer, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
