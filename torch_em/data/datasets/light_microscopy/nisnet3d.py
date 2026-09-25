"""The NISNet3D dataset contains 3D fluorescence microscopy images of nuclei
with manually annotated instance segmentation, curated for training and
evaluating 3D nuclear instance segmentation methods.

The dataset contains annotated subvolumes from eight microscopy volumes:
- BABB-cleared_kidney_1: BABB-cleared rat kidney (confocal, DAPI)
- Cleared_mouse_intestine_1: Cleared mouse intestine (confocal, Hoechst)
- Diabetic_Biopsy_Human_Spectral_1: Diabetic biopsy human spectral (5 subvolumes)
- Diabetic_Biopsy_Human_Spectral_3: Diabetic biopsy human spectral (6 subvolumes)
- Kidney_Cortex_Human_Spectral_1: Kidney cortex human spectral (6 subvolumes)
- Kidney_Human_Nephrectomy_1: Kidney human nephrectomy (4 subvolumes)
- Rat_liver_1: Shallow rat liver (confocal, Hoechst) - entire volume annotated
- Scale-cleared_rat_kidney_1: Scale-cleared rat kidney (confocal, DAPI)

NOTE: The original paper also includes a V5 volume (zebrafish brain EM) sourced
from the NucMM dataset, which is already available in torch-em under
`torch_em.data.datasets.electron_microscopy.nuc_mm`. It is therefore excluded
here to avoid duplication.

The dataset is located at https://zenodo.org/records/7065147.
This dataset is from the publication https://doi.org/10.1038/s41598-023-36243-9.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7065147/files/ground_truth_and_synthetic.zip"
CHECKSUM = "02f8ad4a6e489283548ea4f0c2c39ac975531c09b58e4d6f498b4e49ac73f0d3"

VOLUMES = [
    "BABB-cleared_kidney_1",
    "Cleared_mouse_intestine_1",
    "Diabetic_Biopsy_Human_Spectral_1",
    "Diabetic_Biopsy_Human_Spectral_3",
    "Kidney_Cortex_Human_Spectral_1",
    "Kidney_Human_Nephrectomy_1",
    "Rat_liver_1",
    "Scale-cleared_rat_kidney_1",
]


def get_nisnet3d_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NISNet3D dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "ground_truth_and_synthetic")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "ground_truth_and_synthetic.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    return data_dir


def get_nisnet3d_paths(
    path: Union[os.PathLike, str],
    volumes: Optional[List[Literal[
        "BABB-cleared_kidney_1",
        "Cleared_mouse_intestine_1",
        "Diabetic_Biopsy_Human_Spectral_1",
        "Diabetic_Biopsy_Human_Spectral_3",
        "Kidney_Cortex_Human_Spectral_1",
        "Kidney_Human_Nephrectomy_1",
        "Rat_liver_1",
        "Scale-cleared_rat_kidney_1",
    ]]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the NISNet3D data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volumes: The volume(s) to use. Defaults to all 8 volumes.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if volumes is None:
        volumes = VOLUMES
    else:
        invalid = [v for v in volumes if v not in VOLUMES]
        if invalid:
            raise ValueError(f"Invalid volumes: {invalid}. Valid choices are {VOLUMES}.")

    data_dir = get_nisnet3d_data(path, download)

    raw_paths, label_paths = [], []
    for vol in volumes:
        vol_dir = os.path.join(data_dir, vol)
        if not os.path.exists(vol_dir):
            raise RuntimeError(
                f"Volume directory not found: {vol_dir}. "
                "Please check the dataset structure after downloading."
            )
        # Each subvolume folder contains {name}.tif (raw) and {name}_gt.tif (label).
        # The 'synthetic' subfolder is skipped.
        for sub_dir in natsorted(glob(os.path.join(vol_dir, "*"))):
            if not os.path.isdir(sub_dir) or os.path.basename(sub_dir) == "synthetic":
                continue
            sub_name = os.path.basename(sub_dir)
            raw_file = os.path.join(sub_dir, f"{sub_name}.tif")
            label_file = os.path.join(sub_dir, f"{sub_name}_gt.tif")
            if os.path.exists(raw_file) and os.path.exists(label_file):
                raw_paths.append(raw_file)
                label_paths.append(label_file)

    if len(raw_paths) == 0:
        raise RuntimeError(
            f"No image files found under {data_dir}. "
            "Please check the dataset structure."
        )

    return raw_paths, label_paths


def get_nisnet3d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    volumes: Optional[List[str]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the NISNet3D dataset for 3D nuclear instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        volumes: The volume(s) to use. Defaults to all 8 volumes.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_nisnet3d_paths(path, volumes, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_nisnet3d_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    volumes: Optional[List[str]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the NISNet3D dataloader for 3D nuclear instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        volumes: The volume(s) to use. Defaults to all 8 volumes.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nisnet3d_dataset(path, patch_shape, volumes, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
