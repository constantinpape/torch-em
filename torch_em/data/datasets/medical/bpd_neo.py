"""BPD-Neo is a dataset for segmentation of the lungs and trachea in neonatal MRI.

The dataset consists of 40 free-breathing 3D stack-of-stars radial gradient echo (StarVIBE) MRI
scans of neonates, most of whom are diagnosed with bronchopulmonary dysplasia (BPD), together with
semantic segmentation masks. All 40 scans have a lung mask, and 36 of them additionally have a
trachea mask (4 patients: BPD-Neo-01, BPD-Neo-03, BPD-Neo-11 and BPD-Neo-22 have no trachea mask).

This dataset is located at https://doi.org/10.5281/zenodo.15768091 (Zenodo, CC BY 4.0).
The dataset is from the publication https://doi.org/10.1038/s41597-026-07006-8.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/15768091/files/BPD-Neo-data.zip"
CHECKSUM = "cae6eb7ad6cbd3b1bf488fe88af9b17147f02bf2d754a304ec3b94d0ec1c75a9"


def get_bpd_neo_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BPD-Neo data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Nifti-data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "BPD-Neo-data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_bpd_neo_paths(
    path: Union[os.PathLike, str], structure: Literal['lung', 'trachea'] = "lung", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BPD-Neo data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        structure: The anatomical structure to segment. Either 'lung' or 'trachea'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bpd_neo_data(path, download)

    if structure not in ("lung", "trachea"):
        raise ValueError(f"'{structure}' is not a valid structure. Choose either 'lung' or 'trachea'.")

    patient_dirs = natsorted(glob(os.path.join(data_dir, "BPD-Neo-*")))

    raw_paths, label_paths = [], []
    for patient_dir in patient_dirs:
        raw_path = os.path.join(patient_dir, "image.nii.gz")
        label_path = os.path.join(patient_dir, f"{structure}_seg.nii.gz")
        if os.path.exists(raw_path) and os.path.exists(label_path):
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_bpd_neo_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    structure: Literal['lung', 'trachea'] = "lung",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BPD-Neo dataset for lung and trachea segmentation in neonatal MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        structure: The anatomical structure to segment. Either 'lung' or 'trachea'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bpd_neo_paths(path, structure, download)

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
        is_seg_dataset=True,
        **kwargs
    )


def get_bpd_neo_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    structure: Literal['lung', 'trachea'] = "lung",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BPD-Neo dataloader for lung and trachea segmentation in neonatal MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        structure: The anatomical structure to segment. Either 'lung' or 'trachea'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bpd_neo_dataset(path, patch_shape, structure, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
