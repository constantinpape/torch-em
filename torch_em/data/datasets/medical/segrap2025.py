"""The SegRap2025 dataset contains annotations for lymph node clinical target volume (LN CTV) segmentation
in head and neck CT scans of nasopharyngeal carcinoma patients.

It is the "Task02: LN CTV Segmentation" part of the SegRap2025 challenge
(https://hilab-git.github.io/SegRap2025_Challenge) and is unrelated to the organs-at-risk data of SegRap2023
covered in 'segrap.py': it comprises a different set of
440 CT scans from 262 patients, collected across 5 cohorts (4 centers), each with a pre-aligned pair of a
non-contrast and a contrast-enhanced CT scan. 120 patients from the internal cohort provide the official training
split (labelled), the remaining 4 testing cohorts (142 patients) are distributed with labels as part of this
public release too, and are exposed here as the 'test' split.

NOTE: The label legend is not documented in the released data. Verified on the data: the label volumes contain
the ids 0 to 6, i.e. background and 6 disjoint LN CTV sub-levels.

The dataset is a redistribution of the official SegRap2025 Task02 training data
(https://doi.org/10.6084/m9.figshare.26793622, CC BY 4.0) via Figshare. The archive is password protected, the
password ('lnctvseg@uestc') is published on the challenge dataset page.

This dataset is from the publication https://doi.org/10.48550/arXiv.2601.20575.
Please cite it if you use this dataset in your research.
"""

import os
import zipfile
from glob import glob
from shutil import which
from subprocess import run
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/48684664"
CHECKSUM = "995215f98844f1a5718a1b963fe5615dc3723d506a889503617fcd99aea052a1"

PASSWORD = "lnctvseg@uestc"

MODALITIES = {"ct": "0000", "ct_contrast": "0001"}

COHORTS = {"train": ["Internal_Cohort"], "test": [f"Testing_Cohort_{i}" for i in range(1, 5)]}

SPLIT_DIRS = {"train": "Tr", "test": "Ts"}


def _unzip_with_password(zip_path, dst, password):
    try:
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(dst, pwd=password.encode())
    except (NotImplementedError, RuntimeError) as e:
        # The python zipfile module does not support AES encryption, so we fall back to the 7z CLI.
        if which("7z") is None:
            raise RuntimeError(
                f"Could not extract '{zip_path}' with the zipfile module ({e}). Please install the '7z' CLI "
                "('conda install -c conda-forge p7zip') or extract the archive manually."
            )
        run(["7z", "x", f"-o{dst}", f"-p{password}", "-y", zip_path], check=True)


def get_segrap2025_lnctv_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SegRap2025 LN CTV dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "LNCTVSeg-DataSet")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "LNCTVSeg-DataSet.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    _unzip_with_password(zip_path, path, PASSWORD)
    os.remove(zip_path)

    return data_dir


def get_segrap2025_lnctv_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    modality: Literal["ct", "ct_contrast"] = "ct",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SegRap2025 LN CTV data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (the internal cohort) or 'test' (the 4 external cohorts).
        modality: The CT scan to use as input. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in COHORTS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {list(COHORTS.keys())}.")
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose one of {list(MODALITIES.keys())}.")

    data_dir = get_segrap2025_lnctv_data(path, download)
    suffix = MODALITIES[modality]
    image_dir, label_dir = f"images{SPLIT_DIRS[split]}", f"labels{SPLIT_DIRS[split]}"

    raw_paths, label_paths = [], []
    for cohort in COHORTS[split]:
        cur_raw_paths = natsorted(glob(os.path.join(data_dir, cohort, image_dir, f"*_{suffix}.nii.gz")))
        cur_label_paths = [
            os.path.join(data_dir, cohort, label_dir, os.path.basename(p).replace(f"_{suffix}.nii.gz", ".nii.gz"))
            for p in cur_raw_paths
        ]
        keep = [os.path.exists(p) for p in cur_label_paths]
        raw_paths.extend(p for p, k in zip(cur_raw_paths, keep) if k)
        label_paths.extend(p for p, k in zip(cur_label_paths, keep) if k)

    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_segrap2025_lnctv_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"] = "train",
    modality: Literal["ct", "ct_contrast"] = "ct",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegRap2025 LN CTV dataset for lymph node clinical target volume segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (the internal cohort) or 'test' (the 4 external cohorts).
        modality: The CT scan to use as input. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_segrap2025_lnctv_paths(path, split, modality, download)

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


def get_segrap2025_lnctv_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"] = "train",
    modality: Literal["ct", "ct_contrast"] = "ct",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegRap2025 LN CTV dataloader for lymph node clinical target volume segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (the internal cohort) or 'test' (the 4 external cohorts).
        modality: The CT scan to use as input. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_segrap2025_lnctv_dataset(path, patch_shape, split, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
