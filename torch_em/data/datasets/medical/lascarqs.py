"""The LAScarQS dataset contains annotations for left atrium and atrial scar segmentation
in late gadolinium enhanced (LGE) cardiac MRI of patients with atrial fibrillation.

The data was curated for the LAScarQS 2022 challenge (https://zmiclab.github.io/projects/lascarqs22/),
which was held together with MICCAI 2022. It comprises 194 LGE-MRI scans acquired at three clinical
centers (University of Utah, Beth Israel Deaconess Medical Center and King's College London) and is split
into two tasks, selected with the 'task' argument:
- 'task1': 60 labeled post-ablation scans with annotations of the left atrium cavity and the atrial scar.
- 'task2': 130 labeled pre- and post-ablation scans with annotations of the left atrium cavity only.
Each task also ships additional unlabeled test scans, which are not exposed by this module.

The label ids are described in `LABEL_IDS`: 0 = background, 1 = left atrium cavity, 2 = atrial scar
(only present for 'task1'). This is a different annotation target than the other cardiac MRI datasets
in this library (e.g. `torch_em.data.datasets.medical.atriaseg`), which do not provide scar annotations.

The LGE-MRI are stored as nifti volumes, which are converted to hdf5 volumes with the keys 'raw' and
'labels' by this module.

NOTE: The data is not available for automatic download. It requires registering with the organizers,
see `get_lascarqs_data` for the manual download steps. The data is distributed under the
CC BY-NC-ND license, see https://zmiclab.github.io/projects/lascarqs22/data.html for details.

This dataset is from the publication https://doi.org/10.1007/978-3-031-31778-1.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {"background": 0, "la_cavity": 1, "la_scar": 2}

TASKS = {"task1": 60, "task2": 130}


def get_lascarqs_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the LAScarQS 2022 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded and extracted dataset.
    """
    data_dir = os.path.join(path, "LAScarQS2022")
    if os.path.exists(data_dir):
        return data_dir

    if download:
        raise NotImplementedError(
            "The LAScarQS 2022 dataset cannot be downloaded automatically. See 'get_lascarqs_data' for details."
        )

    zip_path = os.path.join(path, "LAScarQS2022.zip")
    if not os.path.exists(zip_path):
        raise RuntimeError(
            f"Could not find the LAScarQS 2022 dataset at '{path}'. This dataset is not available for automatic "
            "download. To obtain it, please follow these steps:\n"
            "- Visit https://zmiclab.github.io/projects/lascarqs22/data.html and read the data usage agreement.\n"
            "- Register by sending the requested information to LAScarQS2022@outlook.com or LAScarQS2022@163.com.\n"
            "- Once you receive the download link from the organizers, download and extract the archive.\n"
            f"- Place the extracted 'LAScarQS2022' folder (with the 'task1' and 'task2' subfolders) at '{path}', "
            f"or place the downloaded zip archive at '{zip_path}'."
        )

    util.unzip(zip_path=zip_path, dst=path, remove=False)
    return data_dir


def _preprocess_inputs(data_dir, task, preprocessed_dir):
    import h5py
    import nibabel as nib

    case_dirs = natsorted(glob(os.path.join(data_dir, task, "train_data", "train_*")))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for case_dir in tqdm(case_dirs, desc=f"Preprocessing the LAScarQS '{task}' cases"):
        case_id = os.path.basename(case_dir)
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        raw = np.asarray(nib.load(os.path.join(case_dir, "enhanced.nii.gz")).dataobj).T
        cavity = np.asarray(nib.load(os.path.join(case_dir, "atriumSegImgMO.nii.gz")).dataobj).T

        labels = np.zeros(raw.shape, dtype="uint8")
        labels[cavity > 0] = LABEL_IDS["la_cavity"]

        if task == "task1":
            scar = np.asarray(nib.load(os.path.join(case_dir, "scarSegImgM.nii.gz")).dataobj).T
            labels[scar > 0] = LABEL_IDS["la_scar"]

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_lascarqs_paths(
    path: Union[os.PathLike, str], task: Literal["task1", "task2"], download: bool = False
) -> List[str]:
    """Get paths to the LAScarQS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of task. Either 'task1' (LA cavity and scar) or 'task2' (LA cavity only).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    if task not in TASKS:
        raise ValueError(f"'{task}' is not a valid task. Please choose one of {list(TASKS.keys())}.")

    data_dir = get_lascarqs_data(path, download)

    preprocessed_dir = os.path.join(path, "preprocessed", task)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) != TASKS[task]:
        _preprocess_inputs(data_dir, task, preprocessed_dir)

    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."

    return volume_paths


def get_lascarqs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    task: Literal["task1", "task2"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LAScarQS dataset for left atrium and atrial scar segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of task. Either 'task1' (LA cavity and scar) or 'task2' (LA cavity only).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_lascarqs_paths(path, task, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_lascarqs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    task: Literal["task1", "task2"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LAScarQS dataloader for left atrium and atrial scar segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The choice of task. Either 'task1' (LA cavity and scar) or 'task2' (LA cavity only).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lascarqs_dataset(path, patch_shape, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
