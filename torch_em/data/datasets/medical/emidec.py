"""The EMIDEC dataset contains annotations for the left ventricle and for myocardial infarction
in delayed-enhancement (late gadolinium enhanced) cardiac MRI.

The data was curated for the EMIDEC challenge (https://emidec.com), which was held together with the STACOM
workshop at MICCAI 2020. The public release of the segmentation contest consists of 100 training cases,
33 normal cases (case ids with the letter 'N') and 67 pathological cases (case ids with the letter 'P').
The 50 cases of the test set are distributed without ground truth and are therefore not exposed here,
so that this module provides 100 annotated volumes out of the 150 exams of the database.

The label ids are described in `LABEL_IDS`: 0 = background, 1 = cavity, 2 = normal myocardium,
3 = myocardial infarction, 4 = no-reflow (permanent microvascular obstruction). The whole myocardium is
the union of the ids 2, 3 and 4.

The DE-MRI are stored as nifti volumes with the slice axis last, so they are converted to hdf5 volumes
with the slice axis first (the keys are 'raw' and 'labels') by this module.

The data is located at https://emidec.com/dataset and is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.3390/data5040089.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://emidec.com/dataset/download"
CHECKSUM = "9270495a800d717092a6c21feba4b4f20fe90c61be85ac2a0cfc35570e321528"

LABEL_IDS = {"background": 0, "cavity": 1, "myocardium": 2, "infarction": 3, "no_reflow": 4}

PATHOLOGIES = {"normal": "N", "pathological": "P"}


def _preprocess_inputs(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    case_dirs = [p for p in natsorted(glob(os.path.join(data_dir, "Case_*"))) if os.path.isdir(p)]
    os.makedirs(preprocessed_dir, exist_ok=True)

    for case_dir in tqdm(case_dirs, desc="Preprocessing the EMIDEC cases"):
        case_id = os.path.basename(case_dir)
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        raw = np.asarray(nib.load(os.path.join(case_dir, "Images", f"{case_id}.nii.gz")).dataobj).T
        labels = np.asarray(nib.load(os.path.join(case_dir, "Contours", f"{case_id}.nii.gz")).dataobj).T

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_emidec_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the EMIDEC dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == 100:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "emidec-dataset-1.0.1")
    if not os.path.exists(data_dir):
        zip_path = os.path.join(path, "emidec-dataset-1.0.1.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_inputs(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_emidec_paths(
    path: Union[os.PathLike, str],
    pathology: Optional[Literal["normal", "pathological"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the EMIDEC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        pathology: The choice of cases. Either 'normal' or 'pathological'. If None, all cases are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_emidec_data(path, download)

    if pathology is not None and pathology not in PATHOLOGIES:
        raise ValueError(f"'{pathology}' is not a valid pathology. Please choose one of {list(PATHOLOGIES.keys())}.")

    prefix = "*" if pathology is None else PATHOLOGIES[pathology]
    volume_paths = natsorted(glob(os.path.join(data_dir, f"Case_{prefix}*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."

    return volume_paths


def get_emidec_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    pathology: Optional[Literal["normal", "pathological"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EMIDEC dataset for myocardial infarction segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        pathology: The choice of cases. Either 'normal' or 'pathological'. If None, all cases are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_emidec_paths(path, pathology, download)

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


def get_emidec_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    pathology: Optional[Literal["normal", "pathological"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EMIDEC dataloader for myocardial infarction segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        pathology: The choice of cases. Either 'normal' or 'pathological'. If None, all cases are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_emidec_dataset(path, patch_shape, pathology, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
