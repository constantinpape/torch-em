"""The MCT-LTDiag dataset contains annotations for liver tumor segmentation in multi-phase
contrast-enhanced CT.

The dataset consists of 517 patients (one tar archive per patient) with four-phase CT acquisitions
(non-contrast 'nc', arterial 'art', portal-venous 'pvp' and delayed 'delay', stored as
'NIFTI/<phase>.nii.gz'). Manual tumor and whole-liver segmentation masks ('mask_pvp.nii.gz' and
'liver_mask_pvp.nii.gz') are only available for the portal-venous phase, which this loader pairs
with 'NIFTI/pvp.nii.gz'. Each patient is additionally labeled with one of 5 tumor types (e.g. 'BCLM'
for breast-cancer liver metastasis) in 'meta_info_patient.tab', which is not used by this loader but
kept alongside the downloaded data for reference.

The data is located at https://doi.org/10.7910/DVN/S3RW15, released under a CC0-1.0 license, and is
downloadable anonymously via the Harvard Dataverse access API (no personal API token required for
these unrestricted files, despite the dataset-level metadata suggesting otherwise).

This dataset is from the publication https://doi.org/10.1148/radiol.232214.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DATAVERSE_API_URL = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/S3RW15"
DATAVERSE_DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

TARGETS = {"liver": "liver_mask_pvp.nii.gz", "tumor": "mask_pvp.nii.gz"}


def _list_patient_files():
    import requests

    # The default python-requests user agent is blocked by the Dataverse API, unlike a browser-like one.
    response = requests.get(DATAVERSE_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]
    return {
        f["label"][:-len(".tar")]: f["dataFile"]["id"] for f in files if f["label"].endswith(".tar")
    }


def get_mct_ltdiag_data(
    path: Union[os.PathLike, str], n_patients: Optional[int] = None, download: bool = False
) -> str:
    """Download the MCT-LTDiag dataset.

    NOTE: The full collection is about 180 GB. Use `n_patients` to only download a subset for a quick start.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        n_patients: The number of patients to download, sorted by patient id. By default all 517 are downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    import tarfile

    patient_dir = os.path.join(path, "patients")
    patient_files = _list_patient_files()
    patient_ids = sorted(patient_files)
    if n_patients is not None:
        patient_ids = patient_ids[:n_patients]
    missing = [
        patient_id for patient_id in patient_ids
        if not os.path.exists(os.path.join(patient_dir, patient_id, "NIFTI", "pvp.nii.gz"))
    ]
    if not missing:
        return patient_dir

    os.makedirs(patient_dir, exist_ok=True)
    for patient_id in missing:
        tar_path = os.path.join(path, f"{patient_id}.tar")
        url = DATAVERSE_DOWNLOAD_URL.format(file_id=patient_files[patient_id])
        util.download_source(path=tar_path, url=url, download=download)

        out_dir = os.path.join(patient_dir, patient_id)
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=out_dir)
        os.remove(tar_path)

    return patient_dir


def get_mct_ltdiag_paths(
    path: Union[os.PathLike, str],
    target: Literal["liver", "tumor"] = "tumor",
    n_patients: Optional[int] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the MCT-LTDiag data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        target: The choice of segmentation target. Either 'liver' or 'tumor'.
        n_patients: The number of patients to use, sorted by patient id. By default all 517 are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if target not in TARGETS:
        raise ValueError(f"'{target}' is not a valid target. Choose one of {list(TARGETS)}.")

    patient_dir = get_mct_ltdiag_data(path, n_patients, download)

    raw_paths = natsorted(glob(os.path.join(patient_dir, "*", "NIFTI", "pvp.nii.gz")))
    label_paths = [os.path.join(os.path.dirname(os.path.dirname(p)), TARGETS[target]) for p in raw_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_mct_ltdiag_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    target: Literal["liver", "tumor"] = "tumor",
    n_patients: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MCT-LTDiag dataset for liver and tumor segmentation in portal-venous phase CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        target: The choice of segmentation target. Either 'liver' or 'tumor'.
        n_patients: The number of patients to use, sorted by patient id. By default all 517 are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mct_ltdiag_paths(path, target, n_patients, download)

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
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_mct_ltdiag_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    target: Literal["liver", "tumor"] = "tumor",
    n_patients: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MCT-LTDiag dataloader for liver and tumor segmentation in portal-venous phase CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        target: The choice of segmentation target. Either 'liver' or 'tumor'.
        n_patients: The number of patients to use, sorted by patient id. By default all 517 are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mct_ltdiag_dataset(path, patch_shape, target, n_patients, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
