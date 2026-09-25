"""The NPC MRI dataset contains annotations for tumor segmentation in nasopharyngeal
carcinoma (NPC) MRI.

The dataset consists of 831 MRI scans (T1-weighted, T2-weighted and contrast-enhanced T1-weighted
axial series) of 277 untreated primary NPC patients, together with expert radiologist tumor
segmentations delineating the gross tumor volume. The scans are distributed as DICOM series and the
segmentations as binary NIfTI masks (one per patient and modality). The dataset also ships clinical
and laboratory metadata (TNM staging, EBV-DNA concentration, biopsy results, survival), which are
not exposed by this module.

The dataset is located at https://zenodo.org/records/13131827 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-025-05815-x.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/13131827/files/primary_data.zip"
CHECKSUM = "7ed81ca69c92a289ed0d6b8d1387bfe3db4ebe1717750cf94721f992f9ae6b4f"

MODALITIES = ["T1", "T2", "CE-T1"]


def _preprocess_npc_mri(data_root, preprocessed_dir, modality):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)

    patient_dirs = natsorted(glob(os.path.join(data_root, "*")))
    for patient_dir in tqdm(patient_dirs, desc=f"Preprocess NPC MRI ({modality})"):
        patient_id = os.path.basename(patient_dir)
        series_dir = os.path.join(patient_dir, f"{modality}WI")
        mask_path = os.path.join(patient_dir, f"ROI-{modality}.nii")
        if not os.path.isdir(series_dir) or not os.path.exists(mask_path):
            continue

        out_path = os.path.join(preprocessed_dir, f"{patient_id}.h5")
        if os.path.exists(out_path):
            continue

        volume, _ = util.load_dicom_series(series_dir)  # (z, y, x)
        labels = np.asarray(nib.load(mask_path).dataobj).T  # (x, y, z) -> (z, y, x)
        labels = (np.round(labels) > 0).astype("uint8")

        if labels.shape != volume.shape:
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume.astype("float32"), compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_npc_mri_data(
    path: Union[os.PathLike, str], modality: Literal["T1", "T2", "CE-T1"] = "CE-T1", download: bool = False
) -> str:
    """Download the NPC MRI dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of MRI modality. One of 'T1', 'T2' or 'CE-T1'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed (per-patient hdf5) data is stored.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose from {MODALITIES}.")

    preprocessed_dir = os.path.join(path, f"preprocessed_{modality}")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_root = os.path.join(path, "data", "MRI-Segments")
    if not os.path.exists(data_root):
        zip_path = os.path.join(path, "primary_data.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

        import zipfile
        with zipfile.ZipFile(zip_path) as f:
            members = [
                m for m in f.namelist() if m.startswith("data/MRI-Segments/") and "__MACOSX" not in m
            ]
            f.extractall(path, members=members)
        os.remove(zip_path)

    _preprocess_npc_mri(data_root, preprocessed_dir, modality)

    return preprocessed_dir


def get_npc_mri_paths(
    path: Union[os.PathLike, str], modality: Literal["T1", "T2", "CE-T1"] = "CE-T1", download: bool = False
) -> List[str]:
    """Get paths to the NPC MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of MRI modality. One of 'T1', 'T2' or 'CE-T1'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    preprocessed_dir = get_npc_mri_data(path, modality, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find the preprocessed data in '{preprocessed_dir}'."
    return volume_paths


def get_npc_mri_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["T1", "T2", "CE-T1"] = "CE-T1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NPC MRI dataset for nasopharyngeal carcinoma tumor segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of MRI modality. One of 'T1', 'T2' or 'CE-T1'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_npc_mri_paths(path, modality, download)

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


def get_npc_mri_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["T1", "T2", "CE-T1"] = "CE-T1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NPC MRI dataloader for nasopharyngeal carcinoma tumor segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of MRI modality. One of 'T1', 'T2' or 'CE-T1'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_npc_mri_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
