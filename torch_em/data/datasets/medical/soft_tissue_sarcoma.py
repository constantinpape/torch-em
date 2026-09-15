"""The Soft-tissue-Sarcoma dataset contains annotations for tumor segmentation in MRI and FDG-PET/CT
of patients with soft-tissue sarcomas of the extremities.

It consists of 51 patients with a T1-weighted MRI, a T2-weighted fat-suppressed MRI (T2FS, or STIR if T2FS
was not available), a CT and a FDG-PET scan each. The tumor was manually delineated on the T2FS scan by a
radiation oncologist and the contours were propagated to the other scans by rigid registration. The contours
are distributed as DICOM RTSTRUCT and rasterized onto the image grid by this module
(see `torch_em.data.datasets.util.rasterize_rtstruct`). Images and labels are stored in hdf5 files.
The semantic label ids are: 1: tumor ('GTV_Mass'), 2: peritumoral edema ('GTV_Edema' outside of the tumor,
annotated for 32 of the 51 patients).

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/soft-tissue-sarcoma/.

This dataset is from the publication https://doi.org/10.1088/0031-9155/60/14/5471.
The data was released at https://doi.org/10.7937/K9/TCIA.2015.7GO2GSKS.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-zgVcrK7I.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

# The ROI names of the contours. 'GTV_Edema' incorporates the tumor, so the edema id is only assigned outside
# of the tumor (`rasterize_rtstruct` gives precedence to the lower label id). One patient (STS_030) uses the
# names 'GTV_Research' and 'GTV_Res+edema' for the same two structures.
LABEL_IDS = {"GTV_Mass": 1, "GTV_Research": 1, "GTV_Edema": 2, "GTV_Res+edema": 2}

# The RTSTRUCT series descriptions per modality, lower-cased. The T2FS category consists of the T2-weighted
# fat-saturated scans (26 patients) and the STIR scans used where they were not available (25 patients).
# The dataset additionally contains the T1 and T2FS scans registered and resampled to the PET scan
# ('RTstructAlignedT1toPET' etc.), which are not used here.
MODALITIES = {
    "T1": ["rtstructt1"],
    "T2FS": ["rtstructt2fs", "rtstructstir"],
    "CT": ["rtstructct"],
    "PET": ["rtstructpet"],
}


def _get_referenced_series(rtstruct_path):
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    return str(
        rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0].SeriesInstanceUID
    )


def _preprocess_soft_tissue_sarcoma(dicom_dir, csv_path, preprocessed_dir, modality):
    import h5py

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    # The series descriptions are matched case-insensitively, as the collection is not consistent
    # (e.g. 'RTstructCT' and 'RTStructCT').
    rtstruct_series = {
        row["Series UID"]: row["Subject ID"] for row in rows
        if row["Modality"] == "RTSTRUCT" and row["Series Description"].lower() in MODALITIES[modality]
    }

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, subject_id in tqdm(sorted(rtstruct_series.items()), desc=f"Preprocess STS {modality}"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        rtstruct_path = glob(os.path.join(dicom_dir, series_uid, "*.dcm"))[0]
        image_dir = os.path.join(dicom_dir, _get_referenced_series(rtstruct_path))
        volume, geometry = util.load_dicom_series(image_dir)
        if modality == "CT":
            volume = np.round(volume).astype("int16")
        labels = util.rasterize_rtstruct(rtstruct_path, geometry, volume.shape, LABEL_IDS)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_soft_tissue_sarcoma_data(
    path: Union[os.PathLike, str], modality: Literal["T1", "T2FS", "CT", "PET"], download: bool = False
) -> str:
    """Download the Soft-tissue-Sarcoma dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The imaging modality. One of 'T1', 'T2FS', 'CT' or 'PET'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    assert modality in MODALITIES, f"'{modality}' is not a valid modality. Choose one of {list(MODALITIES)}."
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed", modality)
    os.makedirs(path, exist_ok=True)

    # Download the DICOM series (MR, CT, PT and RTSTRUCT) from the TCIA manifest. The series metadata are written
    # after all series are downloaded, so their presence means the download is complete.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "soft_tissue_sarcoma_series")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, os.path.basename(URL)), url=URL, dst=dicom_dir, csv_filename=csv_path,
            download=download,
        )

    _preprocess_soft_tissue_sarcoma(dicom_dir, f"{csv_path}.csv", preprocessed_dir, modality)
    return preprocessed_dir


def get_soft_tissue_sarcoma_paths(
    path: Union[os.PathLike, str], modality: Literal["T1", "T2FS", "CT", "PET"], download: bool = False
) -> List[str]:
    """Get paths to the Soft-tissue-Sarcoma data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The imaging modality. One of 'T1', 'T2FS', 'CT' or 'PET'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_soft_tissue_sarcoma_data(path, modality, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_soft_tissue_sarcoma_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["T1", "T2FS", "CT", "PET"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Soft-tissue-Sarcoma dataset for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality. One of 'T1', 'T2FS', 'CT' or 'PET'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_soft_tissue_sarcoma_paths(path, modality, download)

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


def get_soft_tissue_sarcoma_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["T1", "T2FS", "CT", "PET"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Soft-tissue-Sarcoma dataloader for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality. One of 'T1', 'T2FS', 'CT' or 'PET'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_soft_tissue_sarcoma_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
