"""The NSCLC-Radiomics dataset (Lung1) contains annotations for the gross tumor volume and thoracic organs
in pretreatment CT of non-small cell lung cancer patients.

It consists of 422 CT volumes with manual delineations by a radiation oncologist, which are distributed as
DICOM RTSTRUCT contours. This module rasterizes the contours onto the CT grid (see
`torch_em.data.datasets.util.rasterize_rtstruct`) and stores CT and labels in hdf5 files.
The semantic label ids are: 1: tumor, 2: lung ('Lung-Left', 'Lung-Right' or 'Lungs-Total'), 3: heart,
4: esophagus, 5: spinal cord. The tumor id covers the primary gross tumor volume ('GTV-1', present for all but
one patient) and any further 'gtv-*' contours, which are mostly involved lymph nodes and are named
inconsistently ('gtv-2', 'gtv_10_r', 'gtv_supraclav', ...). Not all structures are annotated for every patient:
of the 422 patients 422 have a tumor, 411 a lung, 355 an esophagus, 411 a spinal cord and 127 a heart contour.
The structures to use can be selected via `structures`.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/nsclc-radiomics/.

This dataset is from the publication https://doi.org/10.1038/ncomms5006.
The data was released at https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Sequence

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.transform.generic import Compose

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-Version-4-Oct-2020-NBIA-manifest.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

STRUCTURE_IDS = {"tumor": 1, "lung": 2, "heart": 3, "esophagus": 4, "spinal_cord": 5}


class SelectStructures:
    """Label transform that keeps only the given label ids and sets all other labels to background.

    Args:
        label_ids: The label ids to keep.
    """
    def __init__(self, label_ids: Sequence[int]):
        self.label_ids = list(label_ids)

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        return np.where(np.isin(labels, self.label_ids), labels, 0)


def _get_structure_label(roi_number, roi_name):
    """Map the (inconsistently named) ROIs of the RTSTRUCT files to the semantic label ids."""
    name = roi_name.lower().replace("_", "-").strip()
    if name.startswith("gtv"):
        return STRUCTURE_IDS["tumor"]
    if name.startswith("lung"):
        return STRUCTURE_IDS["lung"]
    if name.startswith("heart"):
        return STRUCTURE_IDS["heart"]
    if name.startswith("esophagus"):
        return STRUCTURE_IDS["esophagus"]
    if name.startswith("spinal-cord") or name.startswith("spinalcord"):
        return STRUCTURE_IDS["spinal_cord"]
    return None


def _get_referenced_series(rtstruct_path):
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    return str(
        rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0].SeriesInstanceUID
    )


def _preprocess_nsclc_radiomics(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    rtstruct_series = {row["Series UID"]: row["Subject ID"] for row in rows if row["Modality"] == "RTSTRUCT"}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, subject_id in tqdm(sorted(rtstruct_series.items()), desc="Preprocess NSCLC-Radiomics"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        rtstruct_path = glob(os.path.join(dicom_dir, series_uid, "*.dcm"))[0]
        ct_dir = os.path.join(dicom_dir, _get_referenced_series(rtstruct_path))
        volume, geometry = util.load_dicom_series(ct_dir)
        volume = np.round(volume).astype("int16")
        labels = util.rasterize_rtstruct(rtstruct_path, geometry, volume.shape, _get_structure_label)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_nsclc_radiomics_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NSCLC-Radiomics dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    # Download the DICOM series (CT, RTSTRUCT and SEG) from the TCIA manifest. The series metadata are written
    # after all series are downloaded, so their presence means the download is complete.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "nsclc_radiomics_series")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, os.path.basename(URL)), url=URL, dst=dicom_dir, csv_filename=csv_path,
            download=download,
        )

    _preprocess_nsclc_radiomics(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_nsclc_radiomics_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the NSCLC-Radiomics data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_nsclc_radiomics_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_nsclc_radiomics_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    structures: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NSCLC-Radiomics dataset for tumor and thoracic organ segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        structures: The structures to use as labels, a subset of 'tumor', 'lung', 'heart', 'esophagus'
            and 'spinal_cord'. All other structures are set to background. By default all structures are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_nsclc_radiomics_paths(path, download)

    if structures is not None:
        assert all(structure in STRUCTURE_IDS for structure in structures), f"Invalid structures: {structures}"
        select_trafo = SelectStructures([STRUCTURE_IDS[structure] for structure in structures])
        if "label_transform" in kwargs:
            kwargs["label_transform"] = Compose(select_trafo, kwargs["label_transform"], is_multi_tensor=False)
        else:
            kwargs["label_transform"] = select_trafo

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


def get_nsclc_radiomics_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    structures: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NSCLC-Radiomics dataloader for tumor and thoracic organ segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        structures: The structures to use as labels, a subset of 'tumor', 'lung', 'heart', 'esophagus'
            and 'spinal_cord'. All other structures are set to background. By default all structures are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nsclc_radiomics_dataset(path, patch_shape, structures, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
