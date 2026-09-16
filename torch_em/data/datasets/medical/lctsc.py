"""The LCTSC dataset (Lung CT Segmentation Challenge 2017) contains annotations for lung and thoracic
organ segmentation in pretreatment CT of non-small cell lung cancer patients.

It consists of 60 CT volumes with manual delineations by an expert, which are distributed as DICOM
RTSTRUCT contours. This module rasterizes the contours onto the CT grid (see
`torch_em.data.datasets.util.rasterize_rtstruct`) and stores CT and labels in hdf5 files.
The semantic label ids are: 1: left lung ('Lung_L'), 2: right lung ('Lung_R'), 3: esophagus
('Esophagus'), 4: heart ('Heart'), 5: spinal cord ('SpinalCord'). All five structures are
annotated for every patient.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/lctsc/.

This dataset is from the publication https://doi.org/10.1002/mp.13141.
The data was released at https://doi.org/10.7937/K9/TCIA.2017.3R3FVZ08.
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


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LCTSC_v2_20190508.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

STRUCTURE_IDS = {"lung_left": 1, "lung_right": 2, "esophagus": 3, "heart": 4, "spinal_cord": 5}

_ROI_NAME_TO_STRUCTURE = {
    "lung_l": "lung_left",
    "lung_r": "lung_right",
    "esophagus": "esophagus",
    "heart": "heart",
    "spinalcord": "spinal_cord",
}


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
    """Map the ROIs of the RTSTRUCT files to the semantic label ids."""
    name = roi_name.lower().replace("_", "").replace("-", "").strip()
    for key, structure in _ROI_NAME_TO_STRUCTURE.items():
        if name == key.replace("_", ""):
            return STRUCTURE_IDS[structure]
    return None


def _get_referenced_series(rtstruct_path):
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    return str(
        rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0].SeriesInstanceUID
    )


def _preprocess_lctsc(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    rtstruct_series = {row["Series UID"]: row["Subject ID"] for row in rows if row["Modality"] == "RTSTRUCT"}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, subject_id in tqdm(sorted(rtstruct_series.items()), desc="Preprocess LCTSC"):
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


def get_lctsc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LCTSC dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    # Download the DICOM series (CT and RTSTRUCT) from the TCIA manifest. The series metadata are written
    # after all series are downloaded, so their presence means the download is complete.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "lctsc_series")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, os.path.basename(URL)), url=URL, dst=dicom_dir, csv_filename=csv_path,
            download=download,
        )

    _preprocess_lctsc(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_lctsc_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the LCTSC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_lctsc_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_lctsc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    structures: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LCTSC dataset for lung and thoracic organ segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        structures: The structures to use as labels, a subset of 'lung_left', 'lung_right', 'esophagus',
            'heart' and 'spinal_cord'. All other structures are set to background. By default all
            structures are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_lctsc_paths(path, download)

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


def get_lctsc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    structures: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LCTSC dataloader for lung and thoracic organ segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        structures: The structures to use as labels, a subset of 'lung_left', 'lung_right', 'esophagus',
            'heart' and 'spinal_cord'. All other structures are set to background. By default all
            structures are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lctsc_dataset(path, patch_shape, structures, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
