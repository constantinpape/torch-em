"""The Spinal-Multiple-Myeloma-SEG dataset contains annotations for spinal multiple myeloma lesions in CT.

The dataset consists of 144 dual-energy CT studies with instance masks for the focal spinal lesions of
each patient, up to several dozen per case. The masks are stored as DICOM-SEG objects, and this module
pairs them with the conventional ('_konv') reconstruction of their dual-energy CT study.

NOTE: A dual-energy study has several reconstructions on the identical voxel grid (conventional,
several monoenergetic keV levels, several calcium-suppression indices), so the shape and geometry of a
scan cannot disambiguate which one the segmentation was drawn on. The DICOM-SEG series references its
source CT by UID, but that series is not resolvable through TCIA anymore for some cases, so the
conventional reconstruction of the same study is used instead: every one of the 144 studies has exactly
one CT series whose description ends in '_konv', and this is the closest series in the collection to a
routine single-energy CT.

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.7937/k4qv-hh78 and is distributed under the CC BY 4.0
license.
This dataset is from the publication https://doi.org/10.1038/s41597-026-08061-x.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List


from torch.utils.data import Dataset, DataLoader

import torch_em

from .adrenal_acc import _load_dicom_volume, _load_dicom_seg, _resample_labels
from .. import util


COLLECTION = "Spinal-Multiple-Myeloma-SEG"


def _get_series_metadata(path, download):
    """Get the metadata of all series in the collection from the NBIA REST API."""
    import requests

    metadata_path = os.path.join(path, "spinal_mm_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(f"{util.NBIA_API_URL}getSeries", params={"Collection": COLLECTION})
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _find_konv_series(series_metadata, study_uid, patient_id):
    """Find the conventional reconstruction of a dual-energy study."""
    candidates = [
        series for series in series_metadata
        if series.get("StudyInstanceUID") == study_uid and series.get("Modality") == "CT"
        and (series.get("SeriesDescription") or "").lower() == f"{patient_id}_konv".lower()
    ]
    return candidates[0]["SeriesInstanceUID"] if len(candidates) == 1 else None


def _preprocess_spinal_mm(dicom_dir, series_metadata, preprocessed_dir):
    import h5py

    seg_series = [series for series in series_metadata if series.get("Modality") == "SEG"]

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series in tqdm(seg_series, desc="Preprocess Spinal-Multiple-Myeloma-SEG"):
        patient_id = series["PatientID"]
        # A patient can have more than one study (e.g. a follow-up scan), each with its own SEG series,
        # so the output is keyed by the SEG series UID rather than the patient id.
        out_path = os.path.join(preprocessed_dir, f"{series['SeriesInstanceUID']}.h5")
        if os.path.exists(out_path):
            continue

        seg_dir = os.path.join(dicom_dir, series["SeriesInstanceUID"])
        seg_paths = glob(os.path.join(seg_dir, "*.dcm"))
        if not seg_paths:
            continue

        import pydicom
        study_uid = pydicom.dcmread(seg_paths[0], stop_before_pixels=True).StudyInstanceUID
        ct_uid = _find_konv_series(series_metadata, study_uid, patient_id)
        ct_dir = os.path.join(dicom_dir, ct_uid) if ct_uid else None
        if ct_dir is None or not glob(os.path.join(ct_dir, "*.dcm")):
            continue

        volume, ct_affine = _load_dicom_volume(ct_dir)
        seg_labels, seg_affine = _load_dicom_seg(seg_paths[0])
        # The SEG object only covers the slices its lesions appear on, cropped from the full CT extent,
        # so it is placed on the CT's own grid rather than compared to it directly.
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, ct_affine)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_spinal_mm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Spinal-Multiple-Myeloma-SEG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)
    series_metadata = _get_series_metadata(path, download)

    seg_uids = [series["SeriesInstanceUID"] for series in series_metadata if series.get("Modality") == "SEG"]

    # Every SEG series belongs to one study; the conventional CT of that study is downloaded alongside it.
    study_by_seg = {
        series["SeriesInstanceUID"]: (series["StudyInstanceUID"], series["PatientID"])
        for series in series_metadata if series.get("Modality") == "SEG"
    }
    konv_uids = [
        _find_konv_series(series_metadata, study_uid, patient_id)
        for study_uid, patient_id in study_by_seg.values()
    ]
    series_uids = sorted(set(seg_uids) | {uid for uid in konv_uids if uid})

    dicom_dir = os.path.join(path, "dicom")
    if download:  # Series that were downloaded already are skipped.
        util.download_tcia_series(series_uids, dst=dicom_dir, csv_filename=os.path.join(path, "spinal_mm"))
    elif not all(glob(os.path.join(dicom_dir, uid, "*.dcm")) for uid in series_uids):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_spinal_mm(dicom_dir, series_metadata, preprocessed_dir)
    return preprocessed_dir


def get_spinal_mm_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Spinal-Multiple-Myeloma-SEG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_spinal_mm_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_spinal_mm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Spinal-Multiple-Myeloma-SEG dataset for spinal lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_spinal_mm_paths(path, download)

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


def get_spinal_mm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Spinal-Multiple-Myeloma-SEG dataloader for spinal lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_spinal_mm_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
