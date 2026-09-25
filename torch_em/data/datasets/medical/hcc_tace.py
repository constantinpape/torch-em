"""The HCC-TACE-Seg dataset contains annotations for liver, tumor and vessel segmentation in multiphasic
contrast-enhanced CT of patients with hepatocellular carcinoma treated with transarterial chemoembolization (TACE).

It consists of 105 CT volumes (the pre-treatment CT phase that was segmented) with semantic labels for:
1: liver, 2: tumor, 3: tumor necrosis (only annotated for one patient), 4: portal vein,
5: abdominal aorta. The CT scans are distributed as DICOM series and the labels as DICOM-SEG objects, which are
converted and stored in hdf5 files by this module. The segments are painted in the order of their ids, i.e. the
tumor overwrites the liver and the necrosis overwrites the tumor.

The collection contains several CT phases per patient (28.6 GB in total), but only one of them is segmented.
Hence, this module downloads the DICOM-SEG objects and the CT series they reference via the TCIA REST API
instead of the full collection manifest.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/hcc-tace-seg/.

This dataset is from the publication https://doi.org/10.1038/s41597-023-01928-3.
The data was released at https://doi.org/10.7937/TCIA.5FNA-0924.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from warnings import warn
from natsort import natsorted
from collections import Counter, defaultdict
from typing import Union, Tuple, List

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "HCC-TACE-Seg"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

NUM_VOLUMES = 105

LABEL_IDS = {"liver": 1, "tumor": 2, "necrosis": 3, "portal_vein": 4, "abdominal_aorta": 5}

# The segment labels used in the DICOM-SEG objects.
SEGMENT_NAMES = {
    "Liver": "liver", "Mass": "tumor", "Necrosis": "necrosis", "Portal vein": "portal_vein",
    "Abdominal aorta": "abdominal_aorta",
}


def _download_series(series_uid, dicom_dir, download):
    """Download a DICOM series via the NBIA REST API and extract it to '<dicom_dir>/<series_uid>'."""
    series_dir = os.path.join(dicom_dir, series_uid)
    if os.path.exists(series_dir):
        return series_dir

    zip_path = os.path.join(dicom_dir, f"{series_uid}.zip")
    url = f"{NBIA_API_URL}/getImage?SeriesInstanceUID={series_uid}"
    util.download_source(path=zip_path, url=url, download=download, checksum=None)
    util.unzip(zip_path=zip_path, dst=f"{series_dir}.tmp")
    os.rename(f"{series_dir}.tmp", series_dir)
    return series_dir


def _get_series_per_patient(download):
    """Get the metadata of all series in the collection from the NBIA REST API, grouped by patient."""
    if not download:
        raise RuntimeError("Cannot find the data, but download was set to False.")
    response = requests.get(f"{NBIA_API_URL}/getSeries", params={"Collection": COLLECTION})
    response.raise_for_status()

    series_per_patient = defaultdict(list)
    for series in response.json():
        series_per_patient[series["PatientID"]].append(series)
    return series_per_patient


def _get_candidate_series(seg_path, ct_series):
    """Rank the CT series of a patient by how likely it is that the DICOM-SEG object was drawn on them.

    The series referenced by the DICOM-SEG object is tried first. This reference is missing for one patient
    (HCC_048) and points to a series that does not overlap with the segmentation for another one (HCC_089),
    so the other CT series are ranked by the difference between their slice count and the number of segmented
    slices and are used as fall-backs.

    Returns a list of (series UID, SOP instance UIDs of the referenced slices) tuples.
    """
    import pydicom

    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    z_positions = {
        round(float(group.PlanePositionSequence[0].ImagePositionPatient[2]), 2)
        for group in seg.PerFrameFunctionalGroupsSequence
    }

    referenced_uid, referenced_sop_uids = None, set()
    if "ReferencedSeriesSequence" in seg:
        assert len(seg.ReferencedSeriesSequence) == 1, f"Expected a single referenced CT series in {seg_path}."
        referenced_series = seg.ReferencedSeriesSequence[0]
        referenced_uid = str(referenced_series.SeriesInstanceUID)
        referenced_sop_uids = {
            str(instance.ReferencedSOPInstanceUID) for instance in referenced_series.ReferencedInstanceSequence
        }

    ranked = sorted(ct_series, key=lambda series: abs(int(series["ImageCount"]) - len(z_positions)))
    candidates = [(referenced_uid, referenced_sop_uids)] if referenced_uid is not None else []
    candidates += [(str(series["SeriesInstanceUID"]), set()) for series in ranked
                   if str(series["SeriesInstanceUID"]) != referenced_uid]
    return candidates


def _load_dicom_volume(series_dir, referenced_sop_uids):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted by ascending patient z position.

    Some series contain several acquisitions (CT phases) with overlapping slice positions. In this case, only the
    acquisition referenced by the DICOM-SEG object is kept, so that each slice position occurs once.
    Returns the volume in Hounsfield units and the geometry needed to align the DICOM-SEG frames with the volume.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]

    acquisitions = Counter(dcm.get("AcquisitionNumber") for dcm in slices if dcm.SOPInstanceUID in referenced_sop_uids)
    acquisition = acquisitions.most_common(1)[0][0] if acquisitions else None
    slices_per_position = {}
    for dcm in slices:
        z = round(float(dcm.ImagePositionPatient[2]), 2)
        priority = (dcm.SOPInstanceUID in referenced_sop_uids, dcm.get("AcquisitionNumber") == acquisition)
        if z not in slices_per_position or priority > slices_per_position[z][0]:
            slices_per_position[z] = (priority, dcm)
    slices = [dcm for _, dcm in sorted(slices_per_position.values(), key=lambda item: item[1].ImagePositionPatient[2])]

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    geometry = {
        "sop_uids": {str(dcm.SOPInstanceUID): i for i, dcm in enumerate(slices)},
        "z_positions": np.array([float(dcm.ImagePositionPatient[2]) for dcm in slices]),
        "orientation": np.round([float(v) for v in slices[0].ImageOrientationPatient]).astype("int"),
    }
    return volume, geometry


def _load_dicom_seg(seg_path, shape, geometry):
    """Convert a DICOM-SEG object into binary masks (one per segment) aligned with the reference CT volume.

    Each frame is mapped to its CT slice via the source image it was derived from (or its z position).
    """
    import pydicom

    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:  # A segmentation with a single frame.
        frames = frames[None]

    segment_names = {int(segment.SegmentNumber): str(segment.SegmentLabel) for segment in seg.SegmentSequence}

    # The segmentation frames may use a different in-plane orientation than the CT slices,
    # in which case they have to be flipped to align them.
    seg_orientation = seg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
    seg_orientation = np.round([float(v) for v in seg_orientation]).astype("int")
    orientation = geometry["orientation"]
    assert np.all(np.abs(seg_orientation) == np.abs(orientation)), f"Unexpected orientation in {seg_path}."
    flip_axes = []
    if np.any(seg_orientation[3:] != orientation[3:]):  # The direction of the rows differs.
        flip_axes.append(0)
    if np.any(seg_orientation[:3] != orientation[:3]):  # The direction of the columns differs.
        flip_axes.append(1)

    # Frames are matched to CT slices via their z position, if they cannot be matched via the source image.
    z_positions = geometry["z_positions"]
    tolerance = np.diff(z_positions).min() / 2 if len(z_positions) > 1 else 1.0

    masks = {name: np.zeros(shape, dtype="bool") for name in segment_names.values()}
    for frame, frame_group in zip(frames, seg.PerFrameFunctionalGroupsSequence):
        segment_number = int(frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        mask = frame.astype("bool")

        z = None
        derivation = frame_group.get("DerivationImageSequence", [])
        if derivation and derivation[0].get("SourceImageSequence"):
            z = geometry["sop_uids"].get(str(derivation[0].SourceImageSequence[0].ReferencedSOPInstanceUID))
        if z is None:
            frame_z = float(frame_group.PlanePositionSequence[0].ImagePositionPatient[2])
            z = int(np.argmin(np.abs(z_positions - frame_z)))
            if abs(z_positions[z] - frame_z) > tolerance:
                if mask.any():
                    warn(f"Skipping a frame of '{segment_names[segment_number]}' at z={frame_z} in {seg_path}, "
                         "which does not match a CT slice.")
                continue

        if flip_axes:
            mask = np.flip(mask, axis=flip_axes)
        masks[segment_names[segment_number]][z] |= mask

    return masks


def _preprocess_hcc_tace(seg_path, ct_dir, referenced_sop_uids, out_path):
    """Convert a CT series and the segmentation drawn on it into an hdf5 file.

    Returns whether the segmentation could be mapped onto the CT series.
    """
    import h5py

    volume, geometry = _load_dicom_volume(ct_dir, referenced_sop_uids)
    masks = _load_dicom_seg(seg_path, volume.shape, geometry)

    unknown = [name for name in masks if name not in SEGMENT_NAMES]
    if unknown:
        raise ValueError(f"Unknown segment labels {unknown} in {seg_path}.")

    labels = np.zeros(volume.shape, dtype="uint8")
    for name in sorted(masks, key=lambda name: LABEL_IDS[SEGMENT_NAMES[name]]):
        labels[masks[name]] = LABEL_IDS[SEGMENT_NAMES[name]]

    if labels.max() == 0:  # The segmentation does not belong to this CT series.
        return False

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=volume, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    return True


def get_hcc_tace_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HCC-TACE-Seg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == NUM_VOLUMES:
        return preprocessed_dir

    dicom_dir = os.path.join(path, "dicom")
    os.makedirs(dicom_dir, exist_ok=True)

    # Download the DICOM-SEG object of each patient and the CT series it references, then convert them.
    series_per_patient = _get_series_per_patient(download)
    for patient_id, series in tqdm(sorted(series_per_patient.items()), desc="Download and preprocess HCC-TACE-Seg"):
        out_path = os.path.join(preprocessed_dir, f"{patient_id}.h5")
        if os.path.exists(out_path):
            continue

        seg_series = [s for s in series if s["Modality"] == "SEG"]
        assert len(seg_series) == 1, f"Expected a single segmentation for the patient '{patient_id}'."
        seg_dir = _download_series(str(seg_series[0]["SeriesInstanceUID"]), dicom_dir, download)
        seg_path = glob(os.path.join(seg_dir, "*.dcm"))[0]

        os.makedirs(preprocessed_dir, exist_ok=True)
        candidates = _get_candidate_series(seg_path, [s for s in series if s["Modality"] == "CT"])
        for ct_uid, referenced_sop_uids in candidates:
            ct_dir = _download_series(ct_uid, dicom_dir, download)
            if _preprocess_hcc_tace(seg_path, ct_dir, referenced_sop_uids, out_path):
                break
        else:
            raise RuntimeError(f"Could not find the CT series that belongs to {seg_path}.")

    return preprocessed_dir


def get_hcc_tace_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the HCC-TACE-Seg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_hcc_tace_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_hcc_tace_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HCC-TACE-Seg dataset for liver, tumor and vessel segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_hcc_tace_paths(path, download)

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


def get_hcc_tace_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HCC-TACE-Seg dataloader for liver, tumor and vessel segmentation.

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
    dataset = get_hcc_tace_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
