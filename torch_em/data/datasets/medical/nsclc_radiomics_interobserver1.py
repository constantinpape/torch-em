"""The NSCLC-Radiomics-Interobserver1 dataset contains annotations for primary gross tumor volume (GTV)
segmentation in preoperative CT of non-small cell lung cancer patients, from an interobserver-variability
study.

It consists of 22 CT volumes, 21 of which have a primary GTV delineated independently by 5 radiation
oncologists, distributed as DICOM-SEG objects and converted and stored in hdf5 files by this module. Each
radiation oncologist provided two delineations of the same tumor: a purely manual ('vis') one and one
assisted by an autosegmentation tool and then manually edited ('auto'). Radiation oncologists '1' and '3'
were trainees at the time of the study, '2', '4' and '5' were extensively experienced. The semantic label
id is: 1: tumor. This module queries the TCIA REST API for the segmented series and downloads only those
and their referenced CT series, rather than the full collection manifest (which also has the CT series
of the one patient without a usable delineation).

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/.
It is released under the CC BY-NC 3.0 license.

This dataset is from the publication https://doi.org/10.1038/ncomms5006.
The data was released at https://doi.org/10.7937/tcia.2019.cwvlpd26.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from warnings import warn
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List, Literal

import requests
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


COLLECTION = "NSCLC-Radiomics-Interobserver1"

LABEL_IDS = {"tumor": 1}

ANNOTATORS = (1, 2, 3, 4, 5)

# The DICOM-SEG segment descriptions follow the pattern 'GTV-1<annotation_type><annotator>', e.g.
# 'GTV-1vis-3' for the manual delineation of annotator 3, see the module docstring.
_ANNOTATION_TYPE_TO_TAG = {"manual": "vis", "auto": "auto"}


def _get_segmented_series_uids():
    """Query the TCIA REST API for the SEG series of the collection, without downloading any image data."""
    response = requests.get(util.NBIA_API_URL + "getSeries", params={"Collection": COLLECTION, "Modality": "SEG"})
    response.raise_for_status()
    return [row["SeriesInstanceUID"] for row in response.json()]


def _get_referenced_sop_uids(seg_path):
    """Get the SOP instance UIDs of the CT slices referenced by a DICOM-SEG object."""
    import pydicom

    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    assert len(seg.ReferencedSeriesSequence) == 1, f"Expected a single referenced CT series in {seg_path}."
    referenced_instances = seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence
    return {str(instance.ReferencedSOPInstanceUID) for instance in referenced_instances}


def _get_referenced_ct_series_uid(seg_path):
    """Get the series instance UID of the CT series referenced by a DICOM-SEG object."""
    import pydicom

    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    assert len(seg.ReferencedSeriesSequence) == 1, f"Expected a single referenced CT series in {seg_path}."
    return str(seg.ReferencedSeriesSequence[0].SeriesInstanceUID)


def _load_dicom_volume(series_dir, referenced_sop_uids):
    """Stack a DICOM series into a volume with axes (z, y, x), sorted by ascending patient z position.

    Returns the volume in Hounsfield units and the geometry needed to align the DICOM-SEG frames with the volume.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices = [dcm for dcm in slices if dcm.SOPInstanceUID in referenced_sop_uids]
    slices.sort(key=lambda dcm: dcm.ImagePositionPatient[2])

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    geometry = {
        "sop_uids": {str(dcm.SOPInstanceUID): i for i, dcm in enumerate(slices)},
        "z_positions": np.array([float(dcm.ImagePositionPatient[2]) for dcm in slices]),
        "orientation": np.round([float(v) for v in slices[0].ImageOrientationPatient]).astype("int"),
    }
    return volume, geometry


def _parse_segment_description(description):
    """Parse a DICOM-SEG segment description, e.g. 'GTV-1vis-3', into an (annotation_type, annotator) key."""
    for annotation_type, tag in _ANNOTATION_TYPE_TO_TAG.items():
        prefix = f"GTV-1{tag}-"
        if str(description).startswith(prefix):
            annotator = str(description)[len(prefix):]
            if annotator in {str(a) for a in ANNOTATORS}:
                return annotation_type, annotator
    return None


def _load_dicom_seg(seg_path, shape, geometry):
    """Convert a DICOM-SEG object into per-annotator, per-annotation-type binary tumor masks.

    Each frame is mapped to its CT slice via the source image it was derived from (or its z position), and
    to a segment via its 'ReferencedSegmentNumber'. Only the segments matching the 'GTV-1<vis|auto>-<1-5>'
    naming convention are kept (see the module docstring); the additional PET-SUV-threshold-based ROIs that
    some patients also carry are ignored, since they are not tied to a specific annotator.

    Returns a dict mapping '<annotation_type>/<annotator>' (e.g. 'manual/3') to a boolean mask.
    """
    import pydicom

    seg = pydicom.dcmread(seg_path)

    segment_keys = {}
    for segment in seg.SegmentSequence:
        parsed = _parse_segment_description(segment.SegmentDescription)
        if parsed is not None:
            annotation_type, annotator = parsed
            segment_keys[int(segment.SegmentNumber)] = f"{annotation_type}/{annotator}"

    seg_orientation = seg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
    seg_orientation = np.round([float(v) for v in seg_orientation]).astype("int")
    orientation = geometry["orientation"]
    assert np.all(np.abs(seg_orientation) == np.abs(orientation)), f"Unexpected orientation in {seg_path}."
    flip_axes = []
    if np.any(seg_orientation[3:] != orientation[3:]):  # The direction of the rows differs.
        flip_axes.append(0)
    if np.any(seg_orientation[:3] != orientation[:3]):  # The direction of the columns differs.
        flip_axes.append(1)

    z_positions = geometry["z_positions"]
    tolerance = np.diff(z_positions).min() / 2 if len(z_positions) > 1 else 1.0

    frames = seg.pixel_array
    if frames.ndim == 2:  # A segmentation with a single frame.
        frames = frames[None]

    masks = {key: np.zeros(shape, dtype="bool") for key in segment_keys.values()}
    for frame, frame_group in zip(frames, seg.PerFrameFunctionalGroupsSequence):
        segment_number = int(frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        key = segment_keys.get(segment_number)
        if key is None:  # Not one of the 'GTV-1<vis|auto>-<1-5>' segments, e.g. a PET-SUV-threshold-based ROI.
            continue
        frame_mask = frame.astype("bool")

        z = None
        derivation = frame_group.get("DerivationImageSequence", [])
        if derivation and derivation[0].get("SourceImageSequence"):
            z = geometry["sop_uids"].get(str(derivation[0].SourceImageSequence[0].ReferencedSOPInstanceUID))
        if z is None:
            frame_z = float(frame_group.PlanePositionSequence[0].ImagePositionPatient[2])
            z = int(np.argmin(np.abs(z_positions - frame_z)))
            if abs(z_positions[z] - frame_z) > tolerance:
                if frame_mask.any():
                    warn(f"Skipping a frame at z={frame_z} in {seg_path}, which does not match a CT slice.")
                continue

        if flip_axes:
            frame_mask = np.flip(frame_mask, axis=flip_axes)
        masks[key][z] |= frame_mask

    return masks


def _preprocess_nsclc_radiomics_interobserver1(dicom_dir, csv_paths, preprocessed_dir):
    import h5py

    series_per_subject = defaultdict(dict)
    for csv_path in csv_paths:
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                series_per_subject[row["Subject ID"]][row["Modality"]] = os.path.join(dicom_dir, row["Series UID"])

    os.makedirs(preprocessed_dir, exist_ok=True)
    subjects_with_seg = {sid: series for sid, series in series_per_subject.items() if "SEG" in series}
    for subject_id, series_dirs in tqdm(
        sorted(subjects_with_seg.items()), desc="Preprocess NSCLC-Radiomics-Interobserver1"
    ):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue
        if "CT" not in series_dirs:
            warn(f"Skipping {subject_id}, which has a SEG object but no CT series.")
            continue

        seg_path = glob(os.path.join(series_dirs["SEG"], "*.dcm"))[0]
        volume, geometry = _load_dicom_volume(series_dirs["CT"], _get_referenced_sop_uids(seg_path))
        masks = _load_dicom_seg(seg_path, volume.shape, geometry)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            for key, mask in masks.items():
                labels = (mask * LABEL_IDS["tumor"]).astype("uint8")
                f.create_dataset(f"labels/{key}", data=labels, compression="gzip")


def get_nsclc_radiomics_interobserver1_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NSCLC-Radiomics-Interobserver1 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    # Only the segmented series and their referenced CT series are downloaded (see the module docstring),
    # not the full collection manifest. Each download step is skipped once its metadata csv is written.
    dicom_dir = os.path.join(path, "dicom")
    seg_csv_path = os.path.join(path, "nsclc_radiomics_interobserver1_seg_series")
    if not os.path.exists(f"{seg_csv_path}.csv"):
        if not download:
            raise RuntimeError(f"Cannot find the data at {dicom_dir}, but download was set to False.")
        seg_uids = _get_segmented_series_uids()
        util.download_tcia_series(seg_uids, dicom_dir, seg_csv_path)

    ct_csv_path = os.path.join(path, "nsclc_radiomics_interobserver1_ct_series")
    if not os.path.exists(f"{ct_csv_path}.csv"):
        with open(f"{seg_csv_path}.csv", "r") as f:
            seg_uids = [row["Series UID"] for row in csv.DictReader(f)]
        ct_uids = sorted({
            _get_referenced_ct_series_uid(glob(os.path.join(dicom_dir, uid, "*.dcm"))[0]) for uid in seg_uids
        })
        util.download_tcia_series(ct_uids, dicom_dir, ct_csv_path)

    csv_paths = [f"{seg_csv_path}.csv", f"{ct_csv_path}.csv"]
    _preprocess_nsclc_radiomics_interobserver1(dicom_dir, csv_paths, preprocessed_dir)
    return preprocessed_dir


def get_nsclc_radiomics_interobserver1_paths(
    path: Union[os.PathLike, str],
    annotator: int = 1,
    annotation_type: Literal["manual", "auto"] = "manual",
    download: bool = False,
) -> Tuple[List[str], str]:
    """Get paths to the NSCLC-Radiomics-Interobserver1 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotator: The radiation oncologist whose delineation to use, an integer from 1 to 5. Oncologists '1'
            and '3' were trainees at the time of the study, '2', '4' and '5' were extensively experienced.
        annotation_type: The type of delineation to use. Either 'manual' for a purely manual delineation, or
            'auto' for a delineation assisted by an autosegmentation tool and then manually edited. Only 20 of
            the 21 patients have an 'auto' delineation; the one missing it is skipped for that choice.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data.
        The key of the label data in each hdf5 file.
    """
    assert annotator in ANNOTATORS, f"Invalid annotator: {annotator}. Choose from {ANNOTATORS}."
    assert annotation_type in _ANNOTATION_TYPE_TO_TAG, f"Invalid annotation_type: {annotation_type}."

    import h5py

    data_dir = get_nsclc_radiomics_interobserver1_data(path, download)
    label_key = f"labels/{annotation_type}/{annotator}"

    volume_paths = []
    for volume_path in natsorted(glob(os.path.join(data_dir, "*.h5"))):
        with h5py.File(volume_path, "r") as f:
            has_label = label_key in f
        if has_label:
            volume_paths.append(volume_path)
        else:
            warn(f"Skipping {volume_path}, which has no '{label_key}' delineation.")

    return volume_paths, label_key


def get_nsclc_radiomics_interobserver1_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotator: int = 1,
    annotation_type: Literal["manual", "auto"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NSCLC-Radiomics-Interobserver1 dataset for lung tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotator: The radiation oncologist whose delineation to use, an integer from 1 to 5. Oncologists '1'
            and '3' were trainees at the time of the study, '2', '4' and '5' were extensively experienced.
        annotation_type: The type of delineation to use. Either 'manual' for a purely manual delineation, or
            'auto' for a delineation assisted by an autosegmentation tool and then manually edited. Only 20 of
            the 21 patients have an 'auto' delineation; the one missing it is skipped for that choice.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths, label_key = get_nsclc_radiomics_interobserver1_paths(path, annotator, annotation_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_nsclc_radiomics_interobserver1_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotator: int = 1,
    annotation_type: Literal["manual", "auto"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NSCLC-Radiomics-Interobserver1 dataloader for lung tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotator: The radiation oncologist whose delineation to use, an integer from 1 to 5. Oncologists '1'
            and '3' were trainees at the time of the study, '2', '4' and '5' were extensively experienced.
        annotation_type: The type of delineation to use. Either 'manual' for a purely manual delineation, or
            'auto' for a delineation assisted by an autosegmentation tool and then manually edited. Only 20 of
            the 21 patients have an 'auto' delineation; the one missing it is skipped for that choice.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nsclc_radiomics_interobserver1_dataset(
        path, patch_shape, annotator, annotation_type, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
