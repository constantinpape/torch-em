"""The NSCLC-Radiogenomics dataset contains annotations for primary tumor segmentation in preoperative
CT of non-small cell lung cancer patients.

It consists of CT volumes with a manual delineation of the primary gross tumor volume by a radiation
oncologist, distributed as DICOM-SEG objects, which are converted and stored in hdf5 files by this
module. The semantic label id is: 1: tumor. Only 144 of the 211 patients in the collection have a
segmentation; this module queries the TCIA REST API for the segmented series and downloads only those
and their referenced CT series, rather than the full ~98GB collection (which also has PET series and
CT series for the unsegmented patients).

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/.

This dataset is from the publication https://doi.org/10.1038/sdata.2018.202.
The data was released at https://doi.org/10.7937/K9/TCIA.2017.7hs46erv.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from warnings import warn
from natsort import natsorted
from collections import defaultdict, Counter
from typing import Union, Tuple, List

import requests
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


COLLECTION = "NSCLC Radiogenomics"

LABEL_IDS = {"tumor": 1}


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
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted by ascending patient z position.

    Some series contain several acquisitions with overlapping slice positions. In this case, only the
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
    """Convert a DICOM-SEG object into a binary tumor mask aligned with the reference CT volume.

    Each frame is mapped to its CT slice via the source image it was derived from (or its z position).
    All segments in the object are combined into a single tumor mask, since the collection ships one
    primary gross tumor volume per patient.
    """
    import pydicom

    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:  # A segmentation with a single frame.
        frames = frames[None]

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

    mask = np.zeros(shape, dtype="bool")
    for frame, frame_group in zip(frames, seg.PerFrameFunctionalGroupsSequence):
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
        mask[z] |= frame_mask

    return mask


def _preprocess_nsclc_radiogenomics(dicom_dir, csv_paths, preprocessed_dir):
    import h5py

    series_per_subject = defaultdict(dict)
    for csv_path in csv_paths:
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                series_per_subject[row["Subject ID"]][row["Modality"]] = os.path.join(dicom_dir, row["Series UID"])

    os.makedirs(preprocessed_dir, exist_ok=True)
    subjects_with_seg = {sid: series for sid, series in series_per_subject.items() if "SEG" in series}
    for subject_id, series_dirs in tqdm(sorted(subjects_with_seg.items()), desc="Preprocess NSCLC-Radiogenomics"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue
        if "CT" not in series_dirs:
            warn(f"Skipping {subject_id}, which has a SEG object but no CT series.")
            continue

        seg_path = glob(os.path.join(series_dirs["SEG"], "*.dcm"))[0]
        volume, geometry = _load_dicom_volume(series_dirs["CT"], _get_referenced_sop_uids(seg_path))
        mask = _load_dicom_seg(seg_path, volume.shape, geometry)
        labels = (mask * LABEL_IDS["tumor"]).astype("uint8")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_nsclc_radiogenomics_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NSCLC-Radiogenomics dataset.

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
    seg_csv_path = os.path.join(path, "nsclc_radiogenomics_seg_series")
    if not os.path.exists(f"{seg_csv_path}.csv"):
        if not download:
            raise RuntimeError(f"Cannot find the data at {dicom_dir}, but download was set to False.")
        seg_uids = _get_segmented_series_uids()
        util.download_tcia_series(seg_uids, dicom_dir, seg_csv_path)

    ct_csv_path = os.path.join(path, "nsclc_radiogenomics_ct_series")
    if not os.path.exists(f"{ct_csv_path}.csv"):
        with open(f"{seg_csv_path}.csv", "r") as f:
            seg_uids = [row["Series UID"] for row in csv.DictReader(f)]
        ct_uids = sorted({
            _get_referenced_ct_series_uid(glob(os.path.join(dicom_dir, uid, "*.dcm"))[0]) for uid in seg_uids
        })
        util.download_tcia_series(ct_uids, dicom_dir, ct_csv_path)

    _preprocess_nsclc_radiogenomics(dicom_dir, [f"{seg_csv_path}.csv", f"{ct_csv_path}.csv"], preprocessed_dir)
    return preprocessed_dir


def get_nsclc_radiogenomics_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the NSCLC-Radiogenomics data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_nsclc_radiogenomics_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_nsclc_radiogenomics_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NSCLC-Radiogenomics dataset for lung tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_nsclc_radiogenomics_paths(path, download)

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


def get_nsclc_radiogenomics_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NSCLC-Radiogenomics dataloader for lung tumor segmentation.

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
    dataset = get_nsclc_radiogenomics_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
