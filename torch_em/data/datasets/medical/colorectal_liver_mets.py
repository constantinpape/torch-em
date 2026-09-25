"""The Colorectal-Liver-Metastases dataset contains annotations for liver, tumor and vessel segmentation
in contrast-enhanced CT of patients with colorectal liver metastases before hepatic resection.

It consists of 197 CT volumes with semantic labels for: 1: liver, 2: hepatic veins, 3: portal veins, 4: tumors.
The CT scans are distributed as DICOM series and the labels as DICOM-SEG objects, which are converted and stored
in hdf5 files by this module. The segments are painted in the order of their ids, i.e. the vessels and the tumors
overwrite the liver. Besides the semantic labels ('labels'), the hdf5 files contain the tumor instance labels
('tumor_instances', each metastasis is segmented individually) and the projected post-operative liver remnant
('liver_remnant', a binary mask that overlaps with the liver and is hence stored separately).

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/.

This dataset is from the publication https://doi.org/10.1038/s41597-024-02981-2.
The data was released at https://doi.org/10.7937/QXK2-QG03.
Please cite it if you use this dataset in your research.
"""

import os
import re
import csv
from glob import glob
from tqdm import tqdm
from warnings import warn
from natsort import natsorted
from collections import defaultdict, Counter
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Colorectal-Liver-Metastases-November-2022-manifest.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

NUM_VOLUMES = 197

LABEL_IDS = {"liver": 1, "hepatic_vein": 2, "portal_vein": 3, "tumor": 4}

# The segment labels used in the DICOM-SEG objects. The tumors are labeled as 'Tumor_1', 'Tumor_2', etc.
SEGMENT_NAMES = {"Liver": "liver", "Liver Remnant": "liver_remnant", "Hepatic": "hepatic_vein", "Portal": "portal_vein"}
TUMOR_PATTERN = re.compile(r"^Tumor_(\d+)$")


def _get_referenced_sop_uids(seg_path):
    """Get the SOP instance UIDs of the CT slices referenced by a DICOM-SEG object."""
    import pydicom

    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    assert len(seg.ReferencedSeriesSequence) == 1, f"Expected a single referenced CT series in {seg_path}."
    referenced_instances = seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence
    return {str(instance.ReferencedSOPInstanceUID) for instance in referenced_instances}


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


def _preprocess_colorectal_liver_mets(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    series_per_subject = defaultdict(dict)
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            series_per_subject[row["Subject ID"]][row["Modality"]] = os.path.join(dicom_dir, row["Series UID"])

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, series_dirs in tqdm(
        sorted(series_per_subject.items()), desc="Preprocess Colorectal-Liver-Metastases"
    ):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        seg_path = glob(os.path.join(series_dirs["SEG"], "*.dcm"))[0]
        volume, geometry = _load_dicom_volume(series_dirs["CT"], _get_referenced_sop_uids(seg_path))
        masks = _load_dicom_seg(seg_path, volume.shape, geometry)

        labels = np.zeros(volume.shape, dtype="uint8")
        tumor_instances = np.zeros(volume.shape, dtype="uint8")
        liver_remnant = np.zeros(volume.shape, dtype="bool")
        for name in sorted(masks, key=lambda name: (LABEL_IDS.get(SEGMENT_NAMES.get(name), len(LABEL_IDS)), name)):
            tumor_match = TUMOR_PATTERN.match(name)
            if tumor_match:
                labels[masks[name]] = LABEL_IDS["tumor"]
                tumor_instances[masks[name]] = int(tumor_match.group(1))
            elif name == "Liver Remnant":
                liver_remnant = masks[name]
            elif name in SEGMENT_NAMES:
                labels[masks[name]] = LABEL_IDS[SEGMENT_NAMES[name]]
            else:
                raise ValueError(f"Unknown segment label '{name}' in {seg_path}.")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
            f.create_dataset("tumor_instances", data=tumor_instances, compression="gzip")
            f.create_dataset("liver_remnant", data=liver_remnant.astype("uint8"), compression="gzip")


def get_colorectal_liver_mets_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Colorectal-Liver-Metastases dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == NUM_VOLUMES:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    # Download the DICOM series (CT and SEG) from the TCIA manifest.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "colorectal_liver_mets_series")
    util.download_source_tcia(
        path=os.path.join(path, "Colorectal-Liver-Metastases-November-2022-manifest.tcia"), url=URL, dst=dicom_dir,
        csv_filename=csv_path, download=download,
    )

    _preprocess_colorectal_liver_mets(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_colorectal_liver_mets_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Colorectal-Liver-Metastases data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_colorectal_liver_mets_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_colorectal_liver_mets_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Colorectal-Liver-Metastases dataset for liver, tumor and vessel segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_colorectal_liver_mets_paths(path, download)

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


def get_colorectal_liver_mets_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Colorectal-Liver-Metastases dataloader for liver, tumor and vessel segmentation.

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
    dataset = get_colorectal_liver_mets_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
