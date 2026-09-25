"""The Spine-Mets-CT-SEG dataset contains annotations for vertebra segmentation in CT of patients
with spinal metastases (pre and post radiotherapy).

It consists of 55 CT volumes with semantic labels for the individual vertebrae. The CT scans are distributed
as DICOM series and the labels as DICOM-SEG objects, which are converted and stored in hdf5 files by this module.
The vertebra labels are mapped to consistent semantic ids: 1-7: C1-C7, 8-19: T1-T12, 20-24: L1-L5, 25: S1.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/.

This dataset is from the publication https://doi.org/10.7937/kh36-ds04.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Spine-Mets-CT-SEG_v1_2024.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

VERTEBRA_IDS = {
    **{f"C{i}": i for i in range(1, 8)},
    **{f"T{i}": 7 + i for i in range(1, 13)},
    **{f"L{i}": 19 + i for i in range(1, 6)},
    "S1": 25,
}


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted by ascending patient z position.

    Returns the volume in Hounsfield units, the z position of each slice and the image orientation
    (DICOM 'ImageOrientationPatient').
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices.sort(key=lambda dcm: float(dcm.ImagePositionPatient[2]))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    z_positions = np.round([float(dcm.ImagePositionPatient[2]) for dcm in slices], 3)
    orientation = np.round([float(v) for v in slices[0].ImageOrientationPatient]).astype("int")
    return volume, z_positions, orientation


def _load_dicom_seg(seg_path, shape, z_positions, orientation):
    """Convert a DICOM-SEG object into a semantic label volume aligned with the reference CT volume."""
    import pydicom

    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:  # A segmentation with a single frame.
        frames = frames[None]

    segment_ids = {}
    for segment in seg.SegmentSequence:
        name = str(segment.SegmentLabel).replace("vertebra", "").strip()
        if name not in VERTEBRA_IDS:
            raise ValueError(f"Unknown segment label '{segment.SegmentLabel}' in {seg_path}.")
        segment_ids[int(segment.SegmentNumber)] = VERTEBRA_IDS[name]

    # The segmentation frames may use a different in-plane orientation than the CT slices,
    # in which case they have to be flipped to align them.
    seg_orientation = seg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
    seg_orientation = np.round([float(v) for v in seg_orientation]).astype("int")
    assert np.all(np.abs(seg_orientation) == np.abs(orientation)), f"Unexpected orientation in {seg_path}."
    flip_axes = []
    if np.any(seg_orientation[3:] != orientation[3:]):  # The direction of the rows differs.
        flip_axes.append(0)
    if np.any(seg_orientation[:3] != orientation[:3]):  # The direction of the columns differs.
        flip_axes.append(1)

    slice_ids = {z: i for i, z in enumerate(z_positions)}
    labels = np.zeros(shape, dtype="uint8")
    for frame, frame_group in zip(frames, seg.PerFrameFunctionalGroupsSequence):
        z = round(float(frame_group.PlanePositionSequence[0].ImagePositionPatient[2]), 3)
        segment_number = int(frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        mask = frame.astype("bool")
        if flip_axes:
            mask = np.flip(mask, axis=flip_axes)
        labels[slice_ids[z]][mask] = segment_ids[segment_number]

    return labels


def _preprocess_spine_mets(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    series_per_subject = defaultdict(dict)
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            series_per_subject[row["Subject ID"]][row["Modality"]] = os.path.join(dicom_dir, row["Series UID"])

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, series_dirs in tqdm(sorted(series_per_subject.items()), desc="Preprocess Spine-Mets-CT-SEG"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        volume, z_positions, orientation = _load_dicom_volume(series_dirs["CT"])
        seg_path = glob(os.path.join(series_dirs["SEG"], "*.dcm"))[0]
        labels = _load_dicom_seg(seg_path, volume.shape, z_positions, orientation)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_spine_mets_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Spine-Mets-CT-SEG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    # Download the DICOM series (CT and SEG) from the TCIA manifest.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "spine_mets_series")
    util.download_source_tcia(
        path=os.path.join(path, "Spine-Mets-CT-SEG_v1_2024.tcia"), url=URL, dst=dicom_dir,
        csv_filename=csv_path, download=download,
    )

    _preprocess_spine_mets(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_spine_mets_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Spine-Mets-CT-SEG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_spine_mets_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_spine_mets_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Spine-Mets-CT-SEG dataset for vertebra segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_spine_mets_paths(path, download)

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


def get_spine_mets_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Spine-Mets-CT-SEG dataloader for vertebra segmentation.

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
    dataset = get_spine_mets_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
