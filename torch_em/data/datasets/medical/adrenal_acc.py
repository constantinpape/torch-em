"""The Adrenal-ACC-Ki67-Seg dataset contains annotations for adrenocortical carcinoma segmentation
in contrast-enhanced abdominal CT.

It consists of 53 CT volumes (one segmented CT series per patient, the collection also contains 71 additional
unsegmented CT series of the same patients) with binary tumor labels (1: adrenal tumor). The CT scans are
distributed as DICOM series and the labels as DICOM-SEG objects, which are converted and stored in hdf5 files
by this module.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/adrenal-acc-ki67-seg/.

This dataset is from the publication https://doi.org/10.1016/j.crad.2020.01.012.
The data was released at https://doi.org/10.7937/1FPG-VM46.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Adrenal-ACC-Ki67-Seg_v1.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

LABEL_IDS = {"tumor": 1}


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted along the slice normal.

    Returns the volume in Hounsfield units and the affine matrix that maps voxel indices (z, y, x)
    to DICOM patient coordinates.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    orientation = np.array([float(v) for v in slices[0].ImageOrientationPatient])
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda dcm: np.dot([float(v) for v in dcm.ImagePositionPatient], normal))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    positions = np.array([[float(v) for v in dcm.ImagePositionPatient] for dcm in slices])
    spacing = [float(v) for v in slices[0].PixelSpacing]  # The spacing between rows and between columns.
    affine = np.eye(4)
    affine[:3, 0] = (positions[-1] - positions[0]) / (len(slices) - 1)
    affine[:3, 1] = col_dir * spacing[0]
    affine[:3, 2] = row_dir * spacing[1]
    affine[:3, 3] = positions[0]
    return volume, affine


def _load_dicom_seg(seg_path):
    """Load a DICOM-SEG object as a label volume with axes (z, y, x), where the segment number is used as label id.

    Returns the label volume and the affine matrix that maps its voxel indices to DICOM patient coordinates.
    """
    import pydicom

    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:  # A segmentation with a single frame.
        frames = frames[None]

    shared_group = seg.SharedFunctionalGroupsSequence[0]
    orientation = np.array([float(v) for v in shared_group.PlaneOrientationSequence[0].ImageOrientationPatient])
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)
    pixel_measures = shared_group.PixelMeasuresSequence[0]
    spacing = [float(v) for v in pixel_measures.PixelSpacing]  # The spacing between rows and between columns.

    # The frames may be stored in arbitrary order and frames without any foreground may be skipped,
    # so the position of each frame along the slice normal is derived from its patient position.
    frame_groups = seg.PerFrameFunctionalGroupsSequence
    positions = np.array([[float(v) for v in g.PlanePositionSequence[0].ImagePositionPatient] for g in frame_groups])
    projections = positions @ normal
    if "SpacingBetweenSlices" in pixel_measures:
        slice_spacing = float(pixel_measures.SpacingBetweenSlices)
    elif len(projections) > 1:
        slice_spacing = np.min(np.diff(np.unique(np.round(projections, 3))))
    else:
        slice_spacing = float(pixel_measures.SliceThickness)
    slice_ids = np.round((projections - projections.min()) / slice_spacing).astype("int")

    labels = np.zeros((slice_ids.max() + 1, seg.Rows, seg.Columns), dtype="uint8")
    for frame, frame_group, slice_id in zip(frames, frame_groups, slice_ids):
        segment_number = int(frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        labels[slice_id][frame.astype("bool")] = segment_number

    affine = np.eye(4)
    affine[:3, 0] = normal * slice_spacing
    affine[:3, 1] = col_dir * spacing[0]
    affine[:3, 2] = row_dir * spacing[1]
    affine[:3, 3] = positions[np.argmin(projections)]
    return labels, affine


def _resample_labels(labels, affine, target_shape, target_affine):
    """Resample a label volume onto the voxel grid of a reference image with nearest neighbor interpolation.

    This is exact if both volumes are stored on the same grid (e.g. a DICOM-SEG object stored on a cropped
    grid of the reference image) and downsamples segmentations that are stored on a finer grid.
    """
    to_label_index = np.linalg.inv(affine) @ target_affine
    resampled = np.zeros(target_shape, dtype=labels.dtype)
    yy, xx = np.meshgrid(np.arange(target_shape[1]), np.arange(target_shape[2]), indexing="ij")
    for z in range(target_shape[0]):
        target_indices = np.stack([np.full(yy.size, z), yy.ravel(), xx.ravel(), np.ones(yy.size)])
        indices = np.round(to_label_index[:3] @ target_indices).astype("int")
        valid = np.all((indices >= 0) & (indices < np.array(labels.shape)[:, None]), axis=0)
        resampled[z].flat[valid] = labels[tuple(indices[:, valid])]
    return resampled


def _preprocess_adrenal_acc(dicom_dir, csv_path, preprocessed_dir):
    import h5py
    import pydicom

    with open(csv_path, "r") as f:
        seg_series = {row["Subject ID"]: row["Series UID"] for row in csv.DictReader(f) if row["Modality"] == "SEG"}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, seg_uid in tqdm(sorted(seg_series.items()), desc="Preprocess Adrenal-ACC-Ki67-Seg"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        # The segmentation references the CT series it was created for.
        seg_path = glob(os.path.join(dicom_dir, seg_uid, "*.dcm"))[0]
        ct_uid = pydicom.dcmread(seg_path, stop_before_pixels=True).ReferencedSeriesSequence[0].SeriesInstanceUID
        volume, affine = _load_dicom_volume(os.path.join(dicom_dir, ct_uid))
        seg_labels, seg_affine = _load_dicom_seg(seg_path)
        assert seg_labels.max() == 1, f"Expected a single segment in {seg_path}."
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, affine) * LABEL_IDS["tumor"]

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_adrenal_acc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Adrenal-ACC-Ki67-Seg dataset.

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
    csv_path = os.path.join(path, "adrenal_acc_series")
    util.download_source_tcia(
        path=os.path.join(path, "Adrenal-ACC-Ki67-Seg_v1.tcia"), url=URL, dst=dicom_dir,
        csv_filename=csv_path, download=download,
    )

    _preprocess_adrenal_acc(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_adrenal_acc_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Adrenal-ACC-Ki67-Seg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_adrenal_acc_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_adrenal_acc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Adrenal-ACC-Ki67-Seg dataset for adrenal tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_adrenal_acc_paths(path, download)

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


def get_adrenal_acc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Adrenal-ACC-Ki67-Seg dataloader for adrenal tumor segmentation.

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
    dataset = get_adrenal_acc_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
