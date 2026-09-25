"""The ReMIND dataset contains annotations for brain tumor and resection segmentation in pre- and
intra-operative MRI of patients who underwent image-guided tumor resection.

The full collection consists of 114 patients with 369 pre-operative MRI, 301 intra-operative MRI and 320 3D
intra-operative ultrasound series. Segmentations are only provided for 221 MRI series (the ultrasound series are not
annotated), so this module downloads and converts only these MRI series and their segmentations (ca. 8.5 GB instead
of the full 43.5 GB collection). The MRI is distributed as DICOM series and the segmentations as DICOM-SEG objects,
which are converted and stored in hdf5 files by this module. The segmentations of all structures annotated for a MRI
series are merged into a single label volume with the semantic ids 1: cerebrum, 2: ventricles,
3: previous resection cavity, 4: tumor, 5: tumor target, 6: residual tumor (structures with a higher id overwrite
structures with a lower id where they overlap). Only the structures deemed necessary for the surgery are segmented,
so most volumes contain only a subset of these structures.

The MRI sequences are exposed via the 'modality' argument: 't1c' (contrast-enhanced T1), 't1' (native T1,
including MP2RAGE), 't2' and 'flair'. The 'study' argument selects pre-operative or intra-operative MRI.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/remind/.

This dataset is from the publication https://doi.org/10.1038/s41597-024-03295-z.
The data was released at https://doi.org/10.7937/3RAG-D070.
Please cite it if you use this dataset in your research.
"""

import os
import csv
import tempfile
from glob import glob
from tqdm import tqdm
from shutil import copyfileobj
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/ReMIND-Manifest-Sept-2023.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

LABEL_IDS = {
    "cerebrum": 1,
    "ventricles": 2,
    "previous_resection_cavity": 3,
    "tumor": 4,
    "tumor_target": 5,
    "tumor_residual": 6,
}

MODALITIES = ["t1c", "t1", "t2", "flair"]


def _get_modality(series_description):
    """Derive the MRI sequence from the series description, e.g. '3D_AX_T1_postcontrast' or '2D_AX_T2_FLAIR'."""
    if "postcontrast" in series_description:
        return "t1c"
    elif "T1" in series_description:
        return "t1"
    elif "FLAIR" in series_description:
        return "flair"
    elif "T2" in series_description:
        return "t2"
    raise ValueError(f"Could not derive the modality from the series description '{series_description}'.")


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted along the slice normal.

    Returns the volume and the affine matrix that maps voxel indices (z, y, x) to DICOM patient coordinates.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    orientation = np.array([float(v) for v in slices[0].ImageOrientationPatient])
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda dcm: np.dot([float(v) for v in dcm.ImagePositionPatient], normal))

    volume = np.stack([dcm.pixel_array for dcm in slices])
    if "RescaleSlope" in slices[0]:
        volume = volume.astype("float32") * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
        volume = np.round(volume).astype("int16")

    positions = np.array([[float(v) for v in dcm.ImagePositionPatient] for dcm in slices])
    spacing = [float(v) for v in slices[0].PixelSpacing]  # The spacing between rows and between columns.
    if len(slices) > 1:
        slice_step = (positions[-1] - positions[0]) / (len(slices) - 1)
    else:
        slice_step = normal * float(slices[0].SliceThickness)

    affine = np.eye(4)
    affine[:3, 0] = slice_step
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


def _download_series(series_uids, dicom_dir):
    """Download DICOM series from TCIA via the NBIA REST API into '<dicom_dir>/<SeriesInstanceUID>/'."""
    os.makedirs(dicom_dir, exist_ok=True)
    for uid in tqdm(series_uids, desc=f"Download {len(series_uids)} series from TCIA to {dicom_dir}"):
        series_dir = os.path.join(dicom_dir, uid)
        if os.path.exists(series_dir):  # This series has been downloaded already.
            continue

        # The series is downloaded as a zip archive, which is extracted to a temporary folder
        # and only moved to the final location once it is complete.
        with tempfile.TemporaryDirectory(dir=dicom_dir) as tmp_dir:
            zip_path = os.path.join(tmp_dir, "series.zip")
            with requests.get(util.NBIA_API_URL + "getImage", params={"SeriesInstanceUID": uid}, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    copyfileobj(r.raw, f)
            tmp_series_dir = os.path.join(tmp_dir, "series")
            util.unzip(zip_path, tmp_series_dir)
            os.rename(tmp_series_dir, series_dir)


def _get_series_metadata(csv_path, download):
    """Fetch the metadata of all series in the ReMIND collection from TCIA and pair each segmentation
    with the MRI series it references (via the study and the series description, e.g.
    'tumor seg - MR ref: 3D_AX_T1_postcontrast').
    """
    if not os.path.exists(csv_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {csv_path}, but download was set to False.")
        response = requests.get(util.NBIA_API_URL + "getSeries", params={"Collection": "ReMIND"})
        response.raise_for_status()
        metadata = response.json()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in metadata for key in row.keys()}))
            writer.writeheader()
            writer.writerows(metadata)

    with open(csv_path, "r") as f:
        metadata = list(csv.DictReader(f))

    mr_series = {}
    for row in metadata:
        if row["Modality"] == "MR":
            mr_series[(row["StudyInstanceUID"], row["SeriesDescription"])] = row

    segmentations = {}
    for row in metadata:
        if row["Modality"] != "SEG":
            continue
        reference = row["SeriesDescription"].split("MR ref:")[-1].strip()
        mr_row = mr_series[(row["StudyInstanceUID"], reference)]
        segmentations.setdefault(mr_row["SeriesInstanceUID"], (mr_row, []))[1].append(row)

    return segmentations


def _preprocess_remind(dicom_dir, segmentations, preprocessed_dir):
    import h5py

    os.makedirs(preprocessed_dir, exist_ok=True)
    for mr_uid, (mr_row, seg_rows) in tqdm(sorted(segmentations.items()), desc="Preprocess ReMIND"):
        study = mr_row["StudyDesc"].lower()
        modality = _get_modality(mr_row["SeriesDescription"])
        fname = f"{mr_row['PatientID']}_{study}_{modality}_{mr_row['SeriesNumber']}.h5"
        out_path = os.path.join(preprocessed_dir, fname)
        if os.path.exists(out_path):
            continue

        volume, affine = _load_dicom_volume(os.path.join(dicom_dir, mr_uid))

        # Each structure is stored in a separate DICOM-SEG object (possibly on a cropped or finer grid), which is
        # resampled onto the MRI grid. The structures are written in the order of their label ids, so that more
        # specific structures (e.g. the tumor) overwrite larger structures (e.g. the cerebrum) where they overlap.
        labels = np.zeros(volume.shape, dtype="uint8")
        structures = {row["SeriesDescription"].split(" seg")[0]: row["SeriesInstanceUID"] for row in seg_rows}
        for structure, seg_uid in sorted(structures.items(), key=lambda item: LABEL_IDS[item[0]]):
            seg_labels, seg_affine = _load_dicom_seg(glob(os.path.join(dicom_dir, seg_uid, "*.dcm"))[0])
            mask = _resample_labels(seg_labels, seg_affine, volume.shape, affine) > 0
            labels[mask] = LABEL_IDS[structure]

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
            f.attrs["series_description"] = mr_row["SeriesDescription"]
            f.attrs["structures"] = sorted(structures.keys())


def get_remind_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ReMIND dataset.

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

    # Download the manifest for reference and the series metadata of the collection. Only the MRI series
    # with segmentations and the corresponding DICOM-SEG series are downloaded.
    manifest_path = os.path.join(path, "ReMIND-Manifest-Sept-2023.tcia")
    util.download_source(path=manifest_path, url=URL, download=download, checksum=CHECKSUM)
    segmentations = _get_series_metadata(os.path.join(path, "remind_series.csv"), download)

    dicom_dir = os.path.join(path, "dicom")
    series_uids = sorted(segmentations.keys())
    series_uids += sorted(row["SeriesInstanceUID"] for _, seg_rows in segmentations.values() for row in seg_rows)
    if not all(os.path.exists(os.path.join(dicom_dir, uid)) for uid in series_uids):
        if not download:
            raise RuntimeError(f"Cannot find the data at {dicom_dir}, but download was set to False.")
        _download_series(series_uids, dicom_dir)

    _preprocess_remind(dicom_dir, segmentations, preprocessed_dir)
    return preprocessed_dir


def get_remind_paths(
    path: Union[os.PathLike, str],
    modality: Optional[Literal["t1c", "t1", "t2", "flair"]] = None,
    study: Optional[Literal["preop", "intraop"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the ReMIND data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The MRI sequence. One of 't1c', 't1', 't2' or 'flair'. If None, all sequences are returned.
        study: The choice of study. Either 'preop' or 'intraop'. If None, both studies are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_remind_data(path, download)

    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {MODALITIES}.")
    if study is not None and study not in ["preop", "intraop"]:
        raise ValueError(f"'{study}' is not a valid study. Please choose one of 'preop' or 'intraop'.")

    pattern = f"*_{'*' if study is None else study}_{'*' if modality is None else modality}_*.h5"
    volume_paths = natsorted(glob(os.path.join(data_dir, pattern)))
    return volume_paths


def get_remind_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["t1c", "t1", "t2", "flair"]] = None,
    study: Optional[Literal["preop", "intraop"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ReMIND dataset for brain tumor and resection segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. One of 't1c', 't1', 't2' or 'flair'. If None, all sequences are returned.
        study: The choice of study. Either 'preop' or 'intraop'. If None, both studies are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_remind_paths(path, modality, study, download)

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


def get_remind_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["t1c", "t1", "t2", "flair"]] = None,
    study: Optional[Literal["preop", "intraop"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ReMIND dataloader for brain tumor and resection segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. One of 't1c', 't1', 't2' or 'flair'. If None, all sequences are returned.
        study: The choice of study. Either 'preop' or 'intraop'. If None, both studies are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_remind_dataset(path, patch_shape, modality, study, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
