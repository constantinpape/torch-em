"""The Prostate-Anatomical-Edge-Cases dataset contains annotations for pelvic organ segmentation in
radiotherapy planning CT of prostate cancer patients with anatomical edge cases (e.g. hip prostheses,
bowel in the pelvis, large or small bladders, or rectal gas).

It consists of 131 CT volumes with semantic labels for 1: bladder, 2: prostate, 3: rectum, 4: left femoral head,
5: right femoral head. The CT scans are distributed as DICOM series and the labels as DICOM RTSTRUCT contours,
which are rasterized on the CT voxel grid (with `skimage.draw.polygon`, slice by slice) and stored together with the
CT volumes in hdf5 files by this module. The structures are rasterized in the order femoral heads, bladder, rectum,
prostate, so the prostate label takes precedence where contours overlap.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/prostate-anatomical-edge-cases/.

This dataset is from the publication https://doi.org/10.1002/mp.16537.
The data was released at https://doi.org/10.7937/013R-ZM07.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
from skimage.draw import polygon

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Prostate-Anatomical-Edge-Cases-May-2023-manifest.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

LABEL_IDS = {"Bladder": 1, "Prostate": 2, "Rectum": 3, "Femur_Head_L": 4, "Femur_Head_R": 5}

# The order in which the structures are rasterized. Later structures overwrite earlier ones where they overlap.
RASTERIZATION_ORDER = ["Femur_Head_L", "Femur_Head_R", "Bladder", "Rectum", "Prostate"]


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


def _load_rtstruct(rtstruct_path, shape, inverse_affine):
    """Rasterize the closed planar contours of a DICOM RTSTRUCT object on the voxel grid of the reference CT.

    The contour points are mapped from patient coordinates to voxel indices via the inverse affine of the CT,
    and each contour is filled with `skimage.draw.polygon` in the slice it belongs to.
    """
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path)
    roi_names = {int(roi.ROINumber): str(roi.ROIName) for roi in rtstruct.StructureSetROISequence}
    contour_sequences = {
        roi_names[int(roi_contour.ReferencedROINumber)]: roi_contour.ContourSequence
        for roi_contour in rtstruct.ROIContourSequence if "ContourSequence" in roi_contour
    }

    labels = np.zeros(shape, dtype="uint8")
    for structure in RASTERIZATION_ORDER:
        if structure not in contour_sequences:
            continue
        for contour in contour_sequences[structure]:
            if contour.ContourGeometricType != "CLOSED_PLANAR":
                continue
            points = np.array([float(v) for v in contour.ContourData]).reshape(-1, 3)
            indices = inverse_affine[:3, :3] @ points.T + inverse_affine[:3, 3:]
            z = int(np.round(indices[0].mean()))
            assert np.abs(indices[0] - z).max() < 0.1, f"Non-planar contour in {rtstruct_path}."
            if z < 0 or z >= shape[0]:
                continue
            rows, cols = polygon(indices[1], indices[2], shape=shape[1:])
            labels[z, rows, cols] = LABEL_IDS[structure]

    return labels


def _preprocess_prostate_edge_cases(dicom_dir, csv_path, preprocessed_dir):
    import h5py
    import pydicom

    with open(csv_path, "r") as f:
        rtstruct_series = {
            row["Subject ID"]: row["Series UID"] for row in csv.DictReader(f) if row["Modality"] == "RTSTRUCT"
        }

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, rtstruct_uid in tqdm(
        sorted(rtstruct_series.items()), desc="Preprocess Prostate-Anatomical-Edge-Cases"
    ):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        # The structure set references the CT series it was created for.
        rtstruct_path = glob(os.path.join(dicom_dir, rtstruct_uid, "*.dcm"))[0]
        rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        referenced_study = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        ct_uid = referenced_study.RTReferencedSeriesSequence[0].SeriesInstanceUID
        volume, affine = _load_dicom_volume(os.path.join(dicom_dir, ct_uid))
        labels = _load_rtstruct(rtstruct_path, volume.shape, np.linalg.inv(affine))

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_prostate_edge_cases_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Prostate-Anatomical-Edge-Cases dataset.

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

    # Download the DICOM series (CT and RTSTRUCT) from the TCIA manifest.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "prostate_edge_cases_series")
    util.download_source_tcia(
        path=os.path.join(path, "Prostate-Anatomical-Edge-Cases-May-2023-manifest.tcia"), url=URL, dst=dicom_dir,
        csv_filename=csv_path, download=download,
    )

    _preprocess_prostate_edge_cases(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_prostate_edge_cases_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Prostate-Anatomical-Edge-Cases data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_prostate_edge_cases_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_prostate_edge_cases_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Prostate-Anatomical-Edge-Cases dataset for pelvic organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_prostate_edge_cases_paths(path, download)

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


def get_prostate_edge_cases_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Prostate-Anatomical-Edge-Cases dataloader for pelvic organ segmentation.

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
    dataset = get_prostate_edge_cases_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
