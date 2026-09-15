"""The Pediatric-CT-SEG dataset contains annotations for organ segmentation in pediatric CT.

It consists of 359 CT volumes (chest-abdomen-pelvis, patients aged 5 days to 16 years) with contours of up to
29 organs, which are distributed as DICOM RTSTRUCT objects. The CT scans are stacked into volumes, the contours are
rasterized on the CT grid and both are stored together in hdf5 files by this module.
The organs are mapped to the following semantic ids:
1: adrenal left, 2: adrenal right, 3: bladder, 4: breast left, 5: breast right, 6: duodenum, 7: esophagus,
8: femoral head left, 9: femoral head right, 10: gall bladder, 11: gonads, 12: heart, 13: kidney left,
14: kidney right, 15: large intestine, 16: liver, 17: pancreas, 18: prostate, 19: rectum, 20: small intestine,
21: spinal canal, 22: spleen, 23: stomach, 24: thymus, 25: uterocervix, 26: lung left, 27: lung right, 28: skin,
29: bones.
Some volumes are missing structures that are outside of the scan range or could not be identified reliably.
The contours of overlapping structures are rasterized in a fixed order, so that the skin and bone labels are
overwritten by the organs. Additional contours that do not belong to the 29 organs (e.g. 'BODY' or a
'Horseshoe Kidney' in one patient) are not included in the labels.

NOTE: This requires the pydicom and opencv python packages.

The dataset is located at https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/.

This dataset is from the publication https://doi.org/10.1002/mp.15485.
The data was released at https://doi.org/10.7937/TCIA.X0H0-1706.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from warnings import warn
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Pediatric-CT-SEG-Mar-22-2022-manifest.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

NUM_VOLUMES = 359

# The organ ids follow the ROI numbering of the RTSTRUCT files.
ORGAN_IDS = {
    "adrenal_left": 1, "adrenal_right": 2, "bladder": 3, "breast_left": 4, "breast_right": 5, "duodenum": 6,
    "esophagus": 7, "femoral_head_left": 8, "femoral_head_right": 9, "gall_bladder": 10, "gonads": 11, "heart": 12,
    "kidney_left": 13, "kidney_right": 14, "large_intestine": 15, "liver": 16, "pancreas": 17, "prostate": 18,
    "rectum": 19, "small_intestine": 20, "spinal_canal": 21, "spleen": 22, "stomach": 23, "thymus": 24,
    "uterocervix": 25, "lung_left": 26, "lung_right": 27, "skin": 28, "bones": 29,
}

# The ROI names used in the RTSTRUCT files (including the alternative spellings in a few patients).
ROI_NAMES = {
    "Adrenal Left": "adrenal_left", "Lt Adrenal": "adrenal_left", "Adrenal Right": "adrenal_right",
    "Rt Adrenal": "adrenal_right", "Bladder": "bladder", "Breast Left": "breast_left", "Breast Right": "breast_right",
    "Duodenum": "duodenum", "Esophagus": "esophagus", "Femoral Head Lef": "femoral_head_left",
    "Femoral Head Rig": "femoral_head_right", "Gall Bladder": "gall_bladder", "Gonads": "gonads", "Heart": "heart",
    "Kidney Left": "kidney_left", "Kidney Right": "kidney_right", "Large Intestine": "large_intestine",
    "Liver": "liver", "Pancreas": "pancreas", "Prostate": "prostate", "Rectum": "rectum",
    "Small Intestine": "small_intestine", "Spinal Canal": "spinal_canal", "Spleen": "spleen", "Stomach": "stomach",
    "Thymus": "thymus", "UteroCervix": "uterocervix", "Lung_L": "lung_left", "Lung_R": "lung_right", "Skin": "skin",
    "Bones": "bones",
}

# The order in which the structures are rasterized: the organs overwrite the skin and the bones.
PAINT_ORDER = ["skin", "bones"] + [name for name in ORGAN_IDS if name not in ("skin", "bones")]


def _find_dicom_series(path):
    """Find all DICOM series below the given folder and group them by patient and modality.

    This works for the folder layout of the NBIA Data Retriever ('<Collection>/<Patient>/<Study>/<Series>/*.dcm')
    as well as for the layout of `util.download_source_tcia` ('<dst>/<SeriesInstanceUID>/*.dcm').
    """
    import pydicom

    series = defaultdict(dict)
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "preprocessed"]
        dcm_files = [f for f in files if f.endswith(".dcm")]
        if not dcm_files:
            continue
        header = pydicom.dcmread(os.path.join(root, dcm_files[0]), specific_tags=["PatientID", "Modality"])
        series[str(header.PatientID)][str(header.Modality)] = root
    return series


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted by ascending patient z position.

    Returns the volume in Hounsfield units and the geometry needed to map patient coordinates to pixel indices.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices.sort(key=lambda dcm: float(dcm.ImagePositionPatient[2]))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    geometry = {
        "sop_uids": {str(dcm.SOPInstanceUID): i for i, dcm in enumerate(slices)},
        "z_positions": {round(float(dcm.ImagePositionPatient[2]), 2): i for i, dcm in enumerate(slices)},
        "origins": np.array([[float(v) for v in dcm.ImagePositionPatient] for dcm in slices]),
        "orientation": np.array([float(v) for v in slices[0].ImageOrientationPatient]),
        "spacing": np.array([float(v) for v in slices[0].PixelSpacing]),
    }
    return volume, geometry


def _select_rtstruct(rtstruct_dir):
    """Select the RTSTRUCT file of a series.

    103 series contain two RTSTRUCT files, which only differ in their skin contours. We use the most recent one.
    """
    import pydicom

    rtstruct_paths = glob(os.path.join(rtstruct_dir, "*.dcm"))
    if len(rtstruct_paths) == 1:
        return rtstruct_paths[0]
    creation_dates = [
        str(pydicom.dcmread(p, specific_tags=["InstanceCreationDate"]).InstanceCreationDate) for p in rtstruct_paths
    ]
    return rtstruct_paths[int(np.argmax(creation_dates))]


def _rasterize_rtstruct(rtstruct_path, shape, geometry):
    """Rasterize the closed planar contours of a DICOM RTSTRUCT on the grid of the reference CT volume.

    Each contour is mapped to its CT slice via the referenced SOP instance (or its z position) and filled with
    `cv2.fillPoly` (as in rt-utils). Multiple contours of a structure on the same slice are combined with XOR,
    so that inner contours are treated as holes. The structures are then painted in the order given by `PAINT_ORDER`.
    """
    import cv2
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path)
    roi_names = {int(roi.ROINumber): str(roi.ROIName) for roi in rtstruct.StructureSetROISequence}

    row_dir, col_dir = geometry["orientation"][3:], geometry["orientation"][:3]
    row_spacing, col_spacing = geometry["spacing"]

    masks = {}
    for roi_contour in rtstruct.ROIContourSequence:
        roi_name = roi_names[int(roi_contour.ReferencedROINumber)]
        if roi_name not in ROI_NAMES or "ContourSequence" not in roi_contour:
            continue
        organ = ROI_NAMES[roi_name]
        mask = masks.setdefault(organ, np.zeros(shape, dtype="bool"))

        for contour in roi_contour.ContourSequence:
            assert contour.ContourGeometricType == "CLOSED_PLANAR", f"Unexpected contour type in {rtstruct_path}."
            points = np.array(contour.ContourData, dtype="float64").reshape(-1, 3)

            if "ContourImageSequence" in contour:
                z = geometry["sop_uids"].get(str(contour.ContourImageSequence[0].ReferencedSOPInstanceUID))
            else:
                z = geometry["z_positions"].get(round(float(points[0, 2]), 2))
            if z is None:
                warn(f"Skipping a contour of '{roi_name}' in {rtstruct_path}, which does not match a CT slice.")
                continue

            offsets = points - geometry["origins"][z]
            rows = offsets @ row_dir / row_spacing
            cols = offsets @ col_dir / col_spacing
            contour_mask = np.zeros(shape[1:], dtype="uint8")
            cv2.fillPoly(contour_mask, [np.round(np.stack([cols, rows], axis=1)).astype("int32")], 1)
            mask[z] ^= contour_mask.astype("bool")

    labels = np.zeros(shape, dtype="uint8")
    for organ in PAINT_ORDER:
        if organ in masks:
            labels[masks[organ]] = ORGAN_IDS[organ]
    return labels


def _preprocess_pediatric_ct_seg(series, preprocessed_dir):
    import h5py

    os.makedirs(preprocessed_dir, exist_ok=True)
    for patient_id, series_dirs in tqdm(sorted(series.items()), desc="Preprocess Pediatric-CT-SEG"):
        out_path = os.path.join(preprocessed_dir, f"{patient_id}.h5")
        if os.path.exists(out_path):
            continue

        volume, geometry = _load_dicom_volume(series_dirs["CT"])
        rtstruct_path = _select_rtstruct(series_dirs["RTSTRUCT"])
        labels = _rasterize_rtstruct(rtstruct_path, volume.shape, geometry)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_pediatric_ct_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Pediatric-CT-SEG dataset.

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

    # Download the DICOM series (CT and RTSTRUCT) from the TCIA manifest, unless they are present already
    # (e.g. downloaded with the NBIA Data Retriever).
    series = _find_dicom_series(path)
    if not series:
        util.download_source_tcia(
            path=os.path.join(path, "Pediatric-CT-SEG-Mar-22-2022-manifest.tcia"), url=URL,
            dst=os.path.join(path, "dicom"), csv_filename=os.path.join(path, "pediatric_ct_seg_series"),
            download=download,
        )
        series = _find_dicom_series(path)

    missing = [pid for pid, series_dirs in series.items() if {"CT", "RTSTRUCT"} - set(series_dirs)]
    assert not missing, f"The CT or RTSTRUCT series is missing for the patients {missing}."

    _preprocess_pediatric_ct_seg(series, preprocessed_dir)
    return preprocessed_dir


def get_pediatric_ct_seg_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Pediatric-CT-SEG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_pediatric_ct_seg_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_pediatric_ct_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Pediatric-CT-SEG dataset for organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_pediatric_ct_seg_paths(path, download)

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


def get_pediatric_ct_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Pediatric-CT-SEG dataloader for organ segmentation.

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
    dataset = get_pediatric_ct_seg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
