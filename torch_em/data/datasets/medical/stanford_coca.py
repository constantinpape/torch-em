"""The COCA dataset contains annotations for coronary artery calcium segmentation in ECG-gated
cardiac CT.

The dataset is released by the Stanford Center for Artificial Intelligence in Medicine and Imaging
(AIMI) as "COCA - Coronary Calcium and chest CT's". It comprises two parts: gated coronary CT DICOM
series with per-slice coronary artery calcium ROI annotations (stored as a binary property list
'.xml' file per patient), and non-gated chest CT DICOM series with only a coronary artery calcium
score (no pixel-level annotation). This module only supports the gated part, since only that part
has pixel-level segmentation masks; the non-gated part is score-only and does not fit torch_em's
segmentation dataset pattern.

The ROI file lists, per slice ('ImageIndex') and per coronary artery, one or more closed polygons
('Point_px', in pixel coordinates) outlining the calcified regions. The label ids used here are:
- 0: background, 1: right coronary artery, 2: left anterior descending artery,
- 3: left coronary artery, 4: left circumflex artery
(see `LABEL_IDS`; the artery names and this correspondence were taken from a community reimplementation
of the dataset loading, https://github.com/msingh9/cs230-Coronary-Calcium-Scoring-/blob/master/code/my_lib.py,
since the ROI file format itself is undocumented by Stanford AIMI).

NOTE: The slice that a set of ROIs belongs to is identified in the ROI file only by an integer
'ImageIndex', with no reference to a DICOM SOP instance UID or file name. This module assumes that
'ImageIndex' is the 0-based position of the slice when the series is sorted by 'InstanceNumber'.
This assumption could not be verified against the real data, since the dataset requires registration
and a signed data use agreement, so it was not accessible while writing this module. Please verify
the resulting masks against the DICOM series (e.g. by overlaying them) before relying on this dataset.

NOTE: The dataset requires registration and cannot be downloaded automatically. Please follow these steps:
- Visit https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct and follow the link to the
  dataset on Stanford's Redivis platform (https://stanford.redivis.com/datasets/1vm9-30b7p5srg).
- Log in with your institutional account, fill in your contact details and accept the Stanford University
  Dataset Research Use Agreement.
- Download the 'Gated_release_final' folder (e.g. with the Redivis download tool) and place it such that
  '<path>/Gated_release_final/patient/<patient_id>/.../*.dcm' (the DICOM series) and
  '<path>/Gated_release_final/calcium_xml/<patient_id>.xml' (the ROI annotations) exist.

The dataset is located at https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct
(DOI: https://doi.org/10.71718/ge5g-ds80) and is released under the Stanford University Dataset
Research Use Agreement.

This dataset is from the publications https://doi.org/10.1007/s10554-019-01710-8 and
https://doi.org/10.1148/ryai.2020190004. Please cite them if you use this dataset in your research.

NOTE: The DICOM and ROI parsing requires 'pydicom'. Install it with 'pip install pydicom'.
"""

import os
import plistlib
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {
    "background": 0,
    "right_coronary_artery": 1,
    "left_anterior_descending_artery": 2,
    "left_coronary_artery": 3,
    "left_circumflex_artery": 4,
}

ROI_NAME_TO_LABEL_ID = {
    "Right Coronary Artery": LABEL_IDS["right_coronary_artery"],
    "Left Anterior Descending Artery": LABEL_IDS["left_anterior_descending_artery"],
    "Left Coronary Artery": LABEL_IDS["left_coronary_artery"],
    "Left Circumflex Artery": LABEL_IDS["left_circumflex_artery"],
}


def _load_gated_series(patient_dir):
    import pydicom

    dcm_paths = natsorted(glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True))
    assert len(dcm_paths) > 0, f"Could not find any DICOM files in '{patient_dir}'."

    slices = [pydicom.dcmread(p) for p in dcm_paths]
    slices.sort(key=lambda dcm: int(dcm.InstanceNumber))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    volume = np.round(volume * slope + intercept).astype("int16")

    return volume


def _parse_calcium_xml(xml_path):
    with open(xml_path, "rb") as f:
        annotations = plistlib.load(f)

    rois_per_slice = {}
    for image in annotations["Images"]:
        image_index = image["ImageIndex"]
        for roi in image["ROIs"]:
            points = roi.get("Point_px", [])
            if len(points) == 0:
                continue

            label_id = ROI_NAME_TO_LABEL_ID.get(roi["Name"])
            if label_id is None:
                continue

            pixels = np.array([[float(v) for v in point.strip("()").split(",")] for point in points])
            rois_per_slice.setdefault(image_index, []).append((label_id, pixels))

    return rois_per_slice


def _rasterize_labels(shape, rois_per_slice):
    from skimage.draw import polygon

    labels = np.zeros(shape, dtype="uint8")
    for slice_id, rois in rois_per_slice.items():
        if slice_id >= shape[0]:
            continue
        for label_id, pixels in rois:
            rr, cc = polygon(pixels[:, 1], pixels[:, 0], shape=shape[1:])
            labels[slice_id, rr, cc] = label_id

    return labels


def _preprocess_inputs(path, gated_dir):
    import h5py

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    xml_paths = natsorted(glob(os.path.join(gated_dir, "calcium_xml", "*.xml")))
    for xml_path in tqdm(xml_paths, desc="Preprocessing the COCA gated CT scans"):
        patient_id = os.path.basename(xml_path)[:-len(".xml")]
        volume_path = os.path.join(preprocessed_dir, f"{patient_id}.h5")
        if os.path.exists(volume_path):
            continue

        patient_dir = os.path.join(gated_dir, "patient", patient_id)
        if not os.path.isdir(patient_dir):
            continue

        raw = _load_gated_series(patient_dir)
        rois_per_slice = _parse_calcium_xml(xml_path)
        labels = _rasterize_labels(raw.shape, rois_per_slice)

        # The file is written to a temporary path first, so that an interrupted run does not leave a corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)

    return preprocessed_dir


def get_stanford_coca_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the COCA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is preprocessed.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    if download:
        raise RuntimeError(
            "Download is set to True, but 'torch_em' cannot download the COCA dataset automatically: "
            "it requires registration and a signed Stanford University Dataset Research Use Agreement. "
            "See 'torch_em.data.datasets.medical.stanford_coca' for the manual download instructions."
        )

    xml_dirs = [p for p in glob(os.path.join(path, "**", "calcium_xml"), recursive=True) if os.path.isdir(p)]
    if len(xml_dirs) == 0:
        raise RuntimeError(
            f"Could not find the COCA 'Gated_release_final' folder at '{path}'. This dataset requires "
            "registration and a signed Stanford University Dataset Research Use Agreement, so it cannot be "
            "downloaded automatically. Please follow these steps: "
            "1) Visit https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct and follow the link "
            "to the dataset on Stanford's Redivis platform (https://stanford.redivis.com/datasets/1vm9-30b7p5srg). "
            "2) Log in with your institutional account, fill in your contact details and accept the Stanford "
            "University Dataset Research Use Agreement. "
            f"3) Download the 'Gated_release_final' folder and place it such that "
            f"'{path}/Gated_release_final/patient/<patient_id>/.../*.dcm' and "
            f"'{path}/Gated_release_final/calcium_xml/<patient_id>.xml' exist."
        )

    gated_dir = os.path.dirname(xml_dirs[0])
    return _preprocess_inputs(path, gated_dir)


def get_stanford_coca_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the COCA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_stanford_coca_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."
    return volume_paths, volume_paths


def get_stanford_coca_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the COCA dataset for coronary artery calcium segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_stanford_coca_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_stanford_coca_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the COCA dataloader for coronary artery calcium segmentation.

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
    dataset = get_stanford_coca_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
