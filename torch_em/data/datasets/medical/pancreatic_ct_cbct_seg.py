"""The Pancreatic-CT-CBCT-SEG dataset contains annotations for organs at risk in planning CT of
pancreatic cancer patients undergoing ablative radiation therapy.

It consists of 40 patients, each with a breath-hold planning CT and two cone-beam CTs (CBCTs) from
their treatment. This module only uses the planning CT, which is distributed as a DICOM series with a
DICOM RTSTRUCT contour of the organs at risk. The contours are rasterized onto the CT grid (see
`torch_em.data.datasets.util.rasterize_rtstruct`) and CT and labels are stored in hdf5 files.
The semantic label ids are: 1: small bowel, 2: stomach and duodenum. The RTSTRUCT files also contain
lung and volume-of-interest contours that are not used here, since they are not organ-at-risk targets.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/.

This dataset is from the publication https://doi.org/10.1038/s41597-022-01758-9.
The data was released at https://doi.org/10.7937/TCIA.ESHQ-4D90.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Sequence

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.transform.generic import Compose

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Pancreatic-CT-CBCT-SEG_v2_20220823.tcia"

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None

STRUCTURE_IDS = {"bowel": 1, "stomach_duodenum": 2}


class SelectStructures:
    """Label transform that keeps only the given label ids and sets all other labels to background.

    Args:
        label_ids: The label ids to keep.
    """
    def __init__(self, label_ids: Sequence[int]):
        self.label_ids = list(label_ids)

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        return np.where(np.isin(labels, self.label_ids), labels, 0)


def _get_structure_label(roi_number, roi_name):
    """Map the ROIs of the planning CT RTSTRUCT files to the semantic label ids.

    Only the small bowel and stomach/duodenum organ-at-risk contours are used; lung and
    volume-of-interest contours (present in the same file) are ignored.
    """
    name = roi_name.lower()
    if "bowel" in name:
        return STRUCTURE_IDS["bowel"]
    if "stomach" in name or "duo" in name:
        return STRUCTURE_IDS["stomach_duodenum"]
    return None


def _is_planning_ct_rtstruct(rtstruct_path):
    """Check whether an RTSTRUCT file belongs to the planning CT, based on its ROI names.

    The planning CT contours are the only ones with a '_planCT' suffix (e.g. 'Bowel_sm_planCT'),
    which distinguishes them from the CBCT contours in the same collection.
    """
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    roi_names = [str(roi.ROIName).lower() for roi in rtstruct.StructureSetROISequence]
    return any("planct" in name.replace("_", "") for name in roi_names)


def _get_referenced_series(rtstruct_path):
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    return str(
        rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0].SeriesInstanceUID
    )


def _preprocess_pancreatic_ct_cbct_seg(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    rtstruct_series = {row["Series UID"]: row["Subject ID"] for row in rows if row["Modality"] == "RTSTRUCT"}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, subject_id in tqdm(sorted(rtstruct_series.items()), desc="Preprocess Pancreatic-CT-CBCT-SEG"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        rtstruct_path = glob(os.path.join(dicom_dir, series_uid, "*.dcm"))[0]
        if not _is_planning_ct_rtstruct(rtstruct_path):
            continue  # This is a CBCT RTSTRUCT, which is not used by this module.

        ct_dir = os.path.join(dicom_dir, _get_referenced_series(rtstruct_path))
        volume, geometry = util.load_dicom_series(ct_dir)
        volume = np.round(volume).astype("int16")
        labels = util.rasterize_rtstruct(rtstruct_path, geometry, volume.shape, _get_structure_label)

        # Written to a temporary path and renamed atomically, so that a run interrupted mid-write
        # never leaves a stale, incomplete file at 'out_path' for a later run to mistake as done.
        tmp_path = f"{out_path}.tmp"
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        os.replace(tmp_path, out_path)


def get_pancreatic_ct_cbct_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Pancreatic-CT-CBCT-SEG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    # Download the full collection (CT, CBCT and RTSTRUCT series) from the TCIA manifest. Only the
    # planning CT and its RTSTRUCT are used for preprocessing, see `_is_planning_ct_rtstruct`.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "pancreatic_ct_cbct_seg_series")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, os.path.basename(URL)), url=URL, dst=dicom_dir, csv_filename=csv_path,
            download=download,
        )

    _preprocess_pancreatic_ct_cbct_seg(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_pancreatic_ct_cbct_seg_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Pancreatic-CT-CBCT-SEG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_pancreatic_ct_cbct_seg_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_pancreatic_ct_cbct_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    structures: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Pancreatic-CT-CBCT-SEG dataset for organ-at-risk segmentation in planning CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        structures: The structures to use as labels, a subset of 'bowel' and 'stomach_duodenum'.
            All other structures are set to background. By default both structures are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_pancreatic_ct_cbct_seg_paths(path, download)

    if structures is not None:
        assert all(structure in STRUCTURE_IDS for structure in structures), f"Invalid structures: {structures}"
        select_trafo = SelectStructures([STRUCTURE_IDS[structure] for structure in structures])
        if "label_transform" in kwargs:
            kwargs["label_transform"] = Compose(select_trafo, kwargs["label_transform"], is_multi_tensor=False)
        else:
            kwargs["label_transform"] = select_trafo

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


def get_pancreatic_ct_cbct_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    structures: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Pancreatic-CT-CBCT-SEG dataloader for organ-at-risk segmentation in planning CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        structures: The structures to use as labels, a subset of 'bowel' and 'stomach_duodenum'.
            All other structures are set to background. By default both structures are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pancreatic_ct_cbct_seg_dataset(path, patch_shape, structures, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
