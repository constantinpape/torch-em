"""The RIDER Lung CT dataset contains annotations for lung tumor segmentation in same-day repeat CT scans
of non-small cell lung cancer patients.

It consists of 59 CT volumes (test and retest scan of 31 patients, with one scan missing for three patients)
with a delineation of the primary gross tumor volume by a radiation oncologist, which is distributed as DICOM
RTSTRUCT in the 'RIDER-LungCT-Seg' analysis result of the collection. Each RTSTRUCT contains a manual
delineation ('GTVp_<scan>_man') and the result of an in-house autosegmentation method ('GTVp_<scan>_auto').
This module rasterizes the contours onto the CT grid (see `torch_em.data.datasets.util.rasterize_rtstruct`)
and stores the CT ('raw') and the binary tumor labels ('labels/manual' and 'labels/auto') in hdf5 files.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/rider-lung-ct/ and the segmentations
at https://www.cancerimagingarchive.net/analysis-result/rider-lungct-seg/.

This dataset is from the publications https://doi.org/10.1148/radiol.2522081593 (images)
and https://doi.org/10.1038/ncomms5006 (segmentations).
The data was released at https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR and https://doi.org/10.7937/tcia.2020.jit9grk8.
Please cite them if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://www.cancerimagingarchive.net/wp-content/uploads/RIDER-Lung-CT-Original-Scans-for-Leonard-Wee-Feb-10-2020-.tcia",  # noqa
    "labels": "https://www.cancerimagingarchive.net/wp-content/uploads/RIDER-Lung-CT-RTSTRUCTS-DICOM-SEGS-Leonard-Wee-Feb-10-2020.tcia",  # noqa
}

# The DICOM series are downloaded individually from TCIA.
CHECKSUMS = {"images": None, "labels": None}

ANNOTATIONS = ["manual", "auto"]


def _get_referenced_series(rtstruct_path):
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    return str(
        rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0].SeriesInstanceUID
    )


def _preprocess_rider_lung(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    rtstruct_series = {
        row["Series UID"]: (row["Subject ID"], row["Series Description"].lower()) for row in rows
        if row["Modality"] == "RTSTRUCT"
    }

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, (subject_id, scan) in tqdm(sorted(rtstruct_series.items()), desc="Preprocess RIDER Lung CT"):
        assert scan in ("test", "retest"), f"Unexpected RTSTRUCT series description: {scan}"
        out_path = os.path.join(preprocessed_dir, f"{subject_id}_{scan}.h5")
        if os.path.exists(out_path):
            continue

        rtstruct_path = glob(os.path.join(dicom_dir, series_uid, "*.dcm"))[0]
        ct_dir = os.path.join(dicom_dir, _get_referenced_series(rtstruct_path))
        volume, geometry = util.load_dicom_series(ct_dir)
        volume = np.round(volume).astype("int16")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            for annotation in ANNOTATIONS:
                roi_labels = {f"GTVp_{scan}_{'man' if annotation == 'manual' else 'auto'}": 1}
                labels = util.rasterize_rtstruct(rtstruct_path, geometry, volume.shape, roi_labels)
                # Not all RTSTRUCTs contain both delineations, e.g. RIDER-2016615262 has no 'auto' retest contour.
                if labels.any():
                    f.create_dataset(f"labels/{annotation}", data=labels, compression="gzip")


def get_rider_lung_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RIDER Lung CT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    # Download the annotated CT series and the RTSTRUCT (and SEG) series from the two TCIA manifests.
    # The series metadata are written after all series are downloaded, so their presence means it is complete.
    dicom_dir = os.path.join(path, "dicom")
    for name, url in URLS.items():
        csv_filename = os.path.join(path, f"rider_lung_{name}")
        if not os.path.exists(f"{csv_filename}.csv"):
            util.download_source_tcia(
                path=os.path.join(path, os.path.basename(url)), url=url, dst=dicom_dir,
                csv_filename=csv_filename, download=download,
            )

    _preprocess_rider_lung(dicom_dir, os.path.join(path, "rider_lung_labels.csv"), preprocessed_dir)
    return preprocessed_dir


def get_rider_lung_paths(
    path: Union[os.PathLike, str], annotation: Literal["manual", "auto"] = "manual", download: bool = False
) -> List[str]:
    """Get paths to the RIDER Lung CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The tumor delineation to use as labels, either the 'manual' delineation by a radiation
            oncologist or the result of the 'auto' segmentation method.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data
        ('labels/manual' and 'labels/auto').
    """
    import h5py

    assert annotation in ANNOTATIONS, f"'{annotation}' is not a valid annotation. Choose one of {ANNOTATIONS}."
    data_dir = get_rider_lung_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))

    # A few volumes do not have both delineations, they are filtered out for the respective annotation.
    def _has_annotation(volume_path):
        with h5py.File(volume_path, "r") as f:
            return f"labels/{annotation}" in f

    return [volume_path for volume_path in volume_paths if _has_annotation(volume_path)]


def get_rider_lung_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation: Literal["manual", "auto"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RIDER Lung CT dataset for lung tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The tumor delineation to use as labels, either the 'manual' delineation by a radiation
            oncologist or the result of the 'auto' segmentation method.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_rider_lung_paths(path, annotation, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{annotation}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_rider_lung_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation: Literal["manual", "auto"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RIDER Lung CT dataloader for lung tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The tumor delineation to use as labels, either the 'manual' delineation by a radiation
            oncologist or the result of the 'auto' segmentation method.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rider_lung_dataset(path, patch_shape, annotation, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
