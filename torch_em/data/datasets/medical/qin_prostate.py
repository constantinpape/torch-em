"""The QIN-PROSTATE-Repeatability dataset contains annotations for the prostate and its peripheral
zone in MRI.

The dataset consists of a test-retest repeatability study: 15 patients were each scanned twice, with
expert segmentations of the whole prostate, its peripheral zone, a lesion and normal tissue (see
`SEGMENT_IDS`) drawn on three different MRI series per scan (T2-weighted, an ADC map, and a DCE
subtraction image), each as a separate DICOM-SEG object paired here with the exact series it
references. Not every segment is present in every file (e.g. a case without a lesion has no 'Lesion'
segment), and a DICOM-SEG's own segment numbering only reflects the segments it actually contains, so
this module remaps every segment to `SEGMENT_IDS` by its name rather than trusting that numbering.

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.7937/K9/TCIA.2018.MR1CKGND and is distributed under the
CC BY 4.0 license.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .adrenal_acc import _load_dicom_volume, _load_dicom_seg, _resample_labels
from .. import util


COLLECTION = "QIN-PROSTATE-Repeatability"

SEGMENT_IDS = {"normal": 1, "peripheral_zone": 2, "lesion": 3, "prostate": 4}
"""Mapping from the anatomical structure to its label id."""

_SEGMENT_NAME_TO_ID = {
    "normal": SEGMENT_IDS["normal"],
    "peripheral zone of the prostate": SEGMENT_IDS["peripheral_zone"],
    "lesion": SEGMENT_IDS["lesion"],
    "prostate": SEGMENT_IDS["prostate"],
}


def _get_series_metadata(path, download):
    """Get the metadata of all series in the collection from the NBIA REST API."""
    import requests

    metadata_path = os.path.join(path, "qin_prostate_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(f"{util.NBIA_API_URL}getSeries", params={"Collection": COLLECTION})
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _referenced_series_uid(seg):
    return str(seg.ReferencedSeriesSequence[0].SeriesInstanceUID)


def _remap_segments_by_name(seg_labels, seg):
    """Remap a DICOM-SEG label volume from its own (file-specific) segment numbers to `SEGMENT_IDS`,
    since a file that omits a segment does not preserve the numbering of the others.
    """
    remapped = np.zeros_like(seg_labels)
    for segment in seg.SegmentSequence:
        canonical_id = _SEGMENT_NAME_TO_ID.get(str(segment.SegmentLabel).lower())
        if canonical_id is not None:
            remapped[seg_labels == int(segment.SegmentNumber)] = canonical_id
    return remapped


def _preprocess_qin_prostate(dicom_dir, series_metadata, preprocessed_dir):
    import h5py
    import pydicom

    seg_series = [series for series in series_metadata if series.get("Modality") == "SEG"]

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series in tqdm(seg_series, desc="Preprocess QIN-PROSTATE-Repeatability"):
        seg_paths = glob(os.path.join(dicom_dir, series["SeriesInstanceUID"], "*.dcm"))
        if not seg_paths:
            continue

        out_path = os.path.join(preprocessed_dir, f"{series['SeriesInstanceUID']}.h5")
        if os.path.exists(out_path):
            continue

        seg = pydicom.dcmread(seg_paths[0])
        image_dir = os.path.join(dicom_dir, _referenced_series_uid(seg))
        if not glob(os.path.join(image_dir, "*.dcm")):
            continue

        volume, image_affine = _load_dicom_volume(image_dir)
        seg_labels, seg_affine = _load_dicom_seg(seg_paths[0])
        seg_labels = _remap_segments_by_name(seg_labels, seg)
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, image_affine)
        if labels.max() == 0:
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_qin_prostate_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the QIN-PROSTATE-Repeatability dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)
    series_metadata = _get_series_metadata(path, download)

    seg_uids = [series["SeriesInstanceUID"] for series in series_metadata if series.get("Modality") == "SEG"]

    dicom_dir = os.path.join(path, "dicom")
    if download:  # The SEG series are downloaded first, so the image series they reference can be found.
        util.download_tcia_series(seg_uids, dst=dicom_dir, csv_filename=os.path.join(path, "qin_prostate_seg"))

        import pydicom
        image_uids = set()
        for uid in seg_uids:
            seg_paths = glob(os.path.join(dicom_dir, uid, "*.dcm"))
            if seg_paths:
                seg = pydicom.dcmread(seg_paths[0])
                image_uids.add(_referenced_series_uid(seg))
        util.download_tcia_series(
            sorted(image_uids), dst=dicom_dir, csv_filename=os.path.join(path, "qin_prostate_image")
        )
    elif not glob(os.path.join(dicom_dir, "*", "*.dcm")):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_qin_prostate(dicom_dir, series_metadata, preprocessed_dir)
    return preprocessed_dir


def get_qin_prostate_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the QIN-PROSTATE-Repeatability data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_qin_prostate_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_qin_prostate_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the QIN-PROSTATE-Repeatability dataset for prostate and peripheral zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_qin_prostate_paths(path, download)

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


def get_qin_prostate_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the QIN-PROSTATE-Repeatability dataloader for prostate and peripheral zone segmentation.

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
    dataset = get_qin_prostate_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
