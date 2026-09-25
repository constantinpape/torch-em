"""The QIN-LungCT-Seg dataset contains annotations for lung tumors in CT.

It consists of repeated segmentations of the same tumors, drawn either manually or by one of several
semi-automated algorithms, across four source collections (LIDC-IDRI, RIDER Lung CT, QIN LUNG CT and a
CT lung phantom). Each segmentation is a separate DICOM-SEG object paired here with the exact CT
series it references, so the same tumor can appear multiple times with different segmentations - useful
for studying inter-algorithm and inter-rater variability, not just as extra training pairs.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/qin-lungct-seg/.

The data was released at https://doi.org/10.7937/K9/TCIA.2015.1BUVFJR7.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .adrenal_acc import _load_dicom_volume, _load_dicom_seg, _resample_labels
from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/QIN-Multi-site-Lung-CTs-and-SEG-minus-Stanford.tcia"  # noqa

# The DICOM series are downloaded individually from TCIA.
CHECKSUM = None


def _get_referenced_series(seg_path):
    import pydicom

    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    return str(seg.ReferencedSeriesSequence[0].SeriesInstanceUID)


def _preprocess_qin_lungct_seg(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    seg_series = [row["Series UID"] for row in rows if row["Modality"] == "SEG"]

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid in tqdm(sorted(seg_series), desc="Preprocess QIN-LungCT-Seg"):
        out_path = os.path.join(preprocessed_dir, f"{series_uid}.h5")
        if os.path.exists(out_path):
            continue

        seg_paths = glob(os.path.join(dicom_dir, series_uid, "*.dcm"))
        if not seg_paths:
            continue

        ct_dir = os.path.join(dicom_dir, _get_referenced_series(seg_paths[0]))
        if not glob(os.path.join(ct_dir, "*.dcm")):
            continue

        volume, ct_affine = _load_dicom_volume(ct_dir)
        seg_labels, seg_affine = _load_dicom_seg(seg_paths[0])
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, ct_affine)
        if labels.max() == 0:
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_qin_lungct_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the QIN-LungCT-Seg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "qin_lungct_seg_series")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, os.path.basename(URL)), url=URL, dst=dicom_dir, csv_filename=csv_path,
            download=download,
        )

    _preprocess_qin_lungct_seg(dicom_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_qin_lungct_seg_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the QIN-LungCT-Seg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_qin_lungct_seg_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_qin_lungct_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the QIN-LungCT-Seg dataset for lung tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_qin_lungct_seg_paths(path, download)

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


def get_qin_lungct_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the QIN-LungCT-Seg dataloader for lung tumor segmentation.

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
    dataset = get_qin_lungct_seg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
