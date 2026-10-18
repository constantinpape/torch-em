"""The Etch a Cell dataset contains annotations for nuclear envelope segmentation in serial blockface
scanning electron microscopy (SBF-SEM) volumes of HeLa cells.

The labels are consensus segmentations, aggregated from the annotations of volunteers of the citizen science project
'Etch a Cell' (https://www.zooniverse.org/projects/dwright04/etch-a-cell-nuclear-envelope).
The data consists of 18 cropped regions of interest (ROIs) with a voxel size of 10 x 10 x 50 nm.
Each ROI has up to 300 slices of 2000 x 2000 pixels, some of the ROIs are smaller.

The labels are available at https://www.ebi.ac.uk/biostudies/studies/S-BSST448.
The raw data is available at https://www.ebi.ac.uk/empiar/EMPIAR-10478/ (~15.6 GB for all ROIs).
Use `rois` to download only a subset of the ROIs.
The licence of the BioStudies entry is not stated. Please check the licence of EMPIAR-10478 before using the data.

The dataset is from the publication 'Citizen science, cells and CNNs: deep learning for automatic segmentation of the
nuclear envelope in electron microscopy data, trained with volunteer segmentations' (Spiers et al.).
Please cite it if you use this dataset in your research.
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_URL = "https://www.ebi.ac.uk/biostudies/files/S-BSST448/Aggregations.zip"
RAW_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs"
LABEL_CHECKSUM = None
# NOTE: The raw data has no checksums available, so the downloaded files are not verified.

ROIS = [
    "ROI_1416-1932-171", "ROI_1536-3456-213", "ROI_1584-6996-1", "ROI_1608-912-1", "ROI_1656-6756-329",
    "ROI_1716-7800-517", "ROI_2052-5784-112", "ROI_2448-4704-271", "ROI_2820-6780-468", "ROI_2832-1692-1",
    "ROI_3000-3264-393", "ROI_3516-5712-314", "ROI_3576-5232-35", "ROI_3588-3972-1", "ROI_3624-2712-201",
    "ROI_3768-7248-143", "ROI_3972-1956-438", "ROI_4320-1260-95",
]


def _check_rois(rois):
    rois = ROIS if rois is None else rois
    for roi in rois:
        if roi not in ROIS:
            raise ValueError(f"'{roi}' is not a valid ROI. Please choose from {ROIS}.")
    return list(rois)


def get_etchacell_data(
    path: Union[os.PathLike, str], rois: Optional[List[str]] = None, download: bool = False
) -> str:
    """Download the Etch a Cell dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        rois: The ROIs to download. By default all 18 ROIs are used, e.g. 'ROI_1416-1932-171'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    rois = _check_rois(rois)

    label_dir = os.path.join(path, "labels")
    raw_dir = os.path.join(path, "raw")

    if not all(os.path.exists(os.path.join(label_dir, f"{roi}.tiff")) for roi in rois):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "Aggregations.zip")
        util.download_source(zip_path, LABEL_URL, download, checksum=LABEL_CHECKSUM)
        util.unzip(zip_path, label_dir, remove=True)

    os.makedirs(raw_dir, exist_ok=True)
    for roi in rois:
        raw_path = os.path.join(raw_dir, f"{roi}.tiff")
        if not os.path.exists(raw_path):
            util.download_source(raw_path, f"{RAW_URL}/{roi}.tiff", download, checksum=None)

    return path


def get_etchacell_paths(
    path: Union[os.PathLike, str], rois: Optional[List[str]] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Etch a Cell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        rois: The ROIs to use. By default all 18 ROIs are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    rois = _check_rois(rois)
    get_etchacell_data(path, rois, download)

    raw_paths = [os.path.join(path, "raw", f"{roi}.tiff") for roi in rois]
    label_paths = [os.path.join(path, "labels", f"{roi}.tiff") for roi in rois]
    return raw_paths, label_paths


def _binary_labels(labels):
    return (labels > 0).astype(np.uint8)


def get_etchacell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    rois: Optional[List[str]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Etch a Cell dataset for nuclear envelope segmentation in SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        rois: The ROIs to use. By default all 18 ROIs are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_etchacell_paths(path, rois, download)

    kwargs = util.update_kwargs(kwargs, "label_transform", _binary_labels)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_etchacell_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    rois: Optional[List[str]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Etch a Cell dataloader for nuclear envelope segmentation in SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        rois: The ROIs to use. By default all 18 ROIs are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_etchacell_dataset(path, patch_shape, rois, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
