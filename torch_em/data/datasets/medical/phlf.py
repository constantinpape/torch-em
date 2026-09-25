"""PHLF is a dataset for segmentation of the liver, liver tumor, Couinaud liver segments, spleen and
psoas muscle in hepatobiliary-phase Gd-EOB-DTPA-enhanced MRI.

The dataset consists of preoperative MRI scans of 220 patients from three academic medical centers who
underwent hepatectomy, together with 22,342 expert annotations of the whole liver, liver tumors, the
eight Couinaud liver segments, the spleen and the psoas muscle. The annotations enable automated
quantification of future liver remnant (FLR) volume for predicting post-hepatectomy liver failure (PHLF).

The dataset is located at https://doi.org/10.5281/zenodo.18622298 (Zenodo, CC BY 4.0).
The dataset is from the publication https://doi.org/10.1038/s41597-026-07483-x.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/18622298/files/PHLF.zip/content"

CENTERS = ("Center1", "Center2", "Center3")

LABEL_FOLDERS = {
    "liver": "Annotation_Whole liver",
    "tumor": "Annotation_Liver tumor",
    "couinaud": "Annotation_Couinaud liver segments",
    "spleen": "Annotation_Spleen",
    "psoas": "Annotation_Muscle",
}


def get_phlf_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PHLF data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "PHLF")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "PHLF.zip")
    util.download_source(path=zip_path, url=URL, download=download)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_phlf_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["liver", "tumor", "couinaud", "spleen", "psoas"] = "liver",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PHLF data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of annotated structure. One of 'liver', 'tumor', 'couinaud', 'spleen', 'psoas'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    import nibabel as nib

    data_dir = get_phlf_data(path, download)

    if label_choice not in LABEL_FOLDERS:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose from {list(LABEL_FOLDERS.keys())}.")

    raw_paths, label_paths = [], []
    for center in CENTERS:
        label_dir = os.path.join(data_dir, center, LABEL_FOLDERS[label_choice])
        for label_path in natsorted(glob(os.path.join(label_dir, "*.nii.gz"))):
            fname = os.path.basename(label_path)
            raw_path = os.path.join(data_dir, center, "Image", fname)
            if not os.path.exists(raw_path):
                continue
            # A small number of cases (4 / 220) have an annotation volume with a different shape than
            # the released image volume, likely because the annotation was made on the original DICOM
            # resolution rather than the resampled Nifti. These cases are skipped.
            if nib.load(raw_path).shape != nib.load(label_path).shape:
                continue
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_phlf_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["liver", "tumor", "couinaud", "spleen", "psoas"] = "liver",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PHLF dataset for liver, tumor, Couinaud segment, spleen and psoas segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of annotated structure. One of 'liver', 'tumor', 'couinaud', 'spleen', 'psoas'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_phlf_paths(path, label_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_phlf_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["liver", "tumor", "couinaud", "spleen", "psoas"] = "liver",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PHLF dataloader for liver, tumor, Couinaud segment, spleen and psoas segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of annotated structure. One of 'liver', 'tumor', 'couinaud', 'spleen', 'psoas'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_phlf_dataset(path, patch_shape, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
