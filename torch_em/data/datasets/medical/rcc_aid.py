"""The RCC-AID dataset contains annotations for kidney, tumor and cyst segmentation in CT scans of
patients with renal cell carcinoma (RCC).

The dataset consists of 129 CT series from 91 patients across the TCGA-KIRC (clear cell, 85 series),
TCGA-KIRP (papillary, 26 series) and TCGA-KICH (chromophobe, 18 series) cohorts. The images were
converted from the original TCIA DICOM series to NIfTI, and voxel-level segmentation masks were
derived from an automated segmentation model followed by manual quality checks and corrections.

The label ids are - kidney: 1, tumor: 2, cyst: 3

This dataset is from the publication https://doi.org/10.64898/2026.04.22.26351451.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "images": "https://zenodo.org/records/20719257/files/images.zip",
    "labels": "https://zenodo.org/records/20719257/files/labels.zip",
}

CHECKSUM = {
    "images": "7e6a7530e8bce0c34df1cafe00a56bbe579f480b54a9e963d206fe385ff23ade",
    "labels": "692cae50e2de498b63fc6edcc526a57ae19fecfbaadd6f6ad0a4ff928c2da6d1",
}


def get_rcc_aid_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RCC-AID dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    images_dir = os.path.join(path, "images")
    labels_dir = os.path.join(path, "labels")
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        return path

    os.makedirs(path, exist_ok=True)

    for name, out_dir in [("images", images_dir), ("labels", labels_dir)]:
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=URL[name], download=download, checksum=CHECKSUM[name])
        util.unzip(zip_path=zip_path, dst=path)
        assert os.path.exists(out_dir), f"Something went wrong when unzipping '{zip_path}'."

    return path


def get_rcc_aid_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the RCC-AID data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_rcc_aid_data(path, download)

    label_paths = natsorted(glob(os.path.join(data_dir, "labels", "*_seg.nii.gz")))
    image_paths = [
        os.path.join(data_dir, "images", os.path.basename(p)[:-len("_seg.nii.gz")] + ".nii.gz")
        for p in label_paths
    ]

    missing = [p for p in image_paths if not os.path.exists(p)]
    assert not missing, f"Could not find the matching image(s) for {missing}."

    return image_paths, label_paths


def get_rcc_aid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RCC-AID dataset for kidney, tumor and cyst segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_rcc_aid_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_rcc_aid_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RCC-AID dataloader for kidney, tumor and cyst segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rcc_aid_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
