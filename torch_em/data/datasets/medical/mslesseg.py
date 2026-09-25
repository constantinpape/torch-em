"""MSLesSeg is a dataset for segmentation of multiple sclerosis (MS) lesions in brain MRI.

The dataset consists of 115 longitudinal MRI scans from 75 MS patients, with T1-weighted, T2-weighted
and FLAIR sequences, registered to the MNI152 template. Expert-validated lesion segmentation masks are
provided for each scan.

The dataset is located at https://doi.org/10.6084/m9.figshare.27919209 (Figshare, CC BY 4.0).
The dataset is from the publication https://doi.org/10.1038/s41597-025-05250-y.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/52771814"

MODALITIES = ("FLAIR", "T1", "T2")


def get_mslesseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MSLesSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "MSLesSeg Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "MSLesSeg_Dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_mslesseg_paths(
    path: Union[os.PathLike, str], modality: Literal["FLAIR", "T1", "T2"] = "FLAIR", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the MSLesSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of MRI modality. One of 'FLAIR', 'T1', 'T2'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mslesseg_data(path, download)

    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose from {MODALITIES}.")

    # The 'train' split stores scans per timepoint (eg. 'train/P9/T3/P9_T3_MASK.nii.gz'), while the
    # 'test' split has a single timepoint per patient directory (eg. 'test/P54/P54_MASK.nii.gz').
    label_paths = natsorted(glob(os.path.join(data_dir, "*", "P*", "**", "*_MASK.nii.gz"), recursive=True))

    raw_paths = []
    for label_path in label_paths:
        raw_path = label_path.replace("_MASK.nii.gz", f"_{modality}.nii.gz")
        assert os.path.exists(raw_path), raw_path
        raw_paths.append(raw_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_mslesseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["FLAIR", "T1", "T2"] = "FLAIR",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MSLesSeg dataset for multiple sclerosis lesion segmentation in brain MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of MRI modality. One of 'FLAIR', 'T1', 'T2'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mslesseg_paths(path, modality, download)

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


def get_mslesseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["FLAIR", "T1", "T2"] = "FLAIR",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MSLesSeg dataloader for multiple sclerosis lesion segmentation in brain MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of MRI modality. One of 'FLAIR', 'T1', 'T2'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mslesseg_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
