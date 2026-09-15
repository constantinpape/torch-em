"""The Mendeley MS dataset contains annotations for multiple sclerosis lesion segmentation in brain MRI.

It comprises T1-weighted, T2-weighted and T2-FLAIR scans of 60 MS patients with consensus manual lesion
segmentations for each of the three modalities (the scans of a patient are not co-registered and have
different shapes, so the labels are provided per modality). This is the 'MS Lesion' dataset of the
RadioActive benchmark (https://arxiv.org/abs/2411.07885), which uses the FLAIR scans.
The label ids are: 0 = background, 1 = MS lesion.

The dataset is located at https://data.mendeley.com/datasets/8bctsm8jz7/1 (CC BY 4.0).

This dataset is from the publication https://doi.org/10.1016/j.dib.2022.108139.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/8bctsm8jz7/files/9356efeb-dcd8-4213-a2d4-8febe9f1a5db/file_downloaded"  # noqa
CHECKSUM = "c90f0f47c9e1a5e0fafc87b77dfcbcb09ac0cf9ffdaa333aaec1b9c63d31a7b3"

LABEL_IDS = {"background": 0, "ms_lesion": 1}

MODALITIES = {"flair": "Flair", "t1": "T1", "t2": "T2"}


def get_mendeley_ms_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the Mendeley MS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    if len(glob(os.path.join(path, "Patient-*"))) == 60:
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ms_brain_mri.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)


def get_mendeley_ms_paths(
    path: Union[os.PathLike, str], modality: Literal["flair", "t1", "t2"] = "flair", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Mendeley MS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The MRI modality. Either 'flair', 't1' or 't2'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_mendeley_ms_data(path, download)

    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose one of {list(MODALITIES)}.")
    modality = MODALITIES[modality]

    raw_paths = natsorted(glob(os.path.join(path, "Patient-*", f"*-{modality}.nii")))
    raw_paths = [p for p in raw_paths if "LesionSeg" not in os.path.basename(p)]
    label_paths = [p.replace(f"-{modality}.nii", f"-LesionSeg-{modality}.nii") for p in raw_paths]
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_mendeley_ms_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["flair", "t1", "t2"] = "flair",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Mendeley MS dataset for MS lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI modality. Either 'flair', 't1' or 't2'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mendeley_ms_paths(path, modality, download)

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


def get_mendeley_ms_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["flair", "t1", "t2"] = "flair",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Mendeley MS dataloader for MS lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI modality. Either 'flair', 't1' or 't2'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mendeley_ms_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
