"""The AMD-DME-3D-OCT dataset contains annotations for pigment epithelial detachment (PED) and
intraretinal fluid (IRF) segmentation in full 3D optical coherence tomography (OCT) volumes.

The dataset consists of 224 volumes (122 AMD and 102 DME), of which 104 (62 AMD, 42 DME) come with
volumetric annotations. The remaining 120 volumes are unlabeled and are not used by this loader.
Each annotated volume was independently segmented by three junior ophthalmologists, and the
resulting masks were reviewed and corrected by an experienced ophthalmologist. The label volumes
are binary (PED and IRF are merged into a single foreground class), matching each 512x512x512 raw
volume ('<case><eye><picture>.tif') to its mask ('<case><eye><picture>_mask.tif').

NOTE: This is a distinct, volumetric (3D) dataset. It is not the same as the already-integrated
`torch_em.data.datasets.medical.amd_sd`, which ships 2D B-scan annotations for wet AMD lesions only.

The data is located at https://doi.org/10.6084/m9.figshare.30582035, released under a CC-BY-4.0 license.
NOTE: The archive is a single ~19.6 GB 7z file, which requires the 'p7zip' CLI to extract
(see `torch_em.data.datasets.util.unzip_7z`).

This dataset is from the publication https://doi.org/10.1038/s41597-025-06497-1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/59443823"
CHECKSUM = "31d1e2c581be375654a3d757eaa2710f4e91628b0779c39f2a5d04afe4ba0318"

DISEASES = ["AMD", "DME"]


def get_amd_dme_3d_oct_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the AMD-DME-3D-OCT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "AMD_DME_3D_Dataset")
    if all(os.path.exists(os.path.join(data_dir, f"{disease}_labeled")) for disease in DISEASES):
        return data_dir

    os.makedirs(path, exist_ok=True)

    archive_path = os.path.join(path, "AMD_DME_3D_Dataset.7z")
    util.download_source(path=archive_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_7z(path_7z=archive_path, dst=path, remove=False)

    assert all(os.path.exists(os.path.join(data_dir, f"{disease}_labeled")) for disease in DISEASES), \
        f"The extraction of the AMD-DME-3D-OCT archive did not create the expected folders in '{data_dir}'."

    return data_dir


def get_amd_dme_3d_oct_paths(
    path: Union[os.PathLike, str], disease: Literal["AMD", "DME"] = "AMD", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AMD-DME-3D-OCT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        disease: The choice of disease. Either 'AMD' or 'DME'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if disease not in DISEASES:
        raise ValueError(f"'{disease}' is not a valid choice of disease. Choose one of {DISEASES}.")

    data_dir = get_amd_dme_3d_oct_data(path, download)

    label_paths = natsorted(glob(os.path.join(data_dir, f"{disease}_labeled", "label", "*_mask.tif")))
    raw_paths = [
        os.path.join(data_dir, f"{disease}_labeled", "input", os.path.basename(p).replace("_mask.tif", ".tif"))
        for p in label_paths
    ]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_amd_dme_3d_oct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    disease: Literal["AMD", "DME"] = "AMD",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AMD-DME-3D-OCT dataset for PED and IRF segmentation in 3D OCT volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        disease: The choice of disease. Either 'AMD' or 'DME'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_amd_dme_3d_oct_paths(path, disease, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_amd_dme_3d_oct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    disease: Literal["AMD", "DME"] = "AMD",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AMD-DME-3D-OCT dataloader for PED and IRF segmentation in 3D OCT volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        disease: The choice of disease. Either 'AMD' or 'DME'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_amd_dme_3d_oct_dataset(path, patch_shape, disease, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
