"""The TopCoW dataset contains annotations for the vessel components of the circle of Willis
in computed tomography angiography (CTA) and magnetic resonance angiography (MRA).

The data was curated for the TopCoW challenge (https://topcow24.grand-challenge.org). This module downloads the
public training data release of the second edition, which consists of 250 annotated angiographies (125 CTA and
125 MRA of the same 125 patients). The 10 unannotated validation images of the release are not exposed here.
The modality is selected with the 'modality' argument ('ct' or 'mr').

The multi-class segmentations label 13 vessel components of the circle of Willis, see `LABEL_IDS`. Note that the
label id 13 and 14 are not used and that the third A2 segment (an anatomical variant) has the label id 15.

The data is located at https://doi.org/10.5281/zenodo.15692630.

This dataset is from the publication https://doi.org/10.1056/aidbp2500994.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/15692630/files/TopCoW2024_Data_Release.zip"
CHECKSUM = "a23d9d0f05ec439472736f65c47fc7408263eb2f2b01a1d1afc934f2e8f2889b"

LABEL_IDS = {
    "background": 0,
    "BA": 1,
    "R-PCA": 2,
    "L-PCA": 3,
    "R-ICA": 4,
    "R-MCA": 5,
    "L-ICA": 6,
    "L-MCA": 7,
    "R-Pcom": 8,
    "L-Pcom": 9,
    "Acom": 10,
    "R-ACA": 11,
    "L-ACA": 12,
    "3rd-A2": 15,
}

MODALITIES = ["ct", "mr"]


def get_topcow_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TopCoW dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "TopCoW2024_Data_Release")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TopCoW2024_Data_Release.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_topcow_paths(
    path: Union[os.PathLike, str],
    modality: Optional[Literal["ct", "mr"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TopCoW data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_topcow_data(path, download)

    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {MODALITIES}.")

    pattern = f"topcow_{'*' if modality is None else modality}_*.nii.gz"
    label_paths = natsorted(glob(os.path.join(data_dir, "cow_seg_labelsTr", pattern)))
    # The images carry the channel suffix '_0000' of the nnU-Net format, the labels do not.
    raw_paths = [
        os.path.join(data_dir, "imagesTr", os.path.basename(p).replace(".nii.gz", "_0000.nii.gz"))
        for p in label_paths
    ]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_topcow_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["ct", "mr"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TopCoW dataset for circle of Willis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_topcow_paths(path, modality, download)

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


def get_topcow_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["ct", "mr"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TopCoW dataloader for circle of Willis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_topcow_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
