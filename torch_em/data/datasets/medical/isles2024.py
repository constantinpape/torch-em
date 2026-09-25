"""The ISLES 2024 dataset contains annotations for ischemic stroke lesion and large vessel occlusion
segmentation in a multimodal, longitudinal collection of brain CT and MRI scans.

The dataset consists of 149 acute ischemic stroke cases, organized according to the BIDS standard. For each
case, the following data is provided: admission non-contrast CT (NCCT), CT angiography (CTA) and 4D CT
perfusion (CTP) with its derived perfusion maps (Tmax, CBF, CBV, MTT), as well as follow-up MRI (DWI, ADC).
This module uses the 'derivatives' folder, in which all modalities are linearly co-registered to the NCCT
space, so that raw and label volumes of the same case share a common voxel grid.

Two segmentation targets are provided, selected with the 'label_choice' argument:
- 'lesion': the binary infarct mask, derived from the follow-up MRI (session 'ses-02'). The corresponding
  modalities are 'dwi' and 'adc'.
- 'lvo': the binary large vessel occlusion mask, derived from the admission CTA (session 'ses-01'). The
  corresponding modalities are 'ncct', 'cta', 'tmax', 'mtt', 'cbf' and 'cbv'.
The modality is selected with the 'modality' argument. If it is not specified, all modalities of the chosen
'label_choice' are stacked as channels of the raw input.

The data is located at https://doi.org/10.5281/zenodo.16813698.

NOTE: The archive is distributed as a single ~92 GB 7z file, which requires the 'p7zip' CLI to extract
(see `torch_em.data.datasets.util.unzip_7z`).

This dataset is from the publication https://doi.org/10.48550/arXiv.2408.11142.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16813698/files/train.7z"
CHECKSUM = "038920e4dc2011a3f47b8bb8421c67e36d07f1d84f1ba442563077480f75d129"

LABEL_CHOICES = ["lesion", "lvo"]

MODALITIES = {
    "lesion": ["dwi", "adc"],
    "lvo": ["ncct", "cta", "tmax", "mtt", "cbf", "cbv"],
}

SESSIONS = {"lesion": "ses-02", "lvo": "ses-01"}

PERFUSION_MAPS = ["tmax", "mtt", "cbf", "cbv"]


def get_isles2024_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ISLES 2024 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(os.path.join(data_dir, "derivatives")):
        return data_dir

    os.makedirs(path, exist_ok=True)

    archive_path = os.path.join(path, "train.7z")
    util.download_source(path=archive_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_7z(path_7z=archive_path, dst=path, remove=False)

    assert os.path.exists(os.path.join(data_dir, "derivatives")), \
        f"The extraction of the ISLES 2024 archive did not create '{data_dir}'."
    return data_dir


def _get_raw_path(label_path: str, label_choice: str, modality: str) -> str:
    case_dir = os.path.dirname(label_path)
    base_name = os.path.basename(label_path).replace(f"_{label_choice}-msk.nii.gz", f"_{modality}.nii.gz")
    if modality in PERFUSION_MAPS:
        return os.path.join(case_dir, "perfusion-maps", base_name)
    return os.path.join(case_dir, base_name)


def get_isles2024_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["lesion", "lvo"] = "lesion",
    modality: Optional[str] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ISLES 2024 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of segmentation target. Either 'lesion' (infarct) or 'lvo'
            (large vessel occlusion).
        modality: The choice of imaging modality. See `MODALITIES` for the valid choices per 'label_choice'.
            If None, all modalities of the chosen 'label_choice' are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in LABEL_CHOICES:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose one of {LABEL_CHOICES}.")

    valid_modalities = MODALITIES[label_choice]
    if modality is not None and modality not in valid_modalities:
        raise ValueError(f"'{modality}' is not a valid modality for '{label_choice}'. Choose one of {valid_modalities}.")  # noqa

    data_dir = get_isles2024_data(path, download)
    session = SESSIONS[label_choice]

    label_paths = natsorted(
        glob(os.path.join(data_dir, "derivatives", "sub-*", session, f"*_{label_choice}-msk.nii.gz"))
    )
    assert len(label_paths) > 0, f"Could not find any '{label_choice}' labels in '{data_dir}'."

    modalities = valid_modalities if modality is None else [modality]
    if modality is None:
        image_paths = [tuple(_get_raw_path(lp, label_choice, m) for m in modalities) for lp in label_paths]
        for paths_per_case in image_paths:
            assert all(os.path.exists(p) for p in paths_per_case)
    else:
        image_paths = [_get_raw_path(lp, label_choice, modality) for lp in label_paths]
        assert all(os.path.exists(p) for p in image_paths)

    return image_paths, label_paths


def get_isles2024_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["lesion", "lvo"] = "lesion",
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ISLES 2024 dataset for ischemic stroke lesion and large vessel occlusion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of segmentation target. Either 'lesion' (infarct) or 'lvo'
            (large vessel occlusion).
        modality: The choice of imaging modality. See `MODALITIES` for the valid choices per 'label_choice'.
            If None, all modalities of the chosen 'label_choice' are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_isles2024_paths(path, label_choice, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        with_channels=modality is None,
        is_seg_dataset=True,
        **kwargs
    )
    if "sampler" in kwargs:
        for ds in dataset.datasets:
            ds.max_sampling_attempts = 5000

    return dataset


def get_isles2024_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["lesion", "lvo"] = "lesion",
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ISLES 2024 dataloader for ischemic stroke lesion and large vessel occlusion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of segmentation target. Either 'lesion' (infarct) or 'lvo'
            (large vessel occlusion).
        modality: The choice of imaging modality. See `MODALITIES` for the valid choices per 'label_choice'.
            If None, all modalities of the chosen 'label_choice' are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isles2024_dataset(path, patch_shape, label_choice, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
