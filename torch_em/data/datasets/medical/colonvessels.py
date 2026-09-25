"""The ColonVessels 2026 dataset contains annotations for segmentation of mesenteric arteries and veins
in dual-phase contrast-enhanced abdominal CT (CECT).

The dataset consists of 60 CECT studies (50 with both arterial and venous phases, 10 with the venous phase
only) from adult patients, imaged on a Siemens SOMATOM Force scanner. Manual 3D segmentations of the
mesenteric arteries (in the arterial phase) and veins (in the venous phase) were performed in 3D Slicer.

This dataset is from the publication https://doi.org/10.1038/s41597-026-07303-2.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/17407158/files/data.zip"
CHECKSUM = "a5ae3148a54805165388b648a1a45c793ec535ef4eea3b68c679704b3c6db7b5"

VESSEL_TYPES = ["arteries", "veins"]


def _convert_image_to_nifti(nrrd_path: str, nifti_path: str) -> None:
    if os.path.exists(nifti_path):
        return

    import SimpleITK as sitk

    image = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(image, nifti_path, useCompression=True)


def _convert_mask_to_nifti(nrrd_path: str, nifti_path: str) -> None:
    """Convert a 3D Slicer '.seg.nrrd' segmentation to a single-channel binary foreground mask.

    The segmentations combine multiple named vessel segments (e.g. individual named arteries) into a small
    number of shared binary labelmap layers (a 4th array axis), to avoid one layer per segment. Since this
    dataset is used for binary vessel (foreground) segmentation, the layers are collapsed into one channel
    by marking a voxel as foreground if it is non-zero in any layer.
    """
    if os.path.exists(nifti_path):
        return

    import SimpleITK as sitk

    seg = sitk.ReadImage(nrrd_path)
    arr = sitk.GetArrayFromImage(seg)
    if arr.ndim == 4:  # (Z, Y, X, num_layers) -> collapse the layers into one binary foreground mask.
        arr = (arr > 0).any(axis=-1)
    mask = (arr > 0).astype("uint8")

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetSpacing(seg.GetSpacing()[:3])
    mask_image.SetOrigin(seg.GetOrigin()[:3])
    sitk.WriteImage(mask_image, nifti_path, useCompression=True)


def get_colonvessels_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ColonVessels 2026 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_colonvessels_paths(
    path: Union[os.PathLike, str],
    vessel_type: Optional[Literal["arteries", "veins"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ColonVessels 2026 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        vessel_type: The choice of vessel type to segment.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_colonvessels_data(path, download)

    if vessel_type is None:
        vessel_types = VESSEL_TYPES
    else:
        assert vessel_type in VESSEL_TYPES, f"'{vessel_type}' is not a valid vessel type."
        vessel_types = [vessel_type]

    phase_per_vessel_type = {"arteries": "Arterial", "veins": "Venous"}

    nifti_dir = os.path.join(path, "nifti")
    os.makedirs(nifti_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for patient_dir in natsorted(glob(os.path.join(data_dir, "pat_*"))):
        patient_id = os.path.basename(patient_dir)
        for _vessel_type in vessel_types:
            phase = phase_per_vessel_type[_vessel_type]
            label_name = _vessel_type.capitalize()
            image_nrrd = os.path.join(patient_dir, f"{patient_id}_{phase}_Phase_CT.nrrd")
            gt_nrrd = os.path.join(patient_dir, f"{patient_id}_{phase}_Phase_{label_name}.seg.nrrd")
            if not (os.path.exists(image_nrrd) and os.path.exists(gt_nrrd)):
                continue

            image_path = os.path.join(nifti_dir, f"{Path(image_nrrd).stem}.nii.gz")
            gt_path = os.path.join(nifti_dir, f"{Path(gt_nrrd).stem.replace('.seg', '')}.nii.gz")

            _convert_image_to_nifti(image_nrrd, image_path)
            _convert_mask_to_nifti(gt_nrrd, gt_path)

            image_paths.append(image_path)
            gt_paths.append(gt_path)

    assert len(image_paths) > 0 and len(image_paths) == len(gt_paths), \
        f"Could not find a matching number of images and labels in '{data_dir}'."

    return image_paths, gt_paths


def get_colonvessels_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    vessel_type: Optional[Literal["arteries", "veins"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ColonVessels 2026 dataset for segmentation of mesenteric arteries and veins.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        vessel_type: The choice of vessel type to segment.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_colonvessels_paths(path, vessel_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_colonvessels_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    vessel_type: Optional[Literal["arteries", "veins"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ColonVessels 2026 dataloader for segmentation of mesenteric arteries and veins.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        vessel_type: The choice of vessel type to segment.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_colonvessels_dataset(path, patch_shape, vessel_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
