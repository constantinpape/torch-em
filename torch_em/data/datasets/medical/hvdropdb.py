"""The HVDROPDB dataset contains annotations for optic disc, blood vessel, and
demarcation line / ridge segmentation in fundus images of premature infants,
for the task of retinopathy of prematurity (ROP) screening.

This dataset is located at https://doi.org/10.17632/xw5xc7xrmp.3, under the CC BY 4.0 license.
The dataset is from the publication https://doi.org/10.1016/j.dib.2023.109839.
Please cite it if you use this dataset for your research.

The images were acquired with two imaging systems (RetCam and Neo) at PBMA's H.V. Desai Eye
Hospital, Pune, screening preterm infants for ROP. Ground truth masks were prepared manually
(with Adobe Photoshop) by a group of ROP experts, separately for the optic disc, blood vessels,
and the demarcation line / ridge.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/xw5xc7xrmp/files/dfa3ec48-c763-439f-905a-bb0904ea840a/file_downloaded"  # noqa
CHECKSUM = "0a3aa96bbe489fc2f9103bac80fdb3eddc4fc91533e87555446970e9d021c1d2"

# Maps the structure to segment to the folder name and the image / mask subfolder prefixes
# used per imaging device inside the archive.
STRUCTURES = {
    "optic_disc": ("HVDROPDB-OD", {"RetCam": "Retcam_OpticDisc", "Neo": "Neo_OpticDisc"}),
    "vessels": ("HVDROPDB-BV", {"RetCam": "RetCam_Vessels", "Neo": "Neo_Vessels"}),
    "ridge": ("HVDROPDB-RIDGE", {"RetCam": "RetCam_Ridge", "Neo": "Neo_Ridge"}),
}


def get_hvdropdb_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HVDROPDB dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "HVDROPDB_RetCam_Neo_Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "HVDROPDB_RetCam_Neo_Segmentation.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _binarize_mask(mask_path, dst_dir):
    dst_path = os.path.join(dst_dir, Path(mask_path).stem + ".png")
    if os.path.exists(dst_path):
        return dst_path

    os.makedirs(dst_dir, exist_ok=True)
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype("uint8")
    imageio.imwrite(dst_path, mask)
    return dst_path


def get_hvdropdb_paths(
    path: Union[os.PathLike, str],
    structure: Literal["optic_disc", "vessels", "ridge"],
    device: Literal["RetCam", "Neo"] = "RetCam",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HVDROPDB data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        structure: The choice of anatomical structure to segment.
        device: The choice of imaging device used to acquire the fundus images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert structure in STRUCTURES, f"'{structure}' is not a valid structure choice."
    assert device in ["RetCam", "Neo"], f"'{device}' is not a valid device choice."

    data_dir = get_hvdropdb_data(path=path, download=download)

    folder, prefixes = STRUCTURES[structure]
    prefix = prefixes[device]

    image_dir = os.path.join(data_dir, folder, f"{prefix}_images")
    gt_dir = os.path.join(data_dir, folder, f"{prefix}_masks")

    image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
    raw_gt_paths = natsorted(glob(os.path.join(gt_dir, "*.png")))

    assert len(image_paths) == len(raw_gt_paths) and len(image_paths) > 0

    # The shipped masks are RGB images (with an identical value per channel). We binarize
    # them into single-channel masks here, as expected by `torch_em.default_segmentation_dataset`.
    binary_gt_dir = os.path.join(data_dir, folder, f"{prefix}_masks_binary")
    gt_paths = [_binarize_mask(p, binary_gt_dir) for p in raw_gt_paths]

    return image_paths, gt_paths


def get_hvdropdb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    structure: Literal["optic_disc", "vessels", "ridge"],
    device: Literal["RetCam", "Neo"] = "RetCam",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HVDROPDB dataset for optic disc / vessel / demarcation line segmentation in fundus images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        structure: The choice of anatomical structure to segment.
        device: The choice of imaging device used to acquire the fundus images.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_hvdropdb_paths(path=path, structure=structure, device=device, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs,
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_hvdropdb_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    structure: Literal["optic_disc", "vessels", "ridge"],
    device: Literal["RetCam", "Neo"] = "RetCam",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HVDROPDB dataloader for optic disc / vessel / demarcation line segmentation in fundus images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        structure: The choice of anatomical structure to segment.
        device: The choice of imaging device used to acquire the fundus images.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hvdropdb_dataset(path, patch_shape, structure, device, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
