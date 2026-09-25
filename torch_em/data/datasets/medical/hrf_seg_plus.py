"""The HRF-Seg+ dataset contains manual, expert-reviewed multi-structure annotations for the fundus
images of the HRF dataset (`torch_em.data.datasets.medical.hrf`): the optic disc, the optic cup, the
retinal vessels and the peripapillary alpha and beta zones.

NOTE: HRF-Seg+ ships its own copy of the 45 HRF fundus images (resized to 500x500), rather than
reusing the original high-resolution HRF images directly, so this dataset is implemented as a
standalone module rather than as an extension of `torch_em.data.datasets.medical.hrf`.

The archive stores five annotated structures as separate folders: 'Folder_1_Optic_Disc',
'Folder_2_Optic_Cup', 'Folder_3_Vessels' and 'Folder_4_Alpha_Beta_Zones' (each structure as an RGBA
image, with the structure's pixels opaque), plus a merged multi-class mask per image in
'Folder_5_Ground_Truth_Masks' (an RGB image, colored per the 'class_dict.csv' palette). We use the
merged masks and map each pixel to the class with the closest reference color (to be robust to the
lossy-compression-free but anti-aliased mask boundaries), yielding semantic labels: 0 (background),
1 (optic disc), 2 (retinal vessels), 3 (optic cup), 4 (beta zone) and 5 (alpha zone).

This dataset is located at https://doi.org/10.5281/zenodo.16744782.
The dataset is licensed under CC-BY-4.0.
Please cite the dataset if you use it in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16744782/files/HRF-Seg%2B.zip"
CHECKSUM = "e4bf59bb820147aa28158b1ed2228cf59cc426605d023ab2edca651fcf3f3411"

LABEL_COLORS = {
    0: (0, 0, 0),  # unlabeled
    1: (128, 64, 128),  # opticdisc
    2: (254, 148, 12),  # retinalvessels
    3: (130, 76, 0),  # opticcup
    4: (190, 250, 190),  # betazone
    5: (112, 150, 146),  # alphazone
}


def get_hrf_seg_plus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HRF-Seg+ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "HRF-Seg+")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "HRF-Seg+.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_hrf_seg_plus_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the HRF-Seg+ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_hrf_seg_plus_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "Folder_6_Original_Images", "*")))
    mask_paths = natsorted(glob(os.path.join(data_dir, "Folder_5_Ground_Truth_Masks", "*.png")))
    assert image_paths and len(image_paths) == len(mask_paths)

    label_dir = os.path.join(data_dir, "preprocessed_labels")
    os.makedirs(label_dir, exist_ok=True)

    reference_colors = np.array(list(LABEL_COLORS.values()))

    label_paths = []
    for mask_path in tqdm(mask_paths, desc="Preprocessing labels"):
        label_path = os.path.join(label_dir, f"{Path(mask_path).stem}.tif")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue

        mask = imageio.imread(mask_path)[..., :3].astype("float32")
        distances = np.linalg.norm(mask[..., None, :] - reference_colors[None, None, :, :], axis=-1)
        semantic_mask = np.argmin(distances, axis=-1).astype("uint8")
        imageio.imwrite(label_path, semantic_mask, compression="zlib")

    return image_paths, label_paths


def get_hrf_seg_plus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HRF-Seg+ dataset for optic disc, optic cup, vessel and peripapillary zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_hrf_seg_plus_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_hrf_seg_plus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HRF-Seg+ dataloader for optic disc, optic cup, vessel and peripapillary zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hrf_seg_plus_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
