"""The CSC Fundus Segmentation dataset contains annotations for subretinal fluid (SRF)
segmentation in fundus photographs of eyes with central serous chorioretinopathy (CSC).

This dataset is located at https://doi.org/10.17632/4k64fwnp4k.3, under the CC BY 4.0 license.
The dataset is from the publication https://doi.org/10.1167/tvst.11.2.11.
Please cite it if you use this dataset for your research.

The dataset comprises fundus photographs from 194 eyes with CSC, with the SRF area manually
segmented by three ophthalmologists (two licensed ophthalmologists and one ophthalmology
resident), provided here as `grader1`, `grader2` and `grader3`. Each image is shipped as a
paired image, with the raw fundus photograph on the left half and the corresponding
segmentation rendered as a pink overlay on a white background on the right half. We split
these paired images here into the raw image and a binarized foreground / background mask.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/4k64fwnp4k/files/{}/file_downloaded"
FNAMES_TO_IDS = {
    "Grader_1.zip": "e53c7210-9e09-4e0e-8f36-87103128fe38",
    "Grader_2.zip": "2b551de5-1855-47cf-88cf-6f55eb26d001",
    "Grader_3.zip": "fd6214bf-a33a-413c-948c-7430beebe286",
}
CHECKSUMS = {
    "Grader_1.zip": "e9820a9cd2b13490f972e2c58b316b0ab62dccef00564d1c1cc768e55658b435",
    "Grader_2.zip": "78ca3f61b58cf85ef2619b031c6c4a4a95f9af7f8bcbc567e9d295f6ead1120e",
    "Grader_3.zip": "a2c9699125c056cb8f46d073a72f93eec1e3a57b77370eaa15ed1a3475dba07a",
}


def get_csc_fundus_segmentation_data(
    path: Union[os.PathLike, str], grader: Literal["grader1", "grader2", "grader3"], download: bool = False
) -> str:
    """Download the CSC Fundus Segmentation dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        grader: The choice of annotator whose segmentations are used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    assert grader in ["grader1", "grader2", "grader3"], f"'{grader}' is not a valid grader choice."

    fname = f"Grader_{grader[-1]}.zip"
    data_dir = os.path.join(path, Path(fname).stem)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, fname)
    util.download_source(
        path=zip_path, url=URL.format(FNAMES_TO_IDS[fname]), download=download, checksum=CHECKSUMS[fname]
    )
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _split_paired_image(paired_path, image_dir, gt_dir):
    fname = Path(paired_path).stem
    image_path = os.path.join(image_dir, f"{fname}.png")
    gt_path = os.path.join(gt_dir, f"{fname}.png")

    if os.path.exists(image_path) and os.path.exists(gt_path):
        return image_path, gt_path

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    paired_image = imageio.imread(paired_path)
    width = paired_image.shape[1]
    half = width // 2

    raw = paired_image[:, :half]

    # The right half renders the SRF segmentation as a pink overlay on a white background.
    # The seam between the concatenated halves introduces a single column of black JPEG
    # compression bleed at the left edge of the mask half, which we remove before thresholding.
    mask_rgb = paired_image[:, half:].copy()
    mask_rgb[:, 0] = 255
    mask = (np.abs(255 - mask_rgb.astype(int)).sum(axis=-1) > 30).astype("uint8")

    imageio.imwrite(image_path, raw)
    imageio.imwrite(gt_path, mask)

    return image_path, gt_path


def get_csc_fundus_segmentation_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    grader: Literal["grader1", "grader2", "grader3"] = "grader1",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CSC Fundus Segmentation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        grader: The choice of annotator whose segmentations are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in ["train", "test"], f"'{split}' is not a valid split."

    data_dir = get_csc_fundus_segmentation_data(path=path, grader=grader, download=download)

    paired_paths = natsorted(glob(os.path.join(data_dir, split, "*.jpg")))
    assert len(paired_paths) > 0, f"No images found for split '{split}' and grader '{grader}'."

    image_dir = os.path.join(data_dir, "preprocessed", split, "images")
    gt_dir = os.path.join(data_dir, "preprocessed", split, "masks")

    image_paths, gt_paths = [], []
    for paired_path in paired_paths:
        image_path, gt_path = _split_paired_image(paired_path, image_dir, gt_dir)
        image_paths.append(image_path)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_csc_fundus_segmentation_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    grader: Literal["grader1", "grader2", "grader3"] = "grader1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CSC Fundus Segmentation dataset for subretinal fluid segmentation in fundus photographs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        grader: The choice of annotator whose segmentations are used.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_csc_fundus_segmentation_paths(
        path=path, split=split, grader=grader, download=download
    )

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


def get_csc_fundus_segmentation_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    grader: Literal["grader1", "grader2", "grader3"] = "grader1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CSC Fundus Segmentation dataloader for subretinal fluid segmentation in fundus photographs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        grader: The choice of annotator whose segmentations are used.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_csc_fundus_segmentation_dataset(
        path, patch_shape, split, grader, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
