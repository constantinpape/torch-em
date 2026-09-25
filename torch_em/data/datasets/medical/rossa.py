"""The ROSSA dataset contains annotations for retinal vessel segmentation in OCTA images.

It comprises 918 OCTA images: 300 images with manually verified vessel masks (split into
'train_manual', 'val' and 'test' folders of 100 images each) and 618 further images with
vessel masks that were generated semi-automatically with the Segment Anything Model, stored
in the 'train_sam' folder. The manually annotated subset is the reliable one for evaluation
and is used by default; the SAM-assisted subset trades annotation quality for scale.

The dataset is located at https://github.com/nhjydywd/OCTA-FRNet (MIT license).
This dataset is from the publication https://doi.org/10.48550/arXiv.2309.09483.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/nhjydywd/OCTA-FRNet/archive/53e87f8c9b9392c1dcababea07e986f111f4017c.zip"
CHECKSUM = "2d408a9b124aadbb5ebdf49f2aeb5e24c0fee2ef77e6a0be06648f12df8633ca"

ANNOTATION_DIRS = {
    "manual": ["train_manual", "val", "test"],
    "sam_assisted": ["train_sam"],
}


def get_rossa_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ROSSA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded ROSSA images and vessel masks.
    """
    data_dir = os.path.join(path, "OCTA-FRNet-53e87f8c9b9392c1dcababea07e986f111f4017c", "dataset", "ROSSA")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "OCTA-FRNet.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_rossa_paths(
    path: Union[os.PathLike, str],
    annotation: Literal["manual", "sam_assisted", "all"] = "manual",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ROSSA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotation source. Either the 300 manually annotated images
            ('manual', the default), the 618 SAM-assisted images ('sam_assisted') or both ('all').
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_rossa_data(path=path, download=download)

    if annotation == "all":
        sub_dirs = ANNOTATION_DIRS["manual"] + ANNOTATION_DIRS["sam_assisted"]
    elif annotation in ANNOTATION_DIRS:
        sub_dirs = ANNOTATION_DIRS[annotation]
    else:
        raise ValueError(f"'{annotation}' is not a valid annotation choice.")

    image_paths, gt_paths = [], []
    for sub_dir in sub_dirs:
        image_paths.extend(natsorted(glob(os.path.join(data_dir, sub_dir, "image", "*.png"))))
        gt_paths.extend(natsorted(glob(os.path.join(data_dir, sub_dir, "label", "*.png"))))

    return image_paths, gt_paths


def get_rossa_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotation: Literal["manual", "sam_assisted", "all"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ROSSA dataset for retinal vessel segmentation in OCTA images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotation source. Either the 300 manually annotated images
            ('manual', the default), the 618 SAM-assisted images ('sam_assisted') or both ('all').
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_rossa_paths(path, annotation, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
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


def get_rossa_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotation: Literal["manual", "sam_assisted", "all"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ROSSA dataloader for retinal vessel segmentation in OCTA images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotation source. Either the 300 manually annotated images
            ('manual', the default), the 618 SAM-assisted images ('sam_assisted') or both ('all').
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rossa_dataset(path, patch_shape, annotation, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
