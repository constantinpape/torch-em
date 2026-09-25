"""The MMOTU dataset contains annotations for ovarian tumor segmentation in 2d ultrasound
and contrast-enhanced ultrasound (CEUS) images.

The dataset contains 2D B-mode ultrasound images (`OTU_2D`) and CEUS images (`OTU_CEUS`)
of ovarian tumors collected at Beijing Shijitan Hospital, Capital Medical University, with
pixel-wise tumor masks and global tumor-type labels.

This mirror of the dataset is located at https://doi.org/10.6084/m9.figshare.25058690.v2
(CC BY 4.0). The original dataset and code are at https://github.com/cv516Buaa/MMOTU_DS2Net.
This dataset is from the publication https://doi.org/10.1016/j.patcog.2025.112311.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/44222642"
CHECKSUM = "5343647807cf34b507b66acd752dd94de265c6d1a7fde9cf8aa441a7283cde3e"


def get_mmotu_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MMOTU dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_mmotu_paths(
    path: Union[os.PathLike, str],
    modality: Optional[Literal["2d", "ceus"]] = None,
    split: Optional[Literal["train", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the MMOTU data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of imaging modality, either conventional 2d ultrasound ('2d')
            or contrast-enhanced ultrasound ('ceus').
        split: The choice of data split, only valid for the 2d modality.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mmotu_data(path=path, download=download)

    if modality is None:
        modality = "*"
    elif modality not in ["2d", "ceus"]:
        raise ValueError(f"'{modality}' is not a valid modality choice.")

    image_paths, gt_paths = [], []

    if modality in ("2d", "*"):
        if split is None:
            split = "*"
        elif split not in ["train", "test"]:
            raise ValueError(f"'{split}' is not a valid split choice.")

        if split in ("train", "*"):
            image_paths.extend(sorted(glob(os.path.join(data_dir, "OTU_2D", "train", "train_image", "*.JPG"))))
            gt_paths.extend(sorted(glob(os.path.join(data_dir, "OTU_2D", "train", "train_label", "label", "*.PNG"))))

        if split in ("test", "*"):
            image_paths.extend(sorted(glob(os.path.join(data_dir, "OTU_2D", "test", "image", "*.JPG"))))
            gt_paths.extend(sorted(glob(os.path.join(data_dir, "OTU_2D", "test", "label", "black_write", "*.PNG"))))

    if modality in ("ceus", "*"):
        image_paths.extend(sorted(glob(os.path.join(data_dir, "OTU_CEUS", "image", "*.JPG"))))
        gt_paths.extend(sorted(glob(os.path.join(data_dir, "OTU_CEUS", "label", "*.PNG"))))

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_mmotu_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    modality: Optional[Literal["2d", "ceus"]] = None,
    split: Optional[Literal["train", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MMOTU dataset for ovarian tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of imaging modality, either conventional 2d ultrasound ('2d')
            or contrast-enhanced ultrasound ('ceus').
        split: The choice of data split, only valid for the 2d modality.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_mmotu_paths(path, modality, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
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


def get_mmotu_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    modality: Optional[Literal["2d", "ceus"]] = None,
    split: Optional[Literal["train", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MMOTU dataloader for ovarian tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of imaging modality, either conventional 2d ultrasound ('2d')
            or contrast-enhanced ultrasound ('ceus').
        split: The choice of data split, only valid for the 2d modality.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mmotu_dataset(path, patch_shape, modality, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
