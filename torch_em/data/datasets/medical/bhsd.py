"""The BHSD dataset contains annotations for multiclass intracranial hemorrhage segmentation in
brain CT.

The Brain Hemorrhage Segmentation Dataset (BHSD) provides 192 non-contrast head CT volumes with
pixel-level annotations of five intracranial hemorrhage (ICH) subtypes, see `LABEL_IDS`. The
dataset also contains a much larger set of 1980 volumes with only slice-level (i.e. not
pixel-level) annotations, which is not exposed by this module.

The data is hosted on Hugging Face at https://huggingface.co/datasets/Wendy-Fly/BHSD and is
distributed under the MIT license.

This dataset is from the publication https://doi.org/10.1007/978-3-031-45673-2_15.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/Wendy-Fly/BHSD/resolve/main/label_192.zip"
CHECKSUM = "582bf184af993541a4958a4d209a6a44e3bbe702a5daefaf9fb1733a4e7a6e39"

LABEL_IDS = {
    0: "background", 1: "epidural", 2: "intraparenchymal", 3: "intraventricular", 4: "subarachnoid", 5: "subdural",
}
"""The label ids of the intracranial hemorrhage subtypes, as defined by the dataset authors."""


def get_bhsd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BHSD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "label_192")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "label_192.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_bhsd_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the BHSD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bhsd_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "ground truths", "*.nii.gz")))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_bhsd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BHSD dataset for multiclass intracranial hemorrhage segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_bhsd_paths(path, download)

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


def get_bhsd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BHSD dataloader for multiclass intracranial hemorrhage segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bhsd_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
