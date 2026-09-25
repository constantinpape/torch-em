"""The FLARE dataset contains annotations for abdominal organ segmentation in CT scans.

It comprises the labeled training set of the FLARE22 challenge (https://flare22.grand-challenge.org):
50 abdominal CT volumes with a dense annotation of 13 organs. The 2000 unlabeled training volumes, the
50 validation volumes and the 200 test volumes of the challenge are not included here.

NOTE: The label legend is as follows:
- background: 0, liver: 1, right kidney: 2, spleen: 3, pancreas: 4, aorta: 5, inferior vena cava: 6,
  right adrenal gland: 7, left adrenal gland: 8, gallbladder: 9, esophagus: 10, stomach: 11,
  duodenum: 12, left kidney: 13
The ids were verified on the data: all 13 ids are present in every volume, the two adrenal glands (7, 8)
are by far the smallest structures, the liver (1) and stomach (11) the largest, and the gallbladder (9)
is the structure with the largest relative volume variation across cases.

NOTE: The later editions of the challenge (FLARE23, FLARE24, ...) use different data with partial
annotations of additional structures. They are not covered here and would need separate modules.

The dataset is located at https://doi.org/10.5281/zenodo.7860267 (CC BY 4.0).
See https://flare22.grand-challenge.org for the challenge.

This dataset is from the publication https://doi.org/10.1109/TMI.2022.3230667.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7860267/files/FLARE22Train.zip"
CHECKSUM = "d57e201dc9002bd4a6fc1f5e0d4525285f46bcc5be1a3c8f12543151e6675154"

ORGAN_NAMES = [
    "liver", "right_kidney", "spleen", "pancreas", "aorta", "inferior_vena_cava", "right_adrenal_gland",
    "left_adrenal_gland", "gallbladder", "esophagus", "stomach", "duodenum", "left_kidney",
]

LABEL_IDS = {"background": 0, **{name: i + 1 for i, name in enumerate(ORGAN_NAMES)}}


def get_flare_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FLARE22 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "FLARE22Train")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "FLARE22Train.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_flare_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the FLARE22 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_flare_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.nii.gz")))
    label_paths = [
        os.path.join(data_dir, "labels", os.path.basename(p).replace("_0000.nii.gz", ".nii.gz")) for p in raw_paths
    ]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_flare_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FLARE22 dataset for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_flare_paths(path, download)

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


def get_flare_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FLARE22 dataloader for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_flare_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
