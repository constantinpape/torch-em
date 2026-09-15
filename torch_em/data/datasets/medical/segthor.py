"""The SegTHOR dataset contains annotations for thoracic organ-at-risk segmentation in CT scans
of lung or esophageal cancer patients.

It comprises the training set of the SegTHOR challenge (https://competitions.codalab.org/competitions/21145):
40 thoracic CT scans with a dense annotation of the esophagus, heart, trachea and aorta. The 20 test volumes
of the challenge are distributed without annotations and are therefore not included here.

NOTE: The label legend is as follows:
- background: 0, esophagus: 1, heart: 2, trachea: 3, aorta: 4
The ids were verified on the data: label 2 is by far the largest structure and sits anteriorly in the lower
thorax (heart), label 3 is confined to the upper thorax anterior to label 1 (trachea vs. esophagus), and
label 4 spans the whole craniocaudal extent (aorta).

The data is a redistribution of the official challenge data at https://doi.org/10.5281/zenodo.16663661
(CC BY 4.0). The official release at https://competitions.codalab.org/competitions/21145 requires a signed
user agreement, so please make sure that you are allowed to use the data for your purpose.

This dataset is from the publication https://doi.org/10.1109/ICPR48806.2021.9411873.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16663661/files/SegTHOR.zip"
CHECKSUM = "5dba9bcc681c895b7d7b59cf5a26cae570bc8bba02c74f9cb128f3292410fb15"

LABEL_IDS = {"background": 0, "esophagus": 1, "heart": 2, "trachea": 3, "aorta": 4}


def get_segthor_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SegTHOR dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "SegTHOR")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "SegTHOR.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_segthor_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the SegTHOR data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_segthor_data(path, download)

    case_dirs = natsorted(glob(os.path.join(data_dir, "Patient_*")))
    raw_paths = [os.path.join(p, f"{os.path.basename(p)}.nii.gz") for p in case_dirs]
    label_paths = [os.path.join(p, "GT.nii.gz") for p in case_dirs]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_segthor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegTHOR dataset for thoracic organ-at-risk segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_segthor_paths(path, download)

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


def get_segthor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegTHOR dataloader for thoracic organ-at-risk segmentation.

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
    dataset = get_segthor_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
