"""The KiPA dataset contains annotations for kidney, renal tumor, renal artery and renal vein segmentation
in abdominal CT angiography (CTA) scans.

It comprises the training set of the KiPA22 challenge (https://kipa22.grand-challenge.org): 70 CTA volumes
that are cropped around the kidney, with a dense annotation of the four renal structures. The remaining
30 open testing volumes are distributed without annotations and are therefore not included here.

NOTE: The label legend is as follows:
- background: 0, renal vein: 1, kidney: 2, renal artery: 3, renal tumor: 4
The ids were verified on the data by comparing the per-label volumes with the official dataset statistics.

The data is a redistribution of the official challenge data at https://huggingface.co/datasets/YongchengYAO/KiPA22
(CC BY-NC 4.0). See https://kipa22.grand-challenge.org for the official release.

This dataset is from the publication https://doi.org/10.1016/j.media.2021.102055.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/YongchengYAO/KiPA22/resolve/main/train.zip"
CHECKSUM = "43ea3667cb7bb5e0e465f203ce2ea9008034eecc5ddede73c69983763b3a1a65"

LABEL_IDS = {"background": 0, "renal_vein": 1, "kidney": 2, "renal_artery": 3, "renal_tumor": 4}


def get_kipa_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the KiPA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "train.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_kipa_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the KiPA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_kipa_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "image", "*.nii.gz")))
    label_paths = [p.replace(os.sep + "image" + os.sep, os.sep + "label" + os.sep) for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_kipa_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the KiPA dataset for kidney, renal tumor, renal artery and renal vein segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_kipa_paths(path, download)

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


def get_kipa_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the KiPA dataloader for kidney, renal tumor, renal artery and renal vein segmentation.

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
    dataset = get_kipa_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
