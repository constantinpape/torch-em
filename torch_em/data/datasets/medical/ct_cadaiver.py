"""The CadAIver dataset contains annotations for vertebrae segmentation
in CT scans.

The dataset is located at https://zenodo.org/records/10053317.
This dataset is from the publication https://doi.org/10.1038/s41597-024-03191-6.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10053317/files/CadAIver%20study.zip"
CHECKSUM = ""


def get_ct_cadaiver_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CadAIver dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "CadAIver study")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "CadAIver study.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_ct_cadaiver_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[int], List[int]]:
    """Get paths to the CadAIver data.

    Args:
        path: Filepath to a folder where the downloaded data is stored.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ct_cadaiver_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "Images", "*.nii.gz")))
    label_paths = [p.replace("Images", "Segmentations") for p in raw_paths]
    label_paths = [p.replace(".nii.gz", "_seg.nii.gz") for p in label_paths]

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_ct_cadaiver_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CadAIver dataset for vertebrae segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ct_cadaiver_paths(path, download)

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
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_ct_cadaiver_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CadAIver dataset for vertebrae segmentation.

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
    dataset = get_ct_cadaiver_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
