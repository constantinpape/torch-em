"""MSWAL is the first 3D dataset for multi-class segmentation of whole abdominal lesions in CT.

The dataset consists of 484 publicly released training CT volumes (out of 694 total scans acquired at a single
hospital; the 210 held-out test volumes are not released), annotated for seven lesion classes: gallstones,
kidney stones, liver tumors, kidney tumors, pancreatic cancer, liver cysts and kidney cysts. The label ids are
described in `LABEL_IDS`.

The dataset is located at https://huggingface.co/datasets/zhaodongwu/MSWAL (openly accessible, no gating).
This dataset is from the publication https://doi.org/10.1007/978-3-032-04937-7_36 (also on arXiv at
https://doi.org/10.48550/arXiv.2503.13560). Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "zhaodongwu/MSWAL"

LABEL_IDS = {
    "background": 0,
    "gallstone": 1,
    "kidney_stone": 2,
    "liver_tumor": 3,
    "kidney_tumor": 4,
    "pancreatic_cancer": 5,
    "liver_cyst": 6,
    "kidney_cyst": 7,
}
"""The mapping of MSWAL label ids to the corresponding lesion classes."""


def get_mswal_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MSWAL dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(os.path.join(data_dir, "imagesTr")) and os.path.exists(os.path.join(data_dir, "labelsTr")):
        return data_dir

    if not download:
        raise RuntimeError("The dataset is not found and download is set to False.")

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install 'huggingface_hub' to download the MSWAL dataset: 'pip install huggingface_hub'."
        )

    os.makedirs(data_dir, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=data_dir, allow_patterns=["imagesTr/*", "labelsTr/*"]
    )

    return data_dir


def get_mswal_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the MSWAL data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mswal_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    if len(raw_paths) == 0 or len(raw_paths) != len(label_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_mswal_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], resize_inputs: bool = False,
    download: bool = False, **kwargs
) -> Dataset:
    """Get the MSWAL dataset for multi-class whole abdominal lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mswal_paths(path, download)

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


def get_mswal_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], resize_inputs: bool = False,
    download: bool = False, **kwargs
) -> DataLoader:
    """Get the MSWAL dataloader for multi-class whole abdominal lesion segmentation.

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
    dataset = get_mswal_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
