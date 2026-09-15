"""The COVID-19-20 dataset contains annotations for COVID-19 lung lesion segmentation in chest CT scans.

It comprises the training set of the COVID-19-20 Lung CT Lesion Segmentation Challenge
(https://covid-segmentation.grand-challenge.org/COVID-19-20/): 199 non-contrast chest CT volumes of
COVID-19 positive patients with a radiologist-verified binary lesion mask. The 50 validation volumes
and the 46 test volumes of the challenge are distributed without annotations and are not included here.

NOTE: The label legend is as follows:
- background: 0, covid-19 lesion: 1
Verified on the data: the label volumes only contain the ids 0 and 1.

NOTE: This is a different dataset than the one in `torch_em.data.datasets.medical.covid19_seg`, which
contains the 20 annotated volumes from https://doi.org/10.5281/zenodo.3757476.

The data is a redistribution of the official challenge data at
https://huggingface.co/datasets/MedOtter/COVID-19-20 (CC BY 4.0). The official release at
https://covid-segmentation.grand-challenge.org/Data/ requires registration, so please make sure that
you are allowed to use the data for your purpose.

This dataset is from the publication https://doi.org/10.1016/j.media.2022.102605.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HF_REPO = "MedOtter/COVID-19-20"

LABEL_IDS = {"background": 0, "covid19_lesion": 1}


def get_covid19_20_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the COVID-19-20 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "Train")
    if os.path.exists(data_dir):
        return data_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at '{path}', but download was set to False.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("'huggingface_hub' is required to download COVID-19-20. Install it via conda/pip.")

    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=path, allow_patterns=["Train/*"])

    return data_dir


def get_covid19_20_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the COVID-19-20 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_covid19_20_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*_ct.nii.gz")))
    label_paths = [p.replace("_ct.nii.gz", "_seg.nii.gz") for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_covid19_20_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the COVID-19-20 dataset for COVID-19 lung lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_covid19_20_paths(path, download)

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


def get_covid19_20_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the COVID-19-20 dataloader for COVID-19 lung lesion segmentation.

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
    dataset = get_covid19_20_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
