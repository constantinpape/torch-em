"""The HNTS-MRG dataset contains annotations for head and neck tumor segmentation in T2-weighted MRI.

The dataset is the training data of the HNTS-MRG 2024 challenge (https://hntsmrg24.grand-challenge.org).
It consists of 150 patients, each with a pre-radiotherapy (pre-RT) and a mid-radiotherapy (mid-RT) T2-weighted MRI
and the corresponding tumor segmentation. The labels are: 0 = background, 1 = primary gross tumor volume (GTVp),
2 = metastatic lymph nodes (GTVn). NOTE: The mid-RT masks of the patients 21, 25, 29 and 42 are empty
(complete response to the therapy). For the mid-RT timepoint the dataset additionally provides the pre-RT image and
mask registered to the mid-RT image space ('*_preRT_T2_registered.nii.gz', '*_preRT_mask_registered.nii.gz').

The dataset is located at https://doi.org/10.5281/zenodo.11199559.

This dataset is from the publication https://doi.org/10.1007/978-3-031-83274-1_1.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/11199559/files/HNTSMRG24_train.zip"
CHECKSUM = "675e3c56509b0b5bcaad1291990dfa4a8ef054bb468a71b6cd08189a5e67bfb6"


def get_hntsmrg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HNTS-MRG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "HNTSMRG24_train")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "HNTSMRG24_train.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_hntsmrg_paths(
    path: Union[os.PathLike, str], timepoint: Literal['pre', 'mid'] = "pre", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the HNTS-MRG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        timepoint: The timepoint of the scans. Either 'pre' (pre-radiotherapy) or 'mid' (mid-radiotherapy).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_hntsmrg_data(path, download)

    if timepoint not in ("pre", "mid"):
        raise ValueError(f"'{timepoint}' is not a valid timepoint. Choose either 'pre' or 'mid'.")

    tp = f"{timepoint}RT"
    raw_paths = natsorted(glob(os.path.join(data_dir, "*", tp, f"*_{tp}_T2.nii.gz")))
    label_paths = [p.replace(f"_{tp}_T2.nii.gz", f"_{tp}_mask.nii.gz") for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_hntsmrg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    timepoint: Literal['pre', 'mid'] = "pre",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HNTS-MRG dataset for head and neck tumor segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        timepoint: The timepoint of the scans. Either 'pre' (pre-radiotherapy) or 'mid' (mid-radiotherapy).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hntsmrg_paths(path, timepoint, download)

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


def get_hntsmrg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    timepoint: Literal['pre', 'mid'] = "pre",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HNTS-MRG dataloader for head and neck tumor segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        timepoint: The timepoint of the scans. Either 'pre' (pre-radiotherapy) or 'mid' (mid-radiotherapy).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hntsmrg_dataset(path, patch_shape, timepoint, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
