"""The CirrMRI600+ dataset contains annotations for liver segmentation in abdominal MRI of patients
with liver cirrhosis.

The dataset consists of 628 abdominal MRI volumes: 310 T1-weighted and 318 T2-weighted scans of
cirrhotic patients, acquired with multiple scanners and annotated by physicians with binary liver masks
(1 = liver). The scans of the two sequences are distributed as separate archives, each with an official
'train' / 'val' / 'test' split, and can be selected via the 'sequence' argument.

NOTE: This loader uses the 3D archives of the cirrhotic patients ('Cirrhosis_T1_3D' and
'Cirrhosis_T2_3D'). The 2D slice version of the T2 scans and the archive of healthy subjects are not used.

The data is located at https://doi.org/10.17605/OSF.IO/CUK24 and released under a CC-BY-NC-4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-025-05201-7.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "T1": "https://osf.io/download/47rxy/",
    "T2": "https://osf.io/download/72df5/",
}

CHECKSUMS = {
    "T1": "36f533f0496ffc6bfb5c243a25a5e3927c682a538660484540e2cf565ce434d1",
    "T2": "539f0bc3bd3cdb0bb85b465ace8002bcb15631e8b59b81ca04dec0a7c287fda9",
}

SEQUENCES = list(URLS.keys())
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def get_cirrmri600_data(
    path: Union[os.PathLike, str], sequence: Literal["T1", "T2"] = "T2", download: bool = False
) -> str:
    """Download the CirrMRI600+ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence: The choice of MRI sequence. Either 'T1' or 'T2'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if sequence not in SEQUENCES:
        raise ValueError(f"'{sequence}' is not a valid sequence. Choose one of {SEQUENCES}.")

    data_dir = os.path.join(path, f"Cirrhosis_{sequence}_3D")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"Cirrhosis_{sequence}_3D.zip")
    util.download_source(path=zip_path, url=URLS[sequence], download=download, checksum=CHECKSUMS[sequence])
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create the expected folder in '{path}'."

    return data_dir


def get_cirrmri600_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    sequence: Literal["T1", "T2"] = "T2",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CirrMRI600+ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        sequence: The choice of MRI sequence. Either 'T1' or 'T2'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {list(SPLITS)}.")

    data_dir = get_cirrmri600_data(path, sequence, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, f"{SPLITS[split]}_images", "*.nii.gz")))
    label_paths = [p.replace(f"{SPLITS[split]}_images", f"{SPLITS[split]}_masks") for p in raw_paths]

    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_cirrmri600_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"],
    sequence: Literal["T1", "T2"] = "T2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CirrMRI600+ dataset for liver segmentation in cirrhotic MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        sequence: The choice of MRI sequence. Either 'T1' or 'T2'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cirrmri600_paths(path, split, sequence, download)

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
        ndim=3,
        **kwargs
    )


def get_cirrmri600_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"],
    sequence: Literal["T1", "T2"] = "T2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CirrMRI600+ dataloader for liver segmentation in cirrhotic MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        sequence: The choice of MRI sequence. Either 'T1' or 'T2'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cirrmri600_dataset(path, patch_shape, split, sequence, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
