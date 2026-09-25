"""CrossPan is a benchmark for pancreas segmentation in MRI, across MRI sequences and institutions.

It consists of 1336 3D MRI volumes from 8 institutions across three sequences: T1-weighted (T1), T2-weighted
(T2) and Out-of-Phase (OOP), each pre-split into 'train', 'val' and 'test' subsets, with binary pancreas
segmentation masks.

NOTE: The label legend is as follows: background: 0, pancreas: 1. Verified on the data: the label
volumes only contain the ids 0 and 1.

The dataset is located at https://huggingface.co/datasets/linkai-peng/CrossPan (CC BY-NC 4.0).
This dataset is from the publication https://doi.org/10.48550/arXiv.2604.18797.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HF_REPO = "linkai-peng/CrossPan"

SEQUENCES = ["T1", "T2", "OOP"]
SPLITS = ["train", "val", "test"]

LABEL_IDS = {"background": 0, "pancreas": 1}


def get_crosspan_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CrossPan dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if os.path.exists(path):
        return path

    if not download:
        raise RuntimeError(f"Cannot find the data at '{path}', but download was set to False.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("'huggingface_hub' is required to download CrossPan. Install it via conda/pip.")

    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=path)

    return path


def get_crosspan_paths(
    path: Union[os.PathLike, str],
    sequence: Optional[Literal["T1", "T2", "OOP"]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CrossPan data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence: The choice of MRI sequence. Either 'T1', 'T2' or 'OOP'. If None, all sequences are used.
        split: The choice of data split. Either 'train', 'val' or 'test'. If None, all splits are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_crosspan_data(path, download)

    if sequence is None:
        sequences = SEQUENCES
    elif sequence in SEQUENCES:
        sequences = [sequence]
    else:
        raise ValueError(f"'{sequence}' is not a valid sequence.")

    if split is None:
        splits = SPLITS
    elif split in SPLITS:
        splits = [split]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    raw_paths, label_paths = [], []
    for seq in sequences:
        for spl in splits:
            cur_raw_paths = natsorted(glob(os.path.join(data_dir, seq, spl, "images", "*_0000.nii.gz")))
            cur_label_paths = [
                p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace("_0000.nii.gz", ".nii.gz")
                for p in cur_raw_paths
            ]
            assert len(cur_raw_paths) > 0 and all(os.path.exists(p) for p in cur_label_paths)
            raw_paths.extend(cur_raw_paths)
            label_paths.extend(cur_label_paths)

    return raw_paths, label_paths


def get_crosspan_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    sequence: Optional[Literal["T1", "T2", "OOP"]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CrossPan dataset for pancreas segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        sequence: The choice of MRI sequence. Either 'T1', 'T2' or 'OOP'. If None, all sequences are used.
        split: The choice of data split. Either 'train', 'val' or 'test'. If None, all splits are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_crosspan_paths(path, sequence, split, download)

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


def get_crosspan_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    sequence: Optional[Literal["T1", "T2", "OOP"]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CrossPan dataloader for pancreas segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sequence: The choice of MRI sequence. Either 'T1', 'T2' or 'OOP'. If None, all sequences are used.
        split: The choice of data split. Either 'train', 'val' or 'test'. If None, all splits are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_crosspan_dataset(path, patch_shape, sequence, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
