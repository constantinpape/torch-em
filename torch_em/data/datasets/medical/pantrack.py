"""PanTrack is a longitudinal CT benchmark for pancreatic cancer lesion segmentation and tracking.

It consists of 161 portal-venous phase CT scans from 45 patients with pancreatic adenocarcinoma
(2-11 scans per patient), collected at a single institution. The scans are annotated with instance
segmentation masks for 292 lesion instances (pancreatic tumors and hepatic metastases), of which 116
consecutive baseline-to-follow-up pairs form the longitudinal tracking benchmark (see 'tracking.json'
in the downloaded data for the lesion correspondences across timepoints).

NOTE: The label legend is as follows: background: 0, non-zero integer: a lesion instance id. The
instance ids are non-consecutive, but consistent across all timepoints of the same patient, i.e. the
same lesion carries the same instance id at every scan it appears in. Not all liver lesions are
annotated: only hepatic metastases deemed relevant by the radiologist are included.

The dataset is located at https://huggingface.co/datasets/mrokuss/PanTrack (CC BY-NC 4.0).
This dataset is from the publication https://arxiv.org/abs/2605.23118.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HF_REPO = "mrokuss/PanTrack"


def get_pantrack_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PanTrack dataset.

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
        raise ImportError("'huggingface_hub' is required to download PanTrack. Install it via conda/pip.")

    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=path)

    return path


def get_pantrack_paths(
    path: Union[os.PathLike, str], split: Optional[Literal["train", "val"]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PanTrack data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'val', following the recommended split shipped
            with the dataset. If None, all scans are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pantrack_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*_0000.nii.gz")))

    if split is not None:
        import json

        if split not in ("train", "val"):
            raise ValueError(f"'{split}' is not a valid split.")

        with open(os.path.join(data_dir, "recommended_split.json")) as f:
            split_ids = set(json.load(f)[0][split])

        raw_paths = [p for p in raw_paths if os.path.basename(p).replace("_0000.nii.gz", "") in split_ids]

    label_paths = [
        os.path.join(data_dir, "labels", os.path.basename(p).replace("_0000.nii.gz", ".nii.gz")) for p in raw_paths
    ]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_pantrack_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "val"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PanTrack dataset for pancreatic cancer lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'. If None, all scans are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_pantrack_paths(path, split, download)

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


def get_pantrack_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "val"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PanTrack dataloader for pancreatic cancer lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'. If None, all scans are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pantrack_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
