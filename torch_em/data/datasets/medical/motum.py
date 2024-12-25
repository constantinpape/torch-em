"""The MOTUM dataset contains annotations for tumor (brain metastases and high grade glioma) segmentation
in brain multi-modal MRI scans.

The dataset is located at https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/.
This dataset is from the publication https://doi.org/10.1038/s41597-024-03634-0.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/10.12751_g-node.tvzqc5.zip"
CHECKSUM = "2626862599a3fcfe4ac0cefcea3af5b190625275036cc8eb4c9039cbd54e2d7c"


def get_motum_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MOTUM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_motum_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    modality: Literal['flair', 't1ce'],
    download: bool = False
) -> Tuple[List[int], List[int]]:
    """Get paths to the MOTUM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepath for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_motum_data(path, download)

    if modality not in ["flair", "t1ce"]:
        raise ValueError(f"'{modality}' is not a valid modality.")

    raw_paths = natsorted(glob(os.path.join(data_dir, "sub-*", "anat", f"sub-*_{modality}.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "derivatives", "sub-*", f"{modality}_seg_*.nii.gz")))

    # NOTE: Remove labels which are missing preprocessed volumes
    missing_inputs = ["sub-0030", "sub-0031", "sub-0032"]
    label_paths = [p for p in label_paths if all([p.find(_f) == -1 for _f in missing_inputs])]

    if split == "train":
        raw_paths, label_paths = raw_paths[:35], label_paths[:35]
    elif split == "val":
        raw_paths, label_paths = raw_paths[35:45], label_paths[35:45]
    elif split == "test":
        raw_paths, label_paths = raw_paths[45:], label_paths[45:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_motum_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    modality: Literal['flair', 't1ce'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MOTUM dataset for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        modality: The choice of imaging modality.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_motum_paths(path, split, modality, download)

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


def get_motum_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    modality: Literal['flair', 't1ce'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MOTUM dataloader for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.'
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        modality: The choice of imaging modality.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_motum_dataset(path, patch_shape, split, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
