"""RVO-ME (also released as 'RVO-Lesion') is a dataset for segmentation of macular lesions in
retinal vein occlusion (RVO) in optical coherence tomography (OCT) B-scans.

The dataset consists of 3,012 OCT B-scan images from 130 patients (146 eyes). The pixel-level masks
label 4 classes: 0 = background, 1 = SRF (subretinal fluid), 2 = IRF (intraretinal fluid),
3 = ELM (external limiting membrane), 4 = EZ (ellipsoid zone).

This dataset is located at https://doi.org/10.6084/m9.figshare.29804435.v1 (figshare, CC BY 4.0).
The dataset is from the publication https://doi.org/10.1038/s41597-026-06695-5.
Please cite it if you use this dataset for your research.
"""

import os
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/56848025"
CHECKSUM = "ffe522f1b09e1a4c0c8eae10f776b25e3d3e13a1da30755f96f8356c98af3776"


def get_rvo_me_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RVO-ME data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "RVO-Lesion")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "RVO-Lesion.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_rvo_me_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'test'] = "train", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the RVO-ME data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_rvo_me_data(path, download)

    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose either 'train' or 'test'.")

    split_file = os.path.join(data_dir, "Image_Seg", f"{split}.txt")
    with open(split_file) as f:
        fnames = [line.strip() for line in f if line.strip()]

    raw_paths = natsorted(os.path.join(data_dir, "Image_Seg", "images", fname) for fname in fnames)
    label_paths = natsorted(
        os.path.join(data_dir, "Image_Seg", "masks", os.path.splitext(fname)[0] + ".png") for fname in fnames
    )

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_rvo_me_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RVO-ME dataset for segmentation of macular lesions in OCT B-scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_rvo_me_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_rvo_me_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RVO-ME dataloader for segmentation of macular lesions in OCT B-scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rvo_me_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
