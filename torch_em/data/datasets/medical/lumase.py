"""The LumASe dataset contains annotations for lumbar vertebra anatomical substructure segmentation
in computed tomography (CT).

The dataset consists of 663 individual vertebrae (L1 to L5) cropped from lumbar spine CT scans,
acquired at ShengJing Hospital of China Medical University with three different CT manufacturers
(Philips, Siemens and Toshiba). Cases with vertebral fractures, metallic implants, bone tumors or
other foreign materials are excluded. Each vertebra is voxel-wise annotated for 7 anatomical
substructures: superior articular process (SAP), vertebral body (VB), transverse process (TP),
lamina (L), pedicle (P), spinous process (SP) and inferior articular process (IAP).

This dataset is located at https://doi.org/10.5281/zenodo.7181338.
This dataset is from the publication https://doi.org/10.1109/ISBI53787.2023.10230438.
The dataset is licensed under CC-BY-4.0.
Please cite the publication above if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7181338/files/L1-L5FineSegMix-663case.zip"
CHECKSUM = "61a280b446e1f1dd10935e0bed5cc8fa80152fc568f4d6d7dd28d7b857a322d0"


def get_lumase_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LumASe dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "L1-L5FineSegMix-663case")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "L1-L5FineSegMix-663case.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_lumase_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the LumASe data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_lumase_data(path, download)

    label_paths = natsorted(glob(os.path.join(data_dir, "*_seg.nii.gz")))
    raw_paths = [p.replace("_seg.nii.gz", ".nii.gz") for p in label_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_lumase_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LumASe dataset for lumbar vertebra anatomical substructure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lumase_paths(path, download)

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


def get_lumase_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LumASe dataloader for lumbar vertebra anatomical substructure segmentation.

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
    dataset = get_lumase_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
