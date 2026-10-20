"""The CochleaNet dataset contains annotations for 3D instance segmentation of spiral ganglion neurons (SGNs)
and inner hair cells (IHCs) in light-sheet microscopy volumes of cleared cochleae.

The data are 3D crops with isotropic voxel size of 0.38 um. The SGN data has 48 crops of mouse cochleae with
parvalbumin stain and the IHC data has 89 crops of mouse and gerbil cochleae with Vglut3 stain. Some crops contain
no annotated cell. The official train and val splits are the ones used for training the published CochleaNet models.

The data is located at https://doi.org/10.5281/zenodo.23478266, released under the MIT license.
The dataset is from the publication https://doi.org/10.1101/2025.11.16.688700.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "sgn": "https://zenodo.org/records/23478267/files/SGN_v1.zip",
    "ihc": "https://zenodo.org/records/23478267/files/IHC_v1.zip",
}

CHECKSUMS = {
    "sgn": "3d80bce3b7bd635609f15fb64703a1ab4480bef396bfe3d0f11c850091064f10",
    "ihc": "26e0576ede1f1730d77636da0ac75066f5c5dbd7cc3d6ff3be3be9ad973d13ae",
}

FOLDERS = {"sgn": "SGN_v1", "ihc": "IHC_v1"}


def get_cochleanet_data(
    path: Union[os.PathLike, str], target: Literal["sgn", "ihc"] = "sgn", download: bool = False
) -> str:
    """Download the CochleaNet data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        target: The annotated cell type. Either 'sgn' for spiral ganglion neurons or 'ihc' for inner hair cells.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data of the chosen target.
    """
    if target not in URLS:
        raise ValueError(f"'{target}' is not a valid target. Choose from {list(URLS)}.")

    data_dir = os.path.join(path, FOLDERS[target])
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{FOLDERS[target]}.zip")
    util.download_source(path=zip_path, url=URLS[target], download=download, checksum=CHECKSUMS[target])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_cochleanet_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    target: Literal["sgn", "ihc"] = "sgn",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CochleaNet data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'val'.
        target: The annotated cell type. Either 'sgn' for spiral ganglion neurons or 'ihc' for inner hair cells.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "val"):
        raise ValueError(f"'{split}' is not a valid split. Choose from 'train' or 'val'.")

    data_dir = get_cochleanet_data(path, target, download)

    label_paths = natsorted(glob(os.path.join(data_dir, split, "*_annotations.tif")))
    raw_paths = [p.replace("_annotations.tif", ".tif") for p in label_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_cochleanet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val"],
    target: Literal["sgn", "ihc"] = "sgn",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CochleaNet dataset for SGN or IHC segmentation in light-sheet microscopy of cochleae.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        target: The annotated cell type. Either 'sgn' for spiral ganglion neurons or 'ihc' for inner hair cells.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cochleanet_paths(path, split, target, download)

    kwargs = util.ensure_transforms(ndim=3, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_cochleanet_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val"],
    target: Literal["sgn", "ihc"] = "sgn",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CochleaNet dataloader for SGN or IHC segmentation in light-sheet microscopy of cochleae.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        target: The annotated cell type. Either 'sgn' for spiral ganglion neurons or 'ihc' for inner hair cells.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cochleanet_dataset(
        path, patch_shape, split, target, offsets=offsets, boundaries=boundaries, binary=binary,
        download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
