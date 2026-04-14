"""The HepG2 Spheroids dataset contains 3D confocal fluorescence microscopy images
of twelve densely packed HepG2 human carcinoma cell nuclei spheroids, with manually
annotated instance segmentation ground truth created using 3D Slicer.

Original image dimensions are 1024 x 1024 pixels (XY) with 1.01 µm z-step size.

The dataset is located at https://doi.org/10.6084/m9.figshare.16438314.
This dataset is from the publication https://doi.org/10.1186/s12859-022-04827-3.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/30449889"
CHECKSUM = None


def get_spheroids_hepg2_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HepG2 Spheroids dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    # The zip extracts GT/, spheroids/, and seeds/ directly into path.
    if os.path.exists(os.path.join(path, "GT")):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "12spheroids.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    return path


def get_spheroids_hepg2_paths(
    path: Union[os.PathLike, str], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HepG2 Spheroids data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_spheroids_hepg2_data(path, download)

    # Raw: spheroids/{N}_smoothed_spheroid.nrrd (exclude _expanded_3 variants)
    raw_paths = natsorted([
        p for p in glob(os.path.join(data_dir, "spheroids", "*.nrrd"))
        if "expanded" not in os.path.basename(p)
    ])
    # Labels: GT/{N}_GT.nrrd (exclude _expanded_3_DT variants)
    label_paths = natsorted([
        p for p in glob(os.path.join(data_dir, "GT", "*.nrrd"))
        if "expanded" not in os.path.basename(p)
    ])

    if len(raw_paths) == 0:
        raise RuntimeError(
            f"No image files found in {os.path.join(data_dir, 'spheroids')}. "
            "Please check the dataset structure after downloading."
        )
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Number of images ({len(raw_paths)}) and labels ({len(label_paths)}) do not match."
        )

    return raw_paths, label_paths


def get_spheroids_hepg2_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the HepG2 Spheroids dataset for 3D nucleus instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_spheroids_hepg2_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_spheroids_hepg2_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the HepG2 Spheroids dataloader for 3D nucleus instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_spheroids_hepg2_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
