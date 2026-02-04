"""The Wing Disc dataset contains annotations for 3D cell instance segmentation
in confocal microscopy images of Drosophila wing discs.

The dataset is located at https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD843.
This dataset is from the publication https://www.nature.com/articles/s44303-025-00099-7.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/843/S-BIAD843/Files"

VOLUMES = {
    "WD1_15-02_WT_confocalonly": "confocal",
    "WD2.1_21-02_WT_confocalonly": "confocal",
    "WD1.1_17-03_WT_MP": "multiphoton",
    "WD3.2_21-03_WT_MP": "multiphoton",
}


def _preprocess_volumes(path, data_dir):
    """Convert OME-Zarr volumes to HDF5 files with raw and labels datasets."""
    import h5py
    import zarr

    os.makedirs(data_dir, exist_ok=True)

    zarr_dir = os.path.join(path, "zarr")

    for name in VOLUMES:
        h5_path = os.path.join(data_dir, f"{name}.h5")
        if os.path.exists(h5_path):
            continue

        # Read raw volume: shape (1, 1, Z, Y, X) and squeeze to (Z, Y, X).
        raw_zarr = os.path.join(zarr_dir, f"{name}.zarr", "0", "0")
        raw = np.array(zarr.open(store=zarr.storage.LocalStore(raw_zarr)))
        raw = raw.squeeze()

        # Read segmentation: shape (Z, 1, 1, Y, X) and squeeze to (Z, Y, X).
        seg_zarr = os.path.join(zarr_dir, f"{name}_segmented.zarr", "0", "0")
        seg = np.array(zarr.open(store=zarr.storage.LocalStore(seg_zarr)))
        seg = seg.squeeze().astype("uint32")

        assert raw.shape == seg.shape, f"Shape mismatch for {name}: raw={raw.shape}, seg={seg.shape}"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=seg, compression="gzip")


def get_wing_disc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Wing Disc dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed data directory.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir) and len(glob(os.path.join(data_dir, "*.h5"))) == len(VOLUMES):
        return data_dir

    zarr_dir = os.path.join(path, "zarr")
    os.makedirs(zarr_dir, exist_ok=True)

    for name in VOLUMES:
        zarr_path = os.path.join(zarr_dir, f"{name}.zarr")
        if not os.path.exists(zarr_path):
            zip_fname = f"{name}.ome.zarr.zip"
            zip_path = os.path.join(path, zip_fname)
            url = f"{BASE_URL}/{zip_fname}"
            util.download_source(path=zip_path, url=url, download=download, checksum=None)
            util.unzip(zip_path=zip_path, dst=zarr_dir)

        seg_zarr_path = os.path.join(zarr_dir, f"{name}_segmented.zarr")
        if not os.path.exists(seg_zarr_path):
            seg_zip_fname = f"{name}_segmented.ome.zarr.zip"
            seg_zip_path = os.path.join(path, seg_zip_fname)
            seg_url = f"{BASE_URL}/{seg_zip_fname}"
            util.download_source(path=seg_zip_path, url=seg_url, download=download, checksum=None)
            util.unzip(zip_path=seg_zip_path, dst=zarr_dir)

    _preprocess_volumes(path, data_dir)

    return data_dir


def get_wing_disc_paths(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> List[str]:
    """Get paths to the Wing Disc data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    data_dir = get_wing_disc_data(path, download)
    data_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(data_paths) > 0
    return data_paths


def get_wing_disc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Wing Disc dataset for 3D cell segmentation in Drosophila wing discs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_wing_disc_paths(path, download)

    kwargs = util.ensure_transforms(ndim=3, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_wing_disc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Wing Disc dataloader for 3D cell segmentation in Drosophila wing discs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_wing_disc_dataset(
        path=path,
        patch_shape=patch_shape,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
