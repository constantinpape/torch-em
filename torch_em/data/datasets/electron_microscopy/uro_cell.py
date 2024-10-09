"""The UroCell dataset contains segmentation annotations for the following organelles:
- Food Vacuoles
- Golgi Apparatus
- Lysosomes
- Mitochondria
It contains several FIB-SEM volumes with annotations.

This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2020.103693.
Please cite it if you use this dataset for a publication.
"""

import os
import warnings
from glob import glob
from shutil import rmtree
from typing import List, Optional, Union, Tuple

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/MancaZerovnikMekuc/UroCell/archive/refs/heads/master.zip"
CHECKSUM = "a48cf31b06114d7def642742b4fcbe76103483c069122abe10f377d71a1acabc"


def get_uro_cell_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the UroCell training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
    """
    import h5py

    if os.path.exists(path):
        return path

    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("Please install the nibabel package.")

    # Download and unzip the data.
    os.makedirs(path)
    tmp_path = os.path.join(path, "uro_cell.zip")
    util.download_source(tmp_path, URL, download, checksum=CHECKSUM)
    util.unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "UroCell-master")

    files = glob(os.path.join(root, "data", "*.nii.gz"))
    files.sort()
    for data_path in files:
        fname = os.path.basename(data_path)
        data = nib.load(data_path).get_fdata()

        out_path = os.path.join(path, fname.replace("nii.gz", "h5"))
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=data, compression="gzip")

            # Check if we have any of the organelle labels for this volume
            # and also copy them if yes.
            fv_path = os.path.join(root, "fv", "instance", fname)
            if os.path.exists(fv_path):
                fv = nib.load(fv_path).get_fdata().astype("uint32")
                assert fv.shape == data.shape
                f.create_dataset("labels/fv", data=fv, compression="gzip")

            golgi_path = os.path.join(root, "golgi", "precise", fname)
            if os.path.exists(golgi_path):
                golgi = nib.load(golgi_path).get_fdata().astype("uint32")
                assert golgi.shape == data.shape
                f.create_dataset("labels/golgi", data=golgi, compression="gzip")

            lyso_path = os.path.join(root, "lyso", "instance", fname)
            if os.path.exists(lyso_path):
                lyso = nib.load(lyso_path).get_fdata().astype("uint32")
                assert lyso.shape == data.shape
                f.create_dataset("labels/lyso", data=lyso, compression="gzip")

            mito_path = os.path.join(root, "mito", "instance", fname)
            if os.path.exists(mito_path):
                mito = nib.load(mito_path).get_fdata().astype("uint32")
                assert mito.shape == data.shape
                f.create_dataset("labels/mito", data=mito, compression="gzip")

    # Clean Up.
    rmtree(root)
    return path


def get_uro_cell_paths(
    path: Union[os.PathLike], target: str, download: bool = False, return_label_key: bool = False,
) -> List[str]:
    """Get paths to the UroCell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        target: The segmentation target, corresponding to the organelle to segment.
            Available organelles are 'fv', 'golgi', 'lyso' and 'mito'.
        download: Whether to download the data if it is not present.
        return_label_key: Whether to return the label key.

    Returns:
        List of filepaths to the stored data.
    """
    import h5py

    get_uro_cell_data(path, download)

    label_key = f"labels/{target}"
    all_paths = glob(os.path.join(path, "*.h5"))
    all_paths.sort()
    paths = [path for path in all_paths if label_key in h5py.File(path, "r")]

    if return_label_key:
        return paths, label_key
    else:
        return paths


def get_uro_cell_dataset(
    path: Union[os.PathLike, str],
    target: str,
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> Dataset:
    """Get the UroCell dataset for organelle segmentation in FIB-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        target: The segmentation target, corresponding to the organelle to segment.
            Available organelles are 'fv', 'golgi', 'lyso' and 'mito'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert target in ("fv", "golgi", "lyso", "mito")

    paths, label_key = get_uro_cell_paths(path, target, download, return_label_key=True)

    assert sum((offsets is not None, boundaries, binary)) <= 1, f"{offsets}, {boundaries}, {binary}"
    if offsets is not None:
        if target in ("lyso", "golgi"):
            warnings.warn(
                f"{target} does not have instance labels, affinities will be computed based on binary segmentation."
            )
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, ignore_label=None, add_binary_target=True, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, 'label_transform2', label_transform, msg=msg)
    elif boundaries:
        if target in ("lyso", "golgi"):
            warnings.warn(
                f"{target} does not have instance labels, boundaries will be computed based on binary segmentation."
            )
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=label_key,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_uro_cell_loader(
    path: Union[os.PathLike, str],
    target: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> DataLoader:
    """Get the UroCell dataloader for organelle segmentation in FIB-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        target: The segmentation target, corresponding to the organelle to segment.
            Available organelles are 'fv', 'golgi', 'lyso' and 'mito'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_uro_cell_dataset(
        path, target, patch_shape, download=download, offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
