"""MouseCC is a dataset for myelin and intra-axonal space segmentation in 3D SEM images
of the mouse brain genu of corpus callosum.

The volume contains 200 SEM slices at 24 x 24 x 100 nm resolution (36 x 48 x 20 um).
Two segmentation labels are provided:
- myelin: binary mask from a pixel-wise classifier.
- fibers: intra-axonal space, uniquely labeled per axon via random-walker segmentation.

This dataset is from the publication https://doi.org/10.1007/s00429-019-01844-6.
Please cite it if you use this dataset in your research.

The data is listed at https://datacatalog.med.nyu.edu/dataset/10432.
It requires manual download after agreeing to the terms of use at:
https://cai2r.net/resources/intra-axonal-space-segmented-from-3d-scanning-electron-microscopy-of-the-mouse-brain-genu-of-corpus-callosum/

After downloading, place the following four files in a local directory and pass it as `path`:
- datac.nii
- maskc.nii
- myelin_mask.nii
- fibers.nii
"""

import os
from typing import Literal, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


MOUSECC_FILES = ["datac.nii", "maskc.nii", "myelin_mask.nii", "fibers.nii"]
MOUSECC_DOWNLOAD_URL = (
    "https://cai2r.net/resources/"
    "intra-axonal-space-segmented-from-3d-scanning-electron-microscopy-of-the-mouse-brain-genu-of-corpus-callosum/"
)


def _require_mousecc_files(path):
    missing = [f for f in MOUSECC_FILES if not os.path.exists(os.path.join(str(path), f))]
    if missing:
        raise RuntimeError(
            f"MouseCC files not found in {path}: {missing}\n"
            "This dataset requires manual download. Please fill in the form at:\n"
            f"{MOUSECC_DOWNLOAD_URL}\n"
            "and place datac.nii, maskc.nii, myelin_mask.nii, fibers.nii in the path directory."
        )


def _convert_to_h5(path):
    import h5py
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("nibabel is required to process MouseCC data: pip install nibabel")

    h5_path = os.path.join(str(path), "mousecc.h5")
    if os.path.exists(h5_path):
        return h5_path

    _require_mousecc_files(path)

    def load_nii(fname, dtype):
        # nibabel returns (x, y, z); transpose to torch-em convention (z, y, x).
        data = nib.load(os.path.join(str(path), fname)).get_fdata().astype(dtype)
        return np.moveaxis(data, -1, 0)

    raw = load_nii("datac.nii", "uint8")
    foreground = load_nii("maskc.nii", "uint8")
    myelin = load_nii("myelin_mask.nii", "uint8")
    fibers = load_nii("fibers.nii", "uint32")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/foreground", data=foreground, compression="gzip")
        f.create_dataset("labels/myelin", data=myelin, compression="gzip")
        f.create_dataset("labels/fibers", data=fibers, compression="gzip")

    return h5_path


def get_mousecc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Prepare the MouseCC data as an HDF5 file.

    Args:
        path: Filepath to the folder containing the manually downloaded NIfTI files.
        download: Ignored - this dataset requires manual download.

    Returns:
        Path to the converted HDF5 file.
    """
    os.makedirs(str(path), exist_ok=True)
    if download:
        raise RuntimeError(
            "Automatic download is not supported for MouseCC.\n"
            "Please download the data manually from:\n"
            f"{MOUSECC_DOWNLOAD_URL}"
        )
    return _convert_to_h5(path)


def get_mousecc_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["myelin", "fibers"] = "myelin",
    download: bool = False,
) -> Tuple[str, str]:
    """Get paths to the MouseCC HDF5 data.

    Args:
        path: Filepath to the folder containing the manually downloaded NIfTI files.
        label_choice: The segmentation target. Either 'myelin' or 'fibers'.
        download: Ignored - this dataset requires manual download.

    Returns:
        Path to the HDF5 file for raw data.
        Path to the HDF5 file for labels.
    """
    if label_choice not in ("myelin", "fibers"):
        raise ValueError(f"Invalid label_choice: '{label_choice}'. Choose 'myelin' or 'fibers'.")
    h5_path = get_mousecc_data(path, download)
    return h5_path, h5_path


def get_mousecc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["myelin", "fibers"] = "myelin",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MouseCC dataset for myelin or intra-axonal space segmentation in SEM.

    Args:
        path: Filepath to the folder containing the manually downloaded NIfTI files.
        patch_shape: The patch shape to use for training.
        label_choice: The segmentation target. 'myelin' for the binary myelin mask,
            or 'fibers' for the intra-axonal space with unique labels per axon.
        download: Ignored - this dataset requires manual download.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_path, label_path = get_mousecc_paths(path, label_choice, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=raw_path, raw_key="raw",
        label_paths=label_path, label_key=f"labels/{label_choice}",
        patch_shape=patch_shape, **kwargs
    )


def get_mousecc_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    label_choice: Literal["myelin", "fibers"] = "myelin",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for myelin or intra-axonal space segmentation in MouseCC SEM data.

    Args:
        path: Filepath to the folder containing the manually downloaded NIfTI files.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        label_choice: The segmentation target. 'myelin' for the binary myelin mask,
            or 'fibers' for the intra-axonal space with unique labels per axon.
        download: Ignored - this dataset requires manual download.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mousecc_dataset(
        path, patch_shape, label_choice=label_choice, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
