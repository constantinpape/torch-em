"""The Pituitary-Tumor dataset contains annotations for pituitary neuroendocrine tumor and
carotid artery segmentation in contrast-enhanced T1-weighted MRI.

The dataset consists of 136 patients with pituitary adenomas, each with a pre-processed T1-weighted,
contrast-enhanced MRI (and a co-registered T2-weighted MRI for most patients), together with manual /
semi-automated segmentations of (1) the pituitary tumor-gland complex and (2) the bilateral intracranial
carotid arteries.

NOTE: The two binary masks are combined into one semantic label volume with the following ids:
- background: 0
- tumor: 1
- carotids: 2
A few voxels can overlap between the two structures. In this case, the carotids take priority, as they are
written after the tumor mask.

The dataset is located at https://doi.org/10.6084/m9.figshare.27894084.

This dataset is from the publication https://doi.org/10.1038/s41597-024-04218-8.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/50793273"
CHECKSUM = "1aaf1945e08083436a7d5a7614c59f07572d0278a211436978a9c3b818cf9074"

LABEL_IDS = {"tumor": 1, "carotids": 2}

# The order in which the structures are written to the label volume. Later structures overwrite earlier ones.
WRITE_ORDER = ["tumor", "carotids"]


def _find_mask_path(t1_path, structure):
    """Find the mask file for a structure, trying the '_f' (final) suffix used by most patients
    first, then the plain suffix used by a handful of others. Returns None if neither exists or
    the only match on disk is an empty (corrupt/missing-annotation) file.
    """
    for suffix in [f"_{structure}_f.nii.gz", f"_{structure}.nii.gz"]:
        candidate = t1_path.replace("_T1.nii.gz", suffix)
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _convert_case(t1_path, tumor_path, carotids_path, out_path):
    import h5py
    from nibabel.processing import resample_from_to

    t1_img = nib.load(t1_path)
    raw = t1_img.get_fdata()
    labels = np.zeros(raw.shape, dtype="uint8")
    for name, mask_path in zip(WRITE_ORDER, [tumor_path, carotids_path]):
        if mask_path is None:
            continue
        mask_img = nib.load(mask_path)
        if mask_img.shape != raw.shape:
            # A handful of masks were saved on a slightly different crop of the same (isotropic,
            # same-spacing) grid as the T1 volume. Resample onto the T1 grid using the affines,
            # rather than dropping the annotation.
            mask_img = resample_from_to(mask_img, t1_img, order=0)
        mask = mask_img.get_fdata()
        assert mask.shape == raw.shape, f"Shape mismatch for {mask_path}."
        labels[mask > 0] = LABEL_IDS[name]

    # The nifti volumes are stored in (x, y, z) order, we transpose them to (z, y, x).
    raw, labels = raw.transpose(2, 1, 0), labels.transpose(2, 1, 0)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")


def get_pituitary_tumor_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Pituitary-Tumor dataset and convert it to hdf5 volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the preprocessed hdf5 volumes.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    raw_dir = os.path.join(path, "raw")
    if not glob(os.path.join(raw_dir, "**", "*_T1.nii.gz"), recursive=True):
        zip_path = os.path.join(path, "Pituitary_MRI_tumor_carotids.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=raw_dir)

    t1_paths = natsorted(glob(os.path.join(raw_dir, "**", "*_T1.nii.gz"), recursive=True))
    assert len(t1_paths) > 0, f"No T1 volumes found at '{raw_dir}'."

    os.makedirs(data_dir, exist_ok=True)
    for t1_path in tqdm(t1_paths, desc="Converting Pituitary-Tumor volumes to hdf5"):
        pid = os.path.basename(t1_path).replace("_T1.nii.gz", "")
        tumor_path = _find_mask_path(t1_path, "tumor")
        carotids_path = _find_mask_path(t1_path, "carotids")
        out_path = os.path.join(data_dir, f"{pid}.h5")
        _convert_case(t1_path, tumor_path, carotids_path, out_path)

    return data_dir


def get_pituitary_tumor_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Pituitary-Tumor data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 volumes with image and label data.
    """
    data_dir = get_pituitary_tumor_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(volume_paths) > 0
    return volume_paths


def get_pituitary_tumor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Pituitary-Tumor dataset for pituitary tumor and carotid artery segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_pituitary_tumor_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_pituitary_tumor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Pituitary-Tumor dataloader for pituitary tumor and carotid artery segmentation in MRI.

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
    dataset = get_pituitary_tumor_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
