"""The SLIVER07 dataset contains annotations for liver segmentation in CT scans.

The dataset consists of the 20 training CT scans of the MICCAI 2007 liver segmentation challenge with a
binary liver mask. The scans are distributed as MetaImage files, which this module converts into nifti
files once. The 10 test scans of the challenge are not distributed with labels.

The dataset is located at https://doi.org/10.5281/zenodo.2597908. Its license restricts use to
non-commercial liver segmentation research and requires the publication below to be cited; see the
'license.txt' of the release for the complete terms.
This dataset is from the publication https://doi.org/10.1109/TMI.2009.2013851.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .mediastinal_ct import read_mhd
from .. import util


URLS = {
    "training-scans": "https://zenodo.org/records/2597908/files/training-scans.zip?download=1",
    "training-labels": "https://zenodo.org/records/2597908/files/training-labels.zip?download=1",
}

CHECKSUMS = {
    "training-scans": "caf4e43650050543ec59a07c2cb983faa023696dbe3f958019bfa025f13dc6c3",
    "training-labels": "72c41c299ce1392cf810424268186518157c14c361e7dc2220657b1cf7173d78",
}


def _convert_sliver07_to_nifti(data_dir, preprocessed_dir):
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    image_paths = natsorted(glob(os.path.join(data_dir, "**", "liver-orig*.mhd"), recursive=True))
    for image_path in tqdm(image_paths, desc="Converting SLIVER07 to nifti"):
        scan_id = os.path.basename(image_path)[len("liver-orig"):-len(".mhd")]
        label_paths = glob(os.path.join(data_dir, "**", f"liver-seg{scan_id}.mhd"), recursive=True)
        if not label_paths:
            continue

        out_image_path = os.path.join(preprocessed_dir, f"liver-orig{scan_id}.nii.gz")
        out_label_path = os.path.join(preprocessed_dir, f"liver-seg{scan_id}.nii.gz")
        if os.path.exists(out_image_path) and os.path.exists(out_label_path):
            continue

        volume, spacing = read_mhd(image_path)
        labels, _ = read_mhd(label_paths[0])
        assert labels.shape == volume.shape, f"The mask of scan '{scan_id}' does not match its scan."

        affine = np.diag(list(spacing) + [1.0])
        nib.save(nib.Nifti1Image(volume, affine), out_image_path)
        nib.save(nib.Nifti1Image(labels.astype("uint8"), affine), out_label_path)


def get_sliver07_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SLIVER07 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and glob(os.path.join(preprocessed_dir, "liver-orig*.nii.gz")):
        return preprocessed_dir

    if not glob(os.path.join(path, "**", "liver-orig*.mhd"), recursive=True):
        os.makedirs(path, exist_ok=True)
        for name, url in URLS.items():
            zip_path = os.path.join(path, f"{name}.zip")
            util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[name])
            util.unzip(zip_path=zip_path, dst=path, remove=False)

    _convert_sliver07_to_nifti(path, preprocessed_dir)
    return preprocessed_dir


def get_sliver07_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SLIVER07 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    preprocessed_dir = get_sliver07_data(path, download)

    raw_paths = natsorted(glob(os.path.join(preprocessed_dir, "liver-orig*.nii.gz")))
    label_paths = [p.replace("liver-orig", "liver-seg") for p in raw_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_sliver07_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SLIVER07 dataset for liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_sliver07_paths(path, download)

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


def get_sliver07_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SLIVER07 dataloader for liver segmentation.

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
    dataset = get_sliver07_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
