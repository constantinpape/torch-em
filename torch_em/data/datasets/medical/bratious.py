"""The BraTioUS dataset contains 1,669 B-mode 2D intraoperative brain-tumor ultrasound images from
142 glioma patients, collected across 6 hospitals in 5 countries. Every image has a corresponding
binary tumor segmentation mask: the masks started as nnU-Net pseudo-labels and were then manually
reviewed and corrected by neurosurgeons.

The dataset is located at https://doi.org/10.5281/zenodo.16887362, released under a CC-BY-4.0 license.
This loader uses the latest Zenodo version (record 18130394), which includes an additional pass of
label review and correction over the initial release.

This dataset is used in the publications https://doi.org/10.3390/cancers17020280 and
https://doi.org/10.3390/cancers17020315. Please cite them if you use this dataset for your research.

NOTE: A handful of label volumes (4 out of 1,669) ship with an extra trailing singleton axis
compared to their matching image. This loader squeezes them in-place on first use.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://zenodo.org/records/18130394/files/ioUS-BraTioUS-dataset.zip",
    "labels": "https://zenodo.org/records/18130394/files/tumor-segmentation-BraTioUS-dataset.zip",
}

CHECKSUMS = {
    "images": "a26e4cda539a9f8520d7f5032a4a81a72f0978a22e80c47bec62fc8463d93016",
    "labels": "1448b9f28ab28a8a9db8ceb3a9235be54cb6dbc26d48c592b384c42d25ee8d6d",
}


def get_bratious_data(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Download the BraTioUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the image data.
        Filepath to the folder with the label data.
    """
    image_dir = os.path.join(path, "images", "imagesBraTioUS-public-dataset")
    label_dir = os.path.join(path, "labels", "labelsBraTioUS-public-dataset")
    if os.path.exists(image_dir) and os.path.exists(label_dir):
        _squeeze_odd_label_shapes(label_dir)
        return image_dir, label_dir

    os.makedirs(path, exist_ok=True)

    image_zip = os.path.join(path, "images.zip")
    util.download_source(path=image_zip, url=URLS["images"], download=download, checksum=CHECKSUMS["images"])
    util.unzip(zip_path=image_zip, dst=os.path.join(path, "images"))

    label_zip = os.path.join(path, "labels.zip")
    util.download_source(path=label_zip, url=URLS["labels"], download=download, checksum=CHECKSUMS["labels"])
    util.unzip(zip_path=label_zip, dst=os.path.join(path, "labels"))

    assert os.path.exists(image_dir) and os.path.exists(label_dir), \
        f"The extraction of the BraTioUS archives did not create the expected folders in '{path}'."

    _squeeze_odd_label_shapes(label_dir)

    return image_dir, label_dir


def _squeeze_odd_label_shapes(label_dir):
    """A handful of label volumes ship with an extra trailing singleton axis, e.g. (800, 600, 1)
    instead of (800, 600). Squeeze them in-place so that they match their raw image's shape."""
    import nibabel as nib

    for label_path in glob(os.path.join(label_dir, "*.nii.gz")):
        image = nib.load(label_path)
        if len(image.shape) == 2:
            continue
        squeezed = image.get_fdata().squeeze()
        nib.save(nib.Nifti1Image(squeezed, image.affine, image.header), label_path)


def get_bratious_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the BraTioUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, label_dir = get_bratious_data(path, download)

    label_paths = natsorted(glob(os.path.join(label_dir, "*.nii.gz")))
    raw_paths = [os.path.join(image_dir, os.path.basename(p)) for p in label_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_bratious_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BraTioUS dataset for tumor segmentation in intraoperative brain ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bratious_paths(path, download)

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
        ndim=2,
        **kwargs
    )


def get_bratious_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BraTioUS dataloader for tumor segmentation in intraoperative brain ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bratious_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
