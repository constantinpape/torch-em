"""The Lipid Droplet Histology dataset contains annotations for lipid droplet (fat vacuole) segmentation
in H&E stained histological images of liver tissue.

This loader uses the "Corrected Subset B" of the dataset: 700 image patches of size 256 x 256 with manually
corrected masks of lipid droplets. The masks of the full whole-slide images (Datasets A and B) are not used,
since they require the whole-slide images, which are hosted elsewhere.

The dataset is located at https://zenodo.org/records/7802211.
This dataset is from the publication https://openreview.net/forum?id=nTnAm_El0RC.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Tuple, Union, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/7802211/files/Corrected_Subset_B.zip/content"
CHECKSUM = "dc64f674f644f897024a295e23da95b52d82eecb2723a7ee1a3439c9f519ec30"


def _convert_to_tif(data_dir, image_dir, label_dir):
    import nibabel as nib

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    src_dir = os.path.join(data_dir, "Corrected_Subset_B")
    label_paths = sorted(glob(os.path.join(src_dir, "labelsTs_subsetC_B", "*.nii.gz")))
    for label_path in label_paths:
        name = os.path.basename(label_path)[:-len(".nii.gz")]

        # The red, green and blue channels are stored in separate files (suffix 0000, 0001 and 0002).
        channels = [
            np.asarray(nib.load(os.path.join(src_dir, "imagesTs_subsetC_B", f"{name}_000{i}.nii.gz")).dataobj)
            for i in range(3)
        ]
        image = np.stack([ch[..., 0].T for ch in channels], axis=-1).astype("uint8")
        label = (np.asarray(nib.load(label_path).dataobj)[..., 0].T > 0).astype("uint8")

        imageio.imwrite(os.path.join(image_dir, f"{name}.tif"), image, compression="zlib")
        imageio.imwrite(os.path.join(label_dir, f"{name}.tif"), label, compression="zlib")


def get_lipid_droplet_histology_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Lipid Droplet Histology dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    image_dir, label_dir = os.path.join(path, "images"), os.path.join(path, "labels")
    if os.path.exists(image_dir) and os.path.exists(label_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Corrected_Subset_B.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    _convert_to_tif(data_dir, image_dir, label_dir)

    return path


def get_lipid_droplet_histology_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Lipid Droplet Histology data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_lipid_droplet_histology_data(path, download)

    raw_paths = sorted(glob(os.path.join(path, "images", "*.tif")))
    label_paths = sorted(glob(os.path.join(path, "labels", "*.tif")))
    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_lipid_droplet_histology_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Lipid Droplet Histology dataset for lipid droplet segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lipid_droplet_histology_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_lipid_droplet_histology_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Lipid Droplet Histology dataloader for lipid droplet segmentation.

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
    dataset = get_lipid_droplet_histology_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
