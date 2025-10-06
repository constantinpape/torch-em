"""The Parhyale Regen dataset contains nucleus annotations for parhyale images from confocal microscope.

The dataset is located at https://zenodo.org/records/8252039.
This dataset is from the publication https://doi.org/10.7554/eLife.19766.012.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _preprocess_data(root, path):
    import h5py
    import requests

    raw_path = os.path.join(path, "Parhyale_H2B-EGFP_images_tp01-50.tif")
    assert os.path.exists(raw_path)

    raw = imageio.imread(raw_path)

    # We have limited timepoints annotated, let's extract them first.
    tps = [0, 10, 20, 30, 40, 49]
    raw_tps = [raw[i, ...] for i in tps]
    label_tps = [imageio.imread(p) for p in natsorted(glob(os.path.join(path, "*_instance-segmentation-labels_*.tif")))]

    # Get the new folder where we store the h5 files.
    new_path = os.path.join(root, "preprocessed")
    os.makedirs(new_path, exist_ok=True)

    for curr_tp, curr_raw, curr_label in zip(tps, raw_tps, label_tps):
        # Store each 3d volume per timepoint in their individual h5 files.
        fpath = os.path.join(new_path, f"Parhyale_H2B-EGFP_{curr_tp + 1}.h5")
        with h5py.File(fpath, "w") as f:
            f.create_dataset("raw", data=curr_raw, compression="gzip")
            f.create_dataset("labels", data=curr_label, compression="gzip")


def get_parhyale_regen_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Parhyale Regen dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the training data is stored.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return path

    os.makedirs(data_dir, exist_ok=True)

    # Download the data from Zenodo via fetching each file.
    # NOTE: This data download is implemented because all image and label files are scattered in the link.
    url = "https://zenodo.org/api/records/8252039"

    for f in requests.get(url).json()["files"]:
        fpath = os.path.join(data_dir, f["key"])
        print("Downloading:", f["key"])
        r = requests.get(f["links"]["self"])
        with open(fpath, "wb") as out:
            out.write(r.content)

    # Preprocess the images to keep the relevant inputs.
    _preprocess_data(path, data_dir)

    return path


def get_parhyale_regen_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths for the Parhyale Regen data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data.
    """
    data_dir = get_parhyale_regen_data(path, download)
    vol_paths = natsorted(glob(os.path.join(data_dir, "preprocessed", "*.h5")))
    return vol_paths


def get_parhyale_regen_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> Dataset:
    """Get the Parhyale Regen dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_parhyale_regen_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_parhyale_regen_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> DataLoader:
    """Get the Parhyale Regen dataset for nucleus segmentation.

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
    dataset = get_parhyale_regen_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
