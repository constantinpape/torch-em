"""
"""

import os
from glob import glob
from shutil import rmtree
from typing import Tuple, Union

import numpy as np
import torch_em
from elf.io import open_file
from .. import util


ACTIN_ID = 10002


def _process_deepict_actin(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    datasets = ["00004", "00011", "00012"]
    for dataset in datasets:
        ds_folder = os.path.join(input_path, dataset)
        assert os.path.exists(ds_folder)
        ds_out = os.path.join(output_path, f"{dataset}.h5")
        if os.path.exists(ds_out):
            continue

        tomo_folder = glob(os.path.join(ds_folder, "Tomograms", "VoxelSpacing*"))
        assert len(tomo_folder) == 1
        tomo_folder = tomo_folder[0]

        annotation_folder = os.path.join(tomo_folder, "Annotations")
        annotion_files = glob(os.path.join(annotation_folder, "*.zarr"))

        annotations = {}
        for annotation in annotion_files:
            with open_file(annotation, "r") as f:
                annotation_data = f["0"][:].astype("uint8")
            annotation_name = os.path.basename(annotation).split("-")[1]
            annotations[annotation_name] = annotation_data

        tomo_path = os.path.join(tomo_folder, "CanonicalTomogram", f"{dataset}.mrc")
        with open_file(tomo_path, "r") as f:
            data = f["data"][:]

        with open_file(ds_out, "a") as f:
            f.create_dataset("raw", data=data, compression="gzip")
            for name, annotation in annotations.items():
                f.create_dataset(f"labels/original/{name}", data=annotation, compression="gzip")


def get_deepict_actin_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the deepict actin dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
    """
    # Check if the processed data is already present.
    dataset_path = os.path.join(path, "deepict_actin")
    # if os.path.exists(dataset_path):
    #     return dataset_path

    # Otherwise download the data.
    dl_path = util.download_from_cryo_et_portal(path, ACTIN_ID, download)

    # And then process it.
    _process_deepict_actin(dl_path, dataset_path)

    # TODO
    # Clean up the original data after processing.

    return dataset_path


def get_deepict_actin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
):
    """Get the dataset for EM neuron segmentation in ISBI 2012.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3
    data_path = get_deepict_actin_data(path, download)

    raw_key = "raw"
    label_key = "labels/membranes"

    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_deepict_actin_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    """Get the DataLoader for EM neuron segmentation in ISBI 2012.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_deepict_actin_loader(
        path, patch_shape, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
