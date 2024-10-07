"""This dataset contains annotation for nucleus segmentation in 3d fluorescence microscopy.

This dataset is from the publication https://doi.org/10.1242/dev.202800.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from shutil import rmtree
from typing import Optional, Tuple, Union, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD1026/Nuclei_training_segmentation/Training%20image%20dataset_Tiff%20Files.zip"  # noqa
CHECKSUM = "b103388a4aed01c7aadb2d5f49392d2dd08dd7cbeb2357b0c56355384ebb93a9"


def _load_tif(path):
    vol = None

    path_tif = path + ".tif"
    if os.path.exists(path_tif):
        vol = imageio.imread(path_tif)

    path_tiff = path + ".tiff"
    if os.path.exists(path_tiff):
        vol = imageio.imread(path_tiff)

    if vol is None:
        raise RuntimeError("Can't find tif or tiff file for {path}.")

    return vol


def _clip_shape(raw, labels):
    shape = raw.shape
    labels = labels[:shape[0], :shape[1], :shape[2]]

    shape = labels.shape
    raw = raw[:shape[0], :shape[1], :shape[2]]

    assert labels.shape == raw.shape, f"{labels.shape}, {raw.shape}"
    return raw, labels


def _process_data(in_folder, out_folder):
    import h5py

    os.makedirs(out_folder, exist_ok=True)

    sample_folders = glob(os.path.join(in_folder, "*"))
    for folder in sample_folders:
        sample = os.path.basename(folder)
        out_path = os.path.join(out_folder, f"{sample}.h5")

        cell_raw = _load_tif(os.path.join(folder, f"{sample}_cellwall"))
        cell_labels = _load_tif(os.path.join(folder, f"{sample}_cellseg"))
        cell_labels = cell_labels[:, ::-1]
        cell_raw, cell_labels = _clip_shape(cell_raw, cell_labels)

        nucleus_raw = _load_tif(os.path.join(folder, f"{sample}_n_H2BtdTomato"))
        nucleus_labels = _load_tif(os.path.join(folder, f"{sample}_n_stain_StarDist_goldGT"))
        nucleus_labels = nucleus_labels[:, ::-1]
        nucleus_raw, nucleus_labels = _clip_shape(nucleus_raw, nucleus_labels)

        # Remove last frames with artifacts for two volumes (1137 and 1170).
        if sample in ["1137", "1170"]:
            nucleus_raw, nucleus_labels = nucleus_raw[:-1], nucleus_labels[:-1]
            cell_raw, cell_labels = cell_raw[:-1], cell_labels[:-1]

        # Fixing cell labels for one volume (1136) is misaligned.
        if sample == "1136":
            cell_labels = np.fliplr(cell_labels)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw/cells", data=cell_raw, compression="gzip")
            f.create_dataset("raw/nuclei", data=nucleus_raw, compression="gzip")

            f.create_dataset("labels/cells", data=cell_labels, compression="gzip")
            f.create_dataset("labels/nuclei", data=nucleus_labels, compression="gzip")


def get_gonuclear_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the GoNuclear training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    url = URL
    checksum = CHECKSUM

    data_path = os.path.join(path, "gonuclear_datasets")
    if os.path.exists(data_path):
        return data_path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "gonuclear.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)

    extracted_path = os.path.join(path, "Training image dataset_Tiff Files")
    assert os.path.exists(extracted_path), extracted_path
    _process_data(extracted_path, data_path)
    assert os.path.exists(data_path)

    rmtree(extracted_path)
    return data_path


def get_gonuclear_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    segmentation_task: str = "nuclei",
    sample_ids: Optional[Union[int, Tuple[int, ...]]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the GoNuclear dataset for segmenting nuclei in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation_task: The segmentation task. Either 'nuclei' or 'cells'.
        sample_ids: The sample ids to load. The valid sample ids are:
            1135, 1136, 1137, 1139, 1170. If none is given all samples will be loaded.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    data_root = get_gonuclear_data(path, download)

    if sample_ids is None:
        paths = sorted(glob(os.path.join(data_root, "*.h5")))
    else:
        paths = []
        for sample_id in sample_ids:
            sample_path = os.path.join(data_root, f"{sample_id}.h5")
            if not os.path.exists(sample_path):
                raise ValueError(f"Invalid sample id {sample_id}.")
            paths.append(sample_path)

    if segmentation_task == "nuclei":
        raw_key = "raw/nuclei"
        label_key = "labels/nuclei"
    elif segmentation_task == "cells":
        raw_key = "raw/cells"
        label_key = "labels/cells"
    else:
        raise ValueError(f"Invalid segmentation task {segmentation_task}, expect one of 'cells' or 'nuclei'.")

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets,
    )

    return torch_em.default_segmentation_dataset(
        paths, raw_key, paths, label_key, patch_shape, **kwargs
    )


def get_gonuclear_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    segmentation_task: str = "nuclei",
    sample_ids: Optional[Union[int, Tuple[int, ...]]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the GoNuclear dataloader for segmenting nuclei in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        segmentation_task: The segmentation task. Either 'nuclei' or 'cells'.
        sample_ids: The sample ids to load. The valid sample ids are:
            1135, 1136, 1137, 1139, 1170. If none is given all samples will be loaded.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_gonuclear_dataset(
        path=path,
        patch_shape=patch_shape,
        segmentation_task=segmentation_task,
        sample_ids=sample_ids,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
    return loader
