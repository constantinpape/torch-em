"""The DRG-Axon-Mito dataset contains mitochondrial membrane segmentation for cryo-electron tomograms of
axons and varicosities in cultured primary mouse dorsal root ganglion neurons.

The data is hosted on the CryoET Data Portal at https://cryoetdataportal.cziscience.com/datasets/10512
(and the related datasets 10513, 10514, 10515 and 10516 from the same study).

The dataset is part of the publication https://doi.org/10.64898/2026.07.07.737043.
Please cite it if you use this dataset in your research.
"""

import os
import json
import shutil
from typing import Union, Tuple, List

import requests
from tqdm import tqdm

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://files.cryoetdataportal.cziscience.com/{dataset}/{run}/Reconstructions/VoxelSpacing{voxel_spacing}/"
RAW_URL = BASE_URL + "Tomograms/100/{run}.zarr"
OMM_URL = BASE_URL + "Annotations/100/source_label_omm_{omm_scale}_segmentationmask.zarr"
IMM_URL = BASE_URL + "Annotations/101/source_label_imm_{imm_scale}_segmentationmask.zarr"

# The mitochondrial membrane classes and their label value in the merged mask.
CLASSES = {"outer_membrane": 1, "inner_membrane": 2}

# The portal dataset, run name, voxel spacing and the annotation-file scale suffix of each class.
RUNS = [
    (10512, "Position_1", "21.600", "-1.0", "-1.0"),
    (10512, "Position_15", "21.600", "-1.0", "-1.0"),
    (10512, "Position_17", "21.600", "-1.0", "-1.0"),
    (10512, "Position_19", "21.600", "-1.0", "-1.0"),
    (10512, "Position_2", "21.600", "-1.0", "-2.0"),
    (10512, "Position_21", "21.600", "-1.0", "-1.0"),
    (10512, "Position_22", "21.600", "-1.0", "-1.0"),
    (10512, "Position_26", "21.600", "-1.0", "-1.0"),
    (10512, "Position_30", "10.800", "-1.0", "-1.0"),
    (10512, "Position_32", "21.600", "-1.0", "-2.0"),
    (10512, "Position_33", "21.600", "-1.0", "-1.0"),
    (10512, "Position_42", "21.600", "-1.0", "-2.0"),
    (10512, "Position_43", "21.600", "-1.0", "-1.0"),
    (10512, "Position_44", "21.600", "-1.0", "-1.0"),
    (10513, "Position_03", "20.579", "-1.0", "-1.0"),
    (10513, "Position_04", "20.579", "-1.0", "-1.0"),
    (10513, "Position_11", "20.579", "-1.0", "-1.0"),
    (10513, "Position_12", "20.579", "-1.0", "-1.0"),
    (10513, "Position_42", "20.579", "-1.0", "-1.0"),
    (10513, "Position_45", "20.579", "-1.0", "-1.0"),
    (10513, "Position_46", "20.579", "-1.0", "-1.0"),
    (10513, "Position_48", "20.579", "-1.0", "-1.0"),
    (10513, "Position_49", "20.579", "-1.0", "-1.0"),
    (10513, "Position_50", "20.579", "-1.0", "-1.0"),
    (10514, "Position_18", "21.600", "-1.0", "-1.0"),
    (10514, "Position_20", "21.600", "-1.0", "-1.0"),
    (10514, "Position_24", "21.600", "-1.0", "-1.0"),
    (10514, "Position_25", "21.600", "-1.0", "-1.0"),
    (10514, "Position_26", "21.600", "-1.0", "-1.0"),
    (10514, "Position_5", "21.600", "-1.0", "-1.0"),
    (10515, "Position_16B", "21.600", "-1.0", "-1.0"),
    (10515, "Position_17B", "21.600", "-1.0", "-1.0"),
    (10515, "Position_41C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_46C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_50C", "21.600", "-1.0", "-2.0"),
    (10515, "Position_51", "10.800", "-2.0", "-2.0"),
    (10515, "Position_52C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_62C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_63C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_64", "10.800", "-2.0", "-2.0"),
    (10515, "Position_69C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_84C", "21.600", "-1.0", "-1.0"),
    (10515, "Position_9C", "21.600", "-1.0", "-1.0"),
    (10516, "Position_1", "21.600", "-1.0", "-2.0"),
    (10516, "Position_12", "21.600", "-1.0", "-2.0"),
    (10516, "Position_14", "21.600", "-1.0", "-1.0"),
    (10516, "Position_16", "21.600", "-1.0", "-1.0"),
    (10516, "Position_20", "21.600", "-1.0", "-2.0"),
    (10516, "Position_25", "21.600", "-1.0", "-1.0"),
    (10516, "Position_27", "21.600", "-1.0", "-2.0"),
    (10516, "Position_29", "21.600", "-1.0", "-1.0"),
    (10516, "Position_30", "21.600", "-1.0", "-1.0"),
    (10516, "Position_35", "21.600", "-1.0", "-1.0"),
    (10516, "Position_38", "21.600", "-1.0", "-1.0"),
    (10516, "Position_39", "21.600", "-1.0", "-1.0"),
    (10516, "Position_7", "21.600", "-1.0", "-1.0"),
    (10516, "Position_9", "21.600", "-1.0", "-1.0"),
]


def _fetch(url, path, optional=False):
    if os.path.exists(path):
        return True

    with requests.get(url, stream=True, timeout=(20, 300)) as response:
        # A chunk that holds only the fill value is not written by the portal.
        if optional and response.status_code == 404:
            return False
        response.raise_for_status()
        # The chunk is renamed only once it is complete, so an interrupted download is not reused.
        tmp_path = path + ".partial"
        with open(tmp_path, "wb") as f:
            for block in response.iter_content(8 * 1024 ** 2):
                f.write(block)

    os.rename(tmp_path, path)
    return True


def _download_ome_zarr(url, out_path, download):
    array_path = os.path.join(out_path, "0")
    if os.path.exists(array_path):
        return array_path

    if not download:
        raise RuntimeError(f"Cannot find the data at {out_path}, but download was set to False.")

    os.makedirs(out_path, exist_ok=True)
    for name in (".zattrs", ".zgroup"):
        if not os.path.exists(os.path.join(out_path, name)):
            _fetch(f"{url}/{name}", os.path.join(out_path, name))

    tmp_path = os.path.join(out_path, "0.partial")
    os.makedirs(tmp_path, exist_ok=True)
    _fetch(f"{url}/0/.zarray", os.path.join(tmp_path, ".zarray"))
    with open(os.path.join(tmp_path, ".zarray")) as f:
        meta = json.load(f)

    grid = [-(-size // chunk) for size, chunk in zip(meta["shape"], meta["chunks"])]
    for z in range(grid[0]):
        for y in range(grid[1]):
            for x in range(grid[2]):
                chunk_dir = os.path.join(tmp_path, str(z), str(y))
                os.makedirs(chunk_dir, exist_ok=True)
                _fetch(f"{url}/0/{z}/{y}/{x}", os.path.join(chunk_dir, str(x)), optional=True)

    os.rename(tmp_path, array_path)
    return array_path


def _merge_labels(run_dir, dataset, run, voxel_spacing, omm_scale, imm_scale, download):
    """Merge the outer- and inner-membrane masks into one multi-class volume."""
    import zarr

    label_path = os.path.join(run_dir, "labels.zarr")
    if os.path.exists(label_path):
        return label_path

    merged = None
    for name, scale in (("outer_membrane", omm_scale), ("inner_membrane", imm_scale)):
        value = CLASSES[name]
        template = OMM_URL if name == "outer_membrane" else IMM_URL
        url = template.format(dataset=dataset, run=run, voxel_spacing=voxel_spacing, omm_scale=scale, imm_scale=scale)
        class_dir = os.path.join(run_dir, f"class_{name}.zarr")
        array = zarr.open_array(_download_ome_zarr(url, class_dir, download), mode="r")
        mask = array[:] > 0
        if merged is None:
            merged = np.zeros(mask.shape, dtype="uint8")
        elif mask.shape != merged.shape:
            raise ValueError(f"The class masks of run {run} have different shapes.")
        if np.any(merged[mask]):
            raise ValueError(f"The class masks of run {run} overlap, so they cannot be merged.")
        merged[mask] = value

    # The store is built under a temporary name and renamed, so an interrupted merge leaves nothing behind.
    tmp_path = label_path + ".partial"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    store = zarr.open_group(tmp_path, mode="w", zarr_format=2)
    array = store.create_array("0", shape=merged.shape, dtype="uint8", chunks=(64, 256, 256))
    array[:] = merged
    os.rename(tmp_path, label_path)
    return label_path


def get_drg_axon_mito_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DRG-Axon-Mito mitochondrial membrane segmentation dataset.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    os.makedirs(path, exist_ok=True)

    for dataset, run, voxel_spacing, omm_scale, imm_scale in tqdm(RUNS, desc="Downloading the DRG axon tomograms"):
        run_dir = os.path.join(path, f"{dataset}_{run}")
        raw_url = RAW_URL.format(dataset=dataset, run=run, voxel_spacing=voxel_spacing)
        _download_ome_zarr(raw_url, os.path.join(run_dir, "raw.zarr"), download)
        if not download and not os.path.exists(os.path.join(run_dir, "labels.zarr")):
            raise RuntimeError(f"Cannot find the data at {run_dir}, but download was set to False.")
        _merge_labels(run_dir, dataset, run, voxel_spacing, omm_scale, imm_scale, download)

    return path


def get_drg_axon_mito_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DRG-Axon-Mito data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the tomograms.
        List of filepaths to the multi-class mitochondrial membrane masks.
    """
    data_dir = get_drg_axon_mito_data(path, download)
    run_dirs = [f"{dataset}_{run}" for dataset, run, _, _, _ in RUNS]
    raw_paths = [os.path.join(data_dir, run_dir, "raw.zarr") for run_dir in run_dirs]
    label_paths = [os.path.join(data_dir, run_dir, "labels.zarr") for run_dir in run_dirs]
    return raw_paths, label_paths


def get_drg_axon_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for mitochondrial membrane segmentation in cryo-ET tomograms of DRG axons.

    The labels are a multi-class mask with 1: mitochondrial outer membrane and 2: mitochondrial inner membrane.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    raw_paths, label_paths = get_drg_axon_mito_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="0",
        label_paths=label_paths,
        label_key="0",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_drg_axon_mito_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for mitochondrial membrane segmentation in cryo-ET tomograms of DRG axons.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_drg_axon_mito_dataset(path, patch_shape, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
