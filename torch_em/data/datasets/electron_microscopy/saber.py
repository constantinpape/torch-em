"""The SABER dataset contains annotations for bacterial cell segmentation in cryo-ET.

The data is hosted on the CryoET Data Portal at https://cryoetdataportal.czscience.com/depositions/10331.
This loader provides the expert-reviewed subset of the deposition: the semi-manual instance masks of
Legionella pneumophila cell interiors. The automated SABER predictions are not part of it.

The dataset is part of the publication https://doi.org/10.2139/ssrn.6754411.
Please cite it if you use this dataset in your research.
"""

import os
import json
from typing import Union, Tuple, List

import requests
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://files.cryoetdataportal.cziscience.com/{dataset}/{run}/Reconstructions/{spacing}/"
RAW_URL = BASE_URL + "Tomograms/100/{run}.zarr"
LABEL_URL = BASE_URL + "Annotations/{annotation}/intracellular_anatomical_structure-1.0_instancesegmentationmask.zarr"

# The portal dataset, run name, voxel spacing folder and annotation folder of each ground-truth run.
RUNS = [
    (10062, "dga2017-02-02-12", "VoxelSpacing15.600", "104"),
    (10062, "dga2017-02-02-13", "VoxelSpacing15.600", "104"),
    (10062, "dga2017-02-02-23", "VoxelSpacing15.600", "105"),
    (10062, "dga2017-02-02-24", "VoxelSpacing15.600", "105"),
    (10062, "dga2017-02-02-25", "VoxelSpacing15.600", "105"),
    (10062, "dga2017-02-02-26", "VoxelSpacing15.600", "105"),
    (10062, "dga2017-02-02-27", "VoxelSpacing15.600", "104"),
    (10062, "dga2017-02-02-29", "VoxelSpacing15.600", "105"),
    (10062, "dga2017-02-02-31", "VoxelSpacing15.600", "105"),
    (10064, "dga2017-02-08-101", "VoxelSpacing15.600", "106"),
    (10064, "dga2017-02-08-111", "VoxelSpacing15.600", "106"),
    (10064, "dga2017-02-08-14", "VoxelSpacing15.600", "105"),
    (10064, "dga2017-02-08-16", "VoxelSpacing15.600", "105"),
    (10064, "dga2017-02-08-17", "VoxelSpacing15.600", "108"),
    (10064, "dga2017-02-08-18", "VoxelSpacing15.600", "106"),
    (10064, "dga2017-02-08-27", "VoxelSpacing15.600", "107"),
    (10064, "dga2017-02-08-28", "VoxelSpacing15.600", "105"),
    (10064, "dga2017-02-08-49", "VoxelSpacing15.600", "106"),
    (10077, "dga2016-01-12-12", "VoxelSpacing16.800", "105"),
    (10077, "dga2016-03-31-4", "VoxelSpacing16.800", "106"),
    (10077, "dga2016-03-31-41", "VoxelSpacing16.800", "105"),
    (10077, "dga2016-03-31-45", "VoxelSpacing16.800", "106"),
]

SCALES = (0, 1, 2)


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


def _download_ome_zarr(url, out_path, scale, download):
    array_path = os.path.join(out_path, str(scale))
    if os.path.exists(array_path):
        return array_path

    if not download:
        raise RuntimeError(f"Cannot find the data at {out_path}, but download was set to False.")

    os.makedirs(out_path, exist_ok=True)
    for name in (".zattrs", ".zgroup"):
        if not os.path.exists(os.path.join(out_path, name)):
            _fetch(f"{url}/{name}", os.path.join(out_path, name))

    # The scale is downloaded under a temporary name so that an interrupted download is not treated as complete.
    tmp_path = os.path.join(out_path, f"{scale}.partial")
    os.makedirs(tmp_path, exist_ok=True)
    _fetch(f"{url}/{scale}/.zarray", os.path.join(tmp_path, ".zarray"))
    with open(os.path.join(tmp_path, ".zarray")) as f:
        meta = json.load(f)

    grid = [-(-size // chunk) for size, chunk in zip(meta["shape"], meta["chunks"])]
    for z in range(grid[0]):
        for y in range(grid[1]):
            for x in range(grid[2]):
                chunk_dir = os.path.join(tmp_path, str(z), str(y))
                os.makedirs(chunk_dir, exist_ok=True)
                _fetch(f"{url}/{scale}/{z}/{y}/{x}", os.path.join(chunk_dir, str(x)), optional=True)

    os.rename(tmp_path, array_path)
    return array_path


def get_saber_data(path: Union[os.PathLike, str], scale: int = 0, download: bool = False) -> str:
    """Download the SABER cryo-ET dataset.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        scale: The resolution level of the multiscale data. 0 is the native resolution.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if scale not in SCALES:
        raise ValueError(f"The scale must be one of {SCALES}, got {scale}.")

    data_dir = os.path.join(path, "data")
    os.makedirs(data_dir, exist_ok=True)

    for dataset, run, spacing, annotation in tqdm(RUNS, desc="Downloading tomograms"):
        urls = {
            "raw": RAW_URL.format(dataset=dataset, run=run, spacing=spacing),
            "labels": LABEL_URL.format(dataset=dataset, run=run, spacing=spacing, annotation=annotation),
        }
        for name, url in urls.items():
            _download_ome_zarr(url, os.path.join(data_dir, run, f"{name}.zarr"), scale, download)

    return data_dir


def get_saber_paths(
    path: Union[os.PathLike, str], scale: int = 0, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SABER data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        scale: The resolution level of the multiscale data. 0 is the native resolution.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the tomograms.
        List of filepaths to the instance masks.
    """
    data_dir = get_saber_data(path, scale, download)
    raw_paths = [os.path.join(data_dir, run, "raw.zarr") for _, run, _, _ in RUNS]
    label_paths = [os.path.join(data_dir, run, "labels.zarr") for _, run, _, _ in RUNS]
    return raw_paths, label_paths


def get_saber_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    scale: int = 0,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for bacterial cell segmentation in cryo-ET data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        scale: The resolution level of the multiscale data. 0 is the native resolution.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    raw_paths, label_paths = get_saber_paths(path, scale, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=str(scale),
        label_paths=label_paths,
        label_key=str(scale),
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_saber_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    scale: int = 0,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for bacterial cell segmentation in cryo-ET data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        scale: The resolution level of the multiscale data. 0 is the native resolution.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_saber_dataset(path, patch_shape, scale=scale, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
