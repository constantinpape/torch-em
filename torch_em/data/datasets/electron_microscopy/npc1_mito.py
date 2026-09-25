"""The NPC1 mitochondria dataset provides a mitochondrion segmentation mask for one cryo-ET
tomogram of an NPC1-deficient HEK293T cell.

The data is hosted on the CryoET Data Portal at https://cryoetdataportal.czscience.com/datasets/10456,
a standalone Chan Zuckerberg Initiative-funded data release (grant CZII-2023-327779) with no linked
publication. The dataset covers 4 runs (8 tomograms); only this one run carries any annotation.

The mitochondrion mask (as well as the lysosome and membrane masks also present on this run, not
provided here) is a fully automated prediction (nnInteractive followed by mcm-cryoET smoothing),
not expert-verified ground truth. Treat it the same way as the MitoNet auto-labels in
`mitonet_predicted_kidney.py`: a useful pseudo-label, not verified ground truth.

The data is released under CC0-1.0, per the CryoET Data Portal's portal-wide terms of use. No
public corresponding-author email could be found for this dataset (the corresponding authors,
Daniel Serwas and Utz Heinrich Ermel, have no email listed on the portal or their ORCID records).
"""

import os
import json
from typing import Union, Tuple

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DATASET_ID = 10456
RUN = "25jul29a_Position_3"
VOXEL_SPACING = "VoxelSpacing14.985"
ANNOTATION_FOLDER = "102"

BASE_URL = f"https://files.cryoetdataportal.cziscience.com/{DATASET_ID}/{RUN}/Reconstructions/{VOXEL_SPACING}/"
RAW_URL = BASE_URL + f"Tomograms/100/{RUN}.zarr"
LABEL_URL = BASE_URL + f"Annotations/{ANNOTATION_FOLDER}/mitochondrion-1.0_segmentationmask.zarr"


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


def get_npc1_mito_data(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Download the NPC1 mitochondria cryo-ET tomogram and its automated mitochondrion mask.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the tomogram.
        Filepath to the mitochondrion mask.
    """
    run_dir = os.path.join(path, RUN)
    os.makedirs(run_dir, exist_ok=True)

    raw_path = _download_ome_zarr(RAW_URL, os.path.join(run_dir, "raw.zarr"), download)
    label_path = _download_ome_zarr(LABEL_URL, os.path.join(run_dir, "labels.zarr"), download)
    return os.path.dirname(raw_path), os.path.dirname(label_path)


def get_npc1_mito_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Get paths to the NPC1 mitochondria data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the tomogram.
        Filepath to the mitochondrion mask.
    """
    return get_npc1_mito_data(path, download)


def get_npc1_mito_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the dataset for mitochondrion segmentation in the NPC1 cryo-ET tomogram.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    raw_path, label_path = get_npc1_mito_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_path,
        raw_key="0",
        label_paths=label_path,
        label_key="0",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_npc1_mito_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for mitochondrion segmentation in the NPC1 cryo-ET tomogram.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_npc1_mito_dataset(path, patch_shape, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
