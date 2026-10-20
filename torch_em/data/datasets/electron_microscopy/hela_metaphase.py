"""The HeLa Metaphase dataset contains 3D SBF-SEM volumes of two metaphase HeLa cells with
semantic segmentation annotations of organelles.

It contains two sub-datasets, each showing one quadrant of a metaphase cell (30 nm z-resolution):
- control: A control cell (578 slices of 1477 x 1453 pixels) with the labels
  0 (exterior), 1 (mitochondria), 2 (ER), 3 (ER in close proximity to chromosomes, within 320 nm),
  4 (chromosomes), 5 (centrioles) and 6 (peripheral ER). It also has a binary cell outline annotation.
- reep3_4_kd: A REEP3/4 knock-down cell (471 slices of 1586 x 1600 pixels) with the labels
  0 (exterior), 1 (mitochondria), 2 (lipid droplets), 3 (ER) and 4 (chromosomes).

Note that the label indices differ between the two sub-datasets. The annotations were created
with Microscopy Image Browser and cover one quadrant of the cell and the insides of the spindle area.

The data is available at EMPIAR-10994 (https://doi.org/10.6019/EMPIAR-10994) and is distributed under
the CC0 license. It was published in https://doi.org/10.1091/mbc.e18-11-0698.
Please cite this publication if you use the dataset in your research.
"""

import os
import hashlib
import time
from warnings import warn
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Tuple, Union

import numpy as np
import requests
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10994/data"
# The URL patterns of the (processed) raw slices and the label slices for each sub-dataset.
# NOTE: Only the processed tif slices are downloaded (not the raw dm4 files, nor the full 8.7 GB entry).
URLS = {
    "control": {
        "raw": BASE_URL + "/160621_HeLa_Control/02_processed_dataset/160621_HeLa_R01_{i:03d}.tif",
        "organelles": BASE_URL + "/160621_HeLa_Control/03_models/OrganelleModel_TIF/Labels_160621_HeLa_R01_combined3_{i:03d}.tif",  # noqa
        "cell_outline": BASE_URL + "/160621_HeLa_Control/03_models/CellOutlineModel_TIF/Labels_cell_outlines_{i:03d}.tif",  # noqa
    },
    "reep3_4_kd": {
        "raw": BASE_URL + "/161107_HeLa_REEP3-4_KD_LBR/02_processed_dataset/161107_HeLa_REEP3-4_KD_LBR_{i:03d}.tif",
        "organelles": BASE_URL + "/161107_HeLa_REEP3-4_KD_LBR/03_models/Model_TIF/Labels_161107_HeLa_REEP3-4_KD_LBR_{i:03d}.tif",  # noqa
    },
}
NUM_SLICES = {"control": 578, "reep3_4_kd": 471}
# The sha256 checksums of the stacked raw volumes (the bytes of the uint8 array in zyx order).
CHECKSUMS = {
    "control": "882bc5f1794a35b96617964edb294aee243b30ed57758b964738f244bd2b481f",
    "reep3_4_kd": "9bc5f9689f0e96d3aafd0bd97cfb5f50dff2e82fb0d7dbbe2fc8f09832c0597c",
}

# The organelle label of slice 455 of the control cell is an empty folder in the EMPIAR entry.
# It is filled with the label of the previous slice (the z resolution is 30 nm).
MISSING_LABELS = {"control": {"organelles": [455]}}

NAMES = list(URLS.keys())
LABEL_TYPES = {"control": ["organelles", "cell_outline"], "reep3_4_kd": ["organelles"]}


def _download_slice(url, tmp_dir, key, i):
    import tifffile

    out_path = os.path.join(tmp_dir, f"{key}_{i:03d}.tif")
    if not os.path.exists(out_path):
        for attempt in range(5):
            try:
                with requests.get(url, timeout=120) as r:
                    r.raise_for_status()
                    content = r.content
                # The server may answer with an html error page instead of the tif under load.
                if content[:2] not in (b"II", b"MM"):
                    raise requests.ConnectionError(f"Did not receive a tif file from {url}.")
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        with open(out_path + ".incomplete", "wb") as f:
            f.write(content)
        os.replace(out_path + ".incomplete", out_path)
    return tifffile.imread(out_path)


def _create_volume(name, h5_path, tmp_dir):
    import h5py

    n_slices = NUM_SLICES[name]
    sha = hashlib.sha256()
    tmp_h5 = h5_path + ".incomplete"

    with h5py.File(tmp_h5, "w") as f, ThreadPoolExecutor(max_workers=4) as pool:
        datasets = {}
        for key, url in URLS[name].items():
            # Download in blocks of slices, so that tifs are read in order but fetched in parallel.
            ds_name = "raw" if key == "raw" else f"labels/{key}"
            for start in range(1, n_slices + 1, 32):
                ids = range(start, min(start + 32, n_slices + 1))
                missing = MISSING_LABELS.get(name, {}).get(key, [])
                slices = list(pool.map(
                    lambda i: None if i in missing else _download_slice(url.format(i=i), tmp_dir, key, i), ids
                ))
                if ds_name not in datasets:
                    shape = (n_slices,) + slices[0].shape
                    datasets[ds_name] = f.create_dataset(
                        ds_name, shape=shape, dtype="uint8", chunks=(1,) + slices[0].shape, compression="gzip"
                    )
                for i, slice_ in zip(ids, slices):
                    if slice_ is None:
                        warn(f"The {key} label of slice {i} is missing in EMPIAR-10994, using the previous slice.")
                        slice_ = datasets[ds_name][i - 2]
                    assert slice_.shape == datasets[ds_name].shape[1:], f"Unexpected shape of slice {i} for {key}."
                    datasets[ds_name][i - 1] = slice_.astype("uint8")
                    if key == "raw":
                        sha.update(np.ascontiguousarray(slice_, dtype="uint8").tobytes())

    checksum = sha.hexdigest()
    if CHECKSUMS[name] is not None and checksum != CHECKSUMS[name]:
        os.remove(tmp_h5)
        raise RuntimeError(
            f"The checksum of the downloaded data does not match the expected checksum. "
            f"Expected: {CHECKSUMS[name]}, got: {checksum}"
        )
    os.replace(tmp_h5, h5_path)


def get_hela_metaphase_data(
    path: Union[os.PathLike, str], name: Literal["control", "reep3_4_kd"], download: bool = False
) -> str:
    """Download the HeLa Metaphase dataset.

    The slices are downloaded from EMPIAR-10994 and stacked into a single hdf5 file per sub-dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the sub-dataset. Either 'control' or 'reep3_4_kd'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the hdf5 file with the data.
    """
    if name not in URLS:
        raise ValueError(f"'{name}' is not a valid sub-dataset name. Please choose from {NAMES}.")

    h5_path = os.path.join(path, f"hela_metaphase_{name}.h5")
    if os.path.exists(h5_path):
        return h5_path
    if not download:
        raise RuntimeError(f"Cannot find the data at {h5_path}, but download was set to False")

    tmp_dir = os.path.join(path, f"slices_{name}")
    os.makedirs(tmp_dir, exist_ok=True)
    _create_volume(name, h5_path, tmp_dir)

    # The individual slices are not needed anymore.
    for fname in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fname))
    os.rmdir(tmp_dir)

    return h5_path


def get_hela_metaphase_paths(
    path: Union[os.PathLike, str],
    names: Union[Literal["control", "reep3_4_kd"], List[str], None] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the HeLa Metaphase data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        names: The name(s) of the sub-dataset(s). Either 'control' or 'reep3_4_kd', or a list of them.
            By default, both are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the data.
    """
    if names is None:
        names = NAMES
    elif isinstance(names, str):
        names = [names]

    return [get_hela_metaphase_data(path, name, download) for name in names]


def get_hela_metaphase_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    names: Union[Literal["control", "reep3_4_kd"], List[str], None] = None,
    label_type: Literal["organelles", "cell_outline"] = "organelles",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HeLa Metaphase dataset for organelle segmentation in SBF-SEM.

    The 'organelles' labels are semantic (multi-class) labels, see the module description for the label indices.
    The 'cell_outline' labels are binary and only available for the 'control' sub-dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        names: The name(s) of the sub-dataset(s). Either 'control' or 'reep3_4_kd', or a list of them.
            By default, both are used (only possible for 'organelles' labels).
        label_type: The type of labels. Either 'organelles' or 'cell_outline'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_type not in ("organelles", "cell_outline"):
        raise ValueError(f"'{label_type}' is not a valid label type. Please choose 'organelles' or 'cell_outline'.")
    if label_type == "cell_outline":
        if names is None:
            names = "control"
        if names not in ("control", ["control"]):
            raise ValueError("The 'cell_outline' labels are only available for the 'control' sub-dataset.")

    paths = get_hela_metaphase_paths(path, names, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=f"labels/{label_type}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_hela_metaphase_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    names: Union[Literal["control", "reep3_4_kd"], List[str], None] = None,
    label_type: Literal["organelles", "cell_outline"] = "organelles",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in the HeLa Metaphase dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        names: The name(s) of the sub-dataset(s). Either 'control' or 'reep3_4_kd', or a list of them.
            By default, both are used (only possible for 'organelles' labels).
        label_type: The type of labels. Either 'organelles' or 'cell_outline'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hela_metaphase_dataset(path, patch_shape, names, label_type, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
