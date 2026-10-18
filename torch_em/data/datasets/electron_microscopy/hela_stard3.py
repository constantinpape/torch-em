"""The HeLa STARD3 dataset contains FIB-SEM volumes of HeLa cells with manual annotations of
mitochondria, endoplasmic reticulum (ER) and endosomes / lysosomes.

It contains three volumes: control HeLa cells ('ctrl'), cells expressing wild-type STARD3 ('stard3')
and cells expressing the STARD3 FA/YA mutant ('stard3_fa_ya').
Each volume has a region of interest (ROI) with dense annotations. Only this ROI is used here:
the raw data is cropped to the ROI and stored together with the binary labels of the three organelles.
The annotations outside of the ROI are not used.

NOTE: Four slices of the 'ctrl' volume (slices 149, 151 and 152 of the original stack are truncated and
slice 154 is missing in the EMPIAR archive). They are replaced by the closest valid slice (and a warning is raised).

The dataset is located at https://www.ebi.ac.uk/empiar/EMPIAR-13156 (https://doi.org/10.6019/EMPIAR-13156).
The license is not stated in the EMPIAR metadata (EMPIAR entries are usually released under CC0),
please check the entry before using the data.
The data is part of the publication https://doi.org/10.1038/s44318-026-00705-3.
Please cite it if you use this dataset in your research.
"""

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/13156/data"
ORGANELLES = ["mitochondria", "er", "endosomes"]
FILE_NAMES = {"mitochondria": "Mitochondria", "er": "ER", "endosomes": "Endosomes"}

# The EMPIAR folder, the number of slices, the first slices of the sub-folders, the pattern of the slice filenames
# and the region of interest (ROI) of the volumes. The ROI is given as start and shape in (z, y, x).
# The start is zero-based. NOTE: The label volumes of 'stard3' have fewer rows (2112) than the raw slices (2748),
# they are aligned to the top of the slices.
VOLUMES = {
    "ctrl": {
        "folder": "HeLaWT_Raw",
        "n_slices": 994,
        "chunk_starts": [1, 201, 401, 601, 801],
        "pattern": "ROI_CICGFA_{:03d}.tif",
        "roi_start": (5, 64, 1536),
        "roi_shape": (444, 900, 1200),
    },
    "stard3": {
        "folder": "HeLaSTARD3_WT_Raw",
        "n_slices": 810,
        "chunk_starts": [1, 201, 401, 601],
        "pattern": "image_{:03d}.tif",
        "roi_start": (428, 151, 16),
        "roi_shape": (382, 1100, 1400),
    },
    "stard3_fa_ya": {
        "folder": "HeLaSTARD3_FA_YA_Raw",
        "n_slices": 600,
        "chunk_starts": [1, 101, 201, 301, 401, 501],
        "pattern": "image{:03d}.tif",
        "roi_start": (120, 50, 386),
        "roi_shape": (449, 700, 700),
    },
}
# The checksums of the label files with the annotations inside of the ROI.
CHECKSUMS = {
    "ctrl": {
        "mitochondria": "1c05a133f4dbb763db5a9affb1afe3a54622beae55dab1d32a318ed3cd36aec7",
        "er": "3c3ea1b4c9a91aba8b982b712ed48e5f323427fc7d7f2951e05b542577a1dfa6",
        "endosomes": "984201582bd00f307ed2af627a33665168d4f5c35435569e850b979266fff049",
    },
    "stard3": {
        "mitochondria": "32117b402c2d92bafb9a1c7d999af5f05d5e359674dc0f308da788fe448c5d71",
        "er": "f54d5c7ea87d710e9e53ea8d01d062a92c975684237d14a9d336fb61423c8069",
        "endosomes": "b33b88a09c46abdb7786ffd474e1c6a52578efcdc1ed20bac166637ef36a6b2f",
    },
    "stard3_fa_ya": {
        "mitochondria": "b6278389178f881e2ecb0f65713746096b0ef207bc6233a4e63ffbd973695c21",
        "er": "51df8cc2d5d1f45571828d16dd61b53623edd9ba92e32481b4f8bb1188e4bafb",
        "endosomes": "2ce30154604aa751860ae9711c1c49626b6bf3122c97e30dfa8991fe122ea5c4",
    },
}


def _label_url(volume, organelle):
    return f"{BASE_URL}/{VOLUMES[volume]['folder']}/{FILE_NAMES[organelle]}_ROI.tif"


def _slice_url(volume, idx):
    # The slices are stored in sub-folders with a fixed number of slices. The index is one-based.
    info = VOLUMES[volume]
    starts = info["chunk_starts"]
    i = max(j for j, s in enumerate(starts) if s <= idx)
    start = starts[i]
    stop = starts[i + 1] - 1 if i + 1 < len(starts) else info["n_slices"]
    return f"{BASE_URL}/{info['folder']}/{start}-{stop}/{info['pattern'].format(idx)}"


def _download_slices(volume, tmp_dir, download):
    """Download the slices of the ROI. The slice with (zero-based) index z is stored as the file z + 1."""
    info = VOLUMES[volume]
    z0 = info["roi_start"][0]
    indices = range(z0 + 1, z0 + info["roi_shape"][0] + 1)

    def _download(idx):
        path = os.path.join(tmp_dir, f"{idx:04d}.tif")
        with warnings.catch_warnings():
            # There are no checksums for the individual slices.
            warnings.simplefilter("ignore")
            for attempt in range(3):
                try:
                    util.download_source(path, _slice_url(volume, idx), download)
                    break
                except requests.HTTPError as e:
                    # Some slices are missing in the archive.
                    if e.response is not None and e.response.status_code == 404:
                        return None
                    if attempt == 2:
                        raise
                except requests.ConnectionError:  # Retry on transient connection problems.
                    if attempt == 2:
                        raise
                time.sleep(5 * (attempt + 1))
        return path

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(_download, indices))


def _create_volume(volume, h5_path, download):
    import h5py
    import tifffile

    info = VOLUMES[volume]
    (z0, y0, x0), (sz, sy, sx) = info["roi_start"], info["roi_shape"]
    roi = (slice(z0, z0 + sz), slice(y0, y0 + sy), slice(x0, x0 + sx))

    tmp_dir = os.path.join(os.path.dirname(h5_path), f"{volume}_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Download the labels and check that they agree with the ROI.
    labels = {}
    for organelle in ORGANELLES:
        label_path = os.path.join(tmp_dir, f"{FILE_NAMES[organelle]}_ROI.tif")
        util.download_source(label_path, _label_url(volume, organelle), download, CHECKSUMS[volume][organelle])
        label = tifffile.imread(label_path)
        # The label volume has the shape of the full stack, but it is only annotated inside of the ROI.
        assert label.shape[0] == info["n_slices"], f"Unexpected label shape {label.shape} for {label_path}"
        assert int(label.sum()) == int(label[roi].sum()), f"Found annotations outside of the ROI in {label_path}"
        labels[organelle] = label[roi].astype("uint8")
        del label

    slice_paths = _download_slices(volume, tmp_dir, download)

    raw = np.zeros((sz, sy, sx), dtype="uint8")
    valid = np.zeros(sz, dtype=bool)
    for i, path in enumerate(slice_paths):
        if path is None:
            continue
        try:
            image = tifffile.imread(path)
        except (ValueError, tifffile.TiffFileError):  # Truncated files can't be read.
            continue
        raw[i] = image[y0:y0 + sy, x0:x0 + sx]
        valid[i] = True

    if not valid.any():
        raise RuntimeError(f"Could not read any of the slices for {volume}.")
    if not valid.all():
        bad = (np.where(~valid)[0] + z0 + 1).tolist()
        warnings.warn(
            f"The slices {bad} of '{volume}' are corrupted or missing and are replaced by the closest valid slice."
        )
        valid_ids = np.where(valid)[0]
        for i in np.where(~valid)[0]:
            raw[i] = raw[valid_ids[np.argmin(np.abs(valid_ids - i))]]

    tmp_h5 = f"{h5_path}.incomplete"
    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(1, min(sy, 512), min(sx, 512)))
        for organelle, label in labels.items():
            f.create_dataset(
                f"labels/{organelle}", data=label, compression="gzip", chunks=(1, min(sy, 512), min(sx, 512))
            )
    os.replace(tmp_h5, h5_path)

    # Remove the downloaded files.
    for fname in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fname))
    os.rmdir(tmp_dir)


def get_hela_stard3_data(
    path: Union[os.PathLike, str],
    volume: Literal["ctrl", "stard3", "stard3_fa_ya"],
    download: bool = False,
) -> str:
    """Download the HeLa STARD3 dataset.

    Only the slices of the annotated region of interest (ROI) and the corresponding labels are downloaded.
    They are stored in a single hdf5 file per volume.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volume: The volume. Either 'ctrl', 'stard3' or 'stard3_fa_ya'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the hdf5 file with the data.
    """
    if volume not in VOLUMES:
        raise ValueError(f"'{volume}' is not a valid volume. Please choose from {list(VOLUMES)}.")

    h5_path = os.path.join(path, f"{volume}.h5")
    if os.path.exists(h5_path):
        return h5_path

    if not download:
        raise RuntimeError(f"Cannot find the data at {h5_path}, but download was set to False")

    os.makedirs(path, exist_ok=True)
    _create_volume(volume, h5_path, download)
    return h5_path


def get_hela_stard3_paths(
    path: Union[os.PathLike, str],
    volumes: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the HeLa STARD3 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volumes: The volumes to use. Either 'ctrl', 'stard3', 'stard3_fa_ya' or a list of them.
            By default all volumes are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the data.
    """
    if volumes is None:
        volumes = list(VOLUMES)
    elif isinstance(volumes, str):
        volumes = [volumes]

    return [get_hela_stard3_data(path, volume, download) for volume in volumes]


def get_hela_stard3_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    organelle: Literal["mitochondria", "er", "endosomes"] = "mitochondria",
    volumes: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HeLa STARD3 dataset for organelle segmentation in FIB-SEM.

    The labels are binary (1 is the organelle).

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        organelle: The organelle to segment. Either 'mitochondria', 'er' or 'endosomes'.
        volumes: The volumes to use. Either 'ctrl', 'stard3', 'stard3_fa_ya' or a list of them.
            By default all volumes are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if organelle not in ORGANELLES:
        raise ValueError(f"'{organelle}' is not a valid organelle. Please choose from {ORGANELLES}.")

    paths = get_hela_stard3_paths(path, volumes, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=f"labels/{organelle}",
        patch_shape=patch_shape,
        **kwargs
    )


def get_hela_stard3_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    organelle: Literal["mitochondria", "er", "endosomes"] = "mitochondria",
    volumes: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HeLa STARD3 dataloader for organelle segmentation in FIB-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organelle: The organelle to segment. Either 'mitochondria', 'er' or 'endosomes'.
        volumes: The volumes to use. Either 'ctrl', 'stard3', 'stard3_fa_ya' or a list of them.
            By default all volumes are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hela_stard3_dataset(path, patch_shape, organelle, volumes, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
