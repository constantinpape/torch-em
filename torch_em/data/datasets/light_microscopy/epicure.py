"""This dataset contains fluorescence microscopy movies of developing epithelial tissue with per-frame
instance segmentation and full-movie lineage tracking, curated with the EpiCure napari plugin. It covers
four model systems: the Drosophila notum, Drosophila abdomen histoblasts, the zebrafish telencephalon,
and the gastrulating quail.

The dataset is hosted on Zenodo at https://doi.org/10.5281/zenodo.20607705 under the CC BY 4.0 license.
It is from the publication https://doi.org/10.1242/dev.205701.

Please cite it if you use this dataset for your research.
"""

import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import tifffile

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "notum": "https://zenodo.org/records/20607705/files/notumMovie.zip",
    "generalization": "https://zenodo.org/records/20607705/files/MovieEpitheliumFigure5.zip",
}

CHECKSUMS = {
    "notum": "9cbd2399469ea6b267a706490acc3224f4a8f9e23f3bac4020ac39943a4e6dee",
    "generalization": "677c2d39931933a05d5b4aed33891c2edc6f893e0b9444c15559ed6a1e082fe9",
}

SOURCES = ["notum", "histoblast", "telencephalon", "quail_gastrula"]

# Per source: which archive holds it, its raw / label file relative to the extracted archive root,
# and the channel to keep for movies where the raw file has more than one channel.
SOURCE_INFO = {
    "notum": {
        "archive": "notum",
        "raw": "notumMovie/Ecad.tif",
        "label": "notumMovie/epics/Ecad_labels.tif",
        "channel_axis": None,
    },
    "histoblast": {
        "archive": "generalization",
        "raw": "data_generalisations/movie2/abdomen_maxz_z15-24_t1-60_crop.tif",
        "label": "data_generalisations/movie2/epics_corrected/abdomen_maxz_z15-24_t1-60_crop_labels.tif",
        "channel_axis": 1,
        "main_channel": 1,
    },
    "telencephalon": {
        "archive": "generalization",
        "raw": "data_generalisations/movie3/moji_merged_3to13_crop.tif",
        "label": "data_generalisations/movie3/epics_corrected/moji_merged_3to13_crop_labels.tif",
        "channel_axis": 1,
        "main_channel": 0,
    },
    "quail_gastrula": {
        "archive": "generalization",
        "raw": "data_generalisations/movie4/Composite_cropped.tif",
        "label": "data_generalisations/movie4/epics_correctedWithTA/Composite_cropped_labels.tif",
        "channel_axis": 0,
        "main_channel": 0,
        "single_frame": True,
    },
}


def get_epicure_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the EpiCure dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is stored.
    """
    os.makedirs(path, exist_ok=True)

    for archive, marker in [("notum", "notumMovie"), ("generalization", "data_generalisations")]:
        if os.path.exists(os.path.join(path, marker)):
            continue
        zip_path = os.path.join(path, f"{archive}.zip")
        util.download_source(path=zip_path, url=URLS[archive], download=download, checksum=CHECKSUMS[archive])
        util.unzip(zip_path=zip_path, dst=path)

    return path


def _prepare_raw(data_dir, source, info):
    raw_path = os.path.join(data_dir, info["raw"])
    if info["channel_axis"] is None:
        return raw_path

    prepared_path = os.path.join(data_dir, "prepared", f"{source}_raw.tif")
    if os.path.exists(prepared_path):
        return prepared_path

    os.makedirs(os.path.dirname(prepared_path), exist_ok=True)
    raw = tifffile.imread(raw_path)
    raw = np.take(raw, info["main_channel"], axis=info["channel_axis"])
    if info.get("single_frame"):
        raw = raw[None]  # add a singleton frame axis so the array matches the movie sources.
    tifffile.imwrite(prepared_path, raw)
    return prepared_path


def _prepare_label(data_dir, source, info):
    label_path = os.path.join(data_dir, info["label"])
    if not info.get("single_frame"):
        return label_path

    prepared_path = os.path.join(data_dir, "prepared", f"{source}_labels.tif")
    if os.path.exists(prepared_path):
        return prepared_path

    os.makedirs(os.path.dirname(prepared_path), exist_ok=True)
    label = tifffile.imread(label_path)[None]  # add a singleton frame axis, see '_prepare_raw'.
    tifffile.imwrite(prepared_path, label)
    return prepared_path


def get_epicure_paths(
    path: Union[os.PathLike, str], sources: Optional[List[str]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    f"""Get paths for the EpiCure dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sources: The model systems to use. By default uses all of them.
            The available sources are: {', '.join(SOURCES)}.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the raw movies.
        List of filepaths for the instance segmentation and tracking labels.
    """
    sources = SOURCES if sources is None else sources
    for source in sources:
        if source not in SOURCES:
            raise ValueError(f"'{source}' is not a valid source, choose one of {SOURCES}.")

    data_dir = get_epicure_data(path, download)

    raw_paths, label_paths = [], []
    for source in sources:
        info = SOURCE_INFO[source]
        raw_paths.append(_prepare_raw(data_dir, source, info))
        label_paths.append(_prepare_label(data_dir, source, info))

    return raw_paths, label_paths


def get_epicure_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sources: Optional[List[Literal["notum", "histoblast", "telencephalon", "quail_gastrula"]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EpiCure dataset for epithelial cell segmentation and tracking.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sources: The model systems to use. By default uses all of them.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_epicure_paths(path, sources, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_epicure_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    sources: Optional[List[Literal["notum", "histoblast", "telencephalon", "quail_gastrula"]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EpiCure dataloader for epithelial cell segmentation and tracking.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sources: The model systems to use. By default uses all of them.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_epicure_dataset(path, patch_shape, sources, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
