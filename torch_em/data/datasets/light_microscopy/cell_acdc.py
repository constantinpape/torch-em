"""This dataset contains phase-contrast time-lapse images of budding yeast (S. cerevisiae) with
per-frame single-cell instance segmentation, lineage tracking (mother-bud relationships) and
cell-cycle stage annotations, from the Cell-ACDC software test data.

The dataset is hosted on Zenodo at https://doi.org/10.5281/zenodo.6795124.
The dataset is from the publication https://doi.org/10.1186/s12915-022-01372-6.

Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import tifffile

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/6795124/files/test_data_budding_yeast.zip"
CHECKSUM = "391b774888946ccd201be2ad0719a9ddd966b2d2215712ea3f4df9e52fbf9cc6"


def get_cell_acdc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Cell-ACDC budding yeast dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is stored.
    """
    data_dir = os.path.join(path, "test_data_budding_yeast")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "test_data_budding_yeast.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _prepare_position(segm_path):
    # 'last_tracked_i.txt' records the last annotated frame; raw stacks always have more frames.
    last_tracked_path = segm_path.replace("_segm.npz", "_last_tracked_i.txt")
    with open(last_tracked_path) as f:
        n_frames = int(f.read().strip()) + 1

    label_path = segm_path.replace(".npz", ".tif")
    if not os.path.exists(label_path):
        labels = np.load(segm_path)["arr_0"][:n_frames]
        tifffile.imwrite(label_path, labels)

    raw_path = segm_path.replace("_segm.npz", "_phase_contr.tif")
    if tifffile.TiffFile(raw_path).series[0].shape[0] != n_frames:
        matched_raw_path = segm_path.replace("_segm.npz", "_phase_contr_matched.tif")
        if not os.path.exists(matched_raw_path):
            raw = tifffile.imread(raw_path)[:n_frames]
            tifffile.imwrite(matched_raw_path, raw)
        raw_path = matched_raw_path

    return raw_path, label_path


def get_cell_acdc_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths for the Cell-ACDC budding yeast dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the raw phase-contrast images.
        List of filepaths for the instance segmentation and tracking labels.
    """
    data_dir = get_cell_acdc_data(path, download)

    segm_paths = sorted(glob(os.path.join(data_dir, "TimeLapse_2D", "*_labeled", "Position_*", "Images", "*_segm.npz")))
    assert segm_paths, f"No labeled positions found at {data_dir}."

    raw_paths, label_paths = [], []
    for segm_path in segm_paths:
        raw_path, label_path = _prepare_position(segm_path)
        raw_paths.append(raw_path)
        label_paths.append(label_path)

    return raw_paths, label_paths


def get_cell_acdc_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the Cell-ACDC dataset for budding yeast segmentation and tracking.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cell_acdc_paths(path, download)

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


def get_cell_acdc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Cell-ACDC dataloader for budding yeast segmentation and tracking.

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
    dataset = get_cell_acdc_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
