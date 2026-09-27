"""The Murine Phase-Contrast dataset contains annotations for cell instance segmentation in label-free
phase-contrast microscopy images of three murine cell lines: OP9, MC3T3-E1 and NRG.

The dataset consists of 243 RGB images of 1920 x 1200 pixels, each showing a single cell line, with
16,567 manually annotated cell instances in total. The official split is done per image: 223 images
(15,403 cells) for training and 20 images (1,164 cells) for validation. The annotations were made as LabelMe
polygons, which the authors converted to instance masks with sequential polygon filling: a later polygon
overwrites the overlap with an earlier one. This module uses these instance masks (one id per cell, 0 is
background) and converts them once into tif files. A few polygons are completely overwritten by later ones,
so the masks contain 15,371 (train) and 1,160 (val) visible instances instead of 15,403 and 1,164.

The archive additionally contains 16,567 single cell patches, analysis code, model checkpoints and evaluation
tables for a cell type classification study. These are not used and are not extracted by this module.

NOTE: The Zenodo record states the CC-BY-4.0 license, while the README in the archive says that no license was
assigned to its files. Check the terms before you redistribute the data.

The data is located at https://doi.org/10.5281/zenodo.21440736.
Please cite the publication associated with the Zenodo record if you use this dataset in your research.
"""

import os
import csv
import uuid
import zipfile
from natsort import natsorted
from typing import Union, Tuple, Optional, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/21440736/files/Xiong_cell_classification_dataset_v1.1_223_20_verified.zip"
CHECKSUM = "303918988628135edb76de1b00b7e281c83b440e7cb34cda8c81a11d60f0fa5a"

ROOT_IN_ZIP = "Xiong_cell_classification_dataset_v1.1_223_20_verified/dataset/"
CELL_TYPES = ("OP9", "MC3T3-E1", "NRG")
SPLITS = ("train", "val")


def _convert_mask(npz_path, out_path):
    import numpy as np
    import tifffile

    if os.path.exists(out_path):
        return

    label_map = np.load(npz_path)["label_map"].astype("uint16")
    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete.tif"
    tifffile.imwrite(tmp_path, label_map, compression="zlib")
    os.replace(tmp_path, out_path)


def _extract_needed_files(zip_path, data_dir):
    prefixes = tuple(ROOT_IN_ZIP + folder + "/" for folder in ("raw_images", "instance_masks"))
    wanted = ROOT_IN_ZIP + "dataset_splits/image_split.csv"
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not (member.filename.startswith(prefixes) or member.filename == wanted):
                continue
            target = os.path.join(data_dir, member.filename[len(ROOT_IN_ZIP):])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp_path = f"{target}.{uuid.uuid4().hex}.incomplete"
            with archive.open(member) as source, open(tmp_path, "wb") as sink:
                sink.write(source.read())
            os.replace(tmp_path, target)


def get_murine_phase_contrast_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Murine Phase-Contrast dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(os.path.join(data_dir, "dataset_splits", "image_split.csv")):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "murine_phase_contrast.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    _extract_needed_files(zip_path, data_dir)

    assert os.path.exists(os.path.join(data_dir, "dataset_splits", "image_split.csv")), \
        f"The extraction of the archive did not create the expected files in '{data_dir}'."

    return data_dir


def get_murine_phase_contrast_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    cell_type: Optional[Literal["OP9", "MC3T3-E1", "NRG"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Murine Phase-Contrast data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'val'.
        cell_type: The choice of cell line. One of 'OP9', 'MC3T3-E1' or 'NRG'. By default all are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {list(SPLITS)}.")
    if cell_type is not None and cell_type not in CELL_TYPES:
        raise ValueError(f"'{cell_type}' is not a valid cell type. Choose one of {list(CELL_TYPES)}.")

    data_dir = get_murine_phase_contrast_data(path, download)
    label_dir = os.path.join(path, "labels")
    os.makedirs(label_dir, exist_ok=True)

    official_split = "validation" if split == "val" else "train"
    with open(os.path.join(data_dir, "dataset_splits", "image_split.csv")) as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["split"] == official_split and cell_type in (None, row["class_label"])
        ]

    raw_paths, label_paths = [], []
    for row in natsorted(rows, key=lambda row: row["image_id"]):
        raw_path = os.path.join(data_dir, "raw_images", row["filename"])
        npz_path = os.path.join(data_dir, "instance_masks", f"{row['image_id']}_instances.npz")
        label_path = os.path.join(label_dir, f"{row['image_id']}.tif")
        _convert_mask(npz_path, label_path)

        raw_paths.append(raw_path)
        label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_murine_phase_contrast_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    cell_type: Optional[Literal["OP9", "MC3T3-E1", "NRG"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Murine Phase-Contrast dataset for cell instance segmentation in phase-contrast microscopy.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'.
        cell_type: The choice of cell line. One of 'OP9', 'MC3T3-E1' or 'NRG'. By default all are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_murine_phase_contrast_paths(path, split, cell_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_murine_phase_contrast_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    cell_type: Optional[Literal["OP9", "MC3T3-E1", "NRG"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Murine Phase-Contrast dataloader for cell instance segmentation in phase-contrast microscopy.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'.
        cell_type: The choice of cell line. One of 'OP9', 'MC3T3-E1' or 'NRG'. By default all are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_murine_phase_contrast_dataset(
        path, patch_shape, split, cell_type, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
