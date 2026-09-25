"""The PCMMD dataset contains annotations for plasma cell segmentation in
bright-field microscopy images of Wright-Giemsa stained bone marrow smears.

The images were captured with a smartphone camera mounted on a microscope at 100x
magnification, at the feathered edge of bone marrow aspirate smears from patients
investigated for multiple myeloma. The 'segmentation' subset used by this loader holds
single-cell crops, each with one polygon annotation for a plasma cell or a non-plasma
cell: 1,613 plasma cell crops and 1,927 non-plasma cell crops.

NOTE: The full release also has a 'detection' subset with bounding box annotations for
whole slide crops, and per-patient diagnosis data. This loader only prepares the
'segmentation' subset, since it is the only one with pixel-level annotations.

NOTE: The publication defines no split for the 'segmentation' subset. This loader creates
a stratified 80/20 train/test split and caches it to disk, so it stays the same across runs.

The dataset is located at https://doi.org/10.17632/3v2nrxpr9s.1 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-025-04459-1.
Please cite it if you use this dataset in your research.
"""

import os
import json
import zipfile
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-api/zip/3v2nrxpr9s/download/1"
CHECKSUM = "a2e3e2367fc12d1e38bfda28137e86a17c256e29ed73817137fe2ca83207da67"

ARCHIVE_FOLDER = "PCMMD Plasma Cells for Multiple Myeloma Diagnosis"

# The dataset ships two cell folders, each with an 'images' and a 'masks' subfolder.
CELL_FOLDERS = {"plasma": "plasma cells", "non_plasma": "non-plasma cells"}

# The label field in the annotation JSON files marks the class of the segmented cell.
LABEL_IDS = {"plasma_cell": 1, "non_plasma_cell": 2}


def _extract_segmentation_folder(zip_path: str, path: str) -> None:
    """Extract only the segmentation folder of the archive."""
    prefix = f"{ARCHIVE_FOLDER}/data/segmentation/"
    with zipfile.ZipFile(zip_path) as archive:
        members = [n for n in archive.namelist() if n.startswith(prefix)]
        if not members:
            raise RuntimeError(f"The archive {zip_path} does not hold a 'data/segmentation' folder.")
        archive.extractall(path, members=members)


def _rasterize(shapes, shape: Tuple[int, int]) -> np.ndarray:
    """Draw the polygon of the segmented cell, with the pixel value set to its class id."""
    from skimage.draw import polygon as draw_polygon

    semantic = np.zeros(shape, dtype="uint8")
    for item in shapes:
        points = np.array(item["points"], dtype=float)
        rows, columns = draw_polygon(points[:, 1], points[:, 0], shape=shape)
        semantic[rows, columns] = LABEL_IDS.get(item.get("label"), 0)
    return semantic


def _create_labels(cell_dir: str) -> str:
    """Rasterize the polygon of every crop into a label image."""
    from tqdm import tqdm

    label_dir = os.path.join(cell_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    json_paths = natsorted(glob(os.path.join(cell_dir, "masks", "*.json")))
    for json_path in tqdm(json_paths, desc=f"Preprocess the PCMMD annotations in {cell_dir}"):
        stem = Path(json_path).stem
        label_path = os.path.join(label_dir, f"{stem}.tif")
        if os.path.exists(label_path):
            continue

        with open(json_path) as f:
            annotation = json.load(f)

        shape = (annotation["imageHeight"], annotation["imageWidth"])
        semantic = _rasterize(annotation.get("shapes", []), shape)
        imageio.imwrite(label_path, semantic, compression="zlib")

    return label_dir


def _create_split_csv(path: str, sample_ids: List[str]) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    csv_path = os.path.join(path, "pcmmd_split.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    print(f"Creating a new split file at '{csv_path}'.")
    train_ids, test_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)
    split = {sid: "train" for sid in train_ids}
    split.update({sid: "test" for sid in test_ids})
    df = pd.DataFrame({"sample_id": list(split.keys()), "split": list(split.values())})
    df.to_csv(csv_path, index=False)
    return df


def get_pcmmd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PCMMD dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted segmentation data.
    """
    data_dir = os.path.join(path, ARCHIVE_FOLDER, "data", "segmentation")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "pcmmd.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    _extract_segmentation_folder(zip_path, path)

    return data_dir


def get_pcmmd_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    cell_type: Literal["plasma", "non_plasma", "both"] = "both",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PCMMD data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        cell_type: The cell type to load. One of 'plasma', 'non_plasma' or 'both'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose from 'train' or 'test'.")
    if cell_type not in ("plasma", "non_plasma", "both"):
        raise ValueError(f"'{cell_type}' is not a valid cell type. Choose from 'plasma', 'non_plasma' or 'both'.")

    data_dir = get_pcmmd_data(path, download)
    cell_types = list(CELL_FOLDERS.keys()) if cell_type == "both" else [cell_type]

    pairs = []
    for ctype in cell_types:
        cell_dir = os.path.join(data_dir, CELL_FOLDERS[ctype])
        label_dir = _create_labels(cell_dir)
        image_paths = natsorted(glob(os.path.join(cell_dir, "images", "*.jpg")))
        for image_path in image_paths:
            stem = Path(image_path).stem
            label_path = os.path.join(label_dir, f"{stem}.tif")
            if not os.path.exists(label_path):
                continue
            pairs.append((f"{ctype}_{stem}", image_path, label_path))

    if not pairs:
        raise RuntimeError(f"Could not find any PCMMD data for cell_type='{cell_type}' in {data_dir}.")

    split_df = _create_split_csv(data_dir, [sample_id for sample_id, _, _ in pairs])
    split_ids = set(split_df[split_df["split"] == split]["sample_id"])

    image_paths = [image_path for sample_id, image_path, _ in pairs if sample_id in split_ids]
    label_paths = [label_path for sample_id, _, label_path in pairs if sample_id in split_ids]

    if not image_paths:
        raise RuntimeError(f"Could not find any PCMMD data for split='{split}' in {data_dir}.")

    return image_paths, label_paths


def get_pcmmd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    cell_type: Literal["plasma", "non_plasma", "both"] = "both",
    label_choice: Literal["semantic", "binary"] = "semantic",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the PCMMD dataset for plasma cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        cell_type: The cell type to load. One of 'plasma', 'non_plasma' or 'both'.
        label_choice: The label to use. Either 'semantic', where a plasma cell is labeled 1 and a
            non-plasma cell 2, or 'binary', where every cell is labeled 1.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The PCMMD patch shape must be two-dimensional, got {patch_shape}.")
    if label_choice not in ("semantic", "binary"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'semantic' or 'binary'.")

    image_paths, label_paths = get_pcmmd_paths(path, split, cell_type, download)

    if label_choice == "binary":
        kwargs["label_transform"] = torch_em.transform.label.labels_to_binary
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs,
    )


def get_pcmmd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    cell_type: Literal["plasma", "non_plasma", "both"] = "both",
    label_choice: Literal["semantic", "binary"] = "semantic",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the PCMMD dataloader for plasma cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        cell_type: The cell type to load. One of 'plasma', 'non_plasma' or 'both'.
        label_choice: The label to use. Either 'semantic', where a plasma cell is labeled 1 and a
            non-plasma cell 2, or 'binary', where every cell is labeled 1.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pcmmd_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        cell_type=cell_type,
        label_choice=label_choice,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
