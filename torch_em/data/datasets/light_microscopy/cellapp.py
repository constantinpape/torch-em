"""The Cell-APP dataset contains annotations for cell segmentation in
transmitted light microscopy images of cultured mammalian cells.

The dataset provides per cell line subsets for HeLa, RPE1 and U2OS, and eight multi cell line
subsets of increasing size under 'general'. Every cell carries a mitotic or nonmitotic class, at a
ratio of about one to twenty. The loader stores the instance labels and the semantic labels, so you
can train on either target.

NOTE: An automatic pipeline created the annotations, so an image does not show every cell as an
instance. A cell without an annotation becomes background in the label image. Take this into account
when you train on this data, and when you measure the recall of a model on it.

NOTE: The archive does not contain the images of the HeLa train split, although its annotation file
declares 270 images. Only the HeLa test split is available.

The dataset is located at https://doi.org/10.5281/zenodo.16738843 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1091/mbc.E25-02-0076.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/16738843/files/for_zenodo_d.zip/content"
CHECKSUM = "0e875ef3ec5e2937155e21ed0e905da6780108edace64d9c6c94dcfe52e28c18"

# The image folder and the COCO file of each cell line split.
CELL_LINES = {
    ("hela", "test"): ("HeLa/test/images", "HeLa/test/hela_0.1_test.json"),
    ("rpe1", "train"): ("RPE1/train/images", "RPE1/train/rpe1_0.4_train.json"),
    ("rpe1", "test"): ("RPE1/test/images", "RPE1/test/rpe1_0.4_test.json"),
    ("u2os", "train"): ("U2OS/train/images", "U2OS/train/instances_u2os_0.8_train.json"),
    ("u2os", "test"): ("U2OS/test/images", "U2OS/test/instances_u2os_0.8_test.json"),
}

SOURCES = ("hela", "rpe1", "u2os", "general")
GENERAL_SIZES = tuple(range(1, 9))


def _get_subset_layout(source: str, split: Optional[str], size: int) -> Tuple[str, str, str]:
    """Get the image folder, the COCO file and the name of the requested subset."""
    if source not in SOURCES:
        raise ValueError(f"'{source}' is not a valid source. Choose from {list(SOURCES)}.")

    if source == "general":
        if size not in GENERAL_SIZES:
            raise ValueError(f"'{size}' is not a valid size. Choose an integer from {list(GENERAL_SIZES)}.")
        return f"general/dataset_{size}/data", f"general/dataset_{size}/labels.json", f"general_{size}"

    if (source, split) not in CELL_LINES:
        if source == "hela" and split == "train":
            raise ValueError(
                "The Cell-APP archive does not contain the images of the HeLa train split. "
                "Use split='test' for HeLa, or use another source."
            )
        valid = [s for (line, s) in CELL_LINES if line == source]
        raise ValueError(f"'{split}' is not a valid split for '{source}'. Choose from {valid}.")

    image_dir, coco_file = CELL_LINES[(source, split)]
    return image_dir, coco_file, f"{source}_{split}"


def _rasterize(annotation, shape: Tuple[int, int]) -> np.ndarray:
    """Draw all polygons of one annotation into a single binary mask."""
    from skimage.draw import polygon as draw_polygon

    mask = np.zeros(shape, dtype=bool)
    for part in annotation["segmentation"]:
        polygon = np.array(part, dtype=float).reshape(-1, 2)
        rows, columns = draw_polygon(polygon[:, 1], polygon[:, 0], shape=shape)
        mask[rows, columns] = True
    return mask


def _create_labels(data_dir: str, source: str, split: Optional[str], size: int) -> str:
    """Rasterize the COCO polygons into instance labels and semantic labels."""
    import h5py
    from tqdm import tqdm

    image_dir, coco_file, subset = _get_subset_layout(source, split, size)

    preprocessed_dir = os.path.join(data_dir, "preprocessed", subset)
    os.makedirs(preprocessed_dir, exist_ok=True)

    with open(os.path.join(data_dir, coco_file)) as f:
        coco = json.load(f)

    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)

    for image in tqdm(coco["images"], desc=f"Preprocess the '{subset}' subset"):
        stem = Path(image["file_name"]).stem
        output_path = os.path.join(preprocessed_dir, f"{stem}.h5")
        if os.path.exists(output_path):
            continue

        image_path = os.path.join(data_dir, image_dir, image["file_name"])
        if not os.path.exists(image_path):
            raise RuntimeError(f"Could not find the Cell-APP image {image_path}.")

        raw = imageio.imread(image_path)
        if raw.ndim == 3:
            raw = raw[..., 0]  # The channels of the transmitted light images are identical.

        shape = (image["height"], image["width"])
        masks = [(_rasterize(a, shape), a["category_id"]) for a in annotations[image["id"]]]

        # Paint the large cells first, so that a small cell on top of a large one keeps its label.
        labels = np.zeros(shape, dtype="uint16")
        semantic = np.zeros(shape, dtype="uint8")
        for instance_id, (mask, category_id) in enumerate(sorted(masks, key=lambda m: -m[0].sum()), start=1):
            labels[mask] = instance_id
            semantic[mask] = category_id + 1  # 0 is the background, 1 is nonmitotic, 2 is mitotic.

        with h5py.File(output_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
            f.create_dataset("semantic", data=semantic, compression="gzip")

    return preprocessed_dir


def get_cellapp_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Cell-APP dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "for_zenodo_d")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "for_zenodo_d.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_cellapp_paths(
    path: Union[os.PathLike, str],
    source: Literal["hela", "rpe1", "u2os", "general"] = "general",
    split: Optional[Literal["train", "test"]] = None,
    size: int = 8,
    download: bool = False,
) -> List[str]:
    """Get paths to the Cell-APP data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The data source. Either a cell line, 'hela', 'rpe1' or 'u2os', or 'general'.
        split: The data split, 'train' or 'test'. The source 'general' has no split.
            The source 'hela' has no train split, because the archive misses its images.
        size: The size of the 'general' subset, an integer from 1 to 8. Other sources ignore it.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the preprocessed h5 data.
    """
    data_dir = get_cellapp_data(path, download)
    preprocessed_dir = _create_labels(data_dir, source, split, size)

    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    if not volume_paths:
        raise RuntimeError(f"Could not find any preprocessed Cell-APP data in {preprocessed_dir}.")

    return volume_paths


def get_cellapp_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    source: Literal["hela", "rpe1", "u2os", "general"] = "general",
    split: Optional[Literal["train", "test"]] = None,
    size: int = 8,
    label_choice: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Cell-APP dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        source: The data source. Either a cell line, 'hela', 'rpe1' or 'u2os', or 'general'.
        split: The data split, 'train' or 'test'. The source 'general' has no split.
            The source 'hela' has no train split, because the archive misses its images.
        size: The size of the 'general' subset, an integer from 1 to 8. Other sources ignore it.
        label_choice: The target. Either 'instances' for the cell instances, or 'semantic' for the
            mitotic and nonmitotic classes.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The Cell-APP patch shape must be two-dimensional, got {patch_shape}.")

    if label_choice not in ("instances", "semantic"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'instances' or 'semantic'.")

    volume_paths = get_cellapp_paths(path, source, split, size, download)
    label_key = "labels" if label_choice == "instances" else "semantic"

    if label_choice == "instances":
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
        )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs,
    )


def get_cellapp_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    source: Literal["hela", "rpe1", "u2os", "general"] = "general",
    split: Optional[Literal["train", "test"]] = None,
    size: int = 8,
    label_choice: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Cell-APP dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        source: The data source. Either a cell line, 'hela', 'rpe1' or 'u2os', or 'general'.
        split: The data split, 'train' or 'test'. The source 'general' has no split.
            The source 'hela' has no train split, because the archive misses its images.
        size: The size of the 'general' subset, an integer from 1 to 8. Other sources ignore it.
        label_choice: The target. Either 'instances' for the cell instances, or 'semantic' for the
            mitotic and nonmitotic classes.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellapp_dataset(
        path=path,
        patch_shape=patch_shape,
        source=source,
        split=split,
        size=size,
        label_choice=label_choice,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
