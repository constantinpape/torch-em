"""The CCAgT dataset contains annotations for nucleus segmentation in
bright-field microscopy images of cervical cells with the AgNOR stain.

CCAgT stands for Cervical Cells with the AgNOR Technique. The stain marks the nucleolar organizer
regions (NORs) of a nucleus as dark dots, so the annotations cover both the nuclei and the NORs
inside them. The dataset holds 9339 image tiles of 15 slides with 63190 annotated objects.

NOTE: This dataset uses the CC BY-NC 3.0 license, which forbids commercial use. The other datasets
of this module use a permissive license, so check the terms before you use this one in your project.

NOTE: A NOR sits inside a nucleus, and in the released masks the NOR label overwrites the nucleus.
A nucleus therefore looks like a ring with holes. Use the label choice 'nuclei' to merge the nucleus
classes and the NOR classes back into whole nuclei.

NOTE: The class 'Satellite' does not hold a real outline. The annotators clicked a point, and the
export drew a disc of a fixed radius around it, so 6457 of its 6478 objects have the same area. The
label choice 'instances' drops this class.

NOTE: The publication defines no split. This loader splits by slide, so that no slide appears in two
splits. A split over single tiles would put tiles of one patient into several splits.

The dataset is located at https://doi.org/10.17632/wg4bpm33hj.2 under the CC BY-NC 3.0 license.
This dataset is from the publication https://doi.org/10.1109/CBMS49503.2020.00110.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from collections import defaultdict
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-api/zip/wg4bpm33hj/download/2"
CHECKSUM = "eb9f01e8feae0029056dac2dfa6f5e9780692e1a54c0df1bbd33cd279b9eb557"

ARCHIVE_ROOT = "wg4bpm33hj-2"

CLASS_NAMES = {
    1: "nucleus",
    2: "cluster",
    3: "satellite",
    4: "nucleus_out_of_focus",
    5: "overlapped_nuclei",
    6: "non_viable_nucleus",
    7: "leukocyte_nucleus",
}

# The nucleus classes and the NOR classes together form a whole nucleus.
NUCLEUS_CLASSES = (1, 2, 3, 4, 5, 6)

# The annotators clicked a point for this class, so its outline is a disc of a fixed radius.
SYNTHETIC_CLASS = 3

# The slides of one split never appear in another split, so a patient stays in one split.
SPLITS = {
    "train": ("A", "D", "F", "I", "J", "K", "M", "N", "O"),
    "val": ("B", "G", "H"),
    "test": ("C", "E", "L"),
}

LABEL_CHOICES = ("semantic", "nuclei", "instances")


def _extract_archive(zip_path: str, path: str) -> None:
    """Extract the archive and the per slide archives that it holds."""
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(path)

    data_dir = os.path.join(path, ARCHIVE_ROOT)
    for kind in ("images", "masks"):
        for slide_path in natsorted(glob(os.path.join(data_dir, kind, "*.zip"))):
            with zipfile.ZipFile(slide_path) as archive:
                archive.extractall(os.path.join(data_dir, kind))
            os.remove(slide_path)


def _rasterize(annotations, shape: Tuple[int, int]) -> np.ndarray:
    """Draw one label per object, and let a small object win over a large one."""
    from skimage.draw import polygon as draw_polygon

    labels = np.zeros(shape, dtype="uint16")
    # Draw the large objects first, so that a small object on top keeps its label.
    ordered = sorted(annotations, key=lambda a: -a.get("area", 0))
    for instance_id, annotation in enumerate(ordered, start=1):
        for part in annotation["segmentation"]:
            polygon = np.array(part, dtype=float).reshape(-1, 2)
            rows, columns = draw_polygon(polygon[:, 1], polygon[:, 0], shape=shape)
            labels[rows, columns] = instance_id
    return labels


def _create_instance_labels(data_dir: str) -> str:
    """Rasterize the COCO polygons of every tile into an instance label image."""
    from tqdm import tqdm

    label_dir = os.path.join(data_dir, "instance_labels")
    if os.path.exists(label_dir) and len(glob(os.path.join(label_dir, "*", "*.tif"))) > 0:
        return label_dir

    with open(os.path.join(data_dir, "CCAgT_COCO_OD.json")) as f:
        coco = json.load(f)

    per_image = defaultdict(list)
    for annotation in coco["annotations"]:
        # The synthetic class holds a disc around a click, so it is not a real outline.
        if annotation["category_id"] == SYNTHETIC_CLASS:
            continue
        per_image[annotation["image_id"]].append(annotation)

    for image in tqdm(coco["images"], desc="Preprocess the CCAgT annotations"):
        name = image["file_name"]
        slide = name.split("_")[0]
        output_dir = os.path.join(label_dir, slide)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.tif")
        if os.path.exists(output_path):
            continue

        shape = (image["height"], image["width"])
        labels = _rasterize(per_image.get(image["id"], []), shape)
        imageio.imwrite(output_path, labels, compression="zlib")

    return label_dir


def _create_nuclei_labels(data_dir: str) -> str:
    """Merge the nucleus classes and the NOR classes of the released masks into whole nuclei."""
    from tqdm import tqdm

    label_dir = os.path.join(data_dir, "nuclei_labels")
    mask_paths = natsorted(glob(os.path.join(data_dir, "masks", "*", "*.png")))
    if os.path.exists(label_dir) and len(glob(os.path.join(label_dir, "*", "*.tif"))) == len(mask_paths):
        return label_dir

    for mask_path in tqdm(mask_paths, desc="Merge the CCAgT nucleus classes"):
        slide = os.path.basename(os.path.dirname(mask_path))
        output_dir = os.path.join(label_dir, slide)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(mask_path).replace(".png", ".tif"))
        if os.path.exists(output_path):
            continue

        mask = imageio.imread(mask_path)
        nuclei = np.isin(mask, NUCLEUS_CLASSES).astype("uint8")
        imageio.imwrite(output_path, nuclei, compression="zlib")

    return label_dir


def get_ccagt_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CCAgT dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, ARCHIVE_ROOT)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "ccagt.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    _extract_archive(zip_path, path)

    return data_dir


def get_ccagt_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = "train",
    slides: Optional[Sequence[str]] = None,
    label_choice: Literal["semantic", "nuclei", "instances"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CCAgT data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split, which groups the slides. Ignored when you pass `slides`.
        slides: The slides to use, for example ('A', 'D'). Overrides `split`.
        label_choice: The target. Either 'instances' for the single objects, 'nuclei' for the whole
            nuclei, or 'semantic' for the seven classes as the authors released them.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in LABEL_CHOICES:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose from {list(LABEL_CHOICES)}.")

    if slides is None:
        if split not in SPLITS:
            raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}, or pass slides.")
        slides = SPLITS[split]
    else:
        known = {slide for group in SPLITS.values() for slide in group}
        for slide in slides:
            if slide not in known:
                raise ValueError(f"'{slide}' is not a valid slide. Choose from {sorted(known)}.")

    data_dir = get_ccagt_data(path, download)

    if label_choice == "semantic":
        label_dir, extension = os.path.join(data_dir, "masks"), ".png"
    elif label_choice == "nuclei":
        label_dir, extension = _create_nuclei_labels(data_dir), ".tif"
    else:
        label_dir, extension = _create_instance_labels(data_dir), ".tif"

    image_paths, label_paths = [], []
    for slide in slides:
        for image_path in natsorted(glob(os.path.join(data_dir, "images", slide, "*.jpg"))):
            name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, slide, f"{name}{extension}")
            if not os.path.exists(label_path):
                continue
            image_paths.append(image_path)
            label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any CCAgT data in {data_dir}.")

    return image_paths, label_paths


def get_ccagt_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = "train",
    slides: Optional[Sequence[str]] = None,
    label_choice: Literal["semantic", "nuclei", "instances"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CCAgT dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split, which groups the slides. Ignored when you pass `slides`.
        slides: The slides to use, for example ('A', 'D'). Overrides `split`.
        label_choice: The target. Either 'instances' for the single objects, 'nuclei' for the whole
            nuclei, or 'semantic' for the seven classes as the authors released them.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The CCAgT patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_ccagt_paths(path, split, slides, label_choice, download)

    if label_choice == "instances":
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
        )
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


def get_ccagt_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = "train",
    slides: Optional[Sequence[str]] = None,
    label_choice: Literal["semantic", "nuclei", "instances"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CCAgT dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split, which groups the slides. Ignored when you pass `slides`.
        slides: The slides to use, for example ('A', 'D'). Overrides `split`.
        label_choice: The target. Either 'instances' for the single objects, 'nuclei' for the whole
            nuclei, or 'semantic' for the seven classes as the authors released them.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ccagt_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        slides=slides,
        label_choice=label_choice,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
