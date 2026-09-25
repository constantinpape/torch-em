"""The ALFI dataset contains annotations for cell segmentation in label-free
differential interference contrast timelapses of cultured human cells.

ALFI stands for Annotations for Label-Free Images. The dataset holds 796 annotated frames of eight
timelapse sequences of the cell lines U2OS, HeLa and hTERT RPE-1. Every frame comes with a mask that
separates interphase cells from mitotic cells.

The data suits nucleus segmentation in label-free bright-field microscopy, because the annotators
outlined the nucleus of every interphase cell. It also suits cell tracking, because the archive
holds a track id and a lineage parent for every annotated cell over time.

NOTE: The two classes do not mark the same structure. The interphase label covers the nucleus, while
the mitotic label covers the whole rounded cell. The cytoplasm of an interphase cell is background.
So the target is a nucleus for an interphase cell, and a whole cell for a mitotic one.

NOTE: This loader returns segmentation targets only. The tracking annotations sit next to the masks
in '<sequence>/<sequence>_DTLTruth.csv', which stores a frame index, a track id, a class, a bounding
box and the id of the parent cell. A second table, '<sequence>_PhenoTruth.csv', marks the phenotypes
early mitosis, late mitosis, cell death and multipolar division. Read these files directly if you
want to track cells or follow a lineage.

NOTE: The archive holds 29 sequences, but only the eight MI sequences carry masks. The other
sequences provide bounding boxes, which this loader does not use. The loader reads the members of
the eight MI sequences out of the archive, so it transfers about 1.2 GB instead of the full 8.4 GB.

NOTE: The publication defines no train, validation and test split. This loader splits by sequence,
so that a split never shares a sequence with another one. The frames are seven minutes apart and
look almost the same, so a split over single frames would leak.

The dataset is located at https://doi.org/10.6084/m9.figshare.23798451 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-023-02540-1.
Please cite it if you use this dataset in your research.
"""

import os
import zipfile
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/41740227"
CHECKSUM = "fe3326323c10b1748302e962eae26150"

# The folder name holds an ampersand, so quote it whenever it goes into a shell command.
ARCHIVE_ROOT = "Data&Annotations"

SEQUENCES = ("MI01", "MI02", "MI03", "MI04", "MI05", "MI06", "MI07", "MI08")

# The publication has no official split, so the sequences are grouped by cell line.
SPLITS = {
    "train": ("MI01", "MI02", "MI03", "MI04", "MI05"),  # U2OS
    "val": ("MI06",),  # HeLa
    "test": ("MI07", "MI08"),  # hTERT RPE-1
}

# The masks store the background as 0, an interphase cell as 128 and a mitotic cell as 255.
SEMANTIC_IDS = {0: 0, 128: 1, 255: 2}

# A few masks hold tiny blobs at the image border that the annotation tables do not list.
MIN_INSTANCE_SIZE = 200


def _extract_sequences(zip_path: str, path: str) -> None:
    """Extract the images and the masks of the annotated sequences."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name for name in archive.namelist()
            if name.startswith(f"{ARCHIVE_ROOT}/MI")
            and not name.startswith("__MACOSX")
            and not name.endswith(".DS_Store")
        ]
        if not members:
            raise RuntimeError(f"The archive {zip_path} does not hold the annotated MI sequences.")
        archive.extractall(path, members=members)


def _to_semantic(mask: np.ndarray) -> np.ndarray:
    """Map the mask values onto consecutive class ids."""
    semantic = np.zeros(mask.shape, dtype="uint8")
    for value, class_id in SEMANTIC_IDS.items():
        semantic[mask == value] = class_id
    return semantic


def _to_instances(mask: np.ndarray) -> np.ndarray:
    """Split each class of a mask into connected components and give every component one id."""
    from scipy.ndimage import label as connected_components

    instances = np.zeros(mask.shape, dtype="uint16")
    offset = 0
    for value in (128, 255):
        components, n_components = connected_components(mask == value)
        for component_id in range(1, n_components + 1):
            component = components == component_id
            if component.sum() < MIN_INSTANCE_SIZE:
                continue
            offset += 1
            instances[component] = offset
    return instances


def _create_labels(data_dir: str, sequence: str, label_choice: str) -> str:
    """Convert the masks of one sequence into the requested target."""
    from tqdm import tqdm

    mask_dir = os.path.join(data_dir, sequence, "Masks")
    label_dir = os.path.join(data_dir, sequence, f"{label_choice}_labels")
    os.makedirs(label_dir, exist_ok=True)

    mask_paths = natsorted(glob(os.path.join(mask_dir, "*.png")))
    convert = _to_semantic if label_choice == "semantic" else _to_instances

    for mask_path in tqdm(mask_paths, desc=f"Preprocess '{sequence}' for the {label_choice} target"):
        output_path = os.path.join(label_dir, os.path.basename(mask_path).replace(".png", ".tif"))
        if os.path.exists(output_path):
            continue
        imageio.imwrite(output_path, convert(imageio.imread(mask_path)), compression="zlib")

    return label_dir


def get_alfi_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ALFI dataset.

    The loader reads only the eight annotated sequences out of the archive, so it stores about
    1.2 GB instead of the full 8.4 GB.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder that holds the sequences.
    """
    data_dir = os.path.join(path, ARCHIVE_ROOT)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "ALFIdatasetFinal.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    _extract_sequences(zip_path, path)

    return data_dir


def get_alfi_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = "train",
    sequences: Optional[Sequence[str]] = None,
    label_choice: Literal["semantic", "instances"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ALFI data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split, which groups the sequences by cell line. Either 'train' for U2OS,
            'val' for HeLa or 'test' for hTERT RPE-1. Ignored when you pass `sequences`.
        sequences: The sequences to use, for example ('MI01', 'MI06'). Overrides `split`.
        label_choice: The target. Either 'instances' for the single cells, or 'semantic' for the
            classes, where one is an interphase cell and two is a mitotic cell.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in ("semantic", "instances"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'semantic' or 'instances'.")

    if sequences is None:
        if split not in SPLITS:
            raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}, or pass sequences.")
        sequences = SPLITS[split]
    else:
        for sequence in sequences:
            if sequence not in SEQUENCES:
                raise ValueError(f"'{sequence}' is not an annotated sequence. Choose from {list(SEQUENCES)}.")

    data_dir = get_alfi_data(path, download)

    image_paths, label_paths = [], []
    for sequence in sequences:
        label_dir = _create_labels(data_dir, sequence, label_choice)
        for label_path in natsorted(glob(os.path.join(label_dir, "*.tif"))):
            # An image is named I_MI01_0001.png and its mask M_MI01_0001.png.
            image_name = os.path.basename(label_path).replace(".tif", ".png").replace("M_", "I_", 1)
            image_path = os.path.join(data_dir, sequence, "Images", image_name)
            if not os.path.exists(image_path):
                continue
            image_paths.append(image_path)
            label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any ALFI data in {data_dir}.")

    return image_paths, label_paths


def get_alfi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = "train",
    sequences: Optional[Sequence[str]] = None,
    label_choice: Literal["semantic", "instances"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the ALFI dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split, which groups the sequences by cell line. Ignored with `sequences`.
        sequences: The sequences to use, for example ('MI01', 'MI06'). Overrides `split`.
        label_choice: The target. Either 'instances' for the single cells, or 'semantic' for the
            classes, where one is an interphase cell and two is a mitotic cell.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The ALFI patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_alfi_paths(path, split, sequences, label_choice, download)

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


def get_alfi_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = "train",
    sequences: Optional[Sequence[str]] = None,
    label_choice: Literal["semantic", "instances"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the ALFI dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split, which groups the sequences by cell line. Ignored with `sequences`.
        sequences: The sequences to use, for example ('MI01', 'MI06'). Overrides `split`.
        label_choice: The target. Either 'instances' for the single cells, or 'semantic' for the
            classes, where one is an interphase cell and two is a mitotic cell.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_alfi_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        sequences=sequences,
        label_choice=label_choice,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
