"""The SVIA dataset contains annotations for sperm segmentation in
bright-field microscopy videos of human semen.

SVIA stands for Sperm Videos and Images Analysis. The deposit calls the dataset MIaMIA-SVDS. This
loader uses Subset-B, which holds a mask per sperm for 451 frames of ten videos, so 25966 objects.

NOTE: The dataset holds two more label types, which this loader does not return.
- Subset-A holds a bounding box and a class for 125880 objects of 3622 frames. The classes are
  'S' for a sperm and 'Impurity' for everything else, at a ratio of about 27 to 1. Subset-C cuts one
  small image per box out of the frames, for a classification task. Read 'Subset-A/<frame>.xml' and
  'Subset-C/V<video>-F<frame>-<class><index>.png' if you need them.
- The masks of Subset-B carry a track id, because the file name of a mask ends with the id of the
  sperm and that id follows the same sperm over the frames of a video. This loader keeps the id as
  the label value, so the labels of one video track the sperm through time.

NOTE: Only 10 of the 101 videos carry masks. The other 91 videos hold boxes alone.

NOTE: The archive is a rar file, so the extraction needs the 'rarfile' package and the 'unrar'
program. The masks are stored as one full frame image per sperm, which is why the archive holds
25966 mask files for 451 frames.

NOTE: The publication defines no split. This loader splits by video, because the frames of one video
show the same sperm over time and a split over frames would leak.

The dataset is located at https://doi.org/10.6084/m9.figshare.15074253.v1.
This dataset is from the publication https://doi.org/10.1016/j.bbe.2021.12.010.
Please cite it if you use this dataset in your research. The authors welcome non-commercial research
work on this data.
"""

import os
import re
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/28986378"
CHECKSUM = "f6956e21eab440806f76a9805655b8a7e56f2555e8f6a03983d2eebab4b2ba71"

ARCHIVE_ROOT = "Data Set"

# The ten videos of Subset-B, and how many annotated frames each of them holds.
VIDEOS = {
    "S_0001": 45, "S_0003": 38, "S_0005": 36, "S_0006": 26, "S_0007": 10,
    "S_0008": 90, "S_0009": 48, "S_0010": 32, "S_0011": 39, "S_0012": 87,
}

# A video belongs to one split only, so the frames of a video never spread over two splits.
SPLITS = {
    "train": ("S_0001", "S_0005", "S_0008", "S_0010", "S_0011"),  # 242 frames
    "val": ("S_0003", "S_0007"),  # 48 frames
    "test": ("S_0006", "S_0009", "S_0012"),  # 161 frames
}

# The masks are not clean binary images, so the loader compares against the middle of the range.
MASK_THRESHOLD = 127


def _extract_archive(archive_path: str, path: str) -> None:
    """Extract the frames and the masks of Subset-B out of the rar archive."""
    try:
        import rarfile
    except ImportError:
        raise RuntimeError(
            "The 'rarfile' package is required to extract the SVIA archive. "
            "Install it with 'pip install rarfile', and install the 'unrar' program as well."
        )

    with rarfile.RarFile(archive_path) as archive:
        members = [
            name for name in archive.namelist()
            if name.startswith(f"{ARCHIVE_ROOT}/Subset-B/")
            or name.startswith(f"{ARCHIVE_ROOT}/Frames from original videos/")
        ]
        if not members:
            raise RuntimeError(f"The archive {archive_path} does not hold Subset-B.")
        archive.extractall(path, members=members)


def _get_track_id(mask_stem: str, frame_name: str) -> int:
    """Read the track id of a sperm out of the name of its mask.

    A mask of the frame 'S_0001_0026' is normally named 'S_0001_0026_0010'. Two files of the
    archive break that rule, 'S_0001_0026-0010' and 'S_0003_001311png', so the id comes from the
    part of the name that follows the name of the frame.
    """
    remainder = mask_stem[len(frame_name):] if mask_stem.startswith(frame_name) else mask_stem
    digits = re.findall(r"\d+", remainder)
    if not digits:
        raise RuntimeError(f"Could not read a track id from the mask '{mask_stem}'.")
    return int(digits[0])


def _create_instance_labels(data_dir: str, video: str) -> str:
    """Merge the per sperm masks of every frame into one instance label image."""
    from tqdm import tqdm

    label_dir = os.path.join(data_dir, "instance_labels", video)
    os.makedirs(label_dir, exist_ok=True)

    frame_dirs = natsorted(glob(os.path.join(data_dir, "Subset-B", video, "*")))
    for frame_dir in tqdm(frame_dirs, desc=f"Preprocess '{video}'"):
        name = os.path.basename(frame_dir)
        output_path = os.path.join(label_dir, f"{name}.tif")
        if os.path.exists(output_path):
            continue

        image_path = os.path.join(data_dir, "Frames from original videos", video, f"{name}.png")
        if not os.path.exists(image_path):
            continue

        shape = imageio.imread(image_path).shape[:2]
        labels = np.zeros(shape, dtype="uint16")
        for mask_path in natsorted(glob(os.path.join(frame_dir, "*.png"))):
            track_id = _get_track_id(Path(mask_path).stem, name)
            mask = imageio.imread(mask_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            labels[mask > MASK_THRESHOLD] = track_id

        imageio.imwrite(output_path, labels, compression="zlib")

    return label_dir


def get_svia_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SVIA dataset.

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
    archive_path = os.path.join(path, "svia.rar")
    util.download_source(archive_path, URL, download, CHECKSUM)
    _extract_archive(archive_path, path)

    return data_dir


def get_svia_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = "train",
    videos: Optional[Sequence[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SVIA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split, which groups the videos. Ignored when you pass `videos`.
        videos: The videos to use, for example ('S_0001',). Overrides `split`.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if videos is None:
        if split not in SPLITS:
            raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}, or pass videos.")
        videos = SPLITS[split]
    else:
        for video in videos:
            if video not in VIDEOS:
                raise ValueError(f"'{video}' has no masks. Choose from {list(VIDEOS)}.")

    data_dir = get_svia_data(path, download)

    image_paths, label_paths = [], []
    for video in videos:
        label_dir = _create_instance_labels(data_dir, video)
        for label_path in natsorted(glob(os.path.join(label_dir, "*.tif"))):
            name = Path(label_path).stem
            image_path = os.path.join(data_dir, "Frames from original videos", video, f"{name}.png")
            if not os.path.exists(image_path):
                continue
            image_paths.append(image_path)
            label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any SVIA data in {data_dir}.")

    return image_paths, label_paths


def get_svia_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = "train",
    videos: Optional[Sequence[str]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SVIA dataset for sperm segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split, which groups the videos. Ignored when you pass `videos`.
        videos: The videos to use, for example ('S_0001',). Overrides `split`.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The SVIA patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_svia_paths(path, split, videos, download)

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


def get_svia_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = "train",
    videos: Optional[Sequence[str]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the SVIA dataloader for sperm segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split, which groups the videos. Ignored when you pass `videos`.
        videos: The videos to use, for example ('S_0001',). Overrides `split`.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_svia_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        videos=videos,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
