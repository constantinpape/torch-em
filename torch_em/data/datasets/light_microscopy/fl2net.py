"""The FL2-Net dataset contains 3D bright-field microscopy timelapses of mouse embryos
with nuclear instance segmentation annotations.

The dataset holds 84 embryos with 506 timepoints each, so 42504 volumes in total. The authors split
the data by embryo, and no embryo appears in more than one split. The images are label-free, which
makes the nuclei much harder to see than in a fluorescence image.

NOTE: The whole dataset needs about 68 GB for the annotations and a similar amount for the images,
because one volume takes about 1.6 MB and the archives compress well. The loader therefore extracts
only the timepoints that you request. Use `stride` to set how many timepoints it skips, or pass
`timepoints` to select them.

NOTE: You must download both archives manually from the links in
https://github.com/funalab/FL2-Net and place them in `path`. Google Drive limits how often it
serves them, so an automatic download fails once too many users have fetched the file.

The dataset is located at https://github.com/funalab/FL2-Net.
This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2025.111179.
Please cite it if you use this dataset in your research.
"""

import os
import tarfile
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URLS = {
    "images": "https://drive.usercontent.google.com/download?id=1OAMmFM76TputGnU6nell6LU81N0hDmRc&confirm=xxx",
    "labels": "https://drive.usercontent.google.com/download?id=1hdSnCthLtyKMCahFLHUz36Awtj2-OC6T&confirm=xxx",
}

CHECKSUMS = {
    "images": None,  # The archive was not reachable yet, so the checksum is still unknown.
    "labels": "9c12b70978f3995662f377dac8fc173abdc0a350ee3c38e6367096c87c2d2200",
}

ARCHIVE_NAMES = {"images": "raw.tar.gz", "labels": "gt.tar.gz"}

N_TIMEPOINTS = 506

# The authors split the data by embryo in datasets/split_list_411 of the FL2-Net repository.
SPLITS = {
    "train": (
        "F001/Embryo01", "F001/Embryo02", "F001/Embryo03", "F001/Embryo04",
        "F001/Embryo06", "F001/Embryo08", "F001/Embryo09", "F001/Embryo10",
        "F002/Embryo01", "F002/Embryo03", "F002/Embryo04", "F002/Embryo06",
        "F002/Embryo07", "F002/Embryo08", "F002/Embryo10", "F002/Embryo11",
        "F003/Embryo01", "F003/Embryo02", "F003/Embryo04", "F003/Embryo05",
        "F003/Embryo08", "F003/Embryo09", "F003/Embryo10", "F003/Embryo12",
        "F004/Embryo01", "F004/Embryo02", "F004/Embryo05", "F004/Embryo06",
        "F004/Embryo08", "F004/Embryo09", "F004/Embryo10", "F004/Embryo12",
        "F005/Embryo02", "F005/Embryo04", "F005/Embryo05", "F005/Embryo06",
        "F005/Embryo08", "F005/Embryo09", "F005/Embryo10", "F005/Embryo11",
        "F006/Embryo01", "F006/Embryo04", "F006/Embryo05", "F006/Embryo08",
        "F006/Embryo09", "F006/Embryo10", "F006/Embryo11", "F006/Embryo12",
        "F007/Embryo03", "F007/Embryo04", "F007/Embryo05", "F007/Embryo06",
        "F007/Embryo07", "F007/Embryo09", "F007/Embryo10", "F007/Embryo11",
    ),
    "val": (
        "F001/Embryo05", "F001/Embryo12", "F002/Embryo05", "F002/Embryo09",
        "F003/Embryo03", "F003/Embryo06", "F004/Embryo04", "F004/Embryo11",
        "F005/Embryo01", "F005/Embryo03", "F006/Embryo06", "F006/Embryo07",
        "F007/Embryo01", "F007/Embryo12",
    ),
    "test": (
        "F001/Embryo07", "F001/Embryo11", "F002/Embryo02", "F002/Embryo12",
        "F003/Embryo07", "F003/Embryo11", "F004/Embryo03", "F004/Embryo07",
        "F005/Embryo07", "F005/Embryo12", "F006/Embryo02", "F006/Embryo03",
        "F007/Embryo02", "F007/Embryo08",
    ),
}


def _get_archive_root(archive_path: str) -> str:
    """Read the name of the top level folder of an archive.

    The annotation archive stores its files under 'qcanet'. The name of the image archive is not
    documented, so the loader reads it instead of assuming it.
    """
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            root = member.name.split("/")[0]
            if root:
                return root
    raise RuntimeError(f"The archive {archive_path} is empty.")


def _extract_members(archive_path: str, relative_names: Sequence[str], destination: str) -> None:
    """Extract the given files from an archive in one pass, and drop the top level folder."""
    missing = {name for name in relative_names if not os.path.exists(os.path.join(destination, name))}
    if not missing:
        return

    root = _get_archive_root(archive_path)
    wanted = {f"{root}/{name}": name for name in missing}

    found = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            target = wanted.get(member.name)
            if target is None:
                continue
            output_path = os.path.join(destination, target)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            source = archive.extractfile(member)
            with open(output_path, "wb") as f:
                f.write(source.read())
            found.add(target)
            if len(found) == len(wanted):
                break

    if found != missing:
        raise RuntimeError(
            f"Could not find {len(missing - found)} of {len(missing)} files in {archive_path}. "
            f"The first missing file is '{sorted(missing - found)[0]}'."
        )


def _get_timepoints(timepoints: Optional[Sequence[int]], stride: int) -> List[int]:
    """Resolve the requested timepoints. The timepoint index starts at one."""
    if timepoints is not None:
        selected = [int(t) for t in timepoints]
        for timepoint in selected:
            if not 1 <= timepoint <= N_TIMEPOINTS:
                raise ValueError(f"The timepoint {timepoint} is outside the range 1 to {N_TIMEPOINTS}.")
        return selected

    if stride < 1:
        raise ValueError(f"The stride must be at least one, got {stride}.")
    return list(range(1, N_TIMEPOINTS + 1, stride))


def get_fl2net_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FL2-Net dataset.

    NOTE: Google Drive limits how often it serves the archives. Download them manually from the
    links in https://github.com/funalab/FL2-Net and place them in `path` as 'raw.tar.gz' and
    'gt.tar.gz' when the automatic download fails.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder that holds the archives.
    """
    os.makedirs(path, exist_ok=True)

    for key, archive_name in ARCHIVE_NAMES.items():
        archive_path = os.path.join(path, archive_name)
        if os.path.exists(archive_path):
            continue
        util.download_source(archive_path, URLS[key], download, CHECKSUMS[key])

    return path


def get_fl2net_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    embryos: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 25,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FL2-Net data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val' or 'test'.
        embryos: The embryos to use, for example 'F001/Embryo01'. Defaults to all of the split.
        timepoints: The timepoints to use, counted from one. Overrides `stride`.
        stride: The number of timepoints to skip between two extractions.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    if embryos is None:
        embryos = SPLITS[split]
    else:
        for embryo in embryos:
            if embryo not in SPLITS[split]:
                raise ValueError(f"The embryo '{embryo}' is not part of the '{split}' split.")

    get_fl2net_data(path, download)
    selected = _get_timepoints(timepoints, stride)
    relative_names = [f"{embryo}/{timepoint:03d}.tif" for embryo in embryos for timepoint in selected]

    image_dir = os.path.join(path, "images")
    label_dir = os.path.join(path, "labels")
    _extract_members(os.path.join(path, ARCHIVE_NAMES["images"]), relative_names, image_dir)
    _extract_members(os.path.join(path, ARCHIVE_NAMES["labels"]), relative_names, label_dir)

    image_paths = [os.path.join(image_dir, name) for name in relative_names]
    label_paths = [os.path.join(label_dir, name) for name in relative_names]
    return natsorted(image_paths), natsorted(label_paths)


def get_fl2net_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    embryos: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 25,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FL2-Net dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        embryos: The embryos to use, for example 'F001/Embryo01'. Defaults to all of the split.
        timepoints: The timepoints to use, counted from one. Overrides `stride`.
        stride: The number of timepoints to skip between two extractions.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The FL2-Net patch shape must be three-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_fl2net_paths(path, split, embryos, timepoints, stride, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs,
    )


def get_fl2net_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    embryos: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 25,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the FL2-Net dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        embryos: The embryos to use, for example 'F001/Embryo01'. Defaults to all of the split.
        timepoints: The timepoints to use, counted from one. Overrides `stride`.
        stride: The number of timepoints to skip between two extractions.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fl2net_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        embryos=embryos,
        timepoints=timepoints,
        stride=stride,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
