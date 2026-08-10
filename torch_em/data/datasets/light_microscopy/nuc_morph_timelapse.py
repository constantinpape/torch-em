"""The NucMorph timelapse dataset contains 3D fluorescence microscopy timelapses of hiPSC colonies
with nuclear instance segmentation annotations.

The dataset holds 14 colonies over six conditions. Each colony provides a raw timelapse with a
Lamin B1 EGFP channel and a brightfield channel, and a matching nuclear instance segmentation. The
annotations come from a Vision Transformer based segmentation model, and they are much cleaner than
the ones of the related `nuc_morph` dataset.

NOTE: One timepoint holds about 190 MB per array, so the loader downloads a subset of the
timepoints. Use `stride` to set how many timepoints it skips, or pass `timepoints` to select them.

NOTE: The raw level 0 and the segmentation level 1 share one grid. The segmentation level 0 is an
upsampled version with a different shape, so it does not match the raw data.

NOTE: The index of the EGFP channel differs per colony, and the segmentation of most colonies stops
before the raw data ends. This module stores both facts in `COLONIES`, and it pairs the timepoints
without an offset, which was verified against the image data.

The dataset is located at https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/ under
the Allen Institute for Cell Science Terms of Use.
This dataset is from the publication https://doi.org/10.1016/j.cels.2025.101265.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


S3_BASE = (
    "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/"
    "hipsc_fov_nuclei_timelapse_data_used_for_analysis"
)

# colony -> (condition, index of the EGFP channel in the raw data)
COLONIES = {
    "20200323_05_large": ("baseline_colonies", 0),
    "20200323_06_medium": ("baseline_colonies", 0),
    "20200323_09_small": ("baseline_colonies", 0),
    "20220411_03_control": ("dna_replication_inhibitor", 0),
    "20220411_05_aphidicolin": ("dna_replication_inhibitor", 0),
    "20230424_01_control": ("dna_replication_inhibitor", 1),
    "20230424_03_control": ("dna_replication_inhibitor", 1),
    "20230424_05_aphidicolin": ("dna_replication_inhibitor", 1),
    "20230720_01_control": ("feeding_control", 1),
    "20230720_04_pre-starved": ("feeding_control", 1),
    "20230720_07_re-fed": ("feeding_control", 1),
    "20220901_01": ("fixed_control", 1),
    "20230417_01_control": ("nuclear_import_inhibitor", 1),
    "20230417_07_importazole": ("nuclear_import_inhibitor", 1),
}

CONDITIONS = tuple(dict.fromkeys(condition for condition, _ in COLONIES.values()))

CHANNELS = ("egfp", "brightfield", "both")


def _open_array(url: str):
    """Open a remote zarr array over http."""
    import zarr

    try:
        return zarr.open(url, mode="r")
    except Exception:
        from zarr.storage import FsspecStore
        return zarr.open(store=FsspecStore.from_url(url), mode="r")


def _get_colonies(condition: Optional[str], colony: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Resolve the requested colonies."""
    if colony is not None:
        colonies = [colony] if isinstance(colony, str) else list(colony)
        for name in colonies:
            if name not in COLONIES:
                raise ValueError(f"'{name}' is not a valid colony. Choose from {list(COLONIES)}.")
        return colonies

    if condition is None:
        return list(COLONIES)

    if condition not in CONDITIONS:
        raise ValueError(f"'{condition}' is not a valid condition. Choose from {list(CONDITIONS)}.")
    return [name for name, (this_condition, _) in COLONIES.items() if this_condition == condition]


def _download_colony(
    path: str, colony: str, timepoints: Optional[Sequence[int]], stride: int, channel: str, download: bool,
) -> str:
    """Download the selected timepoints of one colony and store them as h5 files."""
    import h5py
    from tqdm import tqdm

    condition, egfp_channel = COLONIES[colony]
    colony_dir = os.path.join(path, colony)
    os.makedirs(colony_dir, exist_ok=True)

    base = f"{S3_BASE}/{condition}_fov_timelapse_dataset/{colony}"
    raw_array = _open_array(f"{base}/raw.ome.zarr/0")
    seg_array = _open_array(f"{base}/seg.ome.zarr/1")

    if raw_array.shape[-3:] != seg_array.shape[-3:]:
        raise RuntimeError(
            f"The raw and the segmentation grid of '{colony}' differ, "
            f"{raw_array.shape[-3:]} against {seg_array.shape[-3:]}."
        )

    # The segmentation stops before the raw data ends, so it limits the valid timepoints.
    n_timepoints = seg_array.shape[0]
    if timepoints is None:
        timepoints = range(0, n_timepoints, stride)
    selected = [int(t) for t in timepoints]
    for timepoint in selected:
        if not 0 <= timepoint < n_timepoints:
            raise ValueError(f"The timepoint {timepoint} is outside the segmented range of '{colony}', "
                             f"which holds {n_timepoints} timepoints.")

    if channel == "egfp":
        channel_ids = [egfp_channel]
    elif channel == "brightfield":
        channel_ids = [1 - egfp_channel]
    else:
        channel_ids = [egfp_channel, 1 - egfp_channel]

    for timepoint in tqdm(selected, desc=f"Download '{colony}'"):
        output_path = os.path.join(colony_dir, f"t{timepoint:04d}.h5")
        if os.path.exists(output_path):
            continue

        if not download:
            raise RuntimeError(f"Cannot find the data at {output_path}, but download was set to False.")

        raw = np.stack([np.asarray(raw_array[timepoint, c]) for c in channel_ids])
        labels = np.asarray(seg_array[timepoint, 0])
        if raw.shape[0] == 1:
            raw = raw[0]

        with h5py.File(output_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

    return colony_dir


def get_nuc_morph_timelapse_data(
    path: Union[os.PathLike, str],
    condition: Optional[str] = "baseline_colonies",
    colony: Optional[Union[str, Sequence[str]]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 50,
    channel: Literal["egfp", "brightfield", "both"] = "egfp",
    download: bool = False,
) -> List[str]:
    """Download the NucMorph timelapse dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        condition: The experimental condition. Ignored when you pass `colony`.
        colony: The colony or colonies to use. Overrides `condition`.
        timepoints: The timepoints to download. Overrides `stride`.
        stride: The number of timepoints to skip between two downloads.
        channel: The raw channel. Either 'egfp', 'brightfield' or 'both'.
        download: Whether to download the data if it is not present.

    Returns:
        List of the folders that hold the data of the requested colonies.
    """
    if channel not in CHANNELS:
        raise ValueError(f"'{channel}' is not a valid channel. Choose from {list(CHANNELS)}.")
    if stride < 1:
        raise ValueError(f"The stride must be at least one, got {stride}.")

    colonies = _get_colonies(condition, colony)
    os.makedirs(path, exist_ok=True)
    return [_download_colony(path, name, timepoints, stride, channel, download) for name in colonies]


def get_nuc_morph_timelapse_paths(
    path: Union[os.PathLike, str],
    condition: Optional[str] = "baseline_colonies",
    colony: Optional[Union[str, Sequence[str]]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 50,
    channel: Literal["egfp", "brightfield", "both"] = "egfp",
    download: bool = False,
) -> List[str]:
    """Get paths to the NucMorph timelapse data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        condition: The experimental condition. Ignored when you pass `colony`.
        colony: The colony or colonies to use. Overrides `condition`.
        timepoints: The timepoints to download. Overrides `stride`.
        stride: The number of timepoints to skip between two downloads.
        channel: The raw channel. Either 'egfp', 'brightfield' or 'both'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    colony_dirs = get_nuc_morph_timelapse_data(path, condition, colony, timepoints, stride, channel, download)

    volume_paths = []
    for colony_dir in colony_dirs:
        volume_paths.extend(natsorted(glob(os.path.join(colony_dir, "*.h5"))))

    if not volume_paths:
        raise RuntimeError(f"Could not find any NucMorph timelapse data in {path}.")

    return volume_paths


def get_nuc_morph_timelapse_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    condition: Optional[str] = "baseline_colonies",
    colony: Optional[Union[str, Sequence[str]]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 50,
    channel: Literal["egfp", "brightfield", "both"] = "egfp",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the NucMorph timelapse dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        condition: The experimental condition. Ignored when you pass `colony`.
        colony: The colony or colonies to use. Overrides `condition`.
        timepoints: The timepoints to download. Overrides `stride`.
        stride: The number of timepoints to skip between two downloads.
        channel: The raw channel. Either 'egfp', 'brightfield' or 'both'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The NucMorph timelapse patch shape must be three-dimensional, got {patch_shape}.")

    volume_paths = get_nuc_morph_timelapse_paths(
        path, condition, colony, timepoints, stride, channel, download
    )

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        with_channels=channel == "both",
        **kwargs,
    )


def get_nuc_morph_timelapse_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    condition: Optional[str] = "baseline_colonies",
    colony: Optional[Union[str, Sequence[str]]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 50,
    channel: Literal["egfp", "brightfield", "both"] = "egfp",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the NucMorph timelapse dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        condition: The experimental condition. Ignored when you pass `colony`.
        colony: The colony or colonies to use. Overrides `condition`.
        timepoints: The timepoints to download. Overrides `stride`.
        stride: The number of timepoints to skip between two downloads.
        channel: The raw channel. Either 'egfp', 'brightfield' or 'both'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nuc_morph_timelapse_dataset(
        path=path,
        patch_shape=patch_shape,
        condition=condition,
        colony=colony,
        timepoints=timepoints,
        stride=stride,
        channel=channel,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
