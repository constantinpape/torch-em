"""This dataset contains spinning-disk confocal time-lapse images of growing *Neisseria meningitidis*
bacterial colonies, with per-cell instance segmentation and lineage tracking (including division events).
The segmentation and tracks are reconstructed here from a TrackMate ilastik-detector tracking session
shipped with the raw data, by rasterizing the per-spot contours and assigning a new instance id to each
daughter cell after a division.

NOTE: The instance contours come from an ilastik pixel classifier (used as the TrackMate detector), not
from manual annotation. They are comparatively rough and not densely accurate at the pixel level.

The dataset is hosted on Zenodo at https://doi.org/10.5281/zenodo.5419619.

Please cite it if you use this dataset for your research.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, Tuple, Union

import numpy as np
import tifffile
from skimage.draw import polygon

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "raw": "https://zenodo.org/records/5419619/files/NeisseriaMeningitidisGrowth.tif",
    "tracks": "https://zenodo.org/records/5419619/files/NeisseriaMeningitidisGrowth.xml",
}
CHECKSUMS = {
    "raw": "491212b547654b0637001ce61e01cf289b63704ced7173b9dc44f2189fdc2ba9",
    "tracks": "a860f67520f7f5435be2d2d0a1759906e6f10c7dddffbad8ca7ce3ef24dd1a3c",
}


def get_neisseria_meningitidis_data(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Download the Neisseria meningitidis bacterial growth dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the raw image stack.
        Filepath to the TrackMate tracking / lineage annotations.
    """
    os.makedirs(path, exist_ok=True)

    raw_path = os.path.join(path, "NeisseriaMeningitidisGrowth.tif")
    util.download_source(path=raw_path, url=URLS["raw"], download=download, checksum=CHECKSUMS["raw"])

    tracks_path = os.path.join(path, "NeisseriaMeningitidisGrowth.xml")
    util.download_source(path=tracks_path, url=URLS["tracks"], download=download, checksum=CHECKSUMS["tracks"])

    return raw_path, tracks_path


def _build_segment_ids(model) -> Dict[str, int]:
    # Assign a new instance id at each lineage root and to each daughter after a division, so that a
    # continuous single-cell trajectory keeps one id until it splits (CTC-style tracking convention).
    all_tracks = model.find("AllTracks")
    filtered_ids = {t.attrib["TRACK_ID"] for t in model.find("FilteredTracks").findall("TrackID")}

    spot_frame = {}
    for frame_elem in model.find("AllSpots").findall("SpotsInFrame"):
        frame = int(frame_elem.attrib["frame"])
        for spot in frame_elem.findall("Spot"):
            spot_frame[spot.attrib["ID"]] = frame

    out_edges, in_edges, kept_spots = defaultdict(list), defaultdict(list), set()
    for track in all_tracks.findall("Track"):
        if track.attrib["TRACK_ID"] not in filtered_ids:
            continue
        for edge in track.findall("Edge"):
            source, target = edge.attrib["SPOT_SOURCE_ID"], edge.attrib["SPOT_TARGET_ID"]
            if spot_frame[source] > spot_frame[target]:
                source, target = target, source
            out_edges[source].append(target)
            in_edges[target].append(source)
            kept_spots.update((source, target))

    segment_id, next_id = {}, 1
    roots = sorted((s for s in kept_spots if not in_edges[s]), key=lambda s: spot_frame[s])
    queue = deque()
    for root in roots:
        segment_id[root] = next_id
        next_id += 1
        queue.append(root)

    while queue:
        spot = queue.popleft()
        children = out_edges.get(spot, [])
        if len(children) == 1:
            segment_id[children[0]] = segment_id[spot]
            queue.append(children[0])
        else:
            for child in children:
                segment_id[child] = next_id
                next_id += 1
                queue.append(child)

    return segment_id


def _rasterize_labels(tracks_path: str, shape: Tuple[int, int, int]) -> np.ndarray:
    root = ET.parse(tracks_path).getroot()
    model = root.find("Model")
    pixel_size = float(root.find("Settings").find("ImageData").attrib["pixelwidth"])
    segment_id = _build_segment_ids(model)

    _, height, width = shape
    labels = np.zeros(shape, dtype=np.uint16)
    for frame_elem in model.find("AllSpots").findall("SpotsInFrame"):
        frame = int(frame_elem.attrib["frame"])
        for spot in frame_elem.findall("Spot"):
            label = segment_id.get(spot.attrib["ID"])
            if label is None:  # Spots outside the filtered tracks are discarded, not segmented.
                continue
            center = np.array([float(spot.attrib["POSITION_X"]), float(spot.attrib["POSITION_Y"])])
            points = np.array((spot.text or "").split(), dtype=float).reshape(-1, 2)
            coords = (points + center) / pixel_size
            rows, cols = polygon(coords[:, 1], coords[:, 0], shape=(height, width))
            labels[frame, rows, cols] = label

    return labels


def get_neisseria_meningitidis_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Get paths for the Neisseria meningitidis bacterial growth dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the raw image stack.
        Filepath to the instance segmentation and tracking labels.
    """
    raw_path, tracks_path = get_neisseria_meningitidis_data(path, download)

    label_path = os.path.join(path, "NeisseriaMeningitidisGrowth_labels.tif")
    if not os.path.exists(label_path):
        raw_shape = tifffile.TiffFile(raw_path).series[0].shape
        tifffile.imwrite(label_path, _rasterize_labels(tracks_path, raw_shape))

    return raw_path, label_path


def get_neisseria_meningitidis_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the Neisseria meningitidis dataset for bacterial cell segmentation and tracking.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_path, label_path = get_neisseria_meningitidis_paths(path, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_path,
        raw_key=None,
        label_paths=label_path,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_neisseria_meningitidis_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Neisseria meningitidis dataloader for bacterial cell segmentation and tracking.

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
    dataset = get_neisseria_meningitidis_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
