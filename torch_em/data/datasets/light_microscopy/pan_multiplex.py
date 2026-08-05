"""Pan-Multiplex (Pan-M) contains annotations for cell segmentation in multiplexed images of tissue.

The dataset covers five subsets from three imaging platforms: MIBI-TOF, CODEX and Vectra.
Every field of view provides many protein marker channels, a nuclei channel and a cell instance mask.
This module builds a nuclei and a membrane composite from the marker channels, which is the common
input format for cell segmentation in multiplexed imaging.
NOTE: The cell instance masks come from the upstream studies and nobody curated them by hand.
NOTE: The subsets are large. See `SUBSET_PARTS` and download one subset at a time.

The dataset is located at https://huggingface.co/datasets/JLrumberger/Pan-Multiplex.
This dataset is from the publication https://doi.org/10.1038/s41592-025-02826-9.
Please cite it if you use this dataset for your research.
"""

import os
import gzip
import json
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import tifffile

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/JLrumberger/Pan-Multiplex/resolve/main"
SPLIT_URL = "https://raw.githubusercontent.com/angelolab/Nimbus/main/configs"

SUBSET_PARTS = {
    "mibi_decidua": ["mibi_decidua.zip"],
    "codex_colon": [f"codex_colon.zip.{i:03d}" for i in range(1, 5)],
    "vectra_colon": [f"vectra_colon.zip.{i:03d}" for i in range(1, 3)],
    "vectra_pancreas": [f"vectra_pancreas.zip.{i:03d}" for i in range(1, 3)],
    "mibi_breast": [f"mibi_breast.zip.{i:03d}" for i in range(1, 12)],
}

CHECKSUMS = {
    "mibi_decidua.zip": "723a4c6863ca49bc063e0f6eead3a7bbf8d4d8ecb3992882739beeb7ad51eaea",
    "codex_colon.zip.001": "cefdc3147eb9f395130b5052ad965880fd97876af32c1fa4e2167a9cbc91c77f",
    "codex_colon.zip.002": "16deede6bb675c61c9c16deb967c89b316ddee8a5f7a2b0af1373cfbea470c40",
    "codex_colon.zip.003": "bbf6d96d3627050b1e227bc448ec3ebb773f5768b62d2e4357141e9355ba9246",
    "codex_colon.zip.004": "3ba68777b5cab20111118c03081b5e4aba67ae4e50e5c58fdd6732ffd49aee2c",
    "vectra_colon.zip.001": "fb23579672991954a0d6be32befa15bb716e14152389ec88c566eefc3c6e3dd1",
    "vectra_colon.zip.002": "4365420a09c6417a361382e44f85d0b8baa51001ff68fc22c46dc57b558d1667",
    "vectra_pancreas.zip.001": "efac85a30c1e01628ac3b0e6bf85844b724e549676ec18a7601f9ce9e1f5b1ed",
    "vectra_pancreas.zip.002": "16e4a5180ecaf1f184c8e1d9b2362873ff10b282d7b39599f27d9e134b63e472",
    "mibi_breast.zip.001": "357c146e37489bca02f52aed68b1736ac434f106771dc1345443720a72a41256",
    "mibi_breast.zip.002": "82ceb444dd7b3501afce3ce2078b2218047ce6bbb005ff063826be2df349720a",
    "mibi_breast.zip.003": "aa3361bf33a66d2358ff01c96583d6d388aaad4da06800384e8493fd1e5a0a72",
    "mibi_breast.zip.004": "6b61d95a37e9895f5933f55f08e65ab516a1ce663efa67928222ffb4dd53feae",
    "mibi_breast.zip.005": "71c250197c99e4ba640ccdbc6231a0207731d8f961b6401caea0497f69eb8474",
    "mibi_breast.zip.006": "77083a417465eea42dca5d60c7f1ff8bedf1fdb33d03ca147150e1a89ef8c6b0",
    "mibi_breast.zip.007": "bbd95daaf51ee5bcdf9c2a9e1b2517513e0fcbe544a2d8e3e702d356c7f67f53",
    "mibi_breast.zip.008": "d0379baef5b30f47a1dcdd4a094898a81906987bc2b9d9cc87ac95ff70b3032e",
    "mibi_breast.zip.009": "9d24f342ac90bd41823aa68f537128ff79be9ee1b321a0faec5fc0c1848cdef6",
    "mibi_breast.zip.010": "862824597edef36782ad77937725118304a77ca3e776ed682cb95facdec1b12d",
    "mibi_breast.zip.011": "bfcae2f5c9396558b33a0629a77105bef8ef63700e17db38968ffa741f9b1692",
}

SPLIT_CONFIGS = {
    "mibi_decidua": "decidua_split.json",
    "codex_colon": "hickey_split.json",
    "vectra_colon": "msk_colon_split.json",
    "vectra_pancreas": "msk_pancreas_split.json",
    "mibi_breast": "tonic_split.json",
}

# The channels that make up the nuclei and membrane composites, taken from the data preparation
# scripts of https://github.com/angelolab/Nimbus. The panel of 'codex_colon' is not documented there,
# so its channels come from the marker names in the archive, where DRAQ5 is the nuclear stain.
NUCLEI_CHANNELS = {
    "mibi_decidua": ["H3"],
    "mibi_breast": ["H3K27me3", "H3K9ac"],
    "vectra_colon": ["DAPI"],
    "vectra_pancreas": ["DAPI"],
    "codex_colon": ["DRAQ5"],
}

MEMBRANE_CHANNELS = {
    "mibi_decidua": ["VIM", "HLAG", "CD3", "CD14", "CD56"],
    "mibi_breast": ["CD45", "ECAD", "CD14", "CD38", "CK17"],
    "vectra_colon": ["CD3", "CD8", "ICOS", "panCK+CK7+CAM5.2"],
    "vectra_pancreas": ["CD8", "CD40", "CD40-L", "panCK"],
    "codex_colon": ["CD45", "Podoplanin", "CD31", "Cytokeratin", "aSMA", "Vimentin"],
}

# The subsets do not share a folder layout, so the image folder is named per subset.
IMAGE_DIRS = {
    "mibi_decidua": "image_data",
    "mibi_breast": "image_data",
    "codex_colon": "raw_structured",
    "vectra_colon": "raw_structured",
    "vectra_pancreas": "raw_structured",
}


def _download_subset(path, subset, download):
    """Download the parts of one subset and join them into a single archive."""
    zip_path = os.path.join(path, f"{subset}.zip")
    if os.path.exists(zip_path):
        return zip_path

    part_paths = []
    for fname in SUBSET_PARTS[subset]:
        part_path = os.path.join(path, fname)
        util.download_source(
            path=part_path, url=f"{URL}/{fname}", download=download, checksum=CHECKSUMS[fname]
        )
        part_paths.append(part_path)

    if len(part_paths) == 1:
        return part_paths[0]

    # The parts are a plain byte split, so joining them yields the original archive.
    # Each part is removed right after it is joined to limit the peak disk usage.
    with open(zip_path, "wb") as dst:
        for part_path in part_paths:
            with open(part_path, "rb") as src:
                shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
            os.remove(part_path)

    return zip_path


def _get_split_assignment(path, subset):
    """Map every field of view to the split that the authors of the publication assigned to it."""
    config = SPLIT_CONFIGS[subset]
    config_path = os.path.join(path, config)
    util.download_source(path=config_path, url=f"{SPLIT_URL}/{config}", download=True, checksum=None)

    with open(config_path, "rb") as f:
        payload = f.read()
    if payload[:2] == b"\x1f\x8b":  # the server may answer with a gzip encoded body
        payload = gzip.decompress(payload)
    splits = json.loads(payload)

    assignment = {}
    for split, fovs in splits.items():
        split = "val" if split == "validation" else split
        for fov in fovs:
            assignment[fov] = split
    return assignment


def _get_marker_paths(image_dir):
    """Map every marker name in a field of view to its file.

    The folder names contain brackets and commas in some subsets, so this avoids glob patterns.
    """
    markers = {}
    for fname in os.listdir(image_dir):
        if not fname.endswith((".tif", ".tiff")):
            continue
        name = fname[:-len(".ome.tif")] if fname.endswith(".ome.tif") else os.path.splitext(fname)[0]
        markers[name] = os.path.join(image_dir, fname)
    return markers


def _get_composite(markers, channels):
    """Sum the given marker channels after normalizing each of them by its upper quantile."""
    stack = []
    for name in channels:
        if name not in markers:
            continue
        image = np.squeeze(tifffile.imread(markers[name])).astype("float32")
        upper = np.quantile(image, 0.999)
        stack.append(image / upper if upper > 0 else image)

    if not stack:
        return None
    return np.clip(np.sum(stack, axis=0), 0, 1)


def _find_instance_path(input_dir, subset, fov):
    """Resolve the cell mask of a field of view, which is named differently in every subset."""
    if subset == "mibi_decidua":
        candidate = os.path.join(input_dir, "segmentation_data", f"{fov}_segmentation_labels.tiff")
        return candidate if os.path.exists(candidate) else None

    if subset == "mibi_breast":
        candidate = os.path.join(input_dir, "segmentation_data", f"{fov}_feature_0.tif")
        return candidate if os.path.exists(candidate) else None

    if subset in ("vectra_colon", "vectra_pancreas"):
        # The mask repeats the folder name of the field of view and appends the deepcell suffix.
        seg_dir = os.path.join(input_dir, "segmentation")
        for suffix in ("feature_0.ome.tif", "feature_0.tif"):
            candidate = os.path.join(seg_dir, f"{fov}{suffix}")
            if os.path.exists(candidate):
                return candidate
        return None

    # For codex_colon the mask name differs from the field of view, so it is matched on the
    # sample id and the region, e.g. 'B012B_reg004_X01_Y01_Z01' -> '.../B012B/B012B_..._reg004_..._labeled.ome.tif'.
    tokens = fov.split("_")
    sample = tokens[0]
    region = next((t for t in tokens if t.startswith("reg")), None)
    sample_dir = os.path.join(input_dir, "masks", sample)
    if region is None or not os.path.isdir(sample_dir):
        return None
    for fname in sorted(os.listdir(sample_dir)):
        if region in fname and fname.endswith("_labeled.ome.tif"):
            return os.path.join(sample_dir, fname)
    return None


def _preprocess_data(input_dir, data_dir, subset):
    import h5py

    os.makedirs(data_dir, exist_ok=True)
    assignment = _get_split_assignment(os.path.dirname(data_dir), subset)

    image_root = os.path.join(input_dir, IMAGE_DIRS[subset])
    if not os.path.isdir(image_root):
        raise RuntimeError(f"Could not find the image folder '{IMAGE_DIRS[subset]}' of '{subset}' in {input_dir}.")

    image_dirs = natsorted(os.path.join(image_root, name) for name in os.listdir(image_root))

    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            continue

        fov = os.path.basename(image_dir)
        out_path = os.path.join(data_dir, f"{fov}.h5")
        if os.path.exists(out_path):
            continue

        instance_path = _find_instance_path(input_dir, subset, fov)
        if instance_path is None:
            continue

        markers = _get_marker_paths(image_dir)
        nuclei = _get_composite(markers, NUCLEI_CHANNELS[subset])
        membrane = _get_composite(markers, MEMBRANE_CHANNELS[subset])
        if nuclei is None:
            raise RuntimeError(
                f"None of the nuclei channels {NUCLEI_CHANNELS[subset]} are present for '{fov}'. "
                f"The channels in the data are {sorted(markers)[:20]}. Please adapt 'NUCLEI_CHANNELS'."
            )
        if membrane is None:
            membrane = np.zeros_like(nuclei)

        instances = np.squeeze(tifffile.imread(instance_path)).astype("uint32")
        stacked = np.stack([nuclei, membrane])

        with h5py.File(out_path, "a") as f:
            # A field of view that the split config does not list stays out of all three splits.
            f.attrs["split"] = assignment.get(fov, "unassigned")
            f.attrs["subset"] = subset
            f.create_dataset("raw/nuclei", data=nuclei, compression="gzip")
            f.create_dataset("raw/membrane", data=membrane, compression="gzip")
            f.create_dataset("raw/stacked", data=stacked, compression="gzip")
            f.create_dataset("labels/cell", data=instances, compression="gzip")


def get_pan_multiplex_data(
    path: Union[os.PathLike, str],
    subset: Literal["mibi_decidua", "mibi_breast", "codex_colon", "vectra_colon", "vectra_pancreas"],
    download: bool = False,
) -> str:
    """Download one subset of the Pan-Multiplex dataset.

    Args:
        path: The folder where the function stores the data.
        subset: The subset of the dataset. See `SUBSET_PARTS` for the valid choices.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the prepared data.
    """
    if subset not in SUBSET_PARTS:
        raise ValueError(f"'{subset}' is not a valid subset. Choose one of {list(SUBSET_PARTS.keys())}.")

    data_dir = os.path.join(path, subset, "data")
    if glob(os.path.join(data_dir, "*.h5")):
        return data_dir

    subset_dir = os.path.join(path, subset)
    os.makedirs(subset_dir, exist_ok=True)

    input_dir = os.path.join(subset_dir, subset)
    if not os.path.exists(input_dir):
        zip_path = _download_subset(subset_dir, subset, download)
        util.unzip(zip_path=zip_path, dst=subset_dir, remove=True)

    _preprocess_data(input_dir, data_dir, subset)

    return data_dir


def get_pan_multiplex_paths(
    path: Union[os.PathLike, str],
    subset: Union[str, List[str]],
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> List[str]:
    """Get the paths to the Pan-Multiplex data.

    Args:
        path: The folder where the function stores the data.
        subset: One subset or a list of subsets. See `SUBSET_PARTS` for the valid choices.
        split: The data split. The function uses all fields of view by default.
        download: Whether to download the data if it is not present.

    Returns:
        The list of filepaths to the input data.
    """
    import h5py

    if split is not None and split not in ("train", "val", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose 'train', 'val' or 'test'.")

    subsets = [subset] if isinstance(subset, str) else subset
    volume_paths = []
    for name in subsets:
        data_dir = get_pan_multiplex_data(path, name, download)
        volume_paths.extend(natsorted(glob(os.path.join(data_dir, "*.h5"))))

    if split is not None:
        selected = []
        for volume_path in volume_paths:
            with h5py.File(volume_path, "r") as f:
                if f.attrs.get("split") == split:
                    selected.append(volume_path)
        volume_paths = selected

    assert len(volume_paths) > 0, f"Could not find data for the subset '{subset}' and split '{split}'."
    return volume_paths


def get_pan_multiplex_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Union[str, List[str]],
    split: Optional[Literal["train", "val", "test"]] = None,
    raw_channel: Literal["stacked", "nuclei", "membrane"] = "stacked",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Pan-Multiplex dataset for cell segmentation in multiplexed images.

    Args:
        path: The folder where the function stores the data.
        patch_shape: The patch shape to use for training.
        subset: One subset or a list of subsets. See `SUBSET_PARTS` for the valid choices.
        split: The data split. The function uses all fields of view by default.
        raw_channel: The input channels. Use 'stacked' for the nuclei and the membrane composite,
            or one of 'nuclei' and 'membrane' for a single channel.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if raw_channel not in ("stacked", "nuclei", "membrane"):
        raise ValueError(f"'{raw_channel}' is not a valid raw channel. Choose 'stacked', 'nuclei' or 'membrane'.")

    volume_paths = get_pan_multiplex_paths(path, subset, split, download)
    kwargs = util.update_kwargs(kwargs, "with_channels", raw_channel == "stacked")

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=f"raw/{raw_channel}",
        label_paths=volume_paths,
        label_key="labels/cell",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_pan_multiplex_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Union[str, List[str]],
    split: Optional[Literal["train", "val", "test"]] = None,
    raw_channel: Literal["stacked", "nuclei", "membrane"] = "stacked",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Pan-Multiplex dataloader for cell segmentation in multiplexed images.

    Args:
        path: The folder where the function stores the data.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: One subset or a list of subsets. See `SUBSET_PARTS` for the valid choices.
        split: The data split. The function uses all fields of view by default.
        raw_channel: The input channels. Use 'stacked' for the nuclei and the membrane composite,
            or one of 'nuclei' and 'membrane' for a single channel.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pan_multiplex_dataset(path, patch_shape, subset, split, raw_channel, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
