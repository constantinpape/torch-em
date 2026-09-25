"""This dataset provides lipid droplet segmentation masks for representative crops from four
mouse liver volumes on the public "OpenOrganelle" `janelia-cosem-datasets` S3 bucket (the same
bucket used by `cellmap.py`, `janelia_nucleus.py`, and `openorganelle_nucleus.py`).

Three of the four datasets (`jrc_mus-liver-4/5/6`) are already covered by
`openorganelle_nucleus.py`, but only for their nuclear-membrane (`nuc_mem`) label; their separate
lipid droplet label, at `labels/inference/segmentations/ld/{level}`, is a distinct annotation not
provided by that module. `jrc_mus-liver-7` is not covered by any existing torch-em loader.

The lipid droplet label is a binary mask (only values 0/255 seen) that exactly matches the raw
FIB-SEM data's multiscale pyramid at every level checked, so no resampling or axis correction is
needed. Lipid droplets in these fatty-liver volumes can be very large (macrovesicular steatosis),
so `jrc_mus-liver-5` and `jrc_mus-liver-6` use a coarser matched pyramid level (`s3`, 64nm) for
their fixed crop, chosen to show a real droplet boundary rather than a single, deep-interior,
fully-saturated block; `jrc_mus-liver-4` and `jrc_mus-liver-7` use the finest level (`s0`, 8nm).

The full matched-resolution raw+label pairs are far too large to download whole (hundreds of GB
to over 1 TB); this module downloads one fixed, manually verified crop per dataset, matching the
crop that was visually reviewed in napari before this loader was added. Data is CC0-1.0, per the
bucket's established convention, except for `jrc_mus-liver-7`, whose license could not be
independently confirmed (no dedicated landing page was found for this specific dataset).

Please cite the CellMap project (https://www.janelia.org/project-team/cellmap) if you use this
data.
"""

import os
from typing import List, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BUCKET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com/"
LABEL_SUBPATH = "labels/inference/segmentations/ld"

# dataset name -> recon folder, EM array name, ld pyramid level to use, and the bounding box
# (in that level's voxel coordinates) of the fixed, visually reviewed crop.
DATASETS = {
    "jrc_mus-liver-4": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_level": "s0",
        "bounding_box": ((704, 832), (1984, 2112), (4672, 4800)),
    },
    "jrc_mus-liver-5": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_level": "s3",
        "bounding_box": ((448, 640), (352, 544), (0, 128)),
    },
    "jrc_mus-liver-6": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_level": "s3",
        "bounding_box": ((288, 480), (0, 96), (736, 928)),
    },
    "jrc_mus-liver-7": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_level": "s0",
        "bounding_box": ((960, 1088), (5056, 5184), (8640, 8768)),
    },
}

# `jrc_mus-liver-7` has no dedicated landing page confirming its license; the others follow the
# bucket's established CC0-1.0 convention (already verified for their existing torch-em entries).
UNVERIFIED_LICENSE = ("jrc_mus-liver-7",)


def _open_remote_zarr(s3_path):
    import zarr
    from zarr.storage import FsspecStore

    # `FsspecStore` performs ranged partial reads, so a bounded crop only pulls the chunks it
    # actually needs, unlike `fsspec.get_mapper` which can fetch whole shard/chunk files.
    store = FsspecStore.from_url(s3_path, storage_options={"anon": True}, read_only=True)
    return zarr.open(store, mode="r")


def _get_json(url):
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _multiscale_levels(group_url):
    attrs = _get_json(f"{group_url}/.zattrs")
    return {
        d["path"]: tuple(d["coordinateTransformations"][0]["scale"])
        for d in attrs["multiscales"][0]["datasets"]
    }


def _find_matching_em_level(dataset_name, recon, em_name, label_scale):
    em_levels = _multiscale_levels(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}")
    for level, scale in em_levels.items():
        if scale == label_scale:
            return level
    raise RuntimeError(f"No EM pyramid level of '{dataset_name}' matches the ld label scale {label_scale}.")


def get_openorganelle_lipid_droplet_data(
    path: Union[os.PathLike, str], dataset_name: str, download: bool = False
) -> str:
    """Download the reviewed lipid droplet crop for one OpenOrganelle dataset.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        dataset_name: The name of the dataset, one of the keys in `DATASETS`.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    if dataset_name not in DATASETS:
        raise ValueError(f"'{dataset_name}' is not a valid dataset name. Choose from {sorted(DATASETS.keys())}.")

    info = DATASETS[dataset_name]
    recon, em_name, label_level = info["recon"], info["em_name"], info["label_level"]
    bounding_box = info["bounding_box"]

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{dataset_name}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached crop found at '{zarr_path}'. Set download=True to stream it from S3.")

    print(f"Streaming a lipid droplet crop for '{dataset_name}' from the janelia-cosem-datasets S3 bucket ...")
    label_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/{LABEL_SUBPATH}/{label_level}"
    label_arr = _open_remote_zarr(label_url)
    label_slices = tuple(slice(*bb) for bb in bounding_box)
    label_data = label_arr[label_slices]

    label_scale = _multiscale_levels(
        f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/{LABEL_SUBPATH}"
    )[label_level]
    em_level = _find_matching_em_level(dataset_name, recon, em_name, label_scale)
    em_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/{em_level}"
    em_arr = _open_remote_zarr(em_url)
    raw_data = em_arr[label_slices]

    common_shape = tuple(min(r, lb) for r, lb in zip(raw_data.shape, label_data.shape))
    raw_data = raw_data[tuple(slice(0, s) for s in common_shape)]
    label_data = label_data[tuple(slice(0, s) for s in common_shape)]

    assert raw_data.shape == label_data.shape, (
        f"Shape mismatch for '{dataset_name}': raw {raw_data.shape} vs labels {label_data.shape}"
    )

    def _make_array(name, data, shuffle):
        arr = root.create_array(
            name, shape=data.shape, chunks=tuple(min(128, s) for s in data.shape), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["dataset"] = dataset_name
    root.attrs["recon"] = recon
    root.attrs["label_target"] = "lipid_droplet"
    root.attrs["label_source"] = label_url
    root.attrs["em_source"] = em_url
    root.attrs["license_verified"] = dataset_name not in UNVERIFIED_LICENSE

    _make_array("raw", raw_data, shuffle="shuffle")
    _make_array("labels", label_data, shuffle="bitshuffle")

    print(f"Cached '{dataset_name}' to '{zarr_path}' (shape {raw_data.shape}).")
    return zarr_path


def get_openorganelle_lipid_droplet_paths(
    path: Union[os.PathLike, str], dataset_names: Optional[List[str]] = None, download: bool = False,
) -> List[str]:
    """Get paths to cached OpenOrganelle lipid droplet zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())
    return [get_openorganelle_lipid_droplet_data(path, name, download) for name in dataset_names]


def get_openorganelle_lipid_droplet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    dataset_names: Optional[List[str]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the OpenOrganelle lipid droplet dataset for lipid droplet segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_openorganelle_lipid_droplet_paths(path, dataset_names, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_openorganelle_lipid_droplet_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    dataset_names: Optional[List[str]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for lipid droplet segmentation in the OpenOrganelle lipid droplet dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_openorganelle_lipid_droplet_dataset(
        path, patch_shape, dataset_names=dataset_names, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
