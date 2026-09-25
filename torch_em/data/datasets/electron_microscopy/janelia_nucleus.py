"""The Janelia CellMap nucleus dataset provides nucleus instance segmentation masks for FIB-SEM
volumes from the public `janelia-cosem-datasets` S3 bucket (the same "OpenOrganelle" bucket used
by other CellMap data), covering several mouse tissues and one Drosophila tissue.

For 13 of the 16 volumes, nuclei were segmented automatically with Cellpose 3.0.9 and only
cursorily corrected by hand, following the protocol
"Generating Nuclei Segmentations for vEM datasets using Cellpose"
(https://dx.doi.org/10.17504/protocols.io.n2bvjnq8pgk5/v1). Treat these labels the same way as
the MitoNet auto-labels in `mitonet_predicted_kidney.py`: useful pseudo-labels, not verified
ground truth.

The 3 `jrc_mus-nacc-*` volumes (mouse nucleus accumbens) are instead densely, manually segmented
in Amira-Avizo, following
"Using Amira to manually segment organelles in vEM for machine learning V.3"
(https://dx.doi.org/10.17504/protocols.io.bp2l61rb5vqe/v3). These can be treated as reliable
ground truth.

Each volume is released as its own Figshare deposit (e.g. https://doi.org/10.6084/m9.figshare.26506555),
under CC-BY-4.0. The Figshare entries are metadata-only pointers: their one "file" is a link-only
stub whose actual content lives on the S3 bucket, at
`s3://janelia-cosem-datasets/{dataset}/{dataset}.zarr/{recon}/em/...` (raw) and
`.../{recon}/labels/inference/segmentations/nuc/...` (nucleus instance labels), where `recon` is
either "recon-1" or "recon-2" depending on the deposit.

The raw EM data is stored as a multiscale OME-Zarr pyramid at up to 4 nm isotropic resolution,
while the nucleus labels are only released at one, coarser resolution (their pyramid level "s0").
This module finds the raw pyramid level whose resolution matches the label's "s0" level (their
shapes then match exactly, so no rescaling is needed, unlike the mito/tissue labels in
`hemibrain.py` which need explicit upsampling) and downloads only that matched pair, in full,
for each requested dataset name.

Please cite the Cellpose or Amira-Avizo protocol above (as appropriate for the chosen dataset)
and the CellMap project (https://www.janelia.org/project-team/cellmap) if you use this data.
"""

import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BUCKET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com/"

# dataset name -> (figshare article id, recon folder, annotation type)
# "cellpose" = automatic Cellpose 3.0.9 prediction with cursory manual correction.
# "manual" = dense manual segmentation in Amira-Avizo.
DATASETS = {
    "jrc_mus-pancreas-1": (26506552, "recon-1", "cellpose"),
    "jrc_mus-pancreas-2": (26506555, "recon-1", "cellpose"),
    "jrc_mus-pancreas-3": (26506561, "recon-1", "cellpose"),
    "jrc_mus-granule-neurons-1": (26506510, "recon-2", "cellpose"),
    "jrc_mus-granule-neurons-2": (26506513, "recon-2", "cellpose"),
    "jrc_mus-granule-neurons-3": (26506516, "recon-2", "cellpose"),
    "jrc_mus-guard-hair-follicle": (26506519, "recon-1", "cellpose"),
    "jrc_mus-kidney": (26506522, "recon-1", "cellpose"),
    "jrc_mus-kidney-2": (26506534, "recon-1", "cellpose"),
    "jrc_mus-liver-2": (26506537, "recon-1", "cellpose"),
    "jrc_mus-meissner-corpuscle-1": (26506540, "recon-1", "cellpose"),
    "jrc_mus-meissner-corpuscle-2": (26506543, "recon-1", "cellpose"),
    "jrc_fly-mb-z0419-20": (26506507, "recon-1", "cellpose"),
    "jrc_mus-nacc-2": (26513209, "recon-2", "manual"),
    "jrc_mus-nacc-3": (26513212, "recon-2", "manual"),
    "jrc_mus-nacc-4": (26513215, "recon-2", "manual"),
}


def _get_json(url):
    import requests
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _list_prefixes(prefix):
    import requests
    resp = requests.get(BUCKET_URL, params={"prefix": prefix, "delimiter": "/"}, timeout=60)
    resp.raise_for_status()
    return re.findall(r"<Prefix>([^<]*)</Prefix>", resp.text)


def _find_em_array_name(dataset_name, recon):
    prefix = f"{dataset_name}/{dataset_name}.zarr/{recon}/em/"
    names = [p[len(prefix):].rstrip("/") for p in _list_prefixes(prefix) if p != prefix]
    if not names:
        raise RuntimeError(f"Could not find an 'em' array for '{dataset_name}' under '{prefix}'.")
    return names[0]


def _find_matching_em_level(dataset_name, recon, em_name):
    em_attrs = _get_json(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/.zattrs")
    nuc_attrs = _get_json(
        f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/labels/inference/segmentations/nuc/.zattrs"
    )
    nuc_scale = nuc_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    for level in em_attrs["multiscales"][0]["datasets"]:
        scale = level["coordinateTransformations"][0]["scale"]
        if scale == nuc_scale:
            return level["path"]
    raise RuntimeError(
        f"No EM resolution level of '{dataset_name}' matches the nucleus label resolution {nuc_scale}."
    )


def _read_zarr_array(s3_path):
    import zarr
    import fsspec

    store = fsspec.get_mapper(s3_path, anon=True)
    array = zarr.open(store, mode="r")
    return np.asarray(array[:])


def get_janelia_nucleus_data(path: Union[os.PathLike, str], dataset_name: str, download: bool = False) -> str:
    """Download nucleus instance segmentation data for one Janelia CellMap dataset.

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

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{dataset_name}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it from S3.")

    _, recon, annotation = DATASETS[dataset_name]
    em_name = _find_em_array_name(dataset_name, recon)
    em_level = _find_matching_em_level(dataset_name, recon, em_name)

    em_s3 = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/{em_level}"
    nuc_s3 = (
        f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/"
        "labels/inference/segmentations/nuc/s0"
    )

    print(f"Streaming '{dataset_name}' ({recon}, {annotation} labels) from the janelia-cosem-datasets S3 bucket ...")
    raw = _read_zarr_array(em_s3)
    labels = _read_zarr_array(nuc_s3)
    assert raw.shape == labels.shape, f"Shape mismatch for '{dataset_name}': raw {raw.shape} vs labels {labels.shape}"

    def _make_array(name, data, shuffle):
        arr = root.create_array(
            name, shape=data.shape, chunks=tuple(min(128, s) for s in data.shape), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["dataset"] = dataset_name
    root.attrs["recon"] = recon
    root.attrs["annotation_type"] = annotation
    root.attrs["labels_are_manual"] = annotation == "manual"
    root.attrs["em_source"] = em_s3
    root.attrs["label_source"] = nuc_s3

    _make_array("raw", raw, shuffle="shuffle")
    _make_array("labels", labels, shuffle="bitshuffle")

    print(f"Cached '{dataset_name}' to '{zarr_path}' (shape {raw.shape}).")
    return zarr_path


def get_janelia_nucleus_paths(
    path: Union[os.PathLike, str], dataset_names: Optional[List[str]] = None, download: bool = False,
) -> List[str]:
    """Get paths to cached Janelia CellMap nucleus zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())
    return [get_janelia_nucleus_data(path, name, download) for name in dataset_names]


def get_janelia_nucleus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    dataset_names: Optional[List[str]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Janelia CellMap nucleus dataset for nucleus instance segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_janelia_nucleus_paths(path, dataset_names, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_janelia_nucleus_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    dataset_names: Optional[List[str]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for nucleus instance segmentation in the Janelia CellMap nucleus dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_janelia_nucleus_dataset(
        path, patch_shape, dataset_names=dataset_names, download=download,
        offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
