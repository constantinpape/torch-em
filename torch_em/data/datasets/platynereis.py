import os
from glob import glob

import numpy as np
import torch_em
from . import util

URLS = {
    "cells": "https://zenodo.org/record/3675220/files/membrane.zip",
    "nuclei": "https://zenodo.org/record/3675220/files/nuclei.zip",
    "cilia": "https://zenodo.org/record/3675220/files/cilia.zip",
    "cuticle": "https://zenodo.org/record/3675220/files/cuticle.zip"
}

CHECKSUMS = {
    "cells": "30eb50c39e7e9883e1cd96e0df689fac37a56abb11e8ed088907c94a5980d6a3",
    "nuclei": "a05033c5fbc6a3069479ac6595b0a430070f83f5281f5b5c8913125743cf5510",
    "cilia": "6d2b47f63d39a671789c02d8b66cad5e4cf30eb14cdb073da1a52b7defcc5e24",
    "cuticle": "464f75d30133e8864958049647fe3c2216ddf2d4327569738ad72d299c991843"
}


#
# TODO data-loader for more classes:
# - mitos
#


def _require_platy_data(path, name, download):
    os.makedirs(path, exist_ok=True)
    url = URLS[name]
    checksum = CHECKSUMS[name]

    zip_path = os.path.join(path, f"data-{name}.zip")
    util.download_source(zip_path, url, download=download, checksum=checksum)
    util.unzip(zip_path, path, remove=True)


def _check_data(path, prefix, extension, n_files):
    if not os.path.exists(path):
        return False
    files = glob(os.path.join(path, f"{prefix}*{extension}"))
    return len(files) == n_files


def get_platynereis_cuticle_dataset(path, patch_shape, sample_ids=None, download=False, **kwargs):
    cuticle_root = os.path.join(path, "cuticle")

    ext = ".n5"
    prefix, n_files = "train_data_", 5
    data_is_complete = _check_data(cuticle_root, prefix, ext, n_files)
    if not data_is_complete:
        _require_platy_data(path, "cuticle", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    paths = [os.path.join(cuticle_root, f"train_data_0{i}.n5") for i in sample_ids]

    raw_key = "volumes/raw"
    label_key = "volumes/labels/segmentation"
    return torch_em.default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_platynereis_cuticle_loader(
    path, patch_shape, batch_size, sample_ids=None, download=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_cuticle_dataset(
        path, patch_shape, sample_ids=sample_ids, download=download, **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_platynereis_cilia_dataset(
    path, split, patch_shape,
    offsets=None, boundaries=False, binary=False,
    download=False, **kwargs
):
    assert split in ("train", "val")
    cilia_root = os.path.join(path, "cilia")

    ext = ".h5"
    prefix, n_files = "train_data_cilia_", 3
    data_is_complete = _check_data(cilia_root, prefix, ext, n_files)
    if not data_is_complete:
        _require_platy_data(path, "cilia", download)

    paths = glob(os.path.join(cilia_root, f"{split}*.h5"))
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )

    raw_key = "volumes/raw"
    label_key = "volumes/labels/segmentation"
    return torch_em.default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_platynereis_cilia_loader(
    path, split, patch_shape, batch_size,
    offsets=None, boundaries=False, binary=False,
    download=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_cilia_dataset(
        path, split, patch_shape,
        offsets=offsets, boundaries=boundaries, binary=binary,
        download=download, **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_platynereis_cell_dataset(
    path, patch_shape,
    sample_ids=None, rois={},
    offsets=None, boundaries=False,
    download=False, **kwargs
):
    cell_root = os.path.join(path, "membrane")

    prefix = "train_data_membrane_"
    ext = ".n5"
    n_files = 9
    data_is_complete = _check_data(cell_root, prefix, ext, n_files)
    if not data_is_complete:
        _require_platy_data(path, "cells", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
    sample_ids.sort()

    data_paths = []
    data_rois = []
    for sample in sample_ids:
        data_paths.append(
            os.path.join(cell_root, f"{prefix}{sample:02}{ext}")
        )
        data_rois.append(rois.get(sample, np.s_[:, :, :]))

    kwargs = util.update_kwargs(kwargs, "rois", data_rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets,
    )

    raw_key = "volumes/raw/s1"
    label_key = "volumes/labels/segmentation/s1"
    return torch_em.default_segmentation_dataset(data_paths, raw_key, data_paths, label_key, patch_shape,  **kwargs)


def get_platynereis_cell_loader(
    path, patch_shape, batch_size,
    sample_ids=None, rois={},
    offsets=None, boundaries=False,
    download=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_cell_dataset(
        path, patch_shape, sample_ids, rois=rois,
        offsets=offsets, boundaries=boundaries, download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_platynereis_nuclei_dataset(
    path, patch_shape, sample_ids=None, rois={},
    offsets=None, boundaries=False, binary=False,
    download=False, **kwargs,
):
    nuc_root = os.path.join(path, "nuclei")
    prefix = "train_data_nuclei_"
    ext = ".h5"
    n_files = 12
    data_is_complete = _check_data(nuc_root, "train_data_nuclei", ".h5", 12)
    if not data_is_complete:
        _require_platy_data(path, "nuclei", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
    sample_ids.sort()

    data_paths = []
    data_rois = []
    for sample in sample_ids:
        data_paths.append(
            os.path.join(nuc_root, f"{prefix}{sample:02}{ext}")
        )
        data_rois.append(rois.get(sample, np.s_[:, :, :]))

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs = util.update_kwargs(kwargs, "rois", data_rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )

    raw_key = "volumes/raw"
    label_key = "volumes/labels/nucleus_instance_labels"
    return torch_em.default_segmentation_dataset(data_paths, raw_key, data_paths, label_key, patch_shape, **kwargs)


def get_platynereis_nuclei_loader(
    path, patch_shape, batch_size,
    sample_ids=None, rois={},
    offsets=None, boundaries=False, binary=False,
    download=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_nuclei_dataset(
        path, patch_shape, sample_ids=sample_ids, rois=rois,
        offsets=offsets, boundaries=boundaries, binary=binary, download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
