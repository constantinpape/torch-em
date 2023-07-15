import os
import torch_em

from . import util


def _require_dataset(path, sample):
    data_folder = os.path.join(path, sample)
    os.makedirs(data_folder, exist_ok=True)


def get_nuc_mm_dataset(path, sample, patch_shape, download=False, **kwargs):
    assert sample in ("mouse", "zebrafish")

    data_folder = os.path.join(path, sample)
    if not os.path.exists(data_folder):
        _require_dataset(path, sample)


def get_nuc_mm_loader(path, sample, patch_shape, batch_size, download=False, **kwargs):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_nuc_mm_dataset(path, sample, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
