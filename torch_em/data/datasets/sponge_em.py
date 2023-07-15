import os
from glob import glob

import torch_em
from . import util

URL = "https://zenodo.org/record/8150818/files/sponge_em_train_data.zip?download=1"
CHECKSUM = "f1df616cd60f81b91d7642933e9edd74dc6c486b2e546186a7c1e54c67dd32a5"


def _require_sponge_em_data(path, download):
    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "data.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path)


def get_sponge_em_dataset(path, mode, patch_shape, sample_ids=None, download=False, **kwargs):
    """Dataset for the segmentation of sponge cells and organelles in EM.

    This dataset is from the publication https://doi.org/10.1126/science.abj2949.
    Please cite it if you use this dataset for a publication.
    """
    assert mode in ("semantic", "instances")

    n_files = len(glob(os.path.join(path, "*.h5")))
    if n_files == 0:
        _require_sponge_em_data(path, download)
        n_files = len(glob(os.path.join(path, "*.h5")))
    assert n_files == 3

    if sample_ids is None:
        sample_ids = range(1, n_files + 1)
    paths = [os.path.join(path, f"train_data_0{i}.h5") for i in sample_ids]

    raw_key = "volumes/raw"
    label_key = f"volumes/labels/{mode}"
    return torch_em.default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_sponge_em_loader(path, mode, patch_shape, batch_size, sample_ids=None, download=False, **kwargs):
    """Dataloader for the segmentation of sponge cells and organelles in EM. See 'get_sponge_em_dataset'."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_sponge_em_dataset(path, mode, patch_shape, sample_ids=sample_ids, download=download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
