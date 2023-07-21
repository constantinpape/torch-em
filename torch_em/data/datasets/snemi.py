import os

import torch_em
from . import util

SNEMI_URLS = {
    "train": "https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download",
    "test": "https://oc.embl.de/index.php/s/aRhphk35H23De2s/download"
}
CHECKSUMS = {
    "train": "5b130a24d9eb23d972fede0f1a403bc05f6808b361cfa22eff23b930b12f0615",
    "test": "3df3920a0ddec6897105845f842b2665d37a47c2d1b96d4f4565682e315a59fa"
}


def get_snemi_dataset(
    path,
    patch_shape,
    sample="train",
    download=False,
    offsets=None,
    boundaries=False,
    **kwargs,
):
    """Dataset for the segmentation of neurons in EM.

    This dataset is from the publication https://doi.org/10.1016/j.cell.2015.06.054.
    Please cite it if you use this dataset for a publication.
    """
    assert len(patch_shape) == 3
    os.makedirs(path, exist_ok=True)

    data_path = os.path.join(path, f"snemi_{sample}.h5")
    util.download_source(data_path, SNEMI_URLS[sample], download, CHECKSUMS[sample])
    assert os.path.exists(data_path), data_path

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    raw_key = "volumes/raw"
    label_key = "volumes/labels/neuron_ids"
    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_snemi_loader(
    path,
    patch_shape,
    batch_size,
    sample="train",
    download=False,
    offsets=None,
    boundaries=False,
    **kwargs,
):
    """Dataloader for the segmentation of neurons in EM. See 'get_snemi_dataset'."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_snemi_dataset(
        path=path,
        patch_shape=patch_shape,
        sample=sample,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
