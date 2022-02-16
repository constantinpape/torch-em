import os

import torch_em
from .util import download_source, update_kwargs

SNEMI_URLS = {
    "train": "https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download",
    "test": "https://oc.embl.de/index.php/s/aRhphk35H23De2s/download"
}
CHECKSUMS = {
    "train": "5b130a24d9eb23d972fede0f1a403bc05f6808b361cfa22eff23b930b12f0615",
    "test": "3df3920a0ddec6897105845f842b2665d37a47c2d1b96d4f4565682e315a59fa"
}


def get_snemi_loader(
    path,
    patch_shape,
    sample="train",
    download=False,
    offsets=None,
    boundaries=False,
    batch_size=1,
    num_workers: int = 0,
    shuffle: bool = False,
    loader_kwargs=None,
    **dataset_kwargs,
):
    """
    """
    ds = get_snemi_dataset(
        path=path,
        patch_shape=patch_shape,
        sample=sample,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **dataset_kwargs,
    )
    return torch_em.get_data_loader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, **(loader_kwargs or {})
    )


def get_snemi_dataset(
    path,
    patch_shape,
    sample="train",
    download=False,
    offsets=None,
    boundaries=False,
    **kwargs,
):
    assert len(patch_shape) == 3
    os.makedirs(path, exist_ok=True)

    url = SNEMI_URLS[sample]
    checksum = CHECKSUMS[sample]

    data_path = os.path.join(path, f"snemi_{sample}.h5")
    download_source(data_path, url, download, checksum)
    assert os.path.exists(data_path), data_path

    kwargs = update_kwargs(kwargs, "patch_shape", patch_shape)
    kwargs = update_kwargs(kwargs, "ndim", 3)
    kwargs = update_kwargs(kwargs, "is_seg_dataset", True)

    raw_key = "volumes/raw"
    label_key = "volumes/labels/neuron_ids"

    assert not ((offsets is not None) and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, ignore_label=None, add_binary_target=False, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform()
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, **kwargs)
