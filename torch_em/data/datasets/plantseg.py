import os
from glob import glob
import torch_em
from .util import download_source, update_kwargs

# TODO add the other URLS and checksums
URLS = {
    "root": {
        "cells": [],
        "nuclei": [
            "https://osf.io/n9y34/download",
            "https://osf.io/su27h/download",
            "https://osf.io/q5rxz/download",
        ]
    },
    "ovules": {
        "cells": []
    }
}
CHECKSUMS = {
    "root": {
        "cells": [],
        "nuclei": [
            "ff9e86cb05d56ae2463e7482ad248a985a2378b1c7f3d92022d1191a6504adfa",
            "b21fd70556591ca04e83b1461324d0a14e31b1dad24fe4b1efe9712dded2281c",
            "c8976fefdc06d92290ba6c2b7686fd2c1a285a800a3b6d8a002e1ec67caca072",
        ]
    },
    "ovules": {
        "cells": []
    }
}


# TODO resizing for pretraining
def _download_plantseg(path, download, name, type_):
    urls = URLS[name][type_]
    checksums = CHECKSUMS[name][type_]
    assert len(urls) == len(checksums)
    os.makedirs(path, exist_ok=True)
    for ii, (url, checksum) in enumerate(zip(urls, checksums)):
        out_path = os.path.join(path, f"{name}_{type_}_{ii}.h5")
        if os.path.exists(out_path):
            continue
        download_source(out_path, url, download, checksum)


# TODO resizing for pretraining
def get_root_nucleus_loader(
    path,
    patch_shape,
    samples=None,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    assert len(patch_shape) == 3
    _download_plantseg(path, download, "root", "nuclei")

    file_paths = glob(os.path.join(path, "*.h5"))
    file_paths.sort()

    if samples is not None:
        assert all(isinstance(sample, int) for sample in samples)
        assert all(sample < len(file_paths) for sample in samples)
        file_paths = [file_paths[sample] for sample in samples]

    assert sum((offsets is not None, boundaries, binary)) <= 1
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     add_binary_target=True,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    kwargs = update_kwargs(kwargs, "patch_shape", patch_shape)
    kwargs = update_kwargs(kwargs, "ndim", 3)

    raw_key, label_key = "raw", "label_uint16_smooth"
    return torch_em.default_segmentation_loader(
        file_paths, raw_key,
        file_paths, label_key,
        **kwargs
    )


# TODO
def get_root_cell_loader():
    pass


# TODO
def get_ovules_loader():
    pass
