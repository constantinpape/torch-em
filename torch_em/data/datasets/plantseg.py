import os
from glob import glob

import numpy as np
import torch_em
from elf.io import open_file, is_group
from skimage.transform import rescale
from .util import download_source, update_kwargs

# TODO just download the full zip from https://osf.io/uzq3w/ instead
# but this is currently broken
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
NATIVE_RESOLUTION = (0.235, 0.075, 0.075)


def _resize(path, native_resolution, target_resolution):
    assert len(native_resolution) == len(target_resolution)
    scale_factor = tuple(nres / tres for nres, tres in zip(native_resolution, target_resolution))
    paths = glob(os.path.join(path, "*.h5"))
    for pp in paths:
        with open_file(pp, "a") as f:
            for name, obj in f.items():
                rescaled_name = f"rescaled/{name}"

                if is_group(obj):
                    continue
                if rescaled_name in f:
                    this_resolution = f[rescaled_name].attrs["resolution"]
                    correct_res = all(
                        np.isclose(this_re, target_re) for this_re, target_re in zip(this_resolution, target_resolution)
                    )
                    if correct_res:
                        continue
                    del f[rescaled_name]

                print("Resizing", pp, name)
                print("from resolution (microns)", native_resolution, "to", target_resolution)
                print("with scale factor", scale_factor)

                vol = obj[:]
                if name == "raw":
                    vol = rescale(vol, scale_factor, preserve_range=True).astype(vol.dtype)
                else:
                    vol = rescale(
                        vol, scale_factor, preserve_range=True, order=0, anti_aliasing=False
                    ).astype(vol.dtype)
                ds = f.create_dataset(rescaled_name, data=vol, compression="gzip")
                ds.attrs["resolution"] = target_resolution


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


def get_root_nucleus_loader(
    path,
    patch_shape,
    samples=None,
    target_resolution=None,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    assert len(patch_shape) == 3
    _download_plantseg(path, download, "root", "nuclei")
    if target_resolution is not None:
        _resize(path, NATIVE_RESOLUTION, target_resolution)

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

    if target_resolution is None:
        raw_key, label_key = "raw", "label_uint16_smooth"
    else:
        raw_key, label_key = "rescaled/raw", "rescaled/label_uint16_smooth"
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
