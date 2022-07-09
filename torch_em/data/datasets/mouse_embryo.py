import os
from glob import glob

import torch_em
from .util import download_source, update_kwargs, unzip

URL = "https://zenodo.org/record/6546550/files/MouseEmbryos.zip?download=1"
CHECKSUM = "bf24df25e5f919489ce9e674876ff27e06af84445c48cf2900f1ab590a042622"


def _require_embryo_data(path, download):
    if os.path.exists(path):
        return
    os.makedirs(path, exist_ok=True)
    tmp_path = os.path.join(path, "mouse_embryo.zip")
    download_source(tmp_path, URL, download, CHECKSUM)
    unzip(tmp_path, path, remove=True)
    # remove empty volume
    os.remove(os.path.join(path, "Membrane", "fused_paral_stack0_chan2_tp00073_raw_crop_bg_noise.h5"))


def get_mouse_embryo_loader(
    path,
    name,
    split,
    patch_shape,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    ndim=3,
    **kwargs,
):
    assert name in ("membrane", "nuclei")
    assert split in ("train", "val")
    assert len(patch_shape) == 3
    data_path = _require_embryo_data(path, download)

    # the naming of the data is inconsistent: membrane has val, nuclei has test;
    # we treat nuclei:test as val
    split_ = "test" if name == "nuclei" and split == "val" else split
    file_paths = glob(os.path.join(data_path, name.capitalize(), split_, "*.h5"))
    file_paths.sort()

    assert not (offsets is not None and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, add_binary_target=binary, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=binary)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    kwargs = update_kwargs(kwargs, "patch_shape", patch_shape)
    kwargs = update_kwargs(kwargs, "ndim", ndim)

    raw_key, label_key = "raw", "label"
    return torch_em.default_segmentation_loader(
        file_paths, raw_key,
        file_paths, label_key,
        **kwargs
    )
