import os
from glob import glob

import torch_em
from . import util

URL = "https://zenodo.org/record/6546550/files/MouseEmbryos.zip?download=1"
CHECKSUM = "bf24df25e5f919489ce9e674876ff27e06af84445c48cf2900f1ab590a042622"


def _require_embryo_data(path, download):
    if os.path.exists(path):
        return
    os.makedirs(path, exist_ok=True)
    tmp_path = os.path.join(path, "mouse_embryo.zip")
    util.download_source(tmp_path, URL, download, CHECKSUM)
    util.unzip(tmp_path, path, remove=True)
    # remove empty volume
    os.remove(os.path.join(path, "Membrane", "train", "fused_paral_stack0_chan2_tp00073_raw_crop_bg_noise.h5"))


def get_mouse_embryo_dataset(
    path,
    name,
    split,
    patch_shape,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    assert name in ("membrane", "nuclei")
    assert split in ("train", "val")
    assert len(patch_shape) == 3
    _require_embryo_data(path, download)

    # the naming of the data is inconsistent: membrane has val, nuclei has test;
    # we treat nuclei:test as val
    split_ = "test" if name == "nuclei" and split == "val" else split
    file_paths = glob(os.path.join(path, name.capitalize(), split_, "*.h5"))
    file_paths.sort()

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=binary, binary=binary, boundaries=boundaries,
        offsets=offsets, binary_is_exclusive=False
    )

    raw_key, label_key = "raw", "label"
    return torch_em.default_segmentation_dataset(file_paths, raw_key, file_paths, label_key, patch_shape, **kwargs)


def get_mouse_embryo_loader(
    path,
    name,
    split,
    patch_shape,
    batch_size,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_mouse_embryo_dataset(
        path, name, split, patch_shape,
        download=download, offsets=offsets, boundaries=boundaries, binary=binary,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
