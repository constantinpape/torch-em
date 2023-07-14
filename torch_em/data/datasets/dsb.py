import os
from shutil import move

import torch_em
from . import util

DSB_URLS = {
    "full": "",  # TODO
    "reduced": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
}
CHECKSUMS = {
    "full": None,
    "reduced": "e44921950edce378063aa4457e625581ba35b4c2dbd9a07c19d48900129f386f"
}


def _download_dsb(path, source, download):
    os.makedirs(path, exist_ok=True)
    url = DSB_URLS[source]
    checksum = CHECKSUMS[source]

    train_out_path = os.path.join(path, "train")
    test_out_path = os.path.join(path, "test")

    if os.path.exists(train_out_path) and os.path.exists(test_out_path):
        return

    zip_path = os.path.join(path, "dsb.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)

    move(os.path.join(path, "dsb2018", "train"), train_out_path)
    move(os.path.join(path, "dsb2018", "test"), test_out_path)


def get_dsb_dataset(
    path, split, patch_shape, download=False,
    offsets=None, boundaries=False, binary=False,
    source="reduced", **kwargs
):
    assert split in ("test", "train"), split
    _download_dsb(path, source, download)

    image_path = os.path.join(path, split, "images")
    label_path = os.path.join(path, split, "masks")

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)
    return torch_em.default_segmentation_dataset(
        image_path, "*.tif", label_path, "*.tif", patch_shape, **kwargs
    )


def get_dsb_loader(
    path, split, patch_shape, batch_size, download=False,
    offsets=None, boundaries=False, binary=False,
    source="reduced", **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_dsb_dataset(
        path, split, patch_shape, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary,
        source=source, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
