import os
from glob import glob
from natsort import natsorted

import torch_em

from .. import util


URL = {
    "images": "https://zenodo.org/records/10159290/files/images.zip?download=1",
    "masks": "https://zenodo.org/records/10159290/files/masks.zip?download=1"
}

CHECKSUMS = {
    "images": "a54cba2905284ff6cc9999f1dd0e4d871c8487187db7cd4b068484eac2f50f17",
    "masks": "13a6e25a8c0d74f507e16ebb2edafc277ceeaf2598474f1fed24fdf59cb7f18f"
}


def get_spider_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "images.zip")
    util.download_source(path=zip_path, url=URL["images"], download=download, checksum=CHECKSUMS["images"])
    util.unzip(zip_path=zip_path, dst=data_dir)

    zip_path = os.path.join(path, "masks.zip")
    util.download_source(path=zip_path, url=URL["images"], download=download, checksum=CHECKSUMS["images"])
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_spider_paths(path, download):
    data_dir = get_spider_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.mha")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.mha")))

    return image_paths, gt_paths


def get_spider_dataset(path, patch_shape, download=False, **kwargs):
    """Dataset for segmentation of vertebrae, intervertebral discs and spinal canal in T1 and T2 MRI series.

    https://zenodo.org/records/10159290
    https://www.nature.com/articles/s41597-024-03090-w

    Please cite it if you use this data in a publication.
    """
    # TODO: expose the choice to choose specific MRI modality, for now this works for our interests.
    image_paths, gt_paths = _get_spider_paths(path, download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_spider_loader(path, patch_shape, batch_size, download=False, **kwargs):
    """Dataloader for segmentation of vertebrae, intervertebral discs and spinal canal in T1 and T2 MRI series.
    See `get_spider_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_spider_dataset(path=path, patch_shape=patch_shape, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
