import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util


URL = {
    "train": "https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa463786541a01e714d390/?zip=",
    "val": "https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa463686541a01eb15048c/?zip=",
    "test": "https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa4635ba010901f0891bd0/?zip="
}

# FIXME the checksums are not reliable (same behaviour spotted in PlantSeg downloads from osf)
CHECKSUM = {
    "train": None, "val": None, "test": None,
}


def get_verse_data(path, split, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data", split)
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, f"verse2020_{split}.zip")
    util.download_source(path=zip_path, url=URL[split], download=download, checksum=CHECKSUM[split])
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_verse_paths(path, split, download):
    data_dir = get_verse_data(path=path, split=split, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "rawdata", "*", "*_ct.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "derivatives", "*", "*_msk.nii.gz")))

    return image_paths, gt_paths


def get_verse_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of vertebrae in CT scans.

    This dataset is from Sekuboyina et al. - https://doi.org/10.1016/j.media.2021.102166
    Please cite it if you use this dataset for a publication.
    """
    assert split in ["train", "val", "test"], f"'{split}' is not a valid split."

    image_paths, gt_paths = _get_verse_paths(path=path, split=split, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_verse_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of vertebrae in CT scans. See `get_verse_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_verse_dataset(
        path=path, split=split, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
