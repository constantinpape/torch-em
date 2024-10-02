import os
import shutil
import zipfile
from glob import glob
from typing import Tuple, Union

import torch_em

from .. import util


URL = "https://zenodo.org/records/10643410/files/OrganoIDNetData.zip?download=1"
CHECKSUM = "3cd9239bf74bda096ecb5b7bdb95f800c7fa30b9937f9aba6ddf98d754cbfa3d"


def get_organoidnet_data(path, split, download):
    splits = ["Training", "Validation", "Test"]
    assert split in splits

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    # Download and extraction.
    zip_path = os.path.join(path, "OrganoIDNetData.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    # Only "Training", "Test", "Validation" from the zip are relevant and need to be extracted.
    # They are in "/OrganoIDNetData/Dataset/"
    prefix = "OrganoIDNetData/Dataset/"
    for dl_split in splits:

        dl_prefix = prefix + dl_split

        with zipfile.ZipFile(zip_path) as archive:
            for ff in archive.namelist():
                if ff.startswith(dl_prefix):
                    archive.extract(ff, path)

    for dl_split in splits:
        shutil.move(
            os.path.join(path, "OrganoIDNetData/Dataset", dl_split),
            os.path.join(path, dl_split)
        )

    assert os.path.exists(data_dir)

    os.remove(zip_path)
    return data_dir


def _get_data_paths(path, split, download):
    data_dir = get_organoidnet_data(path=path, split=split, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "Images", "*.tif")))
    label_paths = sorted(glob(os.path.join(data_dir, "Masks", "*.tif")))

    return image_paths, label_paths


def get_organoidnet_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
):
    """Dataset for the segmentation of panceratic organoids.

    This dataset is from the publication https://doi.org/10.1007/s13402-024-00958-2.
    Please cite it if you use this dataset for a publication.
    """
    image_paths, label_paths = _get_data_paths(path=path, split=split, download=download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_organoidnet_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    """Dataloader for the segmentation of pancreatic organoids in brightfield images.
    See `get_organoidnet_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_organoidnet_dataset(
        path=path,
        split=split,
        patch_shape=patch_shape,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
