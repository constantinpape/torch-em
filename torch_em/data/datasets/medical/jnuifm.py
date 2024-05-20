import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple
from urllib.parse import urljoin

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/records/7851339/files/"
URL = urljoin(BASE_URL, "Pubic%20Symphysis-Fetal%20Head%20Segmentation%20and%20Angle%20of%20Progression.zip")
CHECKSUM = "2b14d1c78e11cfb799d74951b0b985b90777c195f7a456ccd00528bf02802e21"


def get_jnuifm_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, r"Pubic Symphysis-Fetal Head Segmentation and Angle of Progression")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "JNU-IFM.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_jnuifm_paths(path, download):
    data_dir = get_jnuifm_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "image_mha", "*.mha")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "label_mha", "*.mha")))

    return image_paths, gt_paths


def get_jnuifm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of pubic symphysis and fetal head in ultrasound images.

    The label pixels are - 0: background, 1: pubic symphysis, 2: fetal head.

    The database is located at https://doi.org/10.5281/zenodo.7851339

    The dataset is from Lu et al. - https://doi.org/10.1016/j.dib.2022.107904
    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_jnuifm_paths(path=path, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )

    return dataset


def get_jnuifm_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    """
    Dataloader for segmentation of pubic symphysis and fetal head in ultrasound images.
    See `get_jnuifm_loader` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_jnuifm_dataset(path=path, patch_shape=patch_shape, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
