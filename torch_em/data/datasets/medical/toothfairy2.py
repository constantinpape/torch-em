import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util


def get_toothfairy2_data(path, download):
    """This function describes the download functionality.

    The dataset is located at https://ditto.ing.unimore.it/toothfairy2/.

    To download the dataset, please follow the mentioned steps:
    1. Visit the website, scroll down to the `Download` section, which expects you to sign up.
    2. Once you are signed up, use your sign in credentials to login to the home page and visit the `Download` section.
    3. Click on the blue icon: `Download Dataset` to download the zipped files to the desired path.
    """
    if download:
        msg = "Download is set to True, but 'torch_em' cannot download this dataset. "
        msg += "See `get_toothfairy2_data` for details."
        print(msg)

    zip_path = os.path.join(path, "ToothFairy2_Dataset.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"It's expected to place the 'ToothFairy2_Dataset.zip' file in '{path}'.")

    util.unzip(zip_path=zip_path, dst=path, remove=False)
    return os.path.join(path, "Dataset112_ToothFairy2")


def _get_toothfairy2_paths(path, download):
    data_dir = get_toothfairy2_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "imagesTr", "*.mha")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "labelsTr", "*.mha")))

    return image_paths, gt_paths


def get_toothfairy2_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of multiple structures in CBCT scans.

    This dataset is from the following publications:
    - Cipriano et al. - https://doi.org/10.1109/ACCESS.2022.3144840
    - Cipriano et al. - https://doi.org/10.1109/CVPR52688.2022.02046

    Please cite them if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_toothfairy2_paths(path=path, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_toothfairy2_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of multiple structures in CBCT scans.
    See `get_toothfairy2_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_toothfairy2_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
