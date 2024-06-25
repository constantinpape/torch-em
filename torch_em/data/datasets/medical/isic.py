import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


URL = {
    "images": {
        "train": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "val": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "test": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
    },
    "gt": {
        "train": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        "val": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "test": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip",
    },
}

CHECKSUM = {
    "images": {
        "train": "80f98572347a2d7a376227fa9eb2e4f7459d317cb619865b8b9910c81446675f",
        "val": "0ea920fcfe512d12a6e620b50b50233c059f67b10146e1479c82be58ff15a797",
        "test": "e59ae1f69f4ed16f09db2cb1d76c2a828487b63d28f6ab85997f5616869b127d",
    },
    "gt": {
        "train": "99f8b2bb3c4d6af483362010715f7e7d5d122d9f6c02cac0e0d15bef77c7604c",
        "val": "f6911e9c0a64e6d687dd3ca466ca927dd5e82145cb2163b7a1e5b37d7a716285",
        "test": "2e8f6edce454a5bdee52485e39f92bd6eddf357e81f39018d05512175238ef82",
    }
}


def get_isic_data(path, split, download):
    os.makedirs(path, exist_ok=True)

    im_url = URL["images"][split]
    im_checksum = CHECKSUM["images"][split]

    gt_url = URL["gt"][split]
    gt_checksum = CHECKSUM["gt"][split]

    im_zipfile = os.path.split(im_url)[-1]
    gt_zipfile = os.path.split(gt_url)[-1]

    imdir = os.path.join(path, Path(im_zipfile).stem)
    gtdir = os.path.join(path, Path(gt_zipfile).stem)

    im_zip_path = os.path.join(path, im_zipfile)
    gt_zip_path = os.path.join(path, gt_zipfile)

    if os.path.exists(imdir) and os.path.exists(gtdir):
        return imdir, gtdir

    # download the images
    util.download_source(path=im_zip_path, url=im_url, download=download, checksum=im_checksum)
    util.unzip(zip_path=im_zip_path, dst=path, remove=False)
    # download the ground-truth
    util.download_source(path=gt_zip_path, url=gt_url, download=download, checksum=gt_checksum)
    util.unzip(zip_path=im_zip_path, dst=path, remove=False)

    return imdir, gtdir


def _get_isic_paths(path, split, download):
    image_dir, gt_dir = get_isic_data(path=path, split=split, download=download)

    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))

    return image_paths, gt_paths


def get_isic_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    download: bool = False,
    make_rgb: bool = True,
    resize_inputs: bool = False,
    **kwargs
):
    """Dataset for the segmentation of skin lesion in dermoscopy images.

    This dataset is related to the following publication(s):
    - https://doi.org/10.1038/sdata.2018.161
    - https://doi.org/10.48550/arXiv.1710.05006
    - https://doi.org/10.48550/arXiv.1902.03368

    The database is located at https://challenge.isic-archive.com/data/#2018

    Please cite it if you use this dataset for a publication.
    """
    assert split in list(URL["images"].keys()), f"{split} is not a valid split."

    image_paths, gt_paths = _get_isic_paths(path=path, split=split, download=download)

    if make_rgb:
        kwargs["raw_transform"] = to_rgb

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )
    return dataset


def get_isic_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: str,
    download: bool = False,
    make_rgb: bool = True,
    resize_inputs: bool = False,
    **kwargs
):
    """Dataloader for the segmentation of skin lesion in dermoscopy images. See `get_isic_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isic_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        download=download,
        make_rgb=make_rgb,
        resize_inputs=resize_inputs,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
