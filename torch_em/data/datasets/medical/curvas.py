import os
from glob import glob
from natsort import natsorted
from typing import Tuple, Union

import torch_em

from .. import util


URL = "https://zenodo.org/records/11147560/files/training_data.zip"
CHECKSUM = "02e64b0d963c3a8ac7330c3161f5f76f25ae01a4072bd3032fb1c4048baec2df"


def get_curvas_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "training_set")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "training_data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_curvas_paths(path, rater, download):
    data_dir = get_curvas_data(path=path, download=download)

    if not isinstance(rater, list):
        rater = [str(rater)]

    assert len(rater) == 1, "The segmentations for multiple raters is not supported at the moment."

    image_paths = natsorted(glob(os.path.join(data_dir, "*", "image.nii.gz")))
    gt_paths = []
    for _rater in rater:
        gt_paths.extend(natsorted(glob(os.path.join(data_dir, "*", f"annotation_{_rater}.nii.gz"))))

    return image_paths, gt_paths


def get_curvas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    rater: Union[int, list] = ["1"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of pancreas, kidney and liver in abdominal CT scans.

    NOTE: This dataset has multiple raters annotating the aforementioned organs for all patients.

    The dataset is located at:
    - https://www.sycaimedical.com/challenge
    - https://zenodo.org/records/11147560

    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_curvas_paths(path, rater, download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_curvas_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    rater: Union[int, list] = ["1"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of pancreas, kidney and liver in abdominal CT scans.
    See `get_curvas_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_curvas_dataset(
        path=path, patch_shape=patch_shape, rater=rater, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
