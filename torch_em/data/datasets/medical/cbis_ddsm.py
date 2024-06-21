import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, Optional

import torch_em

from .. import util


def get_cbis_ddsm_data(path, split, task, tumour_type, download):
    os.makedirs(path, exist_ok=True)

    assert split in ["Train", "Test"]

    if task is None:
        task = "*"
    else:
        assert task in ["Calc", "Mass"]

    if tumour_type is None:
        tumour_type = "*"
    else:
        assert tumour_type in ["MALIGNANT", "BENIGN"]

    data_dir = os.path.join(path, "DATA")
    if os.path.exists(data_dir):
        return os.path.join(path, "DATA", task, split, tumour_type)

    util.download_source_kaggle(
        path=path, dataset_name="mohamedbenticha/cbis-ddsm/", download=download,
    )
    zip_path = os.path.join(path, "cbis-ddsm.zip")
    util.unzip(zip_path=zip_path, dst=path)
    return os.path.join(path, "DATA", task, split, tumour_type)


def _get_cbis_ddsm_paths(path, split, task, tumour_type, download):
    data_dir = get_cbis_ddsm_data(
        path=path,
        split=split,
        task=task,
        tumour_type=tumour_type,
        download=download
    )

    image_paths = natsorted(glob(os.path.join(data_dir, "*_FULL_*.png")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "*_MASK_*.png")))

    assert len(image_paths) == len(gt_paths)

    return image_paths, gt_paths


def get_cbis_ddsm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["Train", "Test"],
    task: Optional[Literal["Calc", "Mass"]] = None,
    tumour_type: Optional[Literal["MALIGNANT", "BENIGN"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of calcification and mass in mammography.

    This dataset is a preprocessed version of https://www.cancerimagingarchive.net/collection/cbis-ddsm/ available
    at https://www.kaggle.com/datasets/mohamedbenticha/cbis-ddsm/data. The related publication is:
    - https://doi.org/10.1038/sdata.2017.177

    Please cite it if you use this dataset in a publication.
    """
    image_paths, gt_paths = _get_cbis_ddsm_paths(
        path=path, split=split, task=task, tumour_type=tumour_type, download=download
    )

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
        is_seg_dataset=False,
        **kwargs
    )
    return dataset


def get_cbis_ddsm_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["Train", "Test"],
    task: Optional[Literal["Calc", "Mass"]] = None,
    tumour_type: Optional[Literal["MALIGNANT", "BENIGN"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of calcification and mass in mammography. See `get_cbis_ddsm_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cbis_ddsm_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        task=task,
        tumour_type=tumour_type,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
