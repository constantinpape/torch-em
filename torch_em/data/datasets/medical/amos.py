import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Optional

import torch_em

from .. import util


URL = "https://zenodo.org/records/7155725/files/amos22.zip"
CHECKSUM = "d2fbf2c31abba9824d183f05741ce187b17905b8cca64d1078eabf1ba96775c2"


def get_amos_data(path, download, remove_zip=False):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "amos22")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "amos22.zip")

    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=remove_zip)

    return data_dir


def _get_amos_paths(path, split, modality, download):
    data_dir = get_amos_data(path=path, download=download)

    if split == "train":
        im_dir, gt_dir = "imagesTr", "labelsTr"
    elif split == "val":
        im_dir, gt_dir = "imagesVa", "labelsVa"
    elif split == "test":
        im_dir, gt_dir = "imagesTs", "labelsTs"
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    image_paths = sorted(glob(os.path.join(data_dir, im_dir, "*.nii.gz")))
    gt_paths = sorted(glob(os.path.join(data_dir, gt_dir, "*.nii.gz")))

    if modality is None:
        chosen_image_paths, chosen_gt_paths = image_paths, gt_paths
    else:
        ct_image_paths, ct_gt_paths = [], []
        mri_image_paths, mri_gt_paths = [], []
        for image_path, gt_path in zip(image_paths, gt_paths):
            patient_id = Path(image_path.split(".")[0]).stem
            id_value = int(patient_id.split("_")[-1])

            is_ct = id_value < 500

            if is_ct:
                ct_image_paths.append(image_path)
                ct_gt_paths.append(gt_path)
            else:
                mri_image_paths.append(image_path)
                mri_gt_paths.append(gt_path)

        if modality.upper() == "CT":
            chosen_image_paths, chosen_gt_paths = ct_image_paths, ct_gt_paths
        elif modality.upper() == "MRI":
            chosen_image_paths, chosen_gt_paths = mri_image_paths, mri_gt_paths
        else:
            raise ValueError(f"'{modality}' is not a valid modality.")

    return chosen_image_paths, chosen_gt_paths


def get_amos_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, ...],
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for abdominal multi-organ segmentation in CT and MRI scans.

    The database is located at https://doi.org/10.5281/zenodo.7155725
    The dataset is from AMOS 2022 Challenge - https://doi.org/10.48550/arXiv.2206.08023
    Please cite it if you use this dataset for publication.
    """
    image_paths, gt_paths = _get_amos_paths(path=path, split=split, modality=modality, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )

    return dataset


def get_amos_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for abdominal multi-organ segmentation in CT and MRI scans. See `get_amos_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_amos_dataset(
        path=path,
        split=split,
        patch_shape=patch_shape,
        modality=modality,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
