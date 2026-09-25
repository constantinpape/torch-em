"""The HVA-CT dataset contains annotations for intrahepatic veins and the liver in CT scans.

The annotations re-annotate the portal and hepatic venous systems of the 61 thin-slice scans of the
Medical Segmentation Decathlon task 8 (hepatic vessel), so the images are downloaded from there. Two
annotation sets are provided: 'vessels' with 1: portal vein and 2: hepatic vein, and 'liver' with a
binary liver mask. See also `CLASS_IDS`.

NOTE: The ids of the two venous systems are the reverse of the order that the file names suggest. They
were assigned from the per-case voxel counts that the release ships in its metadata, which match id 1 to
the portal and id 2 to the hepatic vein for all 61 scans.

The dataset is located at https://doi.org/10.5281/zenodo.19850108 and is distributed under the
CC BY-SA 4.0 license.
Please cite it, and the Medical Segmentation Decathlon publication
https://doi.org/10.1038/s41467-022-30695-9, if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Optional, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .msd import get_msd_data
from .. import util


URLS = {
    "vessels": "https://zenodo.org/records/19850108/files/hp_masks.zip?download=1",
    "liver": "https://zenodo.org/records/19850108/files/liver_masks.zip?download=1",
}

CHECKSUMS = {
    "vessels": "539d5a83c5b9f6f8d890727923d0666e53b8b585f02651a6f250689055fbb4b1",
    "liver": "25c40378fde4f1a1a52518fec7d930a4588b904136893ada69004ffe231efe96",
}

ANNOTATIONS = {
    "vessels": ("hp_masks", "hepatic_portalvessel_"),
    "liver": ("liver_masks", "liver_"),
}
"""Mapping from the annotation choice to its folder and the prefix of its files."""

CLASS_NAMES = ["portal_vein", "hepatic_vein"]
"""The venous systems of the 'vessels' annotations. The label id of a system is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the venous system to its label id."""


def _get_image_paths(msd_dir):
    image_paths = {}
    for split in ["imagesTr", "imagesTs"]:
        for path in glob(os.path.join(msd_dir, "Task08_HepaticVessel", split, "*.nii.gz")):
            fname = os.path.basename(path)
            # The MSD archives carry macOS resource fork files next to the actual volumes.
            if fname.startswith("._"):
                continue
            image_paths[fname[len("hepaticvessel_"):-len(".nii.gz")]] = path
    return image_paths


def get_hva_ct_data(
    path: Union[os.PathLike, str],
    annotation: Literal["vessels", "liver"] = "vessels",
    msd_path: Optional[Union[os.PathLike, str]] = None,
    download: bool = False,
) -> Tuple[str, str]:
    """Download the HVA-CT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotations. Either 'vessels' or 'liver'.
        msd_path: Filepath to an existing download of the Medical Segmentation Decathlon. The scans are
            downloaded to `path` if it is not given.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the annotations are downloaded.
        Filepath where the scans are downloaded.
    """
    if annotation not in ANNOTATIONS:
        raise ValueError(f"'{annotation}' is not a valid annotation. Choose from {list(ANNOTATIONS.keys())}.")

    folder, _ = ANNOTATIONS[annotation]
    label_dir = os.path.join(path, folder)
    if not os.path.exists(label_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, f"{folder}.zip")
        util.download_source(
            path=zip_path, url=URLS[annotation], download=download, checksum=CHECKSUMS[annotation]
        )
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    # The scans are the hepatic vessel scans of the Medical Segmentation Decathlon.
    msd_dir = get_msd_data(path=path if msd_path is None else msd_path, task_name="hepaticvessel", download=download)

    return label_dir, msd_dir


def get_hva_ct_paths(
    path: Union[os.PathLike, str],
    annotation: Literal["vessels", "liver"] = "vessels",
    msd_path: Optional[Union[os.PathLike, str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HVA-CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotations. Either 'vessels' or 'liver'.
        msd_path: Filepath to an existing download of the Medical Segmentation Decathlon.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    label_dir, msd_dir = get_hva_ct_data(path, annotation, msd_path, download)

    _, prefix = ANNOTATIONS[annotation]
    image_paths = _get_image_paths(msd_dir)

    raw_paths, label_paths = [], []
    for label_path in natsorted(glob(os.path.join(label_dir, "*.nii.gz"))):
        case_id = os.path.basename(label_path)[len(prefix):-len(".nii.gz")]
        image_path = image_paths.get(case_id)
        if image_path is not None:
            raw_paths.append(image_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_hva_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation: Literal["vessels", "liver"] = "vessels",
    msd_path: Optional[Union[os.PathLike, str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HVA-CT dataset for intrahepatic vein and liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. Either 'vessels' or 'liver'.
        msd_path: Filepath to an existing download of the Medical Segmentation Decathlon.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hva_ct_paths(path, annotation, msd_path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_hva_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation: Literal["vessels", "liver"] = "vessels",
    msd_path: Optional[Union[os.PathLike, str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HVA-CT dataloader for intrahepatic vein and liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. Either 'vessels' or 'liver'.
        msd_path: Filepath to an existing download of the Medical Segmentation Decathlon.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hva_ct_dataset(path, patch_shape, annotation, msd_path, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
