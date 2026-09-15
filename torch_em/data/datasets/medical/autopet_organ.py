"""The AutoPET-Organ dataset contains annotations for 11 organs in whole-body FDG-PET scans.

The dataset consists of 100 studies of the AutoPET collection with organ annotations that were added and
expert-examined for the SegAnyPET publication, so the annotations are downloaded from there and the
scans are taken from the existing autopet loader. The label ids are 1: liver, 2: kidney (left),
3: kidney (right), 4: heart, 5: spleen, 6: aorta, 7: lung lower lobe (left), 8: lung lower lobe (right),
9: lung upper lobe (left), 10: lung upper lobe (right), 11: lung middle lobe (right).
See also `CLASS_IDS`.

NOTE: The release stores the right middle lobe under two ids, 11 and 12, which are perfectly
complementary: of the 100 studies, 64 use id 11, 36 use id 12, and no study uses both or neither. The
two carry the same structure, matching in side, in position relative to the other organs of the same
study, and in size. 'harmonize' maps id 12 onto id 11 so that one organ has one id, which is the
default because the raw ids split one class in two and make a broken training target.

NOTE: The source describes a prostate annotation, but no study of this release contains a twelfth
structure once ids 11 and 12 are merged, so the annotations cover 11 organs rather than 12.

NOTE: The label ids were assigned by measuring each annotation in the data, since the source lists the
organs without their ids. Every study is stored in LPS orientation, and the ids follow from the side and
the position of each annotation relative to the other organs of the same study.

The annotations are located at https://github.com/YichiZhang98/SegAnyPET and the scans at
https://autopet.grand-challenge.org.
This dataset is from the publication https://doi.org/10.48550/arXiv.2502.14351.
Please cite it, and the AutoPET publication https://doi.org/10.1038/s41597-022-01718-3, if you use this
dataset in your research.
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import Union, Optional, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .autopet import get_autopet_data
from .. import util


URL = "https://github.com/YichiZhang98/SegAnyPET/raw/main/AutoPET-OrganlabelsTr.zip"

CHECKSUM = "8794137a0f8df0bc192aa8bb98009b3bc706e4591cbe8f289f8d38e1c5574142"

CLASS_NAMES = [
    "liver", "kidney_left", "kidney_right", "heart", "spleen", "aorta",
    "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_upper_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right",
]
"""The organs of the AutoPET-Organ dataset. The label id of an organ is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the organ name to its label id."""

MIDDLE_LOBE_ALIAS = 12
"""The second id that the release uses for the right middle lobe, which `harmonize` maps onto id 11."""


def _prepare_labels(label_path, out_path, harmonize):
    """Store the annotation under a filename without a dot in its stem, optionally harmonizing its ids.

    Two of the study names abbreviate 'nativ und mit' as 'nativ u. mit', and the dot of that abbreviation
    makes the '.nii.gz' extension unrecognizable to the file readers.
    """
    import nibabel as nib

    if os.path.exists(out_path):
        return out_path

    image = nib.load(label_path)
    labels = np.asarray(image.dataobj)
    if harmonize:
        labels = np.where(labels == MIDDLE_LOBE_ALIAS, CLASS_IDS["lung_middle_lobe_right"], labels)

    nib.save(nib.Nifti1Image(labels.astype("uint8"), image.affine, image.header), out_path)
    return out_path


def get_autopet_organ_data(
    path: Union[os.PathLike, str], autopet_path: Optional[Union[os.PathLike, str]] = None, download: bool = False
) -> Tuple[str, str]:
    """Download the AutoPET-Organ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        autopet_path: Filepath to an existing AutoPET download. The scans are downloaded to `path` if it
            is not given, which takes several hundred gigabytes.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the annotations are downloaded.
        Filepath where the scans are downloaded.
    """
    label_dir = os.path.join(path, "AutoPET-OrganlabelsTr")
    if not os.path.exists(label_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "AutoPET-OrganlabelsTr.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    # The scans are the whole-body PET studies of the AutoPET collection, which are several hundred
    # gigabytes, so an existing download of them is used when it is given.
    autopet_path = path if autopet_path is None else autopet_path
    get_autopet_data(path=autopet_path, download=download)
    image_dir = os.path.join(autopet_path, "AutoPET-II", "FDG-PET-CT-Lesions")

    return label_dir, image_dir


def get_autopet_organ_paths(
    path: Union[os.PathLike, str],
    harmonize: bool = True,
    autopet_path: Optional[Union[os.PathLike, str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AutoPET-Organ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        harmonize: Whether to map the second id of the right middle lobe onto its first one.
        autopet_path: Filepath to an existing AutoPET download.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    label_dir, image_dir = get_autopet_organ_data(path, autopet_path, download)

    prepared_dir = os.path.join(path, "harmonized" if harmonize else "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    raw_paths, label_paths = [], []
    for label_path in natsorted(glob(os.path.join(label_dir, "*.nii.gz"))):
        # The annotations are named '<patient id>_<study id>_.nii.gz' for the study folder they belong to.
        stem = os.path.basename(label_path)[:-len(".nii.gz")].rstrip("_")
        match = re.match(r"^(PETCT_[0-9a-f]+)_(.+)$", stem)
        if match is None:
            continue

        image_path = os.path.join(image_dir, match.group(1), match.group(2), "SUV.nii.gz")
        if not os.path.exists(image_path):
            continue

        out_path = os.path.join(prepared_dir, f"{stem.replace('.', '')}.nii.gz")
        raw_paths.append(image_path)
        label_paths.append(_prepare_labels(label_path, out_path, harmonize))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_autopet_organ_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    harmonize: bool = True,
    autopet_path: Optional[Union[os.PathLike, str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AutoPET-Organ dataset for organ segmentation in whole-body PET.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        harmonize: Whether to map the second id of the right middle lobe onto its first one.
        autopet_path: Filepath to an existing AutoPET download.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_autopet_organ_paths(path, harmonize, autopet_path, download)

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


def get_autopet_organ_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    harmonize: bool = True,
    autopet_path: Optional[Union[os.PathLike, str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AutoPET-Organ dataloader for organ segmentation in whole-body PET.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        harmonize: Whether to map the second id of the right middle lobe onto its first one.
        autopet_path: Filepath to an existing AutoPET download.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_autopet_organ_dataset(
        path, patch_shape, harmonize, autopet_path, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
