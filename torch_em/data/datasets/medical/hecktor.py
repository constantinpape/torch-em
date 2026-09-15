"""The HECKTOR dataset contains annotations for head and neck tumor segmentation in FDG-PET/CT scans.

It comprises the training set of the HECKTOR 2022 challenge (https://hecktor.grand-challenge.org):
524 PET/CT studies of patients with histologically proven oropharyngeal head and neck cancer, collected
at 7 centers, with annotations of the primary tumor and of the involved lymph nodes. The 359 test studies
of the challenge are distributed without annotations and are therefore not included here.

Every case provides a co-registered pair of a low-dose non-contrast-enhanced CT scan and an FDG-PET scan
(converted to standardized uptake values). The 'modality' argument selects which of the two is used as
the raw input, the annotation is defined on the CT grid.

NOTE: The label legend is as follows:
- background: 0, primary tumor (GTVp): 1, lymph nodes (GTVn): 2
This is documented on https://hecktor.grand-challenge.org/Data/ and in the challenge overview paper.

NOTE: The dataset requires registration and cannot be downloaded automatically. Please follow these steps:
- Visit https://hecktor.grand-challenge.org/Data/, register for the challenge on grand-challenge.org and
  request access to the data. The organizers grant access after approval of the request and of the signed
  end user agreement. NOTE: At the time of writing, the organizers state on that page that the data is
  temporarily unavailable while they extend the agreement with the partner hospitals.
- Download 'hecktor2022_training.zip' from the data download page you are given access to.
- Extract it into '<path>', such that
  '<path>/hecktor2022_training/imagesTr/<case_id>__CT.nii.gz',
  '<path>/hecktor2022_training/imagesTr/<case_id>__PT.nii.gz' and
  '<path>/hecktor2022_training/labelsTr/<case_id>.nii.gz' exist.
  The case ids are of the form '<center>-<number>', e.g. 'CHUM-001' or 'MDA-042'. The 524 cases are
  distributed over the centers as CHUM: 56, CHUP: 72, CHUS: 72, CHUV: 53, HGJ: 55, HMR: 18, MDA: 198.

NOTE: There is no openly published copy of the 2022 release. Third-party re-uploads of HECKTOR data exist,
but all of the ones we are aware of are a different edition of the challenge, which must not be confused
with this one: HECKTOR 2020 has 201 cases, HECKTOR 2021 has 224 training cases, HECKTOR 2025 has 679
training cases (it adds the center USZ and many more MDA cases) and HECKTOR 2026 has 883 cases. Only the
2022 edition has exactly 524 training cases.

The dataset is located at https://hecktor.grand-challenge.org/Data/.

This dataset is from the publication https://doi.org/10.1007/978-3-031-27420-6_1.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


MODALITIES = {"ct": "__CT.nii.gz", "pt": "__PT.nii.gz"}

LABEL_IDS = {"background": 0, "primary_tumor": 1, "lymph_nodes": 2}


def _find_data_dir(path):
    candidates = [
        os.path.join(path, "hecktor2022_training"), path, *glob(os.path.join(path, "*", "hecktor2022_training")),
    ]
    for candidate in candidates:
        if len(glob(os.path.join(candidate, "imagesTr", "*__CT.nii.gz"))) > 0:
            return candidate
    return None


def get_hecktor_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the HECKTOR dataset.

    Args:
        path: Filepath to a folder where the data is stored.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = _find_data_dir(path)
    if data_dir is not None:
        return data_dir

    msg = f"Could not find the HECKTOR 2022 training data at '{path}'. "
    msg += "'torch_em' cannot download this dataset, as it requires registration for the challenge. "
    msg += "Please register at https://hecktor.grand-challenge.org, request access to the data as described at "
    msg += "https://hecktor.grand-challenge.org/Data/ and download 'hecktor2022_training.zip'. Then extract it "
    msg += f"into '{path}', such that "
    msg += f"'{os.path.join(path, 'hecktor2022_training', 'imagesTr', 'CHUM-001__CT.nii.gz')}' exists."
    if download:
        raise NotImplementedError(msg)
    else:
        raise FileNotFoundError(msg)


def get_hecktor_paths(
    path: Union[os.PathLike, str], modality: Literal["ct", "pt"] = "ct", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the HECKTOR data.

    Args:
        path: Filepath to a folder where the data is stored.
        modality: The imaging modality to use as raw input. Either 'ct' or 'pt'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose one of {list(MODALITIES.keys())}.")

    data_dir = get_hecktor_data(path, download)

    suffix = MODALITIES[modality]
    raw_paths = natsorted(glob(os.path.join(data_dir, "imagesTr", f"*{suffix}")))
    label_paths = [
        os.path.join(data_dir, "labelsTr", os.path.basename(p).replace(suffix, ".nii.gz")) for p in raw_paths
    ]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_hecktor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["ct", "pt"] = "ct",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HECKTOR dataset for head and neck tumor segmentation.

    Args:
        path: Filepath to a folder where the data is stored.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality to use as raw input. Either 'ct' or 'pt'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hecktor_paths(path, modality, download)

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


def get_hecktor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["ct", "pt"] = "ct",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HECKTOR dataloader for head and neck tumor segmentation.

    Args:
        path: Filepath to a folder where the data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality to use as raw input. Either 'ct' or 'pt'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hecktor_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
