"""The BrainPTM dataset contains annotations for white matter tracts in brain MRI.

The dataset consists of the 60 training cases of the BrainPTM 2021 challenge, with binary masks for up
to four tracts per case: the left and right optic radiation ('OR_left', 'OR_right') and the left and
right corticospinal tract ('CST_left', 'CST_right'). 56 of the 60 cases have all four tracts, the
remaining 4 have only the optic radiation. See also `TRACT_NAMES`.

NOTE: The T1 image is used, since it shares its grid with the tracts. The release also provides a
diffusion-weighted series per case, which is not on the same grid and is not used here.

NOTE: The 15 test cases of the challenge are not used, because their released tract files are documented
placeholders rather than the withheld ground truth.

The dataset is located at https://doi.org/10.5281/zenodo.4600679 and is distributed under the UK
Non-Commercial Government Licence v2.0.
Please cite the Zenodo record if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "data": "https://zenodo.org/records/4600679/files/sheba75_data_train.zip?download=1",
    "tracts": "https://zenodo.org/records/4600679/files/sheba75_tracts_train.zip?download=1",
}

CHECKSUMS = {
    "data": "540f3fd72edffe87b06169b796a149429c9e71104341893d3c792d66667ee60e",
    "tracts": "e524cdaa73746bcb31718956bd4ea9e00aab50b8fe20fd1adc80dbd005ba22fb",
}

TRACT_NAMES = ["OR_left", "OR_right", "CST_left", "CST_right"]
"""The white matter tracts of the BrainPTM dataset. Every case has the optic radiation tracts, only 56
of the 60 cases also have the corticospinal tracts."""


def get_brainptm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BrainPTM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data_train")
    tracts_dir = os.path.join(path, "tracts_train")
    if os.path.exists(data_dir) and os.path.exists(tracts_dir):
        return path

    os.makedirs(path, exist_ok=True)
    for name, dst in [("data", data_dir), ("tracts", tracts_dir)]:
        zip_path = os.path.join(path, f"sheba75_{name}_train.zip")
        util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=dst, remove=False)

    return path


def get_brainptm_paths(
    path: Union[os.PathLike, str],
    tract: Literal["OR_left", "OR_right", "CST_left", "CST_right"] = "OR_left",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BrainPTM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        tract: The choice of white matter tract. One of 'OR_left', 'OR_right', 'CST_left', 'CST_right'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if tract not in TRACT_NAMES:
        raise ValueError(f"'{tract}' is not a valid tract. Choose from {TRACT_NAMES}.")

    root = get_brainptm_data(path, download)

    raw_paths, label_paths = [], []
    for label_path in natsorted(glob(os.path.join(root, "tracts_train", "case_*", f"{tract}.nii.gz"))):
        case_id = os.path.basename(os.path.dirname(label_path))
        image_path = os.path.join(root, "data_train", case_id, "T1.nii.gz")
        if os.path.exists(image_path):
            raw_paths.append(image_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_brainptm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    tract: Literal["OR_left", "OR_right", "CST_left", "CST_right"] = "OR_left",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BrainPTM dataset for white matter tract segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        tract: The choice of white matter tract. One of 'OR_left', 'OR_right', 'CST_left', 'CST_right'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_brainptm_paths(path, tract, download)

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


def get_brainptm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    tract: Literal["OR_left", "OR_right", "CST_left", "CST_right"] = "OR_left",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BrainPTM dataloader for white matter tract segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        tract: The choice of white matter tract. One of 'OR_left', 'OR_right', 'CST_left', 'CST_right'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brainptm_dataset(path, patch_shape, tract, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
