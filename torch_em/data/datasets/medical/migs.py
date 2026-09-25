"""The MIGS surgical navigation dataset contains annotations for ocular anatomy
(corneal limbus, iris, trabecular meshwork) and surgical instrument segmentation
in minimally invasive glaucoma surgery (MIGS) videos.

NOTE: The Zenodo record holds two dataset families:
- 'MIGS video dataset <i>.zip' (Task I): raw phase-recognition videos, unannotated.
- 'Task II Raw Data.zip' / 'Task II Annotated Data.zip' (Task II): densely-annotated
  frames for semantic segmentation, which is what this module exposes.

Only a subset of the Task II frames (those from the training and validation splits
of the original paper) have their grayscale class-index masks publicly released in
'Task II Annotated Data.zip': 4,062 annotated frames across 51 video clips from 39
patients, confirmed by inspecting the real archive contents, not assumed.

The dataset is located at https://doi.org/10.5281/zenodo.19438128 and is licensed
under CC-BY-4.0.

This dataset is from the publication https://doi.org/10.1038/s41597-026-07535-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "raw": "https://zenodo.org/records/19438128/files/Task%20II%20Raw%20Data.zip",
    "annotations": "https://zenodo.org/records/19438128/files/Task%20II%20Annotated%20Data.zip",
}

CHECKSUMS = {
    "raw": "7f45ee3f91edee52dc93a78dcb9e6b9d22796e6f7ae2f4a8eaa6c240271cde4d",
    "annotations": "94f6d4329c48104f1304eef85d10353c4b26efb232e3ac000db741d27d0c889b",
}


def get_migs_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MIGS surgical navigation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    raw_dir = os.path.join(path, "raw")
    annotations_dir = os.path.join(path, "annotations")
    if os.path.exists(raw_dir) and os.path.exists(annotations_dir):
        return path

    os.makedirs(path, exist_ok=True)

    raw_zip_path = os.path.join(path, "Task_II_Raw_Data.zip")
    util.download_source(path=raw_zip_path, url=URLS["raw"], download=download, checksum=CHECKSUMS["raw"])
    util.unzip(zip_path=raw_zip_path, dst=raw_dir)

    annotations_zip_path = os.path.join(path, "Task_II_Annotated_Data.zip")
    util.download_source(
        path=annotations_zip_path, url=URLS["annotations"], download=download, checksum=CHECKSUMS["annotations"]
    )
    util.unzip(zip_path=annotations_zip_path, dst=annotations_dir)

    return path


def get_migs_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the MIGS surgical navigation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_migs_data(path, download)

    all_gt_paths = natsorted(glob(os.path.join(data_dir, "annotations", "Grayscale Images", "*", "*", "*.png")))

    # A handful of masks in the release have no corresponding raw frame (confirmed by inspecting the real
    # archive contents, e.g. 'S frame 30/154_S_O/154_S_O_frame_00038' has a mask but the raw frame is missing
    # from 'Task II Raw Data.zip'). Such masks are skipped rather than raising an error.
    image_paths, gt_paths = [], []
    for gt_path in all_gt_paths:
        # e.g. '<data_dir>/annotations/Grayscale Images/F frame 50/144_F_O/144_F_O_frame_00040.png' pairs with
        # '<data_dir>/raw/F frame 50/144_F_O/144_F_O_frame_00040.jpg'.
        relpath = Path(gt_path).relative_to(os.path.join(data_dir, "annotations", "Grayscale Images"))
        image_path = os.path.join(data_dir, "raw", relpath.parent, f"{relpath.stem}.jpg")
        if not os.path.exists(image_path):
            continue

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0, (
        "No image-mask pairs were found. The expected 'raw/<category>/<video>/<frame>.jpg' vs "
        "'annotations/Grayscale Images/<category>/<video>/<frame>.png' layout may not match the actual structure "
        f"of the downloaded data. Please inspect the data at '{data_dir}'."
    )

    return image_paths, gt_paths


def get_migs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MIGS dataset for ocular anatomy and surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_migs_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_migs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MIGS dataloader for ocular anatomy and surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_migs_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
