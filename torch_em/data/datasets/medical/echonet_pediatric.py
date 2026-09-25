"""EchoNet-Pediatric contains annotations for left ventricle segmentation in pediatric
echocardiography videos, split into apical-4-chamber (A4C) and parasternal-short-axis (PSAX) views.

The dataset comprises 7,643 videos from 1,958 patients aged 0-18 years, collected at Lucile Packard
Children's Hospital Stanford between 2014 and 2021. Each video only has expert tracings for the two
labeled frames (end-diastole and end-systole) used to compute the left ventricular ejection fraction,
not dense per-frame masks. The dataset is located at https://echonet.github.io/pediatric/ and
distributed through the Stanford AIMI Center Shared Datasets Portal under a non-commercial research
use agreement: registration as an individual user is required, re-distribution (including sharing the
download link) is forbidden, and re-identification attempts are prohibited. This module cannot
download the dataset automatically; see `get_echonet_pediatric_data` for the manual steps.

This dataset is from the publication https://doi.org/10.1016/j.echo.2023.01.015 (cite the DOI
https://doi.org/10.71718/d05h-gy43 for the data itself). Please cite them if you use this dataset in
your research.

NOTE: Reading the videos requires 'opencv-python' ('cv2'), and the tracing rasterization requires
'scikit-image'.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


VIEWS = ("A4C", "PSAX")
SPLITS = ("TRAIN", "VAL", "TEST")


def _trace_to_mask(x1, y1, x2, y2, shape):
    from skimage.draw import polygon

    # Follows the rasterization used by the EchoNet dataset releases: the first coordinate pair is
    # the long axis of the left ventricle, and the remaining pairs are the perpendicular short-axis
    # distances. Walking the two sides of the short-axis points (one side forwards, the other
    # backwards) traces the outline of the traced region, which is then filled in.
    x = np.concatenate((x1[1:], x2[1:][::-1]))
    y = np.concatenate((y1[1:], y2[1:][::-1]))

    rows, cols = polygon(np.round(y).astype(int), np.round(x).astype(int), shape=shape)
    mask = np.zeros(shape, dtype="uint8")
    mask[rows, cols] = 1
    return mask


def _preprocess_view(data_dir, view, preprocessed_dir):
    import cv2

    file_list = pd.read_csv(os.path.join(data_dir, view, "FileList.csv"))
    tracings = pd.read_csv(os.path.join(data_dir, view, "VolumeTracings.csv"))

    videos_dir = os.path.join(data_dir, view, "Videos")
    view_out_dir = os.path.join(preprocessed_dir, view)
    os.makedirs(view_out_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for _, row in tqdm(file_list.iterrows(), total=len(file_list), desc=f"Preprocess EchoNet-Pediatric {view}"):
        file_name = row["FileName"]
        stem = os.path.splitext(file_name)[0]
        split = str(row["Split"]).upper()

        video_tracings = tracings[tracings["FileName"] == file_name]
        if video_tracings.empty:
            video_tracings = tracings[tracings["FileName"] == stem + ".avi"]
        if video_tracings.empty:
            continue

        video_path = os.path.join(videos_dir, file_name)
        if not os.path.exists(video_path):
            video_path = os.path.join(videos_dir, stem + ".avi")

        capture = cv2.VideoCapture(video_path)

        for frame_idx, frame_tracings in video_tracings.groupby("Frame"):
            image_path = os.path.join(view_out_dir, f"{stem}_{split}_{frame_idx}.tif")
            mask_path = os.path.join(view_out_dir, f"{stem}_{split}_{frame_idx}_mask.tif")

            if os.path.exists(image_path) and os.path.exists(mask_path):
                image_paths.append(image_path)
                gt_paths.append(mask_path)
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = capture.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = _trace_to_mask(
                frame_tracings["X1"].to_numpy(),
                frame_tracings["Y1"].to_numpy(),
                frame_tracings["X2"].to_numpy(),
                frame_tracings["Y2"].to_numpy(),
                frame.shape,
            )

            imageio.imwrite(image_path, frame)
            imageio.imwrite(mask_path, mask)

            image_paths.append(image_path)
            gt_paths.append(mask_path)

        capture.release()

    return image_paths, gt_paths


def get_echonet_pediatric_data(
    path: Union[os.PathLike, str], view: Literal["A4C", "PSAX"], download: bool = False
) -> str:
    """Obtain the EchoNet-Pediatric dataset.

    NOTE: 'torch_em' cannot download this dataset, as it requires individual registration and
    agreement to the EchoNet-Pediatric Research Use Agreement. Please follow these steps:
    - Visit https://echonet.github.io/pediatric/ and click on the download / access instructions.
    - Register (individually, per user) with the Stanford AIMI Center Shared Datasets Portal and
      agree to the Research Use Agreement (non-commercial research use only, no re-distribution).
    - Download the dataset and place it at `path`, so that it has the following structure, with one
      folder per view:
      `path/A4C/Videos`, `path/A4C/FileList.csv`, `path/A4C/VolumeTracings.csv`, and analogously for
      `path/PSAX`.

    Args:
        path: Filepath to a folder where the dataset is stored.
        view: The choice of view. Either 'A4C' or 'PSAX'.
        download: Whether to download the data if it is not present. This is not supported for this
            dataset, and setting it to True raises an error explaining the manual steps above.

    Returns:
        Filepath to the folder where the dataset is stored.
    """
    assert view in VIEWS, f"'{view}' is not a valid view choice for the EchoNet-Pediatric dataset."

    if download:
        raise NotImplementedError(
            "Download is set to True, but 'torch_em' cannot download the EchoNet-Pediatric dataset. "
            "See `get_echonet_pediatric_data` for the manual registration and download steps."
        )

    view_dir = os.path.join(path, view)
    if not (
        os.path.exists(os.path.join(view_dir, "FileList.csv"))
        and os.path.exists(os.path.join(view_dir, "VolumeTracings.csv"))
        and os.path.exists(os.path.join(view_dir, "Videos"))
    ):
        raise RuntimeError(
            f"Cannot find the EchoNet-Pediatric '{view}' data at '{view_dir}'. "
            "This dataset requires manual download, see `get_echonet_pediatric_data` for the steps."
        )

    return path


def get_echonet_pediatric_paths(
    path: Union[os.PathLike, str],
    view: Literal["A4C", "PSAX"],
    split: Literal["TRAIN", "VAL", "TEST", None] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the EchoNet-Pediatric data.

    Args:
        path: Filepath to a folder where the dataset is stored.
        view: The choice of view. Either 'A4C' or 'PSAX'.
        split: The choice of data split. Either 'TRAIN', 'VAL' or 'TEST'. By default, all splits are used.
        download: Whether to download the data if it is not present. See `get_echonet_pediatric_data`.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_echonet_pediatric_data(path, view, download)

    preprocessed_dir = os.path.join(path, "preprocessed")
    view_out_dir = os.path.join(preprocessed_dir, view)

    image_paths = natsorted(glob(os.path.join(view_out_dir, "*.tif")))
    image_paths = [p for p in image_paths if not p.endswith("_mask.tif")]
    if not image_paths:
        image_paths, _ = _preprocess_view(data_dir, view, preprocessed_dir)
        image_paths = natsorted(image_paths)

    gt_paths = [p.replace(".tif", "_mask.tif") for p in image_paths]
    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    if split is not None:
        assert split in SPLITS, f"'{split}' is not a valid split choice for the EchoNet-Pediatric dataset."
        image_paths = [p for p in image_paths if f"_{split}_" in os.path.basename(p)]
        gt_paths = [p for p in gt_paths if f"_{split}_" in os.path.basename(p)]

    return image_paths, gt_paths


def get_echonet_pediatric_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    view: Literal["A4C", "PSAX"],
    split: Literal["TRAIN", "VAL", "TEST", None] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EchoNet-Pediatric dataset for left ventricle segmentation in echocardiography videos.

    Args:
        path: Filepath to a folder where the dataset is stored.
        patch_shape: The patch shape to use for training.
        view: The choice of view. Either 'A4C' or 'PSAX'.
        split: The choice of data split. Either 'TRAIN', 'VAL' or 'TEST'. By default, all splits are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present. See `get_echonet_pediatric_data`.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_echonet_pediatric_paths(path, view, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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


def get_echonet_pediatric_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    view: Literal["A4C", "PSAX"],
    split: Literal["TRAIN", "VAL", "TEST", None] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EchoNet-Pediatric dataloader for left ventricle segmentation in echocardiography videos.

    Args:
        path: Filepath to a folder where the dataset is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        view: The choice of view. Either 'A4C' or 'PSAX'.
        split: The choice of data split. Either 'TRAIN', 'VAL' or 'TEST'. By default, all splits are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present. See `get_echonet_pediatric_data`.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_echonet_pediatric_dataset(path, patch_shape, view, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
