"""EchoNet-Dynamic contains annotations for left ventricle segmentation in apical-4-chamber
echocardiography videos, together with cardiac function labels (ejection fraction, end-systolic
and end-diastolic volume).

The dataset comprises 10,030 videos, collected at Stanford Health Care. Each video only has expert
tracings for the two labeled frames (end-diastole and end-systole) used to compute the left
ventricular ejection fraction, not dense per-frame masks. The dataset is located at
https://echonet.github.io/dynamic/ and distributed through the Stanford AIMI Center Shared Datasets
Portal under a non-commercial research use agreement: registration as an individual user is
required, re-distribution (including sharing the download link) is forbidden, and re-identification
attempts are prohibited. This module cannot download the dataset automatically; see
`get_echonet_dynamic_data` for the manual steps.

This dataset is from the publication https://doi.org/10.1038/s41586-020-2145-8 (cite the DOI
https://doi.org/10.71718/yqp5-y078 for the data itself). Please cite them if you use this dataset in
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


def _find_video_path(videos_dir, file_name):
    stem = os.path.splitext(file_name)[0]
    for candidate in (file_name, f"{stem}.avi", f"{stem}.mp4"):
        candidate_path = os.path.join(videos_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _preprocess_inputs(data_dir, preprocessed_dir):
    import cv2

    file_list = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
    tracings = pd.read_csv(os.path.join(data_dir, "VolumeTracings.csv"))

    videos_dir = os.path.join(data_dir, "Videos")
    os.makedirs(preprocessed_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for _, row in tqdm(file_list.iterrows(), total=len(file_list), desc="Preprocess EchoNet-Dynamic"):
        file_name = str(row["FileName"])
        stem = os.path.splitext(file_name)[0]
        split = str(row["Split"]).upper()

        video_tracings = tracings[tracings["FileName"] == file_name]
        if video_tracings.empty:
            video_tracings = tracings[tracings["FileName"] == stem + ".avi"]
        if video_tracings.empty:
            continue

        video_path = _find_video_path(videos_dir, file_name)
        if video_path is None:
            continue

        capture = cv2.VideoCapture(video_path)

        for frame_idx, frame_tracings in video_tracings.groupby("Frame"):
            image_path = os.path.join(preprocessed_dir, f"{stem}_{split}_{frame_idx}.tif")
            mask_path = os.path.join(preprocessed_dir, f"{stem}_{split}_{frame_idx}_mask.tif")

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


def get_echonet_dynamic_data(path: Union[os.PathLike, str]) -> str:
    """Obtain the EchoNet-Dynamic dataset.

    NOTE: 'torch_em' cannot download this dataset, as it requires individual registration and
    agreement to the EchoNet-Dynamic Research Use Agreement. Please follow these steps:
    - Visit https://echonet.github.io/dynamic/ and follow the access instructions.
    - Register (individually, per user) with the Stanford AIMI Center Shared Datasets Portal
      (https://stanford.redivis.com/datasets/66s1-2hsmzj5rn) and agree to the Research Use Agreement
      (non-commercial research use only, no re-distribution).
    - Download the dataset and place it at `path`, so that it has the following structure:
      `path/Videos`, `path/FileList.csv`, and `path/VolumeTracings.csv`.

    Args:
        path: Filepath to a folder where the dataset is stored.

    Returns:
        Filepath to the folder where the dataset is stored.
    """
    if not (
        os.path.exists(os.path.join(path, "FileList.csv"))
        and os.path.exists(os.path.join(path, "VolumeTracings.csv"))
        and os.path.exists(os.path.join(path, "Videos"))
    ):
        raise RuntimeError(
            f"Cannot find the EchoNet-Dynamic data at '{path}'. "
            "This dataset requires manual download, see `get_echonet_dynamic_data` for the steps."
        )

    return path


def get_echonet_dynamic_paths(
    path: Union[os.PathLike, str], split: Literal["TRAIN", "VAL", "TEST", None] = None,
) -> Tuple[List[str], List[str]]:
    """Get paths to the EchoNet-Dynamic data.

    Args:
        path: Filepath to a folder where the dataset is stored.
        split: The choice of data split. Either 'TRAIN', 'VAL' or 'TEST'. By default, all splits are used.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_echonet_dynamic_data(path)

    preprocessed_dir = os.path.join(path, "preprocessed")

    image_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.tif")))
    image_paths = [p for p in image_paths if not p.endswith("_mask.tif")]
    if not image_paths:
        image_paths, _ = _preprocess_inputs(data_dir, preprocessed_dir)
        image_paths = natsorted(image_paths)

    gt_paths = natsorted([p.replace(".tif", "_mask.tif") for p in image_paths])
    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    if split is not None:
        assert split in SPLITS, f"'{split}' is not a valid split choice for the EchoNet-Dynamic dataset."
        image_paths = [p for p in image_paths if f"_{split}_" in os.path.basename(p)]
        gt_paths = [p for p in gt_paths if f"_{split}_" in os.path.basename(p)]

    return image_paths, gt_paths


def get_echonet_dynamic_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["TRAIN", "VAL", "TEST", None] = None,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the EchoNet-Dynamic dataset for left ventricle segmentation in echocardiography videos.

    Args:
        path: Filepath to a folder where the dataset is stored.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'TRAIN', 'VAL' or 'TEST'. By default, all splits are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_echonet_dynamic_paths(path, split)

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


def get_echonet_dynamic_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["TRAIN", "VAL", "TEST", None] = None,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EchoNet-Dynamic dataloader for left ventricle segmentation in echocardiography videos.

    Args:
        path: Filepath to a folder where the dataset is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'TRAIN', 'VAL' or 'TEST'. By default, all splits are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_echonet_dynamic_dataset(path, patch_shape, split, resize_inputs, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
