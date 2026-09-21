"""The Maternal-Fetal Ultrasound Video dataset contains annotations for segmentation of the pubic
symphysis and the fetal head in transperineal intrapartum ultrasound videos.

The dataset consists of 774 videos (68,106 frames in total) collected from three medical centers
(JNU, SYSU and SMU) with different ultrasound devices. A subset of the frames is manually annotated
with pixel-level semantic segmentation masks for two structures: the pubic symphysis (class 1) and
the fetal head (class 2). This wrapper extracts the annotated frames from the videos (using the frame
index encoded in the annotation filename) and pairs them with the corresponding segmentation mask, so
that the data can be used as a regular 2d image segmentation dataset.

The dataset ships three splits with slightly different storage conventions:
- 'train': videos with annotated frames are stored in 'train/pos/<video_id>/', with the video itself
  ('<video_id>.avi') and the per-frame masks ('mask/<video_id>_<frame_idx>_*.png') in the same folder.
  The masks use pixel values 7 (pubic symphysis) and 8 (fetal head), which are remapped to 1 and 2
  respectively to match the 'test' and 'val' splits (see NOTE below).
- 'test': videos are stored in 'test/video/Source*_*/<video_id>.avi' and the per-frame masks in
  'test/seg_label/<video_id>_<frame_idx>.npy' (pixel values 0, 1, 2).
- 'val': videos are stored in 'val/videos/<video_id>_<frame_idx>.avi' (the frame index is part of the
  video filename) and the single annotated mask per video in 'val/label_seg/<video_id>_seg.npy'
  (pixel values 0, 1, 2).

NOTE: The 'train' split masks use different pixel values (7, 8) than the 'test' and 'val' splits
(1, 2). This wrapper remaps the 'train' masks to the same (1, 2) convention when caching the frames,
so that all splits share one label convention: 0 (background), 1 (pubic symphysis), 2 (fetal head).

This dataset is located at https://doi.org/10.5281/zenodo.16869288.
This dataset is from the publication https://doi.org/10.1038/s41597-026-06900-5.
The dataset is licensed under CC-BY-4.0.
Please cite the publication above if you use this dataset for your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16869288/files/IUGC2024.rar"
CHECKSUM = "048730551419192aeaf7b23bcf6dbdcda26ab2d4acde4f5007b350128d188052"

SPLITS = ["train", "val", "test"]

# The 'train' split masks use 7 / 8 for the two foreground classes, the 'test' / 'val' splits use 1 / 2.
TRAIN_LABEL_MAP = {0: 0, 7: 1, 8: 2}


def get_maternal_fetal_us_video_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Maternal-Fetal Ultrasound Video dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "IUGC2024")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    rar_path = os.path.join(path, "IUGC2024.rar")
    util.download_source(path=rar_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_rarfile(rar_path=rar_path, dst=path)

    return data_dir


def _read_frame(video_path, frame_idx):
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Could not read frame {frame_idx} from '{video_path}'.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _cache_train_split(data_dir, cache_dir):
    video_dirs = natsorted(glob(os.path.join(data_dir, "train", "pos", "*")))

    image_paths, label_paths = [], []
    for video_dir in tqdm(video_dirs, desc="Caching 'train' frames"):
        video_id = os.path.basename(video_dir)
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        mask_paths = natsorted(glob(os.path.join(video_dir, "mask", f"{video_id}_*.png")))

        for mask_path in mask_paths:
            fname = os.path.basename(mask_path)
            rest = fname[len(video_id):-len(".png")].strip("_")
            frame_idx = int(rest.split("_")[0])

            image_path = os.path.join(cache_dir, f"train_{video_id}_{frame_idx}_image.tif")
            label_path = os.path.join(cache_dir, f"train_{video_id}_{frame_idx}_label.tif")
            image_paths.append(image_path)
            label_paths.append(label_path)
            if os.path.exists(image_path) and os.path.exists(label_path):
                continue

            frame = _read_frame(video_path, frame_idx)
            mask = imageio.imread(mask_path)
            label = np.vectorize(TRAIN_LABEL_MAP.get)(mask).astype("uint8")

            imageio.imwrite(image_path, frame, compression="zlib")
            imageio.imwrite(label_path, label, compression="zlib")

    return image_paths, label_paths


def _cache_test_split(data_dir, cache_dir):
    mask_paths = natsorted(glob(os.path.join(data_dir, "test", "seg_label", "*.npy")))
    video_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob(os.path.join(data_dir, "test", "video", "Source*_*", "*.avi"))
    }

    image_paths, label_paths = [], []
    for mask_path in tqdm(mask_paths, desc="Caching 'test' frames"):
        fname = os.path.splitext(os.path.basename(mask_path))[0]
        match = re.match(r"(.+)_(\d+)$", fname)
        video_id, frame_idx = match.group(1), int(match.group(2))

        video_path = video_paths.get(video_id)
        if video_path is None:
            raise RuntimeError(f"Could not find the video for '{video_id}' in the 'test' split.")

        image_path = os.path.join(cache_dir, f"test_{video_id}_{frame_idx}_image.tif")
        label_path = os.path.join(cache_dir, f"test_{video_id}_{frame_idx}_label.tif")
        image_paths.append(image_path)
        label_paths.append(label_path)
        if os.path.exists(image_path) and os.path.exists(label_path):
            continue

        frame = _read_frame(video_path, frame_idx)
        label = np.load(mask_path).astype("uint8")

        imageio.imwrite(image_path, frame, compression="zlib")
        imageio.imwrite(label_path, label, compression="zlib")

    return image_paths, label_paths


def _cache_val_split(data_dir, cache_dir):
    mask_paths = natsorted(glob(os.path.join(data_dir, "val", "label_seg", "*_seg.npy")))
    video_paths = {
        re.match(r"(.+)_(\d+)$", os.path.splitext(os.path.basename(p))[0]).group(1): p
        for p in glob(os.path.join(data_dir, "val", "videos", "*.avi"))
    }

    image_paths, label_paths = [], []
    for mask_path in tqdm(mask_paths, desc="Caching 'val' frames"):
        video_id = os.path.basename(mask_path)[:-len("_seg.npy")]

        video_path = video_paths.get(video_id)
        if video_path is None:
            raise RuntimeError(f"Could not find the video for '{video_id}' in the 'val' split.")

        frame_idx = int(re.match(r".+_(\d+)$", os.path.splitext(os.path.basename(video_path))[0]).group(1))

        image_path = os.path.join(cache_dir, f"val_{video_id}_{frame_idx}_image.tif")
        label_path = os.path.join(cache_dir, f"val_{video_id}_{frame_idx}_label.tif")
        image_paths.append(image_path)
        label_paths.append(label_path)
        if os.path.exists(image_path) and os.path.exists(label_path):
            continue

        frame = _read_frame(video_path, frame_idx)
        label = np.load(mask_path).astype("uint8")

        imageio.imwrite(image_path, frame, compression="zlib")
        imageio.imwrite(label_path, label, compression="zlib")

    return image_paths, label_paths


def get_maternal_fetal_us_video_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test", "all"] = "all", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Maternal-Fetal Ultrasound Video data.

    This extracts the annotated frames from the videos and caches them (together with their
    segmentation masks) as tif files, so that repeated calls avoid re-decoding the videos.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'val', 'test' or 'all' (uses all splits).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split == "all":
        splits = SPLITS
    elif split in SPLITS:
        splits = [split]
    else:
        raise ValueError(f"'{split}' is not a valid split. Choose from {SPLITS + ['all']}.")

    data_dir = get_maternal_fetal_us_video_data(path, download)

    cache_dir = os.path.join(path, "frames")
    os.makedirs(cache_dir, exist_ok=True)

    cache_funcs = {"train": _cache_train_split, "test": _cache_test_split, "val": _cache_val_split}

    image_paths, label_paths = [], []
    for this_split in splits:
        this_image_paths, this_label_paths = cache_funcs[this_split](data_dir, cache_dir)
        image_paths.extend(this_image_paths)
        label_paths.extend(this_label_paths)

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_maternal_fetal_us_video_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Maternal-Fetal Ultrasound Video dataset for pubic symphysis and fetal head segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val', 'test' or 'all' (uses all splits).
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_maternal_fetal_us_video_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_maternal_fetal_us_video_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Maternal-Fetal Ultrasound Video dataloader for pubic symphysis and fetal head segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val', 'test' or 'all' (uses all splits).
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_maternal_fetal_us_video_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
