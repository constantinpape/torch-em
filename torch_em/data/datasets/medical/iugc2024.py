"""IUGC 2024 is the Intrapartum Ultrasound Grand Challenge dataset for fetal head and
pubic symphysis segmentation in transperineal ultrasound videos, recorded to assess the
progression of labor. Videos are stored per-frame and only a subset of frames per video
is annotated with a pixel-wise, 2-class (fetal head, pubic symphysis) segmentation mask.

The dataset is located at
https://www.kaggle.com/datasets/aspirexxx/iugc-ultrasound-video-dataset-miccai-2024.
This dataset is from the publication https://doi.org/10.1007/978-3-031-96318-6_1.
Please cite it if you use this dataset for your research.

NOTE: The dataset stores raw frames as videos (`.avi`) instead of individual images, so
this module extracts the specific annotated frame(s) out of each video and caches it to
disk as an image next to the corresponding mask. Only the "val" and "test" splits are
supported, as the "train" split stores masks in a differently structured per-video layout
that could not be verified with the same approach.
"""

import os
import csv
from glob import glob
from typing import Union, Tuple, List, Literal

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "aspirexxx/iugc-ultrasound-video-dataset-miccai-2024"

# The dataset does not expose a stable top-level folder layout, so the per-split prefixes
# below were resolved once via the Kaggle Files API (`KaggleApi.dataset_list_files`).
SPLIT_PREFIXES = {
    "val": "DatasetV3/val-20251119T054616Z-1-001/val/",
    "test": "DatasetV3/test-20251119T054614Z-1-001/test/",
}


def _get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        msg = "Please install the Kaggle API. You can do this using 'pip install kaggle'. "
        msg += "After you have installed kaggle, you would need an API token. "
        msg += "Follow the instructions at https://www.kaggle.com/docs/api."
        raise ModuleNotFoundError(msg)

    api = KaggleApi()
    api.authenticate()
    return api


def _list_seg_filenames(api, prefix):
    seg_prefix = f"{prefix}seg/"
    filenames = []
    token = None
    while True:
        response = api.dataset_list_files(KAGGLE_DATASET_NAME, page_token=token, page_size=500)
        names = [f.name for f in response.files]
        for name in names:
            if name.startswith(seg_prefix) and name.endswith(".png"):
                filenames.append(os.path.basename(name))

        # The listing is alphabetically ordered, so once we have moved past the "seg/" folder
        # (and already collected some file names) we can stop early.
        if filenames and not any(name.startswith(seg_prefix) for name in names):
            break

        token = response.next_page_token
        if not token:
            break

    return filenames


def get_iugc2024_data(path: Union[os.PathLike, str], split: Literal["val", "test"], download: bool = False) -> str:
    """Download the IUGC 2024 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if split not in SPLIT_PREFIXES:
        raise ValueError(f"'{split}' is not a supported split. Choose one of {list(SPLIT_PREFIXES.keys())}.")

    data_dir = os.path.join(path, split)
    videos_dir = os.path.join(data_dir, "videos")
    seg_dir = os.path.join(data_dir, "seg")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    info_path = os.path.join(data_dir, "seg_info.csv")
    if os.path.exists(info_path):
        return data_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    prefix = SPLIT_PREFIXES[split]
    api = _get_kaggle_api()

    api.dataset_download_file(KAGGLE_DATASET_NAME, f"{prefix}seg/seg_info.csv", path=data_dir, quiet=False)

    with open(info_path) as f:
        rows = list(csv.DictReader(f))

    seg_filenames = _list_seg_filenames(api, prefix)

    for row in rows:
        video_name = row["filename"]
        video_path = os.path.join(videos_dir, video_name)
        if not os.path.exists(video_path):
            api.dataset_download_file(KAGGLE_DATASET_NAME, f"{prefix}videos/{video_name}", path=videos_dir, quiet=False)

        video_stem = os.path.splitext(video_name)[0]
        matches = [name for name in seg_filenames if name.startswith(video_stem)]
        for mask_name in matches:
            mask_path = os.path.join(seg_dir, mask_name)
            if os.path.exists(mask_path):
                continue
            api.dataset_download_file(KAGGLE_DATASET_NAME, f"{prefix}seg/{mask_name}", path=seg_dir, quiet=False)

    return data_dir


def get_iugc2024_paths(
    path: Union[os.PathLike, str], split: Literal["val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the IUGC 2024 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    import cv2

    data_dir = get_iugc2024_data(path=path, split=split, download=download)
    videos_dir = os.path.join(data_dir, "videos")
    seg_dir = os.path.join(data_dir, "seg")

    frames_dir = os.path.join(data_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    mask_paths = sorted(glob(os.path.join(seg_dir, "*.png")))

    image_paths, gt_paths = [], []
    for mask_path in mask_paths:
        mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
        frame_path = os.path.join(frames_dir, f"{mask_stem}.tif")

        image_paths.append(frame_path)
        gt_paths.append(mask_path)
        if os.path.exists(frame_path):
            continue

        # The mask stem is either "<video_stem>" (val split) or "<video_stem>_<frame_index>" (test split).
        video_candidates = glob(os.path.join(videos_dir, f"{mask_stem}.avi"))
        if video_candidates:
            video_path = video_candidates[0]
            frame_index = 0
        else:
            video_stem, frame_index = mask_stem.rsplit("_", 1)
            video_path = os.path.join(videos_dir, f"{video_stem}.avi")
            frame_index = int(frame_index)

        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        capture.release()
        if not success:
            raise RuntimeError(f"Could not read frame {frame_index} from '{video_path}'.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imageio.imwrite(frame_path, frame, compression="zlib")

    return image_paths, gt_paths


def get_iugc2024_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the IUGC 2024 dataset for fetal head and pubic symphysis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_iugc2024_paths(path, split, download)

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


def get_iugc2024_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the IUGC 2024 dataloader for fetal head and pubic symphysis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_iugc2024_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
