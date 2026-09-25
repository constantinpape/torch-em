"""The ThyRLN-PUMCH dataset contains annotations for recurrent laryngeal nerve (RLN) segmentation
in endoscopic thyroidectomy video frames.

The dataset comprises frames extracted from 28 endoscopic thyroidectomy surgeries performed at
Peking Union Medical College Hospital (PUMCH), with 18,178 pixel-level RLN segmentation masks
(binary masks, manually annotated by two endocrine surgeons and reviewed / modified by senior
head-and-neck surgeons), plus 734 explicitly negative frames (no RLN visible). Annotations are
provided as per-frame binary masks paired with the raw jpg frames.

The dataset is located at https://www.kaggle.com/datasets/zhenghuaijin/thyrln-pumch and is
distributed under the CC0 1.0 license. Downloading it requires the Kaggle API and a Kaggle
account with an API token, see `torch_em.data.datasets.util.download_source_kaggle` for details.

This dataset is from the publication https://doi.org/10.1038/s41597-026-06961-6.
Please cite it if you use this dataset in your research.
"""

import os
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "zhenghuaijin/thyrln-pumch"


def get_thyrln_pumch_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ThyRLN-PUMCH data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the extracted ThyRLN-PUMCH data.
    """
    data_dir = os.path.join(path, "ThyRLN-PUMCH")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "thyrln-pumch.zip")
    util.unzip(zip_path=zip_path, dst=path)

    if not os.path.exists(data_dir):
        raise RuntimeError(f"The dataset could not be found at '{data_dir}' after extraction.")

    return data_dir


def _is_junk(fname):
    return fname.startswith("._")


def _strip_mp4_suffix(name):
    return name[:-4] if name.lower().endswith(".mp4") else name


def get_thyrln_pumch_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ThyRLN-PUMCH data.

    This returns the frames with an explicit pixel-level RLN mask ('labeled picture'). Frames are
    matched to their raw jpg either from the 'labeled picture' or, as a fallback, the (more complete)
    'original picture' folder, since not every annotated frame is duplicated into 'labeled picture'.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_thyrln_pumch_data(path, download)

    labeled_dir = os.path.join(data_dir, "labeled picture")
    original_dir = os.path.join(data_dir, "original picture")

    image_paths, mask_paths = [], []
    for surgery in natsorted(os.listdir(labeled_dir)):
        surgery_dir = os.path.join(labeled_dir, surgery)
        if not os.path.isdir(surgery_dir):
            continue

        mask_dir = os.path.join(surgery_dir, "mask")
        if not os.path.isdir(mask_dir):
            continue

        labeled_video_dirs = {
            _strip_mp4_suffix(d): d for d in os.listdir(surgery_dir)
            if d != "mask" and os.path.isdir(os.path.join(surgery_dir, d))
        }

        original_surgery_dir = os.path.join(original_dir, surgery)
        original_video_dirs = {
            _strip_mp4_suffix(d): d for d in os.listdir(original_surgery_dir)
        } if os.path.isdir(original_surgery_dir) else {}

        for mask_video_dir in natsorted(os.listdir(mask_dir)):
            key = _strip_mp4_suffix(mask_video_dir)
            mask_video_path = os.path.join(mask_dir, mask_video_dir)
            if not os.path.isdir(mask_video_path):
                continue

            labeled_img_dir = os.path.join(surgery_dir, labeled_video_dirs.get(key, ""))
            original_img_dir = os.path.join(original_surgery_dir, original_video_dirs.get(key, ""))

            for mfname in natsorted(os.listdir(mask_video_path)):
                if _is_junk(mfname) or not mfname.endswith("_mask.png"):
                    continue

                stem = mfname[:-len("_mask.png")]
                image_fname = f"{stem}.jpg"

                image_path = os.path.join(labeled_img_dir, image_fname)
                if not os.path.exists(image_path):
                    image_path = os.path.join(original_img_dir, image_fname)
                if not os.path.exists(image_path):
                    continue

                image_paths.append(image_path)
                mask_paths.append(os.path.join(mask_video_path, mfname))

    if len(image_paths) == 0 or len(image_paths) != len(mask_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, mask_paths


def get_thyrln_pumch_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ThyRLN-PUMCH dataset for recurrent laryngeal nerve segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, mask_paths = get_thyrln_pumch_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=mask_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_thyrln_pumch_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ThyRLN-PUMCH dataloader for recurrent laryngeal nerve segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_thyrln_pumch_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
