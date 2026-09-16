"""The Intraretinal Cystoid Fluid dataset contains annotations for cystoid macular edema (CME)
segmentation in optical coherence tomography (OCT) images of diabetic macular edema (DME) patients.

The dataset is located at https://www.kaggle.com/datasets/zeeshanahmed13/intraretinal-cystoid-fluid.
This dataset is from the publication https://doi.org/10.1002/ima.22662.
Please cite it if you use this dataset for your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_intraretinal_cystoid_fluid_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Intraretinal Cystoid Fluid dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "2021-training-data-ZA", "2021-training-data-ZA")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(
        path=path, dataset_name="zeeshanahmed13/intraretinal-cystoid-fluid", download=download,
    )
    zip_path = os.path.join(path, "intraretinal-cystoid-fluid.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_intraretinal_cystoid_fluid_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Intraretinal Cystoid Fluid data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_intraretinal_cystoid_fluid_data(path=path, download=download)

    neu_gt_dir = os.path.join(path, "preprocessed_masks")
    os.makedirs(neu_gt_dir, exist_ok=True)

    sample_dirs = sorted(glob(os.path.join(data_dir, "*")))

    image_paths, gt_paths = [], []
    for sample_dir in tqdm(sample_dirs, desc="Preprocessing labels"):
        if not os.path.isdir(sample_dir):
            continue

        image_path = glob(os.path.join(sample_dir, "images", "*"))[0]
        mask_paths = sorted(glob(os.path.join(sample_dir, "masks", "*")))

        # A single sample folder ('815DME_F') ships a stray extra mask file that belongs to another
        # sample ('MASK-DME511.png'). Filter to the mask matching the sample's leading numeric id.
        if len(mask_paths) > 1:
            sample_name = os.path.basename(sample_dir)
            prefix_match = re.match(r"^(\d+)", sample_name)
            if prefix_match is not None:
                prefix = prefix_match.group(1)
                mask_paths = [p for p in mask_paths if os.path.basename(p).startswith(prefix)]

        mask_path = mask_paths[0]

        neu_gt_path = os.path.join(neu_gt_dir, f"{Path(mask_path).stem}.tif")
        if not os.path.exists(neu_gt_path):
            gt = imageio.imread(mask_path)
            gt = (gt > 0).astype("uint8")
            imageio.imwrite(neu_gt_path, gt, compression="zlib")

        image_paths.append(image_path)
        gt_paths.append(neu_gt_path)

    return image_paths, gt_paths


def get_intraretinal_cystoid_fluid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Intraretinal Cystoid Fluid dataset for CME segmentation in OCT images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_intraretinal_cystoid_fluid_paths(path, download)

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


def get_intraretinal_cystoid_fluid_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Intraretinal Cystoid Fluid dataloader for CME segmentation in OCT images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_intraretinal_cystoid_fluid_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
