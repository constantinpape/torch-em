"""The AFIO dataset contains annotations for retinal vessel, artery and vein segmentation
in fundus images.

The dataset consists of 100 colour fundus images (86 macula-centred and 14 optic disc-centred)
acquired at the Armed Forces Institute of Ophthalmology (AFIO), Rawalpindi, Pakistan, and manually
annotated by four expert ophthalmologists. The publicly downloadable archive ships pixel-level
annotations for the retinal vessel network and for the separated artery / vein networks (plus a
combined "both" overlay). The optic nerve head, hard exudate and cotton-wool spot annotations that
are mentioned in the associated publication are not part of the downloadable archive.

The dataset is located at https://data.mendeley.com/datasets/3csr652p9y/2 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1016/j.dib.2020.105282.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/3csr652p9y/files/5c07e45a-5f3f-407b-8bdb-16332a84fa23/file_downloaded"  # noqa
CHECKSUM = "f0af3cc8714e1eaff5d2b5a3e0b77f8c6166a0d18322dd2685c2c6ed325fc230"

TASKS = ["vessels", "arteries", "veins", "both"]


def get_afio_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the AFIO dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "AV")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "AV.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _match_annotation(annotation_paths: List[str], task: str) -> str:
    # The annotations were exported from Illustrator by hand, so the suffixes have several typos,
    # e.g. 'arteries' / 'artery' / 'artry' / 'atertries' and 'veins' / 'vein' / 'veinds' / 'veisn'.
    # None of the artery variants contain the letter 'v', so this is used to disambiguate them
    # from the vein variants once the unambiguous 'vessels' and 'both' / 'map' suffixes are removed.
    vessel_paths = [p for p in annotation_paths if "vessel" in Path(p).stem.lower()]
    both_paths = [p for p in annotation_paths if "both" in Path(p).stem.lower() or "map" in Path(p).stem.lower()]
    remaining_paths = [p for p in annotation_paths if p not in vessel_paths and p not in both_paths]
    vein_paths = [p for p in remaining_paths if "v" in Path(p).stem.lower().split("--")[-1]]
    artery_paths = [p for p in remaining_paths if p not in vein_paths]

    task_to_paths = {"vessels": vessel_paths, "both": both_paths, "veins": vein_paths, "arteries": artery_paths}
    matches = task_to_paths[task]
    assert len(matches) == 1, f"Expected exactly one '{task}' annotation, found {matches}."
    return matches[0]


def get_afio_paths(
    path: Union[os.PathLike, str],
    task: Literal["vessels", "arteries", "veins", "both"] = "vessels",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AFIO data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of annotation. One of 'vessels', 'arteries', 'veins' or 'both'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_afio_data(path=path, download=download)

    assert task in TASKS, f"'{task}' is not a valid task. Please choose from {TASKS}."

    image_dirs = natsorted(glob(os.path.join(data_dir, "IM*")))
    assert len(image_dirs) == 100, f"Expected 100 image folders, found {len(image_dirs)}."

    neu_gt_dir = os.path.join(data_dir, "preprocessed", task)
    os.makedirs(neu_gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for image_dir in tqdm(image_dirs, desc=f"Preprocessing '{task}' labels"):
        name = os.path.basename(image_dir)
        # A couple of image folders have an extra (redundant) nesting level, e.g.
        # 'AV/IM000189/IM000189/IM000189.JPG' instead of 'AV/IM000189/IM000189.JPG',
        # so the raw image and annotations are searched for recursively.
        image_matches = glob(os.path.join(image_dir, "**", f"{name}.JPG"), recursive=True)
        assert len(image_matches) == 1, f"Expected exactly one raw image for '{name}', found {image_matches}."
        image_path = image_matches[0]

        annotation_paths = natsorted(glob(os.path.join(image_dir, "**", f"{name}--*.jpg"), recursive=True))
        raw_gt_path = _match_annotation(annotation_paths, task)

        gt_path = os.path.join(neu_gt_dir, f"{name}.tif")
        if not os.path.exists(gt_path):
            # The masks are lightly JPEG-compressed overlays with a bright background and a dark
            # foreground structure (vessel / artery / vein / combined network), so they are
            # binarized into a uint8 (0, 1) label map with the foreground being the darker pixels.
            raw_gt = imageio.imread(raw_gt_path)
            gray_gt = raw_gt.mean(axis=-1) if raw_gt.ndim == 3 else raw_gt
            binary_gt = (gray_gt < 128).astype(np.uint8)
            imageio.imwrite(gt_path, binary_gt)

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_afio_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: Literal["vessels", "arteries", "veins", "both"] = "vessels",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AFIO dataset for segmentation of retinal vessels, arteries and veins in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of annotation. One of 'vessels', 'arteries', 'veins' or 'both'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_afio_paths(path, task, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_afio_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    task: Literal["vessels", "arteries", "veins", "both"] = "vessels",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AFIO dataloader for segmentation of retinal vessels, arteries and veins in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The choice of annotation. One of 'vessels', 'arteries', 'veins' or 'both'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_afio_dataset(path, patch_shape, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
