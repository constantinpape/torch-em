"""The STARE dataset contains annotations for retinal vessel segmentation in fundus images.

STARE (STructured Analysis of the Retina) is located at http://cecas.clemson.edu/~ahoover/stare/.
It provides 20 fundus images along with two independent sets of hand-labeled vessel maps, created
by Adam Hoover ('ah') and Valentina Kouznetsova ('vk').

The dataset is from the publication https://doi.org/10.1109/42.845178.
Please cite it if you use this dataset for your research.
"""

import os
import gzip
import shutil
from glob import glob
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "images": "http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar",
    "ah": "http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar",
    "vk": "http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar",
}

CHECKSUM = {
    "images": "5f7b509b6067cad4f1be84933145d783de9ec087b3eaaf4db0103dd0144dd433",
    "ah": "ebf2f1e17ca955f24579d9edd990e2dae79a5c82def69f0985d8e24f826ddd2f",
    "vk": "47474a701536b0cfdb369fdce012be36141e9f44d80387f0179446b5cb0f5576",
}


def _unpack_gz_files(dir_path):
    gz_paths = sorted(glob(os.path.join(dir_path, "*.gz")))
    for gz_path in gz_paths:
        out_path = gz_path[:-len(".gz")]
        if os.path.exists(out_path):
            continue
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_stare_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the STARE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    images_dir = os.path.join(path, "images")
    if os.path.exists(images_dir):
        return path

    os.makedirs(path, exist_ok=True)

    for name, target_dir in [("images", "images"), ("ah", "labels-ah"), ("vk", "labels-vk")]:
        tar_path = os.path.join(path, f"{name}.tar")
        util.download_source(path=tar_path, url=URL[name], download=download, checksum=CHECKSUM[name])

        extract_dir = os.path.join(path, target_dir)
        os.makedirs(extract_dir, exist_ok=True)
        shutil.unpack_archive(tar_path, extract_dir, format="tar")

        _unpack_gz_files(extract_dir)

    return path


def get_stare_paths(
    path: Union[os.PathLike, str],
    annotator: Literal["ah", "vk"] = "ah",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the STARE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotator: The choice of annotator for the ground-truth vessel maps. There are two independent
            manual annotations, provided by 'ah' (Adam Hoover) and 'vk' (Valentina Kouznetsova).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_stare_data(path=path, download=download)

    assert annotator in ["ah", "vk"], f"'{annotator}' is not a valid annotator choice."

    image_paths = sorted(glob(os.path.join(data_dir, "images", "*.ppm")))
    gt_paths = sorted(glob(os.path.join(data_dir, f"labels-{annotator}", f"*.{annotator}.ppm")))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_stare_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotator: Literal["ah", "vk"] = "ah",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the STARE dataset for segmentation of retinal blood vessels in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotator: The choice of annotator for the ground-truth vessel maps.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_stare_paths(path=path, annotator=annotator, download=download)

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


def get_stare_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotator: Literal["ah", "vk"] = "ah",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the STARE dataloader for segmentation of retinal blood vessels in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotator: The choice of annotator for the ground-truth vessel maps.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_stare_dataset(path, patch_shape, annotator, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
