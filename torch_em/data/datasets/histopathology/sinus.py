"""The SiNuS dataset contains annotations for singular nucleus segmentation in
Dual In Situ Hybridization (DISH) images of breast cancer tissue.

NOTE: This dataset is sparsely annotated. It contains annotations for expert-selected
singular nuclei suitable for HER2 grading, rather than for all nuclei in each image.

The dataset is located at https://data.mendeley.com/datasets/gtjrgwbntc/2.
This dataset is from the publication https://doi.org/10.1016/j.dib.2026.112934.
Please cite it if you use this dataset for your research.
"""

import json
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np
import imageio.v3 as imageio
from skimage.draw import polygon

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-api/zip/gtjrgwbntc/download/2"
CHECKSUM = "aecd1399192ee511ba29f6c23e6f858b4e6a8328028c1ae91f9ee5a826728c5c"


def get_sinus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SiNuS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded.
    """
    data_dir = os.path.join(path, "SiNuS A Comprehensive Dataset for Singular Nuclei", "SiNuS")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "sinus.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _create_instance_labels(annotation_path: str, label_path: str) -> None:
    with open(annotation_path) as f:
        annotation = json.load(f)["annotation"]

    shape = (annotation["size"]["height"], annotation["size"]["width"])
    labels = np.zeros(shape, dtype="uint16")
    for label_id, annotated_object in enumerate(annotation["objects"], 1):
        points = np.asarray(annotated_object["points"]["exterior"])
        rr, cc = polygon(points[:, 1], points[:, 0], shape=shape)
        labels[rr, cc] = label_id

        for interior in annotated_object["points"]["interior"]:
            points = np.asarray(interior)
            rr, cc = polygon(points[:, 1], points[:, 0], shape=shape)
            labels[rr, cc] = 0

    imageio.imwrite(label_path, labels, compression="zlib")


def get_sinus_paths(
    path: Union[os.PathLike, str],
    annotation_choice: Literal["inclusive", "exclusive"] = "inclusive",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SiNuS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        annotation_choice: The annotation selection. The inclusive annotations contain nuclei selected by at least
            one expert, while the exclusive annotations contain nuclei selected by all experts.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if annotation_choice not in ("inclusive", "exclusive"):
        raise ValueError(f"'{annotation_choice}' is not a valid annotation choice.")

    data_dir = get_sinus_data(path, download)
    raw_paths = natsorted(glob(os.path.join(data_dir, "Original", "*.JPG")))

    selection = f"{annotation_choice.capitalize()} Nuclei Selection"
    suffix = "ins" if annotation_choice == "inclusive" else "ens"
    annotation_paths = natsorted(glob(os.path.join(data_dir, selection, "Image *", f"*_annotation_{suffix}.json")))

    label_dir = os.path.join(data_dir, "preprocessed_labels", annotation_choice)
    os.makedirs(label_dir, exist_ok=True)
    label_paths = []
    for annotation_path in tqdm(annotation_paths, desc=f"Preprocessing {annotation_choice} SiNuS labels"):
        image_name = Path(annotation_path).name.split("_annotation")[0]
        label_path = os.path.join(label_dir, f"{image_name}.tif")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue

        _create_instance_labels(annotation_path, label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(Path(raw_path).stem == Path(label_path).stem for raw_path, label_path in zip(raw_paths, label_paths))

    return raw_paths, label_paths


def get_sinus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotation_choice: Literal["inclusive", "exclusive"] = "inclusive",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SiNuS dataset for singular nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        annotation_choice: The annotation selection. The inclusive annotations contain nuclei selected by at least
            one expert, while the exclusive annotations contain nuclei selected by all experts.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_sinus_paths(path, annotation_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_sinus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotation_choice: Literal["inclusive", "exclusive"] = "inclusive",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the SiNuS dataloader for singular nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation_choice: The annotation selection. The inclusive annotations contain nuclei selected by at least
            one expert, while the exclusive annotations contain nuclei selected by all experts.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sinus_dataset(
        path=path,
        patch_shape=patch_shape,
        annotation_choice=annotation_choice,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
