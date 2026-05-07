"""The PanopTILs dataset contains panoptic segmentation annotations for tumor-infiltrating
lymphocyte (TIL) assessment in H&E stained breast cancer histopathology images.

The dataset provides 1,349 ROIs (1024x1024 pixels at 0.25 MPP) from TCGA invasive breast
cancer cases with three annotation types: nuclei instance segmentation, nuclei semantic
segmentation (type), and tissue semantic segmentation.

Nuclei classes: background (0), neoplastic (1), stromal (2), inflammatory (3),
epithelial (4), other (5), unknown (6).

Tissue classes: background (0), tumor (1), stroma (2), epithelium (3),
junk/debris (4), blood (5), other (6).

NOTE: This uses the refined version from https://huggingface.co/datasets/histolytics-hub/panoptils_refined.
The original dataset is at https://sites.google.com/view/panoptils/.
This dataset is from the publication https://doi.org/10.1038/s41523-024-00663-1.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/histolytics-hub/panoptils_refined/resolve/main/panoptils_refined.parquet"

LABEL_CHOICES = ["instances", "type", "semantic"]


def _create_images_from_parquet(path):
    """Extract images and masks from the parquet file and save as TIF."""
    import imageio.v3 as imageio
    import pandas as pd
    from io import BytesIO
    from PIL import Image
    from tqdm import tqdm

    image_dir = os.path.join(path, "images")
    inst_dir = os.path.join(path, "instances")
    type_dir = os.path.join(path, "types")
    sem_dir = os.path.join(path, "semantic")
    for d in [image_dir, inst_dir, type_dir, sem_dir]:
        os.makedirs(d, exist_ok=True)

    parquet_path = os.path.join(path, "panoptils_refined.parquet")
    df = pd.read_parquet(parquet_path)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting PanopTILs images"):
        sample_id = f"{idx:05d}"
        img_path = os.path.join(image_dir, f"{sample_id}.tif")

        if os.path.exists(img_path):
            continue

        img = np.array(Image.open(BytesIO(row["image"])).convert("RGB"))
        inst = np.array(Image.open(BytesIO(row["inst"])))
        ntype = np.array(Image.open(BytesIO(row["type"])))
        sem = np.array(Image.open(BytesIO(row["sem"])))

        imageio.imwrite(img_path, img, compression="zlib")
        imageio.imwrite(os.path.join(inst_dir, f"{sample_id}.tif"), inst.astype("uint32"), compression="zlib")
        imageio.imwrite(os.path.join(type_dir, f"{sample_id}.tif"), ntype.astype("uint8"), compression="zlib")
        imageio.imwrite(os.path.join(sem_dir, f"{sample_id}.tif"), sem.astype("uint8"), compression="zlib")


def get_panoptils_data(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> str:
    """Download the PanopTILs dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    parquet_path = os.path.join(path, "panoptils_refined.parquet")
    if not os.path.exists(parquet_path):
        os.makedirs(path, exist_ok=True)
        util.download_source(path=parquet_path, url=URL, download=download, checksum=None)

    image_dir = os.path.join(path, "images")
    if not os.path.exists(image_dir) or len(glob(os.path.join(image_dir, "*.tif"))) == 0:
        _create_images_from_parquet(path)

    return path


def get_panoptils_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["instances", "type", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PanopTILs data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        label_choice: The type of labels to use. One of 'instances' (nuclei instance segmentation),
            'type' (nuclei semantic segmentation), or 'semantic' (tissue semantic segmentation).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    from natsort import natsorted

    assert label_choice in LABEL_CHOICES, f"'{label_choice}' is not valid. Choose from {LABEL_CHOICES}."

    get_panoptils_data(path, download)

    label_dir = label_choice if label_choice != "type" else "types"
    image_paths = natsorted(glob(os.path.join(path, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(path, label_dir, "*.tif")))

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_panoptils_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    label_choice: Literal["instances", "type", "semantic"] = "instances",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the PanopTILs dataset for panoptic segmentation of tumor-infiltrating lymphocytes.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        label_choice: The type of labels to use. One of 'instances' (nuclei instance segmentation),
            'type' (nuclei semantic segmentation), or 'semantic' (tissue semantic segmentation).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_panoptils_paths(path, label_choice, download)

    if label_choice == "instances":
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True,
        )

    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=label_choice != "instances",
        **kwargs,
    )


def get_panoptils_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    label_choice: Literal["instances", "type", "semantic"] = "instances",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the PanopTILs dataloader for panoptic segmentation of tumor-infiltrating lymphocytes.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The type of labels to use. One of 'instances' (nuclei instance segmentation),
            'type' (nuclei semantic segmentation), or 'semantic' (tissue semantic segmentation).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_panoptils_dataset(path, patch_shape, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
