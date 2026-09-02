"""This dataset ontains annotation for cell segmentation in
differential interference contrast (DIC) microscopy images.

This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2024.109151.
Please cite it if you use this dataset in your research.
"""

import os
from tqdm import tqdm
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, Optional, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from .. import util
from .livecell import _annotations_to_instances


URL = "https://zenodo.org/records/13120679/files/2021-11-15_HepG2_Calcein_AM.zip"
CHECKSUM = "42b939d01c5fc2517dc3ad34bde596ac38dbeba2a96173f37e1b6dfe14cbe3a2"


def get_dic_hepg2_data(path: Union[str, os.PathLike], download: bool = False) -> str:
    """Download the DIC HepG2 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be stored.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the folder where data is stored.
    """
    if os.path.exists(path):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "2021-11-15_HepG2_Calcein_AM.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path, True)

    return path


def _create_segmentations_from_coco_annotation(path, split):
    assert COCO is not None, "pycocotools is required for processing the LiveCELL ground-truth."

    base_dir = os.path.join(path, "2021-11-15_HepG2_Calcein_AM", "coco_format", split)
    image_folder = os.path.join(base_dir, "images")
    gt_folder = os.path.join(base_dir, "annotations")
    if os.path.exists(gt_folder):
        return image_folder, gt_folder

    os.makedirs(gt_folder, exist_ok=True)

    ann_file = os.path.join(base_dir, "annotations.json")
    assert os.path.exists(ann_file)
    coco = COCO(ann_file)
    category_ids = coco.getCatIds(catNms=["cell"])
    image_ids = coco.getImgIds(catIds=category_ids)

    for image_id in tqdm(
        image_ids, desc="Creating DIC HepG2 segmentations from coco-style annotations"
    ):
        image_metadata = coco.loadImgs(image_id)[0]
        fname = image_metadata["file_name"]

        gt_path = os.path.join(gt_folder, Path(fname).with_suffix(".tif"))

        gt = _annotations_to_instances(coco, image_metadata, category_ids)
        imageio.imwrite(gt_path, gt, compression="zlib")

    return image_folder, gt_folder


def get_dic_hepg2_paths(
    path: Union[os.PathLike, str], split: str, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to DIC HepG2 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    path = get_dic_hepg2_data(path=path, download=download)

    image_folder, gt_folder = _create_segmentations_from_coco_annotation(path=path, split=split)
    gt_paths = natsorted(glob(os.path.join(gt_folder, "*.tif")))
    image_paths = [os.path.join(image_folder, f"{Path(gt_path).stem}.png") for gt_path in gt_paths]

    return image_paths, gt_paths


def get_dic_hepg2_dataset(
    path: Union[str, os.PathLike],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DIC HepG2 dataset for segmenting cells in differential interference contrast microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_dic_hepg2_paths(path=path, split=split)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
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


def get_dic_hepg2_loader(
    path: Union[str, os.PathLike],
    split: Literal['train', 'val', 'test'],
    patch_shape: Tuple[int, int],
    batch_size: int,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DIC HepG2 dataloader for segmenting cells in differential interference contrast microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dic_hepg2_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
