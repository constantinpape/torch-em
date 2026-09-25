"""The CHASE_DB1 dataset contains annotations for retinal vessel segmentation
in fundus images.

This dataset is located at https://researchdata.kingston.ac.uk/96/.
The dataset is from the publication https://doi.org/10.1109/TBME.2012.2205687.
The dataset is licensed under CC BY 4.0 (see https://researchdata.kingston.ac.uk/96/
for details). Please cite the publication above if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://researchdata.kingston.ac.uk/96/1/CHASEDB1.zip"
CHECKSUM = None


def get_chase_db1_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CHASE_DB1 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "CHASEDB1")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "CHASEDB1.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_chase_db1_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    annotator: Literal['1st', '2nd'] = '1st',
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CHASE_DB1 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. The dataset does not ship an official split. We use the first
            20 images for training (of which the last 4 are held out for validation) and the
            remaining 8 images for testing, following the split convention used in the vessel
            segmentation literature.
        annotator: The choice of annotator for the ground-truth vessel maps. There are two independent
            manual annotations ('1st' and '2nd') per image.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_chase_db1_data(path=path, download=download)

    assert annotator in ["1st", "2nd"], f"'{annotator}' is not a valid annotator choice."

    image_paths = sorted(glob(os.path.join(data_dir, "Image_*.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, f"Image_*_{annotator}HO.png")))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    if split == "train":
        image_paths, gt_paths = image_paths[:16], gt_paths[:16]
    elif split == "val":
        image_paths, gt_paths = image_paths[16:20], gt_paths[16:20]
    elif split == "test":
        image_paths, gt_paths = image_paths[20:], gt_paths[20:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_chase_db1_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    annotator: Literal['1st', '2nd'] = '1st',
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CHASE_DB1 dataset for segmentation of retinal blood vessels in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotator: The choice of annotator for the ground-truth vessel maps.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_chase_db1_paths(path=path, split=split, annotator=annotator, download=download)

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


def get_chase_db1_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    annotator: Literal['1st', '2nd'] = '1st',
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CHASE_DB1 dataloader for segmentation of retinal blood vessels in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotator: The choice of annotator for the ground-truth vessel maps.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chase_db1_dataset(path, patch_shape, split, annotator, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
