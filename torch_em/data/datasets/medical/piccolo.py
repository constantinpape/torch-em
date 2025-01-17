"""The PICCOLO dataset contains annotations for polyp segmentation
in narrow band imaging colonoscopy.

NOTE: Automatic download is not supported with this dataset. See 'get_piccolo_data' for details.

The dataset is from the publication https://doi.org/10.3390/app10238501.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_piccolo_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Get the PICCOLO dataset.

    The database is located at:
    - https://www.biobancovasco.bioef.eus/en/Sample-and-data-e-catalog/Databases/PD178-PICCOLO-EN1.html

    Follow the instructions below to get access to the dataset.
    - Visit the attached website above
    - Fill up the access request form: https://labur.eus/EzJUN
    - Send an email to Basque Biobank at solicitudes.biobancovasco@bioef.eus, requesting access to the dataset.
    - The team will request you to follow-up with some formalities.
    - Then, you will gain access to the ".rar" file.
    - Finally, provide the path where the rar file is stored, and you should be good to go.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, r"piccolo dataset-release0.1")
    if os.path.exists(data_dir):
        return data_dir

    if download:
        raise NotImplementedError(
            "Automatic download is not possible for this dataset. See 'get_piccolo_data' for details."
        )

    rar_file = os.path.join(path, r"piccolo dataset_widefield-release0.1.rar")
    if not os.path.exists(rar_file):
        raise FileNotFoundError(
            "You must download the PICCOLO dataset from the Basque Biobank, see 'get_piccolo_data' for details."
        )

    util.unzip_rarfile(rar_path=rar_file, dst=path, remove=False)
    return data_dir


def get_piccolo_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'validation', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PICCOLO data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_piccolo_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, split, "polyps", "*")))
    gt_paths = natsorted(glob(os.path.join(data_dir, split, "masks", "*")))

    return image_paths, gt_paths


def get_piccolo_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PICCOLO dataset for polyp segmentation in narrow band imaging colonoscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_piccolo_paths(path, split, download)

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


def get_piccolo_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PICCOLO dataloader for polyp segmentation in narrow band imaging colonoscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_piccolo_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
