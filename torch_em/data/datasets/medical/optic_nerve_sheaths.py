"""The OpticNerveSheaths dataset contains annotations for optic nerve and optic nerve
sheath segmentation in transorbital ultrasound images.

The dataset contains 464 B-mode transorbital ultrasound images collected on a multidevice,
multicenter cohort, with pixel-level annotations of the optic nerve and its sheath, used to
estimate the optic nerve sheath diameter (ONSD), a marker of increased intracranial pressure.

This dataset is located at https://doi.org/10.17632/kw8gvp8m8x.1 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1016/j.ultrasmedbio.2023.05.011.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/kw8gvp8m8x/files/2cc2372d-9edb-4b61-aa84-9513a0156cf0/file_downloaded"  # noqa
CHECKSUM = "f8fc47b345462aa585e511bd73cf66eba05d317c2bbc86025c8b875f9b4fdcff"


def get_optic_nerve_sheaths_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OpticNerveSheaths dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Ultrasound-OpticNerveSheaths")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Ultrasound-OpticNerveSheaths.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_optic_nerve_sheaths_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the OpticNerveSheaths data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_optic_nerve_sheaths_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "DATA", "IMAGES_256", "*.png")))
    gt_paths = sorted(glob(os.path.join(data_dir, "DATA", "LABELS_256", "*.png")))

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_optic_nerve_sheaths_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OpticNerveSheaths dataset for optic nerve and sheath segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_optic_nerve_sheaths_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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


def get_optic_nerve_sheaths_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OpticNerveSheaths dataloader for optic nerve and sheath segmentation.

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
    dataset = get_optic_nerve_sheaths_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
