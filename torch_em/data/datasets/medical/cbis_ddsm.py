"""The CBIS DDSM contains annotations for lesion segmentation in
mammography images.

This dataset is a preprocessed version of https://www.cancerimagingarchive.net/collection/cbis-ddsm/ available
at https://www.kaggle.com/datasets/mohamedbenticha/cbis-ddsm/data.
The dataset is related to the publication https://doi.org/10.1038/sdata.2017.177.
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_cbis_ddsm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CBIS DDSM dataset.
     Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded for the selected task.
    """
    data_dir = os.path.join(path, "DATA")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "cbis-ddsm.zip")
    util.download_source_kaggle(path=path, dataset_name="mohamedbenticha/cbis-ddsm/", download=download)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _check_if_size_matches(image_path, gt_path):
    from PIL import Image
    return Image.open(image_path).size == Image.open(gt_path).size


def get_cbis_ddsm_paths(
    path: Union[os.PathLike, str],
    split: Literal['Train', 'Val', 'Test'],
    task: Literal['Calc', 'Mass'],
    tumour_type: Optional[Literal["MALIGNANT", "BENIGN"]] = None,
    download: bool = False,
    ignore_mismatching_pairs: bool = False,
):
    """Get paths to the CBIS DDSM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for the specified task.
        tumour_type: The choice of tumour type.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cbis_ddsm_data(path, download)

    if split not in ["Train", "Val", "Test"]:
        raise ValueError(f"'{split}' is not a valid split.")

    if task is None:
        task = "*"
    else:
        assert task in ["Calc", "Mass"], f"'{task}' is not a valid task."

    if tumour_type is None:
        tumour_type = "*"
    else:
        assert tumour_type in ["MALIGNANT", "BENIGN"], f"'{tumour_type}' is not a tumor type."

    if split == "Test":
        target_dir = os.path.join(data_dir, task, split, tumour_type)
        image_paths = natsorted(glob(os.path.join(target_dir, "*_FULL_*.png")))
        gt_paths = natsorted(glob(os.path.join(target_dir, "*_MASK_*.png")))
    else:
        target_dir = os.path.join(data_dir, task, "Train", tumour_type)
        image_paths = natsorted(glob(os.path.join(target_dir, "*_FULL_*.png")))
        gt_paths = natsorted(glob(os.path.join(target_dir, "*_MASK_*.png")))

        if ignore_mismatching_pairs:
            input_paths = [
                (ip, gp) for ip, gp in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc="Validate inputs")
                if _check_if_size_matches(ip, gp)
            ]
            image_paths = [p[0] for p in input_paths]
            gt_paths = [p[1] for p in input_paths]

        if split == "Train":
            image_paths, gt_paths = image_paths[125:], gt_paths[125:]
        else:  # validation split (take the first 125 samples for validation)
            image_paths, gt_paths = image_paths[:125], gt_paths[:125]

    assert len(image_paths) == len(gt_paths)

    return image_paths, gt_paths


def get_cbis_ddsm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['Train', 'Val', 'Test'],
    task: Optional[Literal["Calc", "Mass"]] = None,
    tumour_type: Optional[Literal["MALIGNANT", "BENIGN"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CBIS DDSM dataset for lesion segmentation in mammograms.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specified task.
        tumour_type: The choice of tumour type.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_cbis_ddsm_paths(path, split, task, tumour_type, download)

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


def get_cbis_ddsm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['Train', 'Val', 'Test'],
    task: Optional[Literal["Calc", "Mass"]] = None,
    tumour_type: Optional[Literal["MALIGNANT", "BENIGN"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CBIS DDSM dataloader for lesion segmentation in mammograms.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specified task.
        tumour_type: The choice of tumour type.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cbis_ddsm_dataset(path, patch_shape, split, task, tumour_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
