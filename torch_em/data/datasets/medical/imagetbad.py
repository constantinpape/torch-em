"""The ImageTBAD dataset contains annotations for type-B aortic dissection segmentation in
computed tomography angiography (CTA).

The dataset consists of 100 3D CTA scans collected at the Guangdong Provincial People's Hospital between
January 2013 and April 2019. The segmentation was performed by one cardiovascular radiologist and checked
by a second one. It labels three substructures of the aortic dissection: the true lumen (TL), the false
lumen (FL) and the false lumen thrombus (FLT), see `LABEL_IDS`. 68 of the 100 cases contain an FLT, the
remaining 32 cases are free of it. The data is distributed as one 'image.nii.gz' / 'label.nii.gz' pair per
case and hosted on Kaggle as a single, split zip archive (https://www.kaggle.com/datasets/xiaoweixumedicalai/
imagetbad), because the official GitHub repository (https://github.com/XiaoweiXu/Dataset_Type-B-Aortic-
Dissection) does not host the data itself.

The Kaggle archive is itself split into 19 parts ('imageTBAD.change2zip', 'imageTBAD.z01' to 'imageTBAD.z18'):
this module downloads all parts, joins them into a single zip with the 'zip' CLI (Info-ZIP) and extracts it.

NOTE: This requires a Kaggle account and API credentials (see https://www.kaggle.com/docs/api), as well as
the 'zip' CLI (Info-ZIP) to join the split archives.

This dataset is from the publication https://doi.org/10.3389/fphys.2021.732711.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from shutil import which
from subprocess import run
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET = "xiaoweixumedicalai/imagetbad"

N_PARTS = 18
"""The number of split zip parts ('imageTBAD.z01' to 'imageTBAD.z18') on top of 'imageTBAD.change2zip'."""

LABEL_IDS = {"background": 0, "true_lumen": 1, "false_lumen": 2, "false_lumen_thrombus": 3}


def _download_kaggle_file(filename: str, dst_dir: str, download: bool) -> str:
    """Download a single file from the ImageTBAD Kaggle dataset.

    Kaggle wraps every single-file download in an outer zip container (even if the file is itself
    already an archive), which is unpacked here to recover the original file.
    """
    out_path = os.path.join(dst_dir, filename)
    if os.path.exists(out_path):
        return out_path
    if not download:
        raise RuntimeError(f"Cannot find the data at {out_path}, but download was set to False.")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        msg = "Please install the Kaggle API. You can do this using 'pip install kaggle'. "
        msg += "After you have installed kaggle, you would need an API token. "
        msg += "Follow the instructions at https://www.kaggle.com/docs/api."
        raise ModuleNotFoundError(msg)

    os.makedirs(dst_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(KAGGLE_DATASET, filename, path=dst_dir)

    wrapper_path = os.path.join(dst_dir, f"{filename}.zip")
    util.unzip(zip_path=wrapper_path, dst=dst_dir)
    return out_path


def get_imagetbad_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ImageTBAD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    raw_dir = os.path.join(path, "data")
    if len(glob(os.path.join(raw_dir, "**", "*_image.nii.gz"), recursive=True)) >= 100:
        return raw_dir

    os.makedirs(raw_dir, exist_ok=True)

    zip_dir = os.path.join(path, "zips")
    parts = ["imageTBAD.change2zip"] + [f"imageTBAD.z{i:02d}" for i in range(1, N_PARTS + 1)]
    for part in parts:
        _download_kaggle_file(part, zip_dir, download)

    base_zip = os.path.join(zip_dir, "imageTBAD.zip")
    if not os.path.exists(base_zip):
        os.rename(os.path.join(zip_dir, "imageTBAD.change2zip"), base_zip)

    merged_zip = os.path.join(zip_dir, "imageTBAD.merged.zip")
    if not os.path.exists(merged_zip):
        if which("zip") is None:
            raise RuntimeError(
                "Need the 'zip' CLI (Info-ZIP) to join the split zip archive of the ImageTBAD dataset. "
                "You can install it via 'conda install -c conda-forge zip'."
            )
        run(["zip", "-s", "0", base_zip, "--out", merged_zip], check=True, cwd=zip_dir)

    util.unzip(zip_path=merged_zip, dst=raw_dir, remove=False)

    return raw_dir


def get_imagetbad_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ImageTBAD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    raw_dir = get_imagetbad_data(path, download)

    image_paths = natsorted(glob(os.path.join(raw_dir, "**", "*_image.nii.gz"), recursive=True))
    label_paths = natsorted(glob(os.path.join(raw_dir, "**", "*_label.nii.gz"), recursive=True))
    assert len(image_paths) > 0 and len(image_paths) == len(label_paths), \
        f"Could not find a matching number of images and labels in '{raw_dir}'."

    return image_paths, label_paths


def get_imagetbad_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ImageTBAD dataset for aortic dissection segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_imagetbad_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_imagetbad_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ImageTBAD dataloader for aortic dissection segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_imagetbad_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
