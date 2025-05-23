"""The CURVAS dataset contains annotations for pancreas, kidney and liver
in abdominal CT scans.

This dataset is from the challenge: https://curvas.grand-challenge.org.
The dataset is located at: https://zenodo.org/records/12687192.
Please cite tem if you use this dataset for your research.
"""

import os
import shutil
import subprocess
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/12687192/files/training_set.zip"
CHECKSUM = "1126a2205553ae1d4fe5fbaee7ea732aacc4f5a92b96504ed521c23e5a0e3f89"


def _preprocess_data(data_dir):
    import h5py
    import nibabel as nib

    h5_dir = os.path.join(os.path.dirname(data_dir), "data")
    os.makedirs(h5_dir, exist_ok=True)

    image_paths = natsorted(glob(os.path.join(data_dir, "*", "image.nii.gz")))
    for image_path in tqdm(image_paths, desc="Processing data"):
        rater1_path = os.path.join(os.path.dirname(image_path), "annotation_1.nii.gz")
        rater2_path = os.path.join(os.path.dirname(image_path), "annotation_2.nii.gz")
        rater3_path = os.path.join(os.path.dirname(image_path), "annotation_3.nii.gz")

        assert os.path.exists(rater1_path) and os.path.exists(rater2_path) and os.path.exists(rater3_path)

        image = nib.load(image_path).get_fdata().astype("float32").transpose(2, 0, 1)

        label_r1 = np.rint(nib.load(rater1_path).get_fdata()).astype("uint8").transpose(2, 0, 1)
        label_r2 = np.rint(nib.load(rater2_path).get_fdata()).astype("uint8").transpose(2, 0, 1)
        label_r3 = np.rint(nib.load(rater3_path).get_fdata()).astype("uint8").transpose(2, 0, 1)

        fname = os.path.basename(os.path.dirname(image_path))
        chunks = (8, 512, 512)
        with h5py.File(os.path.join(h5_dir, f"{fname}.h5"), "w") as f:
            f.create_dataset("raw", data=image, compression="gzip", chunks=chunks)
            f.create_dataset("labels/rater_1", data=label_r1, compression="gzip", chunks=chunks)
            f.create_dataset("labels/rater_2", data=label_r2, compression="gzip", chunks=chunks)
            f.create_dataset("labels/rater_3", data=label_r3, compression="gzip", chunks=chunks)

    # Remove the nifti files as we don't need them anymore!
    shutil.rmtree(data_dir)


def get_curvas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CURVAS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "training_set.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    # HACK: The zip file is broken. We fix it using the following script.
    fixed_zip_path = os.path.join(path, "training_set_fixed.zip")
    subprocess.run(["zip", "-FF", zip_path, "--out", fixed_zip_path])
    subprocess.run(["unzip", fixed_zip_path, "-d", path])

    _preprocess_data(os.path.join(path, "training_set"))

    # Remove the zip files as we don't need them anymore.
    os.remove(zip_path)
    os.remove(fixed_zip_path)

    return data_dir


def get_curvas_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> List[str]:
    """Get paths to the CURVAS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data.
    """
    data_dir = get_curvas_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))

    if split == "train":
        volume_paths = volume_paths[:10]
    elif split == "val":
        volume_paths = volume_paths[10:13]
    elif split == "test":
        volume_paths = volume_paths[13:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return volume_paths


def get_curvas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    rater: Literal["1", "2", "3"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CURVAS dataset for pancreas, kidney and liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of rater providing the annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_curvas_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/rater_{rater}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs,
    )


def get_curvas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    rater: Literal["1", "2", "3"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CURVAS dataloader for pancreas, kidney and liver segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of rater providing the annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_curvas_dataset(path, patch_shape, split, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
