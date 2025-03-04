"""The IRCADb dataset contains annotations for liver segmentation (and several other organs and structures)
in 3D CT scans.

The dataset is located at https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/.
This dataset is from the publication, referenced in the dataset link above.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
CHECKSUM = None  # NOTE: checksums are mismatching for some reason with every new download instance :/


def _preprocess_inputs(path):
    data_dir = os.path.join(path, "3Dircadb1")
    patient_dirs = glob(os.path.join(data_dir, "*"))

    # Store all preprocessed images in one place
    preprocessed_dir = os.path.join(path, "data")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Let's extract all files per patient, preprocess them, store the final version and remove the zip files.
    for pdir in tqdm(patient_dirs, desc="Preprocessing files"):

        patient_name = os.path.basename(pdir)

        # Get all zipfiles
        masks_file = os.path.join(pdir, "MASKS_DICOM.zip")
        patient_file = os.path.join(pdir, "PATIENT_DICOM.zip")

        # Unzip all.
        util.unzip(masks_file, pdir, remove=False)
        util.unzip(patient_file, pdir, remove=False)

        # Get all files and stack each slice together.
        import pydicom as dicom
        images = [dicom.dcmread(p).pixel_array for p in natsorted(glob(os.path.join(pdir, "PATIENT_DICOM", "*")))]
        images = np.stack(images, axis=0)

        # Get masks per slice per class.
        masks, mask_names = [], []
        for mask_dir in glob(os.path.join(pdir, "MASKS_DICOM", "*")):
            mask_names.append(os.path.basename(mask_dir))
            curr_mask = np.stack(
                [dicom.dcmread(p).pixel_array for p in natsorted(glob(os.path.join(mask_dir, "*")))], axis=0,
            )
            assert curr_mask.shape == images.shape, "The shapes for images and labels don't match."
            masks.append(curr_mask)

        # Store them in one place
        import h5py
        with h5py.File(os.path.join(preprocessed_dir, f"{patient_name}.h5"), "a") as f:
            f.create_dataset("raw", data=images, compression="gzip")
            # Add labels one by one
            for name, _mask in zip(mask_names, masks):
                f.create_dataset(f"labels/{name}", data=_mask, compression="gzip")


def get_ircadb_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the IRCADb dataset.

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

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=True)

    _preprocess_inputs(path)

    return data_dir


def get_ircadb_paths(
    path: Union[os.PathLike, str], split: Optional[Literal["train", "val", "test"]] = None, download: bool = False,
) -> List[str]:
    """Get paths to the IRCADb data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data.
    """

    data_dir = get_ircadb_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))

    # Create splits on-the-fly, if desired.
    if split is not None:
        if split == "train":
            volume_paths = volume_paths[:12]
        elif split == "val":
            volume_paths = volume_paths[12:15]
        elif split == "test":
            volume_paths = volume_paths[15:]
        else:
            raise ValueError(f"'{split}' is not a valid split.")

    return volume_paths


def get_ircadb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: str,
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the IRCADb dataset for liver (and other organ) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of labelled organs.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_ircadb_paths(path, split, download)

    # Get the labels in the expected hierarchy name.
    assert isinstance(label_choice, str)
    label_choice = f"labels/{label_choice}"

    # Get the parameters for resizing inputs
    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_choice,
        patch_shape=patch_shape,
        **kwargs
    )


def get_ircadb_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: str,
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the IRCADb dataloader for liver (and other organ) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of labelled organs.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ircadb_dataset(path, patch_shape, label_choice, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
