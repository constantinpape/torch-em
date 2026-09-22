"""The PlaTiF dataset contains annotations for tibia bone segmentation in anteroposterior knee
radiographs, together with Schatzker classification labels for tibial plateau fractures.

The dataset consists of 421 anteroposterior knee radiographs from 186 patients (normal and
fractured knees), collected at Shariati Hospital, Tehran University of Medical Sciences. Each
image comes with a manually produced (and MATLAB-refined) binary tibia segmentation mask, expert
validated, plus a Schatzker fracture type label (1-6) or a "no fracture" label (7).

The dataset is located at https://doi.org/10.5281/zenodo.18007397 and is distributed under the
CC BY 4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-026-06560-5.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "Patient Data_Part 1.zip": "https://zenodo.org/records/18007397/files/Patient%20Data_Part%201.zip",
    "Patient Data_Part 2.zip": "https://zenodo.org/records/18007397/files/Patient%20Data_Part%202.zip",
    "Patient Data_Part 3.zip": "https://zenodo.org/records/18007397/files/Patient%20Data_Part%203.zip",
    "Patient Data_Part 4.zip": "https://zenodo.org/records/18007397/files/Patient%20Data_Part%204.zip",
    "Patient Data_Part 5.zip": "https://zenodo.org/records/18007397/files/Patient%20Data_Part%205.zip",
}

CHECKSUMS = {
    "Patient Data_Part 1.zip": "df72a6e0d988492eb87928105596001c6551db38d21dfb0a5d46942c45bd9d49",
    "Patient Data_Part 2.zip": "ef0f3315dbd9d47361dad23840cd4a8fbf36fbeb32955deddebc5fe29e14134b",
    "Patient Data_Part 3.zip": "63b77dc6e929df4d5c810a324d079905158d9196796ed9696ee5f1d81094ae09",
    "Patient Data_Part 4.zip": "d6135f67f5d7d76fab2b93ce059d56b6647e5bac87cf3b0d5ab0329121406756",
    "Patient Data_Part 5.zip": "dce5a29107a2e64eb7c25016e8cb65ba46d022f555983b144172602f2920416c",
}


def _preprocess_platif(mat_dir, preprocessed_dir):
    import scipy.io as sio

    os.makedirs(os.path.join(preprocessed_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_dir, "masks"), exist_ok=True)

    mat_paths = natsorted(glob(os.path.join(mat_dir, "**", "*.mat"), recursive=True))
    assert len(mat_paths) > 0, f"No '.mat' files were found at '{mat_dir}'."

    for mat_path in tqdm(mat_paths, desc="Preprocessing inputs"):
        data = sio.loadmat(mat_path, simplify_cells=True)
        patient_keys = [k for k in data if not k.startswith("__")]
        assert len(patient_keys) == 1, f"Unexpected structure in '{mat_path}'."
        patient = data[patient_keys[0]]

        view_keys = natsorted([k for k in patient if k.startswith("im")])
        for view_key in view_keys:
            view = patient[view_key]

            image_path = os.path.join(preprocessed_dir, "images", f"{patient_keys[0]}_{view_key}.tif")
            mask_path = os.path.join(preprocessed_dir, "masks", f"{patient_keys[0]}_{view_key}.tif")
            if os.path.exists(image_path) and os.path.exists(mask_path):
                continue

            image = (view["OriginalImage"] * 255).astype("uint8")
            mask = view["BW"].astype("uint8")

            imageio.imwrite(image_path, image, compression="zlib")
            imageio.imwrite(mask_path, mask, compression="zlib")

    shutil.rmtree(mat_dir)


def get_platif_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PlaTiF data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and glob(os.path.join(preprocessed_dir, "images", "*.tif")):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    mat_dir = os.path.join(path, "mat_files")
    os.makedirs(mat_dir, exist_ok=True)
    for fname, url in URLS.items():
        zip_path = os.path.join(path, fname)
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[fname])
        util.unzip(zip_path=zip_path, dst=mat_dir, remove=False)

    _preprocess_platif(mat_dir, preprocessed_dir)
    return preprocessed_dir


def get_platif_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PlaTiF data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    preprocessed_dir = get_platif_data(path, download)

    image_paths = natsorted(glob(os.path.join(preprocessed_dir, "images", "*.tif")))
    mask_paths = natsorted(glob(os.path.join(preprocessed_dir, "masks", "*.tif")))

    if len(image_paths) == 0 or len(image_paths) != len(mask_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, mask_paths


def get_platif_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PlaTiF dataset for tibia segmentation in knee radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, mask_paths = get_platif_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=mask_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_platif_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PlaTiF dataloader for tibia segmentation in knee radiographs.

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
    dataset = get_platif_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
