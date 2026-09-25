"""The OCUTOX dataset contains annotations for active and inactive lesion segmentation
in fundus images of patients with ocular toxoplasmosis.

The dataset contains 412 fundus images collected at the Hospital de Clinicas and the
Hospital General Pediatrico Acosta Nu medical centers in Asuncion, Paraguay, of which
280 images (with active and / or inactive toxoplasmosis lesions) have pixel-level lesion
masks delineated by ophthalmologists. The remaining images are labeled 'healthy' and do
not have a lesion mask.

This dataset is located at https://doi.org/10.5281/zenodo.5156940 (CC BY 4.0).
Please cite it if you use this dataset for your research.
"""

import os
import re
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/5156940/files/Ocular_Toxoplasmosis_Data_V3.zip"
CHECKSUM = "d088dcfe6c678923eed20e18052935cff70a3e0ede8f87f06406caa734727ece"

# Masks carry suffixes for lesion sub-regions of the same image, eg. '-a' (active lesion),
# '-i' (inactive lesion) and numeric variants ('-2', '-3', '-a-2', ...).
MASK_SUFFIX_PATTERN = re.compile(r"-(?:a|i)(?:-\d+)?$|-\d+$", flags=re.IGNORECASE)


def get_ocutox_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OCUTOX dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "images")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Ocular_Toxoplasmosis_Data_V3.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_ocutox_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the OCUTOX data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ocutox_data(path=path, download=download)

    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    image_files = {os.path.basename(p).lower(): p for p in glob(os.path.join(image_dir, "*.*"))}
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.*")))

    image_paths, matched_mask_paths = [], []
    for mask_path in mask_paths:
        fname = os.path.basename(mask_path)
        stem, ext = os.path.splitext(fname.lower())
        base_stem = MASK_SUFFIX_PATTERN.sub("", stem)
        image_path = image_files.get(base_stem + ext)
        if image_path is None:
            raise RuntimeError(f"Could not find the matching image for the mask at '{mask_path}'.")

        image_paths.append(image_path)
        matched_mask_paths.append(mask_path)

    if len(image_paths) == 0 or len(image_paths) != len(matched_mask_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, matched_mask_paths


def get_ocutox_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OCUTOX dataset for segmentation of ocular toxoplasmosis lesions in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ocutox_paths(path, download)

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


def get_ocutox_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OCUTOX dataloader for segmentation of ocular toxoplasmosis lesions in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ocutox_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
