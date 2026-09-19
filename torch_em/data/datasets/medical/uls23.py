"""ULS23 (Universal Lesion Segmentation Challenge 2023) provides a unified 3D lesion segmentation task
that combines novel lesion annotations with volumes-of-interest (VOIs) processed from several existing
CT lesion datasets. This module covers the 'processed_data' part of the challenge's training data, which
re-uses the source images of three existing lesion datasets and adds new 3D VOI-level lesion masks for them:

- 'kits21': 332 kidney lesion VOIs, images taken from the KiTS21 kidney tumor segmentation challenge.
- 'lits': 888 liver lesion VOIs, images taken from the LiTS liver tumor segmentation challenge.
- 'lidc-idri': 2246 lung lesion VOIs, images taken from the LIDC-IDRI lung nodule dataset.

Each VOI is a 3D CT sub-volume cropped around one lesion, with a binary lesion segmentation mask.
The novel lesion annotations of ULS23 (eg. the DeepLesion3D subset) are covered by the separate
`torch_em.data.datasets.medical.deeplesion` module.

The images (~44 GB, split into a multi-part zip archive) are located at
https://doi.org/10.5281/zenodo.10050960 (part 2 of the ULS23 training data).
The labels are located at https://github.com/DIAGNijmegen/ULS23.
NOTE: The images are distributed as (nested) multi-part zip archives, so the '7z' CLI is required to extract
them (install it via 'conda install -c conda-forge p7zip'). Both the images and labels are stored with a
trailing singleton dimension (eg. shape (256, 256, 128, 1)), which this module squeezes in-place after
extraction so that the volumes are plain 3D arrays.
The data is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.1016/j.media.2025.103525.
Please cite it (and the publication of the source dataset you use: KiTS21 https://doi.org/10.48550/arXiv.2307.01984,
LiTS https://doi.org/10.1016/j.media.2022.102680, or LIDC-IDRI https://doi.org/10.1118/1.3528204) if you use
this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from shutil import which
from subprocess import run
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "ULS23_Part2.zip": "https://zenodo.org/records/10050960/files/ULS23_Part2.zip?download=1",
    "ULS23_Part2.z01": "https://zenodo.org/records/10050960/files/ULS23_Part2.z01?download=1",
    "ULS23_Part2.z02": "https://zenodo.org/records/10050960/files/ULS23_Part2.z02?download=1",
    "ULS23_Part2.z03": "https://zenodo.org/records/10050960/files/ULS23_Part2.z03?download=1",
    "ULS23_Part2.z04": "https://zenodo.org/records/10050960/files/ULS23_Part2.z04?download=1",
    "ULS23_Part2.z05": "https://zenodo.org/records/10050960/files/ULS23_Part2.z05?download=1",
    "ULS23_Part2.z06": "https://zenodo.org/records/10050960/files/ULS23_Part2.z06?download=1",
    "ULS23_Part2.z07": "https://zenodo.org/records/10050960/files/ULS23_Part2.z07?download=1",
    "ULS23_Part2.z08": "https://zenodo.org/records/10050960/files/ULS23_Part2.z08?download=1",
    "ULS23_Part2.z09": "https://zenodo.org/records/10050960/files/ULS23_Part2.z09?download=1",
    "ULS23_Part2.z10": "https://zenodo.org/records/10050960/files/ULS23_Part2.z10?download=1",
    "annotations": "https://github.com/DIAGNijmegen/ULS23/archive/06a2bffc433418f72d04f7ecbb23b28694c81e6b.zip",
}

CHECKSUMS = {
    "ULS23_Part2.zip": "67e9343c58b25ba871ae35aeac5db9cc89acde01d2ce64b3fa0fdf9d17b19bd5",
    "ULS23_Part2.z01": "803b18b4ddd24175589e52dcf847000daa80c1e471bd955bb1f27624f51ace05",
    "ULS23_Part2.z02": "e743e4fc733ae7608fb24f4057399e14aea89aef761ccc275492167b53861529",
    "ULS23_Part2.z03": "44d3092e8fe8f1bf7c8e23fdd93b035967d0771ba6905af0e61cb294d56d6c08",
    "ULS23_Part2.z04": "56d0a40862d82d5f4fce39baeca32714563b2c1ba0768e43d7eb9af9331c2fc6",
    "ULS23_Part2.z05": "dc9696532cad2bd3ed6d3c31fcf4afa044a2c5b0bda804d8fd2bb647cd56d102",
    "ULS23_Part2.z06": "f13ff8402e7e08cc0989b47f115255339061d09d628d06fc5d09fee4983f146d",
    "ULS23_Part2.z07": "d491fd122425add97a951dd7e8edb2cc79abab94e9be9c94b6f1e70a1eaf6e68",
    "ULS23_Part2.z08": "669811801c36514faa906feee953d5b0e970554ad7f0ca382a80996a65b45c18",
    "ULS23_Part2.z09": "0d880b1e22686becbccbbf6debe12076a6ed5b87d4fd64651967e4b1463ef456",
    "ULS23_Part2.z10": "efd22fcff15049cdaac5c002c9e90d15afaae62a9fe5a60fadc79f65d369a8cd",
    "annotations": "19ae6b84aae1a94aa8329a0ce4c6586d1e7cf28532b27367d923773adc3ebe28",
}

SOURCES = {
    "kits21": os.path.join("ULS23", "processed_data", "fully_annotated", "kits21"),
    "lits": os.path.join("ULS23", "processed_data", "fully_annotated", "LiTS"),
    "lidc-idri": os.path.join("ULS23", "processed_data", "fully_annotated", "LIDC-IDRI"),
}
"""Mapping from the source dataset name to its folder in the ULS23 archives."""


def _run_7z_x(archive, dst, members=None):
    if which("7z") is None:
        raise RuntimeError(
            "The ULS23 images are distributed as (nested) multi-part zip archives, which require the '7z' CLI. "
            "You can install it via 'conda install -c conda-forge p7zip'."
        )
    cmd = ["7z", "x", f"-o{dst}", "-y", archive]
    if members is not None:
        cmd += members
    run(cmd, check=True)


def _squeeze_trailing_singleton_dims(root):
    """The ULS23 VOIs are stored with a trailing singleton dimension (eg. shape (256, 256, 128, 1)), which
    is squeezed here in-place so that the volumes are plain 3D arrays."""
    import nibabel as nib

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".nii.gz"):
                continue
            fpath = os.path.join(dirpath, fname)
            nifti = nib.load(fpath)
            if nifti.shape[-1] != 1 or len(nifti.shape) <= 3:
                continue
            data = np.squeeze(np.asarray(nifti.dataobj), axis=-1)
            nib.save(nib.Nifti1Image(data, nifti.affine), fpath)


def _index_by_basename(root):
    """Build a mapping from filename to filepath for all '*.nii.gz' files under 'root' (searched recursively,
    since the inner per-source archives do not consistently nest their images under the same sub-folder)."""
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".nii.gz"):
                index[fname] = os.path.join(dirpath, fname)
    return index


def _extract_uls23_images(path, source, download):
    image_dir = os.path.join(path, SOURCES[source], "images")
    if os.path.exists(image_dir) and _index_by_basename(image_dir):
        return image_dir

    outer_zip = os.path.join(path, "ULS23_Part2.zip")
    for name, url in URLS.items():
        if not name.startswith("ULS23_Part2"):
            continue
        util.download_source(path=os.path.join(path, name), url=url, download=download, checksum=CHECKSUMS[name])

    inner_zip = os.path.join(SOURCES[source], "images.zip")
    _run_7z_x(outer_zip, path, members=[inner_zip + "*"])

    inner_zip_path = os.path.join(path, inner_zip)
    _run_7z_x(inner_zip_path, image_dir)

    # Remove the (potentially multi-part) inner zip archive, but not the extracted 'images.zip' folder itself.
    inner_zip_stem = inner_zip_path[:-len(".zip")]
    for part in glob(inner_zip_stem + ".z[0-9][0-9]"):
        os.remove(part)
    os.remove(inner_zip_path)

    _squeeze_trailing_singleton_dims(image_dir)

    return image_dir


def _extract_uls23_labels(path, source, download):
    label_dir = os.path.join(path, SOURCES[source], "labels")
    if os.path.exists(label_dir) and glob(os.path.join(label_dir, "*.nii.gz")):
        return label_dir

    annotation_zip = os.path.join(path, "ULS23_annotations.zip")
    util.download_source(
        path=annotation_zip, url=URLS["annotations"], download=download, checksum=CHECKSUMS["annotations"]
    )
    extracted_dir = os.path.join(path, "ULS23_annotations")
    if not os.path.exists(extracted_dir):
        util.unzip(zip_path=annotation_zip, dst=extracted_dir, remove=False)

    src_label_dir = glob(os.path.join(extracted_dir, "*", "annotations", SOURCES[source], "labels"))[0]
    for label_zip in tqdm(natsorted(glob(os.path.join(src_label_dir, "*.zip"))), desc=f"Preparing {source} labels"):
        util.unzip(zip_path=label_zip, dst=label_dir, remove=False)

    _squeeze_trailing_singleton_dims(label_dir)

    return label_dir


def get_uls23_data(
    path: Union[os.PathLike, str], source: Literal["kits21", "lits", "lidc-idri"], download: bool = False
) -> Tuple[str, str]:
    """Download the ULS23 processed-data images and labels for one source dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The source dataset to fetch the VOIs for. One of 'kits21', 'lits', 'lidc-idri'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the image data.
        Filepath to the folder with the label data.
    """
    if source not in SOURCES:
        raise ValueError(f"'{source}' is not a supported source. Please choose one of {list(SOURCES.keys())}.")

    os.makedirs(path, exist_ok=True)
    image_dir = _extract_uls23_images(path, source, download)
    label_dir = _extract_uls23_labels(path, source, download)

    return image_dir, label_dir


def get_uls23_paths(
    path: Union[os.PathLike, str], source: Literal["kits21", "lits", "lidc-idri"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ULS23 processed-data VOIs for one source dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The source dataset to fetch the VOIs for. One of 'kits21', 'lits', 'lidc-idri'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, label_dir = get_uls23_data(path, source, download)
    image_index = _index_by_basename(image_dir)

    raw_paths, label_paths = [], []
    for label_path in natsorted(glob(os.path.join(label_dir, "*.nii.gz"))):
        fname = os.path.basename(label_path)
        raw_path = image_index.get(fname)
        if raw_path is None:
            continue
        raw_paths.append(raw_path)
        label_paths.append(label_path)

    if len(raw_paths) == 0 or len(raw_paths) != len(label_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_uls23_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Literal["kits21", "lits", "lidc-idri"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ULS23 dataset for universal lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        source: The source dataset to fetch the VOIs for. One of 'kits21', 'lits', 'lidc-idri'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_uls23_paths(path, source, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_uls23_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    source: Literal["kits21", "lits", "lidc-idri"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ULS23 dataloader for universal lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The source dataset to fetch the VOIs for. One of 'kits21', 'lits', 'lidc-idri'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_uls23_dataset(path, patch_shape, source, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
