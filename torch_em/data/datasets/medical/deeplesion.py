"""The DeepLesion3D dataset contains 3D lesion annotations for CT volumes-of-interest from the DeepLesion dataset.

The original DeepLesion dataset (https://nihcc.app.box.com/v/DeepLesion) only provides RECIST diameters and 2D
bounding boxes on the key slice of each lesion. This module provides the 'ULS23_DeepLesion3D' subset that was
created for the Universal Lesion Segmentation Challenge 2023 (ULS23): 750 lesions from DeepLesion were segmented
in 3D by trained (bio-)medical students (each lesion in triplicate, the majority vote is the final label).
Each sample is a 256 x 256 x 128 volume-of-interest cropped around one lesion, with a binary label mask (1: lesion).
The volumes are padded with a constant value along the z-axis, so that the lesion is centered around slice 64.
743 of the 750 volumes have a label; the remaining 7 are skipped. The lesions are grouped into 7 categories:
200 abdominal, 100 bone, 50 kidney, 50 liver, 100 lung, 100 mediastinal and 150 other lesions.

The images are located at https://doi.org/10.5281/zenodo.10035161 (part 1 of the ULS23 training data, which also
contains the bone and pancreas lesions from Radboudumc).
The labels are located at https://github.com/DIAGNijmegen/ULS23.
NOTE: The images are distributed as a multi-part zip archive, so the '7z' CLI is required to extract them
(install it via 'conda install -c conda-forge p7zip').
The data is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.1016/j.media.2025.103525.
Please cite it (and the DeepLesion publication https://doi.org/10.1117/1.JMI.5.3.036501) if you use this dataset
in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from shutil import which, rmtree
from subprocess import run
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "ULS23_Part1.zip": "https://zenodo.org/records/10035161/files/ULS23_Part1.zip?download=1",
    "ULS23_Part1.z01": "https://zenodo.org/records/10035161/files/ULS23_Part1.z01?download=1",
    "ULS23_Part1.z02": "https://zenodo.org/records/10035161/files/ULS23_Part1.z02?download=1",
    "ULS23_Part1.z03": "https://zenodo.org/records/10035161/files/ULS23_Part1.z03?download=1",
    "annotations": "https://github.com/DIAGNijmegen/ULS23/archive/06a2bffc433418f72d04f7ecbb23b28694c81e6b.zip",
}

CHECKSUMS = {
    "ULS23_Part1.zip": "b0beff9cedd09b212087f8fb678e07de8c41126ba11ee2c2ad14d7e210b8fabb",
    "ULS23_Part1.z01": "d23eefdfec9c394f9ce7effb0daa3c9dac37d38802c063eed2aa5e2a6b7aa354",
    "ULS23_Part1.z02": "d0b464b5ff0b099a85809ed354203bab2830b90208d17df6b30344dc277a519c",
    "ULS23_Part1.z03": "12f2ecef0788267a1f96b21e413647e15014cc8a2f40191dfa5d55426c337200",
    "annotations": "19ae6b84aae1a94aa8329a0ce4c6586d1e7cf28532b27367d923773adc3ebe28",
}

CATEGORIES = ["Abdominal", "Bone", "Kidney", "Liver", "Lung", "Mediastinal", "Other"]
"""The lesion categories of the DeepLesion3D dataset."""

ULS_DIR = os.path.join("ULS23", "novel_data", "ULS23_DeepLesion3D")


def _extract_uls_archives(path, download):
    zip_path = os.path.join(path, "ULS23_Part1.zip")
    for name, url in URLS.items():
        if name == "annotations":
            continue
        util.download_source(path=os.path.join(path, name), url=url, download=download, checksum=CHECKSUMS[name])

    if which("7z") is None:
        raise RuntimeError(
            "The DeepLesion3D images are distributed as a multi-part zip archive, which requires the '7z' CLI. "
            "You can install it via 'conda install -c conda-forge p7zip'."
        )

    inner_zips = [os.path.join(ULS_DIR, "images.zip"), os.path.join(ULS_DIR, "categories.zip")]
    run(["7z", "x", f"-o{path}", "-y", zip_path] + inner_zips, check=True)
    for inner_zip in inner_zips:
        util.unzip(zip_path=os.path.join(path, inner_zip), dst=os.path.join(path, ULS_DIR), remove=True)

    annotation_zip = os.path.join(path, "ULS23_annotations.zip")
    util.download_source(
        path=annotation_zip, url=URLS["annotations"], download=download, checksum=CHECKSUMS["annotations"]
    )
    util.unzip(zip_path=annotation_zip, dst=os.path.join(path, "ULS23_annotations"), remove=True)


def _preprocess_deeplesion3d(path, data_dir):
    import nibabel as nib

    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    src_image_dir = os.path.join(path, ULS_DIR, "images")
    category_dir = os.path.join(path, ULS_DIR, "categories")
    annotation_dir = os.path.join(path, "ULS23_annotations", "*", "annotations", ULS_DIR, "labels")
    label_zips = glob(os.path.join(annotation_dir, "*.zip"))
    assert len(label_zips) > 0, "Could not find the DeepLesion3D annotations."

    categories = {}
    for category in CATEGORIES:
        for fname in os.listdir(os.path.join(category_dir, category)):
            categories[os.path.splitext(fname)[0]] = category

    for label_zip in tqdm(natsorted(label_zips), desc="Preprocessing DeepLesion3D"):
        fname = os.path.basename(label_zip)[:-len(".zip")]
        src_image_path = os.path.join(src_image_dir, fname)
        assert os.path.exists(src_image_path), src_image_path

        util.unzip(zip_path=label_zip, dst=label_dir, remove=False)
        label_path = os.path.join(label_dir, fname)
        image_path = os.path.join(image_dir, fname)

        # The volumes are stored with a trailing singleton dimension, which we remove.
        for src, dst, dtype in [(src_image_path, image_path, "float32"), (label_path, label_path, "uint8")]:
            nifti = nib.load(src)
            data = np.squeeze(np.asarray(nifti.dataobj)).astype(dtype)
            assert data.ndim == 3, data.shape
            nib.save(nib.Nifti1Image(data, nifti.affine), dst)

    # The category of a lesion is given by the first four fields of its name (patient, study, series, key slice).
    lesion_categories = {}
    for label_path in glob(os.path.join(label_dir, "*.nii.gz")):
        fname = os.path.basename(label_path)
        lesion_categories[fname] = categories["_".join(fname.split("_")[:4])]
    with open(os.path.join(data_dir, "categories.json"), "w") as f:
        json.dump(lesion_categories, f, indent=2, sort_keys=True)

    # Remove the intermediate data.
    rmtree(os.path.join(path, "ULS23"))
    rmtree(os.path.join(path, "ULS23_annotations"))


def get_deeplesion_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DeepLesion3D dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "DeepLesion3D")
    if os.path.exists(os.path.join(data_dir, "categories.json")):
        return data_dir

    os.makedirs(path, exist_ok=True)
    _extract_uls_archives(path, download)
    _preprocess_deeplesion3d(path, data_dir)

    return data_dir


def get_deeplesion_paths(
    path: Union[os.PathLike, str],
    category: Optional[Literal["Abdominal", "Bone", "Kidney", "Liver", "Lung", "Mediastinal", "Other"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DeepLesion3D data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        category: The lesion category. By default, the lesions of all categories are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_deeplesion_data(path, download)

    with open(os.path.join(data_dir, "categories.json")) as f:
        categories = json.load(f)

    if category is not None and category not in CATEGORIES:
        raise ValueError(f"'{category}' is not a valid category. Choose from {CATEGORIES}.")

    fnames = natsorted(fname for fname, cat in categories.items() if category is None or cat == category)
    raw_paths = [os.path.join(data_dir, "images", fname) for fname in fnames]
    label_paths = [os.path.join(data_dir, "labels", fname) for fname in fnames]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_deeplesion_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    category: Optional[Literal["Abdominal", "Bone", "Kidney", "Liver", "Lung", "Mediastinal", "Other"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DeepLesion3D dataset for lesion segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        category: The lesion category. By default, the lesions of all categories are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_deeplesion_paths(path, category, download)

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


def get_deeplesion_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    category: Optional[Literal["Abdominal", "Bone", "Kidney", "Liver", "Lung", "Mediastinal", "Other"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DeepLesion3D dataloader for lesion segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        category: The lesion category. By default, the lesions of all categories are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deeplesion_dataset(path, patch_shape, category, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
