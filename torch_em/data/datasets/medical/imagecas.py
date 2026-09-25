"""The ImageCAS dataset contains annotations for coronary artery segmentation in cardiac CT angiography (CCTA).

The dataset consists of 1000 3D CCTA scans collected at the Guangdong Provincial People's Hospital between
April 2012 and December 2018. The left and right coronary arteries were independently annotated by two
radiologists and cross-verified; disagreements were resolved by a third radiologist. The data is distributed
as one 'img.nii.gz' / 'label.nii.gz' pair per case (label id 1 is the coronary artery, 0 is background) and
hosted on Kaggle as five zip archives of 200 cases each (https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas),
because the official GitHub repository (https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)  # noqa
does not host the data itself and otherwise requires emailing the authors for access.

Each Kaggle archive is itself split into five parts ('<group>.change2zip', '<group>.z01' to '<group>.z04'):
this module downloads all parts, joins them into a single zip with the 'zip' CLI (Info-ZIP) and extracts it.

NOTE: This requires a Kaggle account and API credentials (see https://www.kaggle.com/docs/api), as well as
the 'zip' CLI (Info-ZIP) to join the split archives.

This dataset is from the publication https://doi.org/10.1016/j.compmedimag.2023.102287.
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


KAGGLE_DATASET = "xiaoweixumedicalai/imagecas"

GROUPS = ["1-200", "201-400", "401-600", "601-800", "801-1000"]
"""The five Kaggle archives that together make up the 1000 cases of the dataset."""


def _download_kaggle_file(filename: str, dst_dir: str, download: bool) -> str:
    """Download a single file from the ImageCAS Kaggle dataset.

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


def _rename_nifti_files(case_dir: str) -> None:
    """Rename '<id>.img.nii.gz' / '<id>.label.nii.gz' to '<id>_img.nii.gz' / '<id>_label.nii.gz'.

    'elf.io.open_file' (used by `torch_em.data.SegmentationDataset`) only recognizes '.nii.gz' files that have
    exactly two suffixes, e.g. '<id>.nii.gz'. The extra '.img' / '.label' suffix in the original file names would
    otherwise be mistaken for the file extension, so the files are renamed once after extraction.
    """
    for suffix in ("img", "label"):
        for path in glob(os.path.join(case_dir, f"*.{suffix}.nii.gz")):
            new_path = path[:-len(f".{suffix}.nii.gz")] + f"_{suffix}.nii.gz"
            if not os.path.exists(new_path):
                os.rename(path, new_path)


def _merge_and_extract_group(group: str, zip_dir: str, raw_dir: str, download: bool) -> None:
    """Download, join and extract the split zip archive of one group (200 cases) of the dataset.

    Groups that were already extracted (e.g. by a previous, interrupted run) are skipped.
    """
    if glob(os.path.join(raw_dir, group, "*_img.nii.gz")):
        return

    parts = [f"{group}.change2zip"] + [f"{group}.z0{i}" for i in range(1, 5)]
    for part in parts:
        _download_kaggle_file(part, zip_dir, download)

    base_zip = os.path.join(zip_dir, f"{group}.zip")
    if not os.path.exists(base_zip):
        os.rename(os.path.join(zip_dir, f"{group}.change2zip"), base_zip)

    merged_zip = os.path.join(zip_dir, f"{group}.merged.zip")
    if not os.path.exists(merged_zip):
        if which("zip") is None:
            raise RuntimeError(
                "Need the 'zip' CLI (Info-ZIP) to join the split zip archive of the ImageCAS dataset. "
                "You can install it via 'conda install -c conda-forge zip'."
            )
        run(["zip", "-s", "0", base_zip, "--out", merged_zip], check=True, cwd=zip_dir)

    util.unzip(zip_path=merged_zip, dst=raw_dir, remove=False)
    _rename_nifti_files(os.path.join(raw_dir, group))

    # The split zip parts and the joined zip are removed once the group has been extracted, so that the ~18 GB
    # per group of intermediate files do not pile up on disk (the extraction itself is not repeated afterwards).
    for part in parts[1:]:
        part_path = os.path.join(zip_dir, part)
        if os.path.exists(part_path):
            os.remove(part_path)
    for leftover in (base_zip, merged_zip):
        if os.path.exists(leftover):
            os.remove(leftover)


def get_imagecas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ImageCAS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    raw_dir = os.path.join(path, "data")
    if len(glob(os.path.join(raw_dir, "**", "*_img.nii.gz"), recursive=True)) >= 1000:
        return raw_dir

    os.makedirs(raw_dir, exist_ok=True)

    zip_dir = os.path.join(path, "zips")
    for group in GROUPS:
        _merge_and_extract_group(group, zip_dir, raw_dir, download)

    return raw_dir


def get_imagecas_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ImageCAS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    raw_dir = get_imagecas_data(path, download)

    image_paths = natsorted(glob(os.path.join(raw_dir, "**", "*_img.nii.gz"), recursive=True))
    label_paths = natsorted(glob(os.path.join(raw_dir, "**", "*_label.nii.gz"), recursive=True))
    assert len(image_paths) > 0 and len(image_paths) == len(label_paths), \
        f"Could not find a matching number of images and labels in '{raw_dir}'."

    return image_paths, label_paths


def get_imagecas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ImageCAS dataset for coronary artery segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_imagecas_paths(path, download)

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


def get_imagecas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ImageCAS dataloader for coronary artery segmentation.

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
    dataset = get_imagecas_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
