"""The RECIST CT dataset contains annotations for instance segmentation of tumors, metastases and
lymph nodes in CT scans, together with the corresponding RECIST 1.1 diameter measurements.

The dataset consists of 1,246 manually segmented lesions from 58 CT scans of 22 cancer patients treated
at the Clinical Hospital of the University of Chile (HCUCH). The data is split by anatomical region
('abdomen', 'thorax') and by 'train' / 'test' subset.

This dataset is from the publication https://doi.org/10.1038/s41597-026-06597-6.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/17788162/files/final-formatted.zip"
CHECKSUM = "39e2fc8a34ef7f519617901e4683462a75c4d97e5c46eedd50d3b35b22b96be5"

REGIONS = ["abdomen", "thorax"]
SPLITS = ["train", "test"]


def get_recist_ct_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RECIST CT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "final-formatted")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "final-formatted.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_recist_ct_paths(
    path: Union[os.PathLike, str],
    region: Optional[Literal["abdomen", "thorax"]] = None,
    split: Optional[Literal["train", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RECIST CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        region: The choice of anatomical region.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_recist_ct_data(path, download)

    if region is None:
        regions = REGIONS
    else:
        assert region in REGIONS, f"'{region}' is not a valid region."
        regions = [region]

    if split is None:
        splits = SPLITS
    else:
        assert split in SPLITS, f"'{split}' is not a valid split."
        splits = [split]

    # The original filenames combine the patient id with the DICOM Series Instance UID, which contains many
    # dots. 'elf.io.open_file' only recognizes '.nii.gz' files that have exactly two suffixes, so symlinks
    # with the extra dots replaced by underscores are created here to make the files resolvable.
    clean_dir = os.path.join(path, "clean")

    image_paths, gt_paths = [], []
    for _region in regions:
        for _split in splits:
            base_dir = os.path.join(data_dir, "images", _region, _split)
            raw_image_paths = natsorted(glob(os.path.join(base_dir, "images", "*.nii.gz")))
            raw_gt_paths = natsorted(glob(os.path.join(base_dir, "masks", "*.nii.gz")))

            for raw_path, sub_dir, out_paths in [
                (raw_image_paths, "images", image_paths), (raw_gt_paths, "masks", gt_paths)
            ]:
                out_dir = os.path.join(clean_dir, _region, _split, sub_dir)
                os.makedirs(out_dir, exist_ok=True)
                for orig_path in raw_path:
                    clean_name = os.path.basename(orig_path)[:-len(".nii.gz")].replace(".", "_") + ".nii.gz"
                    clean_path = os.path.join(out_dir, clean_name)
                    if not os.path.exists(clean_path):
                        os.symlink(os.path.abspath(orig_path), clean_path)
                    out_paths.append(clean_path)

    assert len(image_paths) > 0 and len(image_paths) == len(gt_paths), \
        f"Could not find a matching number of images and labels in '{data_dir}'."

    return image_paths, gt_paths


def get_recist_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    region: Optional[Literal["abdomen", "thorax"]] = None,
    split: Optional[Literal["train", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RECIST CT dataset for instance segmentation of tumors, metastases and lymph nodes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        region: The choice of anatomical region.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_recist_ct_paths(path, region, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_recist_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    region: Optional[Literal["abdomen", "thorax"]] = None,
    split: Optional[Literal["train", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RECIST CT dataloader for instance segmentation of tumors, metastases and lymph nodes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        region: The choice of anatomical region.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_recist_ct_dataset(path, patch_shape, region, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
