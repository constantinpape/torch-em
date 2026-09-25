"""The Ovarian Reserve dataset contains annotations for 3d oocyte segmentation in
light-sheet microscopy volumes of whole-mount mouse ovaries.

The ovaries come from C57BL/6J mice between 5 and 60 weeks of age. They were cleared, stained for
DDX4 and imaged on a SPIM light-sheet microscope. Every volume holds 40 planes of 256 x 256 pixels
at a voxel size of 5.0 x 0.867 x 0.867 micrometer, together with hand curated instance labels of the
oocytes. The archive holds 66 volumes for training and 7 for validation.

NOTE: This is the representative labeled subset of the study. The whole ovaries of the study carry no
labels. They are available at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3593 and at
https://doi.org/10.5281/zenodo.19085211 .

NOTE: The masks come as uint16, int32, uint32 and float32, so this loader writes them all as uint32.

The dataset is distributed with the segmentation tutorial at
https://biapy.readthedocs.io/en/latest/tutorials/instance_seg/ovarian-reserve.html , and the BioImage
Model Zoo lists it under the CC BY 4.0 license at https://bioimage.io/#/artifacts/splendid-falafel .
This dataset is from the publication https://doi.org/10.1038/s43587-026-01178-z .
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = (
    "https://upvehueus-my.sharepoint.com/:u:/g/personal/ignacio_arganda_ehu_eus/"
    "IQBlTg1-y8MlSqwgDpLZuPAgAU5oE0HOqc6vjDK7vVh_xBM?e=MMgzZf&download=1"
)
CHECKSUM = "d9774be882229451ddf415475a7a404caf190714a7091a679f8d5df97ae00a7c"

SPLITS = ("train", "val")

# The voxel size in micrometer, in the order of the axes of the arrays.
RESOLUTION = (5.0, 0.867, 0.867)


def _create_h5(data_dir: str, split: str) -> str:
    """Write one h5 file per volume, with the raw data, the labels and the voxel size."""
    import h5py
    import tifffile
    from tqdm import tqdm

    output_dir = os.path.join(data_dir, "preprocessed", split)
    os.makedirs(output_dir, exist_ok=True)

    raw_paths = natsorted(glob(os.path.join(data_dir, split, "raw", "*.tif")))
    if not raw_paths:
        raise RuntimeError(f"Could not find any raw data for the split '{split}' in {data_dir}.")

    for raw_path in tqdm(raw_paths, desc=f"Preprocess '{split}'"):
        name = os.path.basename(raw_path)
        # The archive gives an image and its mask the same file name.
        label_path = os.path.join(data_dir, split, "label", name)
        if not os.path.exists(label_path):
            raise RuntimeError(f"Could not find the mask for the image '{name}' at {label_path}.")

        output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.h5")
        if os.path.exists(output_path):
            continue

        raw = tifffile.imread(raw_path)
        labels = tifffile.imread(label_path)
        if raw.shape != labels.shape:
            raise RuntimeError(
                f"The image '{name}' has the shape {raw.shape}, but its mask has the shape {labels.shape}."
            )

        temporary_path = f"{output_path}.tmp"
        with h5py.File(temporary_path, "w") as f:
            f.attrs["modality"] = "selective plane illumination fluorescence microscopy"
            f.attrs["tissue"] = "whole-mount mouse ovary"
            f.attrs["stain"] = "DDX4"
            f.attrs["split"] = split
            f.attrs["resolution"] = RESOLUTION
            f.attrs["axes"] = "zyx"
            f.attrs["image_file"] = name

            raw_dataset = f.create_dataset("raw", data=raw, compression="gzip")
            raw_dataset.attrs["resolution"] = RESOLUTION
            label_dataset = f.create_dataset("labels", data=labels.astype("uint32"), compression="gzip")
            label_dataset.attrs["resolution"] = RESOLUTION
        os.replace(temporary_path, output_path)

    return output_dir


def get_ovarian_reserve_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Ovarian Reserve dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "oocyte_training")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "oocyte_training.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_ovarian_reserve_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"] = "train", download: bool = False,
) -> List[str]:
    """Get paths to the Ovarian Reserve data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    data_dir = get_ovarian_reserve_data(path, download)
    output_dir = _create_h5(data_dir, split)
    volume_paths = natsorted(glob(os.path.join(output_dir, "*.h5")))

    if not volume_paths:
        raise RuntimeError(f"Could not find any Ovarian Reserve data in {data_dir}.")

    return volume_paths


def get_ovarian_reserve_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Ovarian Reserve dataset for 3d oocyte segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train' or 'val'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The Ovarian Reserve patch shape must be three-dimensional, got {patch_shape}.")

    volume_paths = get_ovarian_reserve_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs,
    )


def get_ovarian_reserve_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Ovarian Reserve dataloader for 3d oocyte segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train' or 'val'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ovarian_reserve_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
