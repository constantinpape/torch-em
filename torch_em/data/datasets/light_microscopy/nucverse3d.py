"""The NucVerse3D dataset contains annotations for 3d nucleus segmentation in
two-photon and confocal microscopy volumes.

The dataset holds three collections that the authors annotated by hand:
- 'liver': two-photon volumes of adult mouse liver, with mono- and binucleated hepatocytes,
- 'liver_hcc': two-photon volumes of mouse hepatocellular carcinoma,
- 'drosophila_glia': confocal volumes of Drosophila melanogaster brain, stained for glial nuclei.

Together they hold 26 volumes with 6226 nuclei. Every volume becomes one h5 file that stores the
raw data, the instance labels and the voxel size.

NOTE: The three collections name their masks in three ways. The liver collection appends '_gt' or
'_gt_uint16', the carcinoma collection appends '_labels', and the fly collection replaces a
'Denoised-' prefix with a '-label' suffix. The loader therefore pairs an image with a mask by
position after sorting, and it checks that the shape of a pair agrees.

NOTE: The masks come as int32, uint32 and uint16, even inside one split. The loader writes them all
as uint32.

NOTE: The readme of the repository says that the carcinoma collection serves for testing alone, but
the archive splits it into 8 training volumes and 4 test volumes. This loader follows the archive.

NOTE: The tiff files carry no usable voxel size. The carcinoma files report a resolution of one with
the unit 'none', and the fly files report no resolution at all, so the voxel size of this module
comes from the readme of the repository.

The dataset is located at https://doi.org/10.5281/zenodo.18517324 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41598-026-51994-x.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://zenodo.org/records/18517324/files/raw.zip?download=1"
CHECKSUM = "ba4be8b887fe839bb4b16f50e87b93292f95b959e9dc2d161e234521cc9a6d19"

# The name of a collection in the archive, and what the readme says about it.
DATASETS = {
    "liver": {
        "folder": "2_livernuclei",
        "modality": "two-photon microscopy",
        "tissue": "adult mouse liver",
        # The readme gives the voxel size as x, y, z in micrometer. The arrays are z, y, x.
        "resolution": (0.3, 0.3, 0.3),
    },
    "liver_hcc": {
        "folder": "7_liver_hcc_dataset",
        "modality": "two-photon microscopy",
        "tissue": "mouse liver hepatocellular carcinoma",
        "resolution": (0.3, 0.3, 0.3),
    },
    "drosophila_glia": {
        "folder": "6_Drosophila_denoised",
        "modality": "confocal microscopy",
        "tissue": "Drosophila melanogaster brain glia",
        "resolution": (0.3, 0.15, 0.15),
    },
}

SPLITS = ("train", "test")


def _pair_split(data_dir: str, folder: str, split: str) -> List[Tuple[str, str]]:
    """Pair every image of a split with its mask, by position after sorting."""
    image_paths = natsorted(glob(os.path.join(data_dir, folder, split, "images", "*.tif")))
    mask_paths = natsorted(glob(os.path.join(data_dir, folder, split, "masks", "*.tif")))
    if len(image_paths) != len(mask_paths):
        raise RuntimeError(
            f"The split '{folder}/{split}' holds {len(image_paths)} images but {len(mask_paths)} masks."
        )
    return list(zip(image_paths, mask_paths))


def _create_h5(data_dir: str, name: str, split: str) -> str:
    """Write one h5 file per volume, with the raw data, the labels and the voxel size."""
    import h5py
    import tifffile
    from tqdm import tqdm

    info = DATASETS[name]
    output_dir = os.path.join(data_dir, "preprocessed", name, split)
    os.makedirs(output_dir, exist_ok=True)

    pairs = _pair_split(data_dir, info["folder"], split)
    for image_path, mask_path in tqdm(pairs, desc=f"Preprocess '{name}/{split}'"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}.h5")
        if os.path.exists(output_path):
            continue

        raw = tifffile.imread(image_path)
        labels = tifffile.imread(mask_path)
        if raw.shape != labels.shape:
            raise RuntimeError(
                f"The image {os.path.basename(image_path)} has the shape {raw.shape}, "
                f"but its mask {os.path.basename(mask_path)} has the shape {labels.shape}."
            )

        temporary_path = f"{output_path}.tmp"
        with h5py.File(temporary_path, "w") as f:
            f.attrs["dataset"] = name
            f.attrs["modality"] = info["modality"]
            f.attrs["tissue"] = info["tissue"]
            f.attrs["split"] = split
            # The voxel size in micrometer, in the order of the axes of the arrays.
            f.attrs["resolution"] = info["resolution"]
            f.attrs["axes"] = "zyx"
            f.attrs["image_file"] = os.path.basename(image_path)
            f.attrs["label_file"] = os.path.basename(mask_path)

            raw_dataset = f.create_dataset("raw", data=raw, compression="gzip")
            raw_dataset.attrs["resolution"] = info["resolution"]
            # The masks come in three integer types, so they all become uint32 here.
            label_dataset = f.create_dataset("labels", data=labels.astype("uint32"), compression="gzip")
            label_dataset.attrs["resolution"] = info["resolution"]
        os.replace(temporary_path, output_path)

    return output_dir


def get_nucverse3d_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NucVerse3D dataset.

    The repository also holds trained models and preprocessed patches, which take 31 GB together.
    This loader reads the raw archive alone, which takes 0.68 GB.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "raw")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "raw.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_nucverse3d_paths(
    path: Union[os.PathLike, str],
    dataset: Optional[Union[str, Sequence[str]]] = None,
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the NucVerse3D data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset: The collection or collections to use, out of 'liver', 'liver_hcc' and
            'drosophila_glia'. Defaults to all of them.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    if dataset is None:
        names = list(DATASETS)
    else:
        names = [dataset] if isinstance(dataset, str) else list(dataset)
        for name in names:
            if name not in DATASETS:
                raise ValueError(f"'{name}' is not a valid dataset. Choose from {list(DATASETS)}.")

    data_dir = get_nucverse3d_data(path, download)

    volume_paths = []
    for name in names:
        output_dir = _create_h5(data_dir, name, split)
        volume_paths.extend(natsorted(glob(os.path.join(output_dir, "*.h5"))))

    if not volume_paths:
        raise RuntimeError(f"Could not find any NucVerse3D data in {data_dir}.")

    return volume_paths


def get_nucverse3d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    dataset: Optional[Union[str, Sequence[str]]] = None,
    split: Literal["train", "test"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the NucVerse3D dataset for 3d nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        dataset: The collection or collections to use, out of 'liver', 'liver_hcc' and
            'drosophila_glia'. Defaults to all of them.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The NucVerse3D patch shape must be three-dimensional, got {patch_shape}.")

    volume_paths = get_nucverse3d_paths(path, dataset, split, download)

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


def get_nucverse3d_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    dataset: Optional[Union[str, Sequence[str]]] = None,
    split: Literal["train", "test"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the NucVerse3D dataloader for 3d nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        dataset: The collection or collections to use, out of 'liver', 'liver_hcc' and
            'drosophila_glia'. Defaults to all of them.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset_ = get_nucverse3d_dataset(
        path=path,
        patch_shape=patch_shape,
        dataset=dataset,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset_, batch_size=batch_size, **loader_kwargs)
