"""The CELLULAR dataset contains annotations for cell segmentation and autophagy classification
in fluorescence microscopy images of Drosophila cells.

The cells express the autophagy reporter mRFP-EGFP-Atg8a. The EGFP signal fades once an
autophagosome fuses with a lysosome, while the mRFP signal stays, so the ratio of the two channels
shows whether autophagy is active. An expert marked every cell with one of three classes:
- 'fed' for basal autophagy,
- 'unfed' for activated autophagy,
- 'Unidentified' for a cell that the expert could not assign.

The dataset holds 14005 annotated cells over 53 fields of view. Every field comes as one h5 file
with the three raw channels, an instance label image and a semantic label image.

NOTE: The raw data has three channels, which the microscope metadata names FITC, Texas Red and
TL 25. This loader stores them as 'raw/fitc' for the EGFP signal, 'raw/texas_red' for the mRFP
signal and 'raw/brightfield' for the transmitted light.

NOTE: The archive stores every cell as its own image of the full field, so a field of 264 cells
holds 264 images of 2048 x 2048 pixels. The preprocessing therefore reads about 170 GB of image
data to build 53 label images, and it runs over several cores.

NOTE: The class of a cell comes from the folder that holds its mask. The masks are colored, but the
color varies inside one class, so it does not encode the class.

NOTE: Only 53 of the 6240 fields carry masks. The loader reads the members of those fields out of
the archives, so it transfers about 1.4 GB instead of the full 110 GB.

NOTE: Almost half of the cells are 'Unidentified'. Treat that class as unlabelled rather than as a
third biological state.

The dataset is located at https://doi.org/10.5281/zenodo.8315423 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-023-02687-x.
Please cite it if you use this dataset in your research.
"""

import os
import zipfile
from glob import glob
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URLS = {
    "images": "https://zenodo.org/records/8315423/files/images.zip?download=1",
    "masks": "https://zenodo.org/records/8315423/files/masks.zip?download=1",
}

CHECKSUMS = {
    "images": None,  # The archive holds 105 GB, and the loader reads 159 members out of it.
    "masks": "8e640ba69e627e223a3703a883b5dda426bbbd069dddd61a5122f447bf880987",
}

# The raw file of a channel ends with this token, and the microscope metadata names the channel.
CHANNELS = {"w1": "fitc", "w2": "texas_red", "w3": "brightfield"}

CLASS_IDS = {"fed": 1, "unfed": 2, "Unidentified": 3}

RAW_KEYS = tuple(f"raw/{name}" for name in CHANNELS.values())


def _list_fields(mask_dir: str) -> List[str]:
    """List the fields of view that carry masks."""
    return natsorted(os.path.basename(p) for p in glob(os.path.join(mask_dir, "*")) if os.path.isdir(p))


def _fetch_members(url: str, members: Sequence[str], destination: str) -> None:
    """Read the given members out of a remote zip archive, without downloading all of it."""
    import fsspec

    with zipfile.ZipFile(fsspec.open(url, "rb").open()) as archive:
        available = set(archive.namelist())
        missing = [name for name in members if name not in available]
        if missing:
            raise RuntimeError(f"The archive {url} misses {len(missing)} members, e.g. '{missing[0]}'.")
        archive.extractall(destination, members=list(members))


def _build_field(arguments) -> Tuple[str, int]:
    """Write the h5 file of one field, and report how many cells it holds."""
    import h5py
    import tifffile

    data_dir, field = arguments
    output_path = os.path.join(data_dir, "preprocessed", f"{field}.h5")
    if os.path.exists(output_path):
        return field, 0

    raw = {
        name: tifffile.imread(os.path.join(data_dir, "images", f"{field}_{channel}.TIF"))
        for channel, name in CHANNELS.items()
    }
    shape = raw["fitc"].shape

    instances = np.zeros(shape, dtype="uint16")
    semantic = np.zeros(shape, dtype="uint8")

    instance_id = 0
    for class_name, class_id in CLASS_IDS.items():
        mask_paths = natsorted(glob(os.path.join(data_dir, "masks", field, class_name, "*.png")))
        for mask_path in mask_paths:
            mask = imageio.imread(mask_path)
            mask = mask.max(axis=-1) > 0 if mask.ndim == 3 else mask > 0
            # One mask of the archive holds no object at all.
            if not mask.any():
                continue
            instance_id += 1
            instances[mask] = instance_id
            semantic[mask] = class_id

    # Write to a temporary name, so that an interrupted run leaves no half written file behind.
    temporary_path = f"{output_path}.tmp"
    with h5py.File(temporary_path, "w") as f:
        for name, array in raw.items():
            f.create_dataset(f"raw/{name}", data=array, compression="gzip")
        f.create_dataset("labels/instances", data=instances, compression="gzip")
        f.create_dataset("labels/semantic", data=semantic, compression="gzip")
    os.replace(temporary_path, output_path)

    return field, instance_id


def _preprocess(data_dir: str, n_workers: Optional[int] = None) -> str:
    """Pack every field into one h5 file. The fields are independent, so they run in parallel."""
    from tqdm import tqdm

    output_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    fields = _list_fields(os.path.join(data_dir, "masks"))
    todo = [f for f in fields if not os.path.exists(os.path.join(output_dir, f"{f}.h5"))]
    if not todo:
        return output_dir

    if n_workers is None:
        n_workers = max(1, min(12, (os.cpu_count() or 1) - 2))

    arguments = [(data_dir, field) for field in todo]
    with ProcessPoolExecutor(n_workers) as executor:
        list(tqdm(executor.map(_build_field, arguments), total=len(todo), desc="Preprocess the fields"))

    return output_dir


def get_cellular_data(
    path: Union[os.PathLike, str], download: bool = False, n_workers: Optional[int] = None,
) -> str:
    """Download the CELLULAR dataset.

    The loader reads the 53 annotated fields out of the archives, so it transfers about 1.4 GB
    instead of the 110 GB that the repository holds.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
        n_workers: The number of processes for the preprocessing. Defaults to the core count.

    Returns:
        The filepath to the folder with the h5 files.
    """
    data_dir = str(path)
    output_dir = os.path.join(data_dir, "preprocessed")
    if os.path.exists(output_dir) and glob(os.path.join(output_dir, "*.h5")):
        return output_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {data_dir}, but download was set to False.")

    os.makedirs(data_dir, exist_ok=True)

    mask_dir = os.path.join(data_dir, "masks")
    if not os.path.exists(mask_dir):
        zip_path = os.path.join(data_dir, "masks.zip")
        util.download_source(zip_path, URLS["masks"], download, CHECKSUMS["masks"])
        util.unzip(zip_path=zip_path, dst=data_dir)

    fields = _list_fields(mask_dir)
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        members = [f"images/{field}_{channel}.TIF" for field in fields for channel in CHANNELS]
        _fetch_members(URLS["images"], members, data_dir)

    return _preprocess(data_dir, n_workers)


def get_cellular_paths(
    path: Union[os.PathLike, str],
    fields: Optional[Sequence[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the CELLULAR data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        fields: The fields of view to use. Defaults to all of them.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    output_dir = get_cellular_data(path, download)
    volume_paths = natsorted(glob(os.path.join(output_dir, "*.h5")))

    if fields is not None:
        wanted = set(fields)
        volume_paths = [p for p in volume_paths if os.path.splitext(os.path.basename(p))[0] in wanted]

    if not volume_paths:
        raise RuntimeError(f"Could not find any CELLULAR data in {output_dir}.")

    return volume_paths


def get_cellular_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    fields: Optional[Sequence[str]] = None,
    channel: Literal["fitc", "texas_red", "brightfield", "all"] = "all",
    label_choice: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CELLULAR dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        fields: The fields of view to use. Defaults to all of them.
        channel: The raw channel. Either 'fitc' for the EGFP signal, 'texas_red' for the mRFP
            signal, 'brightfield' for the transmitted light, or 'all' for the three of them.
        label_choice: The target. Either 'instances' for the single cells, or 'semantic' for the
            autophagy classes, where one is fed, two is unfed and three is unidentified.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The CELLULAR patch shape must be two-dimensional, got {patch_shape}.")
    if label_choice not in ("instances", "semantic"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'instances' or 'semantic'.")

    valid_channels = tuple(CHANNELS.values()) + ("all",)
    if channel not in valid_channels:
        raise ValueError(f"'{channel}' is not a valid channel. Choose from {list(valid_channels)}.")

    volume_paths = get_cellular_paths(path, fields, download)
    raw_key = list(RAW_KEYS) if channel == "all" else f"raw/{channel}"
    label_key = f"labels/{label_choice}"

    if label_choice == "instances":
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
        )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=raw_key,
        label_paths=volume_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=channel == "all",
        **kwargs,
    )


def get_cellular_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    fields: Optional[Sequence[str]] = None,
    channel: Literal["fitc", "texas_red", "brightfield", "all"] = "all",
    label_choice: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CELLULAR dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        fields: The fields of view to use. Defaults to all of them.
        channel: The raw channel. Either 'fitc', 'texas_red', 'brightfield' or 'all'.
        label_choice: The target. Either 'instances' or 'semantic'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellular_dataset(
        path=path,
        patch_shape=patch_shape,
        fields=fields,
        channel=channel,
        label_choice=label_choice,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
