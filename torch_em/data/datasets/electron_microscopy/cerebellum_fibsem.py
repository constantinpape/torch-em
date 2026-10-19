"""The Cerebellum FIB-SEM dataset contains a FIB-SEM volume of mouse cerebellum (10 nm pixel size)
with instance segmentations of nuclei and mitochondria.

The volume has 530 x 1010 x 1014 voxels. It contains:
- nuclei: 5 nucleus instances.
- mitochondria: ~370 mitochondria instances.

The labels were created by students in the practical course BIO407 2026 (University of Zurich) and were not
necessarily proofread. The dataset is located at https://zenodo.org/records/18886823
and is distributed under the CC BY 4.0 license. Please cite it if you use this dataset in your research.
"""

import os
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/18886823/files/BIO407_Sideproject_vEM_Group5.zip/content"
CHECKSUM = "d24a8cbfeb985c7dfa1be9d4a92884c0805c3d57c11060d06334056281c4e163"

FILES = {
    "raw": "260304_FIB_SEM_aligned_nucleus_AR_BIO407.tif",
    "nuclei": "260304_FIB_SEM_segmentation_nucleus_AR_407.tif",
    "mitochondria": "260305_FIB_SEM_segmented_mitochondria_ARN_407.tif",
}
LABEL_KEYS = {"nuclei": "labels/nuclei", "mitochondria": "labels/mitochondria"}

# The volume is split along the z-axis: the first slices are used for training, the remaining ones for validation.
VAL_START = 424


def _convert_to_h5(tif_dir, h5_path):
    import h5py
    import tifffile

    with h5py.File(h5_path, "w") as f:
        for name, fname in FILES.items():
            data = tifffile.imread(os.path.join(tif_dir, fname))
            key = "raw" if name == "raw" else LABEL_KEYS[name]
            if name != "raw":
                data = data.astype("uint32")
            f.create_dataset(key, data=data, compression="gzip", chunks=(32, 256, 256))


def get_cerebellum_fibsem_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Cerebellum FIB-SEM dataset.

    Note: The data is only available as a 4.7 GB zip archive, which also contains raw volumes that are not used.
    Only the aligned volume and the two label volumes are extracted and converted to a single hdf5 file.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the hdf5 file with the data.
    """
    h5_path = os.path.join(path, "cerebellum_fibsem.h5")
    if os.path.exists(h5_path):
        return h5_path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "BIO407_Sideproject_vEM_Group5.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    import zipfile
    tif_dir = os.path.join(path, "tifs")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(tif_dir, members=list(FILES.values()))
    _convert_to_h5(tif_dir, h5_path)

    # Clean up the archive and the intermediate tif files.
    os.remove(zip_path)
    for fname in FILES.values():
        os.remove(os.path.join(tif_dir, fname))
    os.rmdir(tif_dir)
    return h5_path


def get_cerebellum_fibsem_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get the path to the Cerebellum FIB-SEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List with the filepath to the hdf5 file.
    """
    return [get_cerebellum_fibsem_data(path, download)]


def get_cerebellum_fibsem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    segmentation_type: Literal["nuclei", "mitochondria"] = "mitochondria",
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Cerebellum FIB-SEM dataset for nucleus or mitochondria segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation_type: The segmentation task. Either 'nuclei' or 'mitochondria'.
        split: The data split to use. Either 'train' (first 424 slices) or 'val' (remaining slices).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert segmentation_type in LABEL_KEYS, f"'{segmentation_type}' is not valid. Choose from {list(LABEL_KEYS)}."
    assert split in ("train", "val"), f"'{split}' is not a valid data split."

    paths = get_cerebellum_fibsem_paths(path, download)

    roi = np.s_[:VAL_START, :, :] if split == "train" else np.s_[VAL_START:, :, :]
    kwargs = util.update_kwargs(kwargs, "rois", [roi] * len(paths))
    kwargs = util.update_kwargs(kwargs, "ndim", 3)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=LABEL_KEYS[segmentation_type],
        patch_shape=patch_shape,
        **kwargs
    )


def get_cerebellum_fibsem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    segmentation_type: Literal["nuclei", "mitochondria"] = "mitochondria",
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for nucleus or mitochondria segmentation in the Cerebellum FIB-SEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        segmentation_type: The segmentation task. Either 'nuclei' or 'mitochondria'.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cerebellum_fibsem_dataset(path, patch_shape, segmentation_type, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
