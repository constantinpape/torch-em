"""The Brain PFIB-SEM dataset contains a PFIB-SEM volume of mouse brain (5 x 5 x 5 nm voxel size)
with expert annotations for mitochondria and synapses.

The volume has 181 x 1300 x 1300 voxels and was acquired at two dwell times: 0.5 us (noisy) and 2 us (reference).
The annotations (binary semantic masks) were made on the 2 us reference stack.

The dataset also contains further volumes (SBF-SEM and PFIB-SEM at different dwell times) for denoising,
which are not used here. The dataset is located at https://zenodo.org/records/20139642
and is distributed under the CC BY 4.0 license. It is part of the publication
https://doi.org/10.1101/2025.08.26.672334. Please cite it if you use this dataset in your research.
"""

import os
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/api/records/20139642/files"
FILES = {
    "noisy": "PFIB_seg_0d5us_Raw_data_5x5x5nm.tif",
    "reference": "PFIB_seg_2us_Reference_5x5x5nm.tif",
    "mitochondria": "mitos_human_label.tif",
    "synapses": "synapses_human_label.tif",
}
CHECKSUMS = {
    "noisy": "c9f799e9a3de701c4bbce0c98c3ba3ea7b26d68f70361b589804e30ec99f7dbd",
    "reference": "eb4517d99010f04d79acc0ad4cd0ae4ff7880df833548f1cc34ed5078a75bacb",
    "mitochondria": "8ecb184e465b94459fdf3fa0fd36313010d51b7ba56faa077dcd99fa13f4e2f6",
    "synapses": "11e851d8b074b03f0c79320457b22b3a2cb38759779820c0bcc610839e1b3e5d",
}
LABEL_KEYS = {"mitochondria": "labels/mitochondria", "synapses": "labels/synapses"}

# The volume is split along the z-axis: the first slices are used for training, the remaining ones for validation.
VAL_START = 144


def _convert_to_h5(path, h5_path):
    import h5py
    import tifffile

    with h5py.File(h5_path, "w") as f:
        for name, fname in FILES.items():
            data = tifffile.imread(os.path.join(path, fname))
            if name in ("noisy", "reference"):
                key = f"raw/{name}"
            else:
                key = LABEL_KEYS[name]
                data = (data > 0).astype("uint8")
            f.create_dataset(key, data=data, compression="gzip", chunks=(32, 256, 256))


def get_brain_pfibsem_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Brain PFIB-SEM dataset.

    Note: Only the PFIB-SEM segmentation volumes (4 x ~300 MB) are downloaded and converted to a single hdf5 file.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the hdf5 file with the data.
    """
    h5_path = os.path.join(path, "brain_pfibsem.h5")
    if os.path.exists(h5_path):
        return h5_path

    os.makedirs(path, exist_ok=True)
    for name, fname in FILES.items():
        util.download_source(
            path=os.path.join(path, fname), url=f"{BASE_URL}/{fname}/content",
            download=download, checksum=CHECKSUMS[name],
        )

    _convert_to_h5(path, h5_path)

    # Clean up the intermediate tif files.
    for fname in FILES.values():
        os.remove(os.path.join(path, fname))
    return h5_path


def get_brain_pfibsem_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get the path to the Brain PFIB-SEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List with the filepath to the hdf5 file.
    """
    return [get_brain_pfibsem_data(path, download)]


def get_brain_pfibsem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    segmentation_type: Literal["mitochondria", "synapses"] = "mitochondria",
    input_type: Literal["noisy", "reference"] = "reference",
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Brain PFIB-SEM dataset for mitochondria or synapse segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation_type: The segmentation task. Either 'mitochondria' or 'synapses'.
        input_type: The input volume. Either 'reference' (2 us dwell time, on which the labels were made)
            or 'noisy' (0.5 us dwell time).
        split: The data split to use. Either 'train' (first 144 slices) or 'val' (remaining slices).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert segmentation_type in LABEL_KEYS, f"'{segmentation_type}' is not valid. Choose from {list(LABEL_KEYS)}."
    assert input_type in ("noisy", "reference"), f"'{input_type}' is not a valid input type."
    assert split in ("train", "val"), f"'{split}' is not a valid data split."

    paths = get_brain_pfibsem_paths(path, download)

    roi = np.s_[:VAL_START, :, :] if split == "train" else np.s_[VAL_START:, :, :]
    kwargs = util.update_kwargs(kwargs, "rois", [roi] * len(paths))
    kwargs = util.update_kwargs(kwargs, "ndim", 3)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key=f"raw/{input_type}",
        label_paths=paths,
        label_key=LABEL_KEYS[segmentation_type],
        patch_shape=patch_shape,
        **kwargs
    )


def get_brain_pfibsem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    segmentation_type: Literal["mitochondria", "synapses"] = "mitochondria",
    input_type: Literal["noisy", "reference"] = "reference",
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for mitochondria or synapse segmentation in the Brain PFIB-SEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        segmentation_type: The segmentation task. Either 'mitochondria' or 'synapses'.
        input_type: The input volume. Either 'reference' or 'noisy'.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brain_pfibsem_dataset(
        path, patch_shape, segmentation_type, input_type, split, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
