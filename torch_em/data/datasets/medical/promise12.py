"""The PROMISE12 dataset contains annotations for prostate segmentation in T2-weighted MRI.

The data comes from the MICCAI 2012 'Prostate MR Image Segmentation' challenge (https://promise12.grand-challenge.org/)
and covers 50 labeled training cases from four centers with different scanners, field strengths and protocols
(with and without an endorectal coil). The label ids are: background: 0 and prostate: 1.
The challenge test cases are not labeled and are therefore not exposed here.

The original MetaImage ('.mhd' / '.raw') volumes are converted to hdf5 once and then loaded from there.

The dataset is located at https://zenodo.org/records/8026660.

This dataset is from the publication https://doi.org/10.1016/j.media.2013.12.002.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/8026660/files/training_data.zip?download=1"
CHECKSUM = "150287d0c74cd0105d8b70b43af4a7bf4f1fd3e1748829779177a0ccaf948f45"

METAIMAGE_DTYPES = {
    "MET_CHAR": "int8", "MET_UCHAR": "uint8", "MET_SHORT": "int16", "MET_USHORT": "uint16",
    "MET_INT": "int32", "MET_UINT": "uint32", "MET_FLOAT": "float32", "MET_DOUBLE": "float64",
}


def _read_metaimage(path):
    """Read an uncompressed MetaImage ('.mhd' + '.raw') volume with numpy, returning it in 'zyx' order."""
    header = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                header[key.strip()] = value.strip()

    if header.get("CompressedData", "False") == "True":
        raise NotImplementedError(f"Compressed MetaImage data is not supported: '{path}'.")

    shape = tuple(int(s) for s in header["DimSize"].split())[::-1]
    dtype = np.dtype(METAIMAGE_DTYPES[header["ElementType"]])
    if header.get("BinaryDataByteOrderMSB", "False") == "True":
        dtype = dtype.newbyteorder(">")

    raw_path = os.path.join(os.path.dirname(path), header["ElementDataFile"])
    return np.fromfile(raw_path, dtype=dtype).reshape(shape)


def get_promise12_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PROMISE12 dataset and convert it to hdf5.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the converted hdf5 volumes.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "training_data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    raw_dir = os.path.join(path, "training_data")
    util.unzip(zip_path=zip_path, dst=raw_dir)

    import h5py

    os.makedirs(data_dir, exist_ok=True)
    label_paths = natsorted(glob(os.path.join(raw_dir, "Case*_segmentation.mhd")))
    for label_path in tqdm(label_paths, desc="Converting PROMISE12 volumes to hdf5"):
        raw_path = label_path.replace("_segmentation.mhd", ".mhd")
        raw = _read_metaimage(raw_path)
        labels = _read_metaimage(label_path).astype("uint8")
        assert raw.shape == labels.shape, f"Shape mismatch for '{raw_path}': {raw.shape} vs. {labels.shape}."

        case_id = os.path.basename(raw_path).replace(".mhd", "")
        with h5py.File(os.path.join(data_dir, f"{case_id}.h5"), "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

    return data_dir


def get_promise12_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the PROMISE12 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 volumes, which store the image data at 'raw' and the label data at 'labels'.
    """
    data_dir = get_promise12_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "Case*.h5")))
    assert len(volume_paths) > 0, f"No PROMISE12 volumes found at '{data_dir}'."
    return volume_paths


def get_promise12_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PROMISE12 dataset for prostate segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_promise12_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_promise12_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PROMISE12 dataloader for prostate segmentation.

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
    dataset = get_promise12_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
