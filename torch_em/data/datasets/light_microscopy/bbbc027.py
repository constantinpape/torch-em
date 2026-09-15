"""The BBBC027 dataset contains synthetic 3D fluorescence microscopy images of colon tissue
with foreground segmentation ground truth for the cell nuclei.

The images were generated with a virtual microscope (CytoPacq) imitating a Zeiss S100 confocal
microscope. The dataset contains 30 images of shape (129, 1030, 1300) voxels (ZYX), each provided
in a low SNR and a high SNR variant. The ground truth is a binary foreground / background mask
of all nuclei inside the tissue region (label 1 for nuclei, 0 for background); individual nuclei are
not separated into instances.

The images and ground truth are distributed in the ICS format and are converted to HDF5 files
(with the keys 'raw' and 'labels') when the data is prepared.

The dataset is located at https://bbbc.broadinstitute.org/BBBC027.
This dataset is from the publication https://doi.org/10.1007/978-3-642-21596-4_4.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
import zlib
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.broadinstitute.org/bbbc/BBBC027/BBBC027_{snr}SNR_{kind}_part{part}.zip"
CHECKSUMS = {
    "low": {
        "images": [
            "bf537d5e63c63f3b86e31d0dfe1b888a5bac8df78d9562f7c4fb738d9ad4e5d1",
            "4e581c373a2656d50265345a0cdaa1ec32926028c56a46ac9a6cc33bd8fca295",
            "b0ca1814101ea7c8f71802c762195f3e69a639bf4ba24065d4efbb780ea94d63",
        ],
        "foreground": [
            "1530cd10f40479045e54c6a56e1ba0081f2b0383b7008376abcb9a774ebdf33e",
            "96a59ff604e515c084ecf8a8c45ac96685ff7259cb210759a06a8a4d4dff9cd3",
            "073c53019799ee40f9dfe2d5f93b841f8e0728f3c8419ae6f211b991ad2f3c84",
        ],
    },
    "high": {
        "images": [
            "f07d6aa56b990dfe564380e004e0b414336e610bfd8e93d3bfff1ea15843a1f0",
            "0ad770285041e53cc6a5950b22b82d23c50a2a4e053e31dcb9653b300fd03ab5",
            "1454e75882d7de39898e54cb02d6502d051cf2a75e7229031d30be6910090eb2",
        ],
        "foreground": [
            "0dd7c5e03fa216e71add8983e4c4a05c21b4208b800055d3c41596188308af5e",
            "d04115940d17d09918a0b2116a3389ced753c549ea792d0352e886fa7eeaab0c",
            "71dff9e5cc2fc202418db85084104ef729d36c61b5c79b8085c63cc186bacdd3",
        ],
    },
}


def _read_ics(ics_path):
    """Read an ICS 2.0 file (header and data in the same file, optionally gzip compressed).
    """
    with open(ics_path, "rb") as f:
        data = f.read()

    # The first two bytes define the field and the line separator of the header.
    field_sep, line_sep = data[0:1], data[1:2]
    end_marker = b"end" + field_sep + line_sep
    header_end = data.index(end_marker)

    header = {}
    for line in data[2:header_end].split(line_sep):
        fields = line.split(field_sep)
        if fields[0] in (b"layout", b"representation") and len(fields) > 2:
            header[(fields[0], fields[1])] = fields[2:]

    order = [dim.decode() for dim in header[(b"layout", b"order")]]
    sizes = [int(size) for size in header[(b"layout", b"sizes")]]
    assert order[0] == "bits"
    bits, sizes, order = sizes[0], sizes[1:], order[1:]

    fmt = header[(b"representation", b"format")][0]
    sign = header[(b"representation", b"sign")][0]
    if fmt == b"integer":
        dtype = np.dtype(f"{'u' if sign == b'unsigned' else 'i'}{bits // 8}")
    else:
        dtype = np.dtype(f"f{bits // 8}")
    byte_order = header[(b"representation", b"byte_order")]
    dtype = dtype.newbyteorder("<" if byte_order[0] == b"1" else ">")

    raw = data[header_end + len(end_marker):]
    if header.get((b"representation", b"compression"), [b"uncompressed"])[0] == b"gzip":
        raw = zlib.decompress(raw, 16 + zlib.MAX_WBITS)

    # The ICS layout lists the fastest varying axis first, so we reverse to get a C-ordered array.
    volume = np.frombuffer(raw, dtype=dtype).reshape(sizes[::-1])
    # The volumes in BBBC027 are stored in the order x, y, z, i.e. we get a zyx array after reversing.
    assert order[::-1] == ["z", "y", "x"], f"Unexpected axis order in {ics_path}: {order}"
    return volume


def _convert_to_h5(image_dir, label_dir, data_dir):
    import h5py

    image_paths = natsorted(glob(os.path.join(image_dir, "**", "image-final_*.ics"), recursive=True))
    label_paths = natsorted(glob(os.path.join(label_dir, "**", "image-labels_*.ics"), recursive=True))
    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    for image_path, label_path in zip(image_paths, label_paths):
        image_id = os.path.basename(image_path).replace("image-final_", "").replace(".ics", "")
        assert os.path.basename(label_path) == f"image-labels_{image_id}.ics"

        raw = _read_ics(image_path)
        labels = _read_ics(label_path)
        assert raw.shape == labels.shape, f"{raw.shape}, {labels.shape}"

        with h5py.File(os.path.join(data_dir, f"image_{image_id}.h5"), "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_bbbc027_data(
    path: Union[os.PathLike, str], snr: Literal["low", "high"] = "high", download: bool = False
) -> str:
    """Download the BBBC027 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data for the chosen SNR level is stored.
    """
    if snr not in ("low", "high"):
        raise ValueError(f"'{snr}' is not a valid SNR level. Choose from 'low' or 'high'.")

    data_dir = os.path.join(path, "BBBC027", f"{snr}SNR")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    tmp_dir = os.path.join(path, "BBBC027", f"{snr}SNR_ics")
    for kind in ("images", "foreground"):
        for part in (1, 2, 3):
            zip_path = os.path.join(path, f"BBBC027_{snr}SNR_{kind}_part{part}.zip")
            url = URL.format(snr=snr, kind=kind, part=part)
            util.download_source(zip_path, url, download, CHECKSUMS[snr][kind][part - 1])
            util.unzip(zip_path, os.path.join(tmp_dir, kind))

    os.makedirs(data_dir, exist_ok=True)
    _convert_to_h5(os.path.join(tmp_dir, "images"), os.path.join(tmp_dir, "foreground"), data_dir)
    shutil.rmtree(tmp_dir)

    return data_dir


def get_bbbc027_paths(
    path: Union[os.PathLike, str], snr: Literal["low", "high"] = "high", download: bool = False
) -> List[str]:
    """Get paths to the BBBC027 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the HDF5 files, which contain the image data (key 'raw') and the labels (key 'labels').
    """
    data_dir = get_bbbc027_data(path, snr, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "image_*.h5")))
    assert len(volume_paths) > 0
    return volume_paths


def get_bbbc027_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    snr: Literal["low", "high"] = "high",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC027 dataset for nucleus foreground segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_bbbc027_paths(path, snr, download)

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


def get_bbbc027_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    snr: Literal["low", "high"] = "high",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC027 dataloader for nucleus foreground segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc027_dataset(path, patch_shape, snr, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
