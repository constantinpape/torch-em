"""The RETOUCH dataset contains annotations for intraretinal fluid (IRF), subretinal fluid (SRF)
and pigment epithelial detachment (PED) segmentation in 3D retinal OCT volumes.

The dataset was curated for the RETOUCH challenge (MICCAI 2017, https://retouch.grand-challenge.org).
It comprises 112 OCT volumes acquired with three different device manufacturers (Cirrus, Spectralis
and Topcon), out of which 70 volumes (24 Cirrus, 24 Spectralis, 22 Topcon) form the labeled training
set and 42 volumes form the unlabeled test set. Only the labeled training set is exposed here, as the
test set annotations are not publicly available.

NOTE: The dataset is gated and cannot be downloaded automatically by 'torch_em'. To obtain it:
- Visit https://retouch.grand-challenge.org and click on 'Join' to register for the challenge.
- Complete the registration form and sign the 'RETOUCH-Agreement_of_Data_Confidentiality' document.
- Once your registration is approved (contact hrvoje.bogunovic@meduniwien.ac.at for questions),
  you will get access to the 'Download' page, from which the training data must be downloaded.
- Extract the downloaded archive(s) to the desired 'path'. The expected layout per vendor is:
  '<path>/RETOUCH-TrainingSet-<vendor>/TRAIN<id>/oct.mhd', 'oct.raw', 'reference.mhd', 'reference.raw',
  where '<vendor>' is one of 'Cirrus', 'Spectralis', 'Topcon'.

The original MetaImage ('.mhd' / '.raw') volumes are converted to hdf5 once and then loaded from there.

The reference annotations use the label ids: 0 = background, 1 = IRF, 2 = SRF, 3 = PED.

The dataset is from the publication https://doi.org/10.1109/TMI.2019.2901398.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Sequence

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


VENDORS = ["Cirrus", "Spectralis", "Topcon"]

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


def get_retouch_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the RETOUCH dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the converted hdf5 volumes.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    vendor_dirs = [os.path.join(path, f"RETOUCH-TrainingSet-{vendor}") for vendor in VENDORS]
    if not any(os.path.exists(vendor_dir) for vendor_dir in vendor_dirs):
        msg = "The RETOUCH dataset is gated and cannot be downloaded automatically. To obtain it:\n" \
            "- Visit https://retouch.grand-challenge.org and click on 'Join' to register for the challenge.\n" \
            "- Complete the registration form and sign the 'RETOUCH-Agreement_of_Data_Confidentiality' " \
            "document. Contact hrvoje.bogunovic@meduniwien.ac.at for registration questions.\n" \
            "- Once approved, download the training data from the challenge 'Download' page.\n" \
            f"- Extract the downloaded archive(s) so that '{path}' contains one folder per vendor, " \
            "named 'RETOUCH-TrainingSet-Cirrus', 'RETOUCH-TrainingSet-Spectralis' and " \
            "'RETOUCH-TrainingSet-Topcon', each with 'TRAIN<id>' subfolders containing " \
            "'oct.mhd', 'oct.raw', 'reference.mhd' and 'reference.raw'."
        if download:
            msg = "Download is set to True, but 'torch_em' cannot download this dataset.\n" + msg
        raise RuntimeError(msg)

    import h5py

    os.makedirs(data_dir, exist_ok=True)
    for vendor_dir in vendor_dirs:
        if not os.path.exists(vendor_dir):
            continue

        vendor = os.path.basename(vendor_dir).replace("RETOUCH-TrainingSet-", "")
        case_dirs = natsorted(glob(os.path.join(vendor_dir, "TRAIN*")))
        for case_dir in tqdm(case_dirs, desc=f"Converting RETOUCH volumes for '{os.path.basename(vendor_dir)}'"):
            case_id = os.path.basename(case_dir)
            out_path = os.path.join(data_dir, f"{vendor}_{case_id}.h5")
            if os.path.exists(out_path):
                continue

            raw = _read_metaimage(os.path.join(case_dir, "oct.mhd"))
            labels = _read_metaimage(os.path.join(case_dir, "reference.mhd")).astype("uint8")
            assert raw.shape == labels.shape, f"Shape mismatch for '{case_dir}': {raw.shape} vs. {labels.shape}."

            with h5py.File(out_path, "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels", data=labels, compression="gzip")

    return data_dir


def get_retouch_paths(
    path: Union[os.PathLike, str], vendor: Union[Literal["Cirrus", "Spectralis", "Topcon"], Sequence[str], None] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the RETOUCH data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        vendor: The choice of device vendor(s). By default, volumes from all vendors are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 volumes, which store the image data at 'raw' and the label data at 'labels'.
    """
    data_dir = get_retouch_data(path, download)

    if vendor is None:
        vendors = VENDORS
    else:
        vendors = [vendor] if isinstance(vendor, str) else list(vendor)
        for v in vendors:
            if v not in VENDORS:
                raise ValueError(f"'{v}' is not a valid vendor. Choose from {VENDORS}.")

    volume_paths = []
    for v in vendors:
        volume_paths.extend(natsorted(glob(os.path.join(data_dir, f"{v}_TRAIN*.h5"))))

    assert len(volume_paths) > 0, f"No RETOUCH volumes found at '{data_dir}'."
    return volume_paths


def get_retouch_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    vendor: Union[Literal["Cirrus", "Spectralis", "Topcon"], Sequence[str], None] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RETOUCH dataset for retinal fluid segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        vendor: The choice of device vendor(s). By default, volumes from all vendors are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_retouch_paths(path, vendor, download)

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


def get_retouch_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    vendor: Union[Literal["Cirrus", "Spectralis", "Topcon"], Sequence[str], None] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RETOUCH dataloader for retinal fluid segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        vendor: The choice of device vendor(s). By default, volumes from all vendors are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_retouch_dataset(path, patch_shape, vendor, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
