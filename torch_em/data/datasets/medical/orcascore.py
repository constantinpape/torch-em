"""The orCaScore dataset contains annotations for coronary artery calcification in cardiac CT.

The data was curated for the orCaScore challenge (https://orcascore.grand-challenge.org), which was held at
MICCAI 2014 and evaluates automatic coronary artery calcium scoring. The training set consists of 32 patients,
8 from each of four CT scanner vendors, and the test set of 40 patients without a reference standard.
Each examination consists of a non-contrast enhanced CT image (the file suffix 'CTI') and a contrast enhanced
CTA image (the file suffix 'CTAI'). The reference standard (the file suffix 'R') was annotated on the
non-contrast CT images only, so this module pairs the labels with the CT images.

The reference standard labels every lesion with intensities above 130 HU by the coronary artery it belongs to,
see `LABEL_IDS`: 1 = left anterior descending artery (calcifications of the left main coronary artery are
labeled as LAD as well), 2 = left circumflex artery, 3 = right coronary artery.

NOTE: The data is only handed out to registered participants and the challenge has been closed for new
registrations since August 2025, so it cannot be downloaded automatically. There is no openly published
copy of it: the data was behind a signed confidentiality agreement from the start, and the original
challenge website 'orcascore.isi.uu.nl' no longer resolves. To obtain the data, please follow these steps:
- Ask the challenge organizers for access to the training data, either through
  https://orcascore.grand-challenge.org or by writing to 'j.m.wolterink@utwente.nl'.
- Place the four downloaded archives 'Train_V1.rar', 'Train_V2.rar', 'Train_V3.rar' and 'Train_V4.rar' in the
  folder passed as 'path'. This module extracts them, or picks up the extracted MetaImage files
  ('TRV<vendor>P<patient>CTI.mhd' with the matching '.raw' or '.zraw') if they are already extracted.

NOTE: The filenames and the archive names are taken from the archived download page of the original
challenge website, http://web.archive.org/web/20180418103434/http://orcascore.isi.uu.nl:80/download/. It
documents 'TR' for the training set (the test set uses 'TE'), 'V' for the vendor, 'P' for the patient,
the suffix 'CTI' for the non-contrast CT, 'CTAI' for the CTA and 'R' for the reference standard, and the
archive sizes 390 MiB, 815 MiB, 500 MiB and 608 MiB for the four training archives.

The MetaImage volumes are converted to hdf5 volumes (the keys are 'raw' and 'labels') by this module.

The dataset is located at https://orcascore.grand-challenge.org.

This dataset is from the publication https://doi.org/10.1118/1.4945696.
Please cite it if you use this dataset in your research.
"""

import os
import zlib
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {"background": 0, "LAD": 1, "LCX": 2, "RCA": 3}

ARCHIVE_NAMES = ["Train_V1.rar", "Train_V2.rar", "Train_V3.rar", "Train_V4.rar"]

N_VOLUMES = 32

METAIMAGE_DTYPES = {
    "MET_CHAR": "int8",
    "MET_UCHAR": "uint8",
    "MET_SHORT": "int16",
    "MET_USHORT": "uint16",
    "MET_INT": "int32",
    "MET_UINT": "uint32",
    "MET_FLOAT": "float32",
    "MET_DOUBLE": "float64",
}


def _read_metaimage(mhd_path):
    """Read a MetaImage volume (a '.mhd' header with a '.raw' or a zlib compressed '.zraw' data file)."""
    header = {}
    with open(mhd_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()

    shape = [int(v) for v in header["DimSize"].split()][::-1]  # The data is stored with the first axis last.
    dtype = np.dtype(METAIMAGE_DTYPES[header["ElementType"]])
    dtype = dtype.newbyteorder(">" if header.get("BinaryDataByteOrderMSB", "False") == "True" else "<")

    with open(os.path.join(os.path.dirname(mhd_path), header["ElementDataFile"]), "rb") as f:
        data = f.read()
    if header.get("CompressedData", "False") == "True":
        data = zlib.decompress(data)

    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _preprocess_inputs(data_dir, preprocessed_dir):
    import h5py

    label_paths = natsorted(glob(os.path.join(data_dir, "**", "TRV*P*R.mhd"), recursive=True))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for label_path in tqdm(label_paths, desc="Preprocessing the orCaScore cases"):
        case_id = os.path.basename(label_path)[:-len("R.mhd")]
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        raw = _read_metaimage(os.path.join(os.path.dirname(label_path), f"{case_id}CTI.mhd"))
        labels = _read_metaimage(label_path)

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_orcascore_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the orCaScore dataset.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present. The data cannot be downloaded
            automatically, so this raises if the data has not been downloaded manually.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_VOLUMES:
        return preprocessed_dir

    if not glob(os.path.join(path, "**", "TRV*P*R.mhd"), recursive=True):
        archive_paths = [os.path.join(path, name) for name in ARCHIVE_NAMES]
        if not any(os.path.exists(p) for p in archive_paths):
            msg = "'torch_em' cannot download this dataset, because the orCaScore data is only handed out to "
            msg += "registered participants and the challenge has been closed for new registrations. Please ask "
            msg += "the organizers at 'https://orcascore.grand-challenge.org' or at 'j.m.wolterink@utwente.nl' for "
            msg += f"the training data and place {ARCHIVE_NAMES} in '{path}'."
            raise NotImplementedError(msg)

        for archive_path in archive_paths:
            util.unzip_rarfile(rar_path=archive_path, dst=path, remove=False)

    _preprocess_inputs(path, preprocessed_dir)
    return preprocessed_dir


def get_orcascore_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the orCaScore data.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_orcascore_data(path, download)
    return natsorted(glob(os.path.join(data_dir, "*.h5")))


def get_orcascore_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the orCaScore dataset for coronary artery calcification segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_orcascore_paths(path, download)

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


def get_orcascore_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the orCaScore dataloader for coronary artery calcification segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orcascore_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
