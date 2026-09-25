"""The TopAneu dataset contains annotations for vessel-specific intracranial aneurysm
classification and segmentation in computed tomography angiography (CTA) and magnetic
resonance angiography (MRA).

The data was curated for the TopAneu 2026 challenge (https://topaneu-26.grand-challenge.org), which extends
the TopBrain / TopCoW vessel anatomy challenges (`torch_em.data.datasets.medical.topbrain`,
`torch_em.data.datasets.medical.topcow`) to vessel-specific aneurysm classification and segmentation. This
module downloads the training data release (batch-1 and batch-2, published June 15 and July 31 2026), which
consists of 415 annotated angiographies (415 scans from 408 patients) from Lausanne University Hospital
(CHUV), Geneva University Hospitals (HUG), Mie Chuo Medical Center, and public data reused from INSTED and
the Lausanne TOF-MRA aneurysm cohort on OpenNeuro. The modality is selected with the 'modality' argument
('ct' or 'mr').

Three types of voxel-level annotations are provided, selected with the 'label_choice' argument:
- 'location': multiclass aneurysm segmentation, where each aneurysm voxel is assigned to one of 52
  vessel-specific location classes (laterality x anatomical position), see the 'location_mapping.json'
  file of the downloaded data.
- 'type': multiclass aneurysm segmentation by morphological type (saccular, dissecting, fusiform), see the
  'type_mapping.json' file of the downloaded data.
- 'vessel': the vessel anatomy mask predicted by the TopBrain organizer model, see the 'vessel_mapping.json'
  file of the downloaded data. This is a silver-standard annotation, not a manual ground-truth.

The data is hosted in a public SWITCHdrive share at https://drive.switch.ch/index.php/s/O36U43RkChkNcHd (see
also https://topaneu-26.grand-challenge.org/data). It is released under a non-commercial open-use license
that requires attribution, see the 'Terms_of_use.txt' file of the downloaded data for the full terms.

The challenge design is described in https://doi.org/10.5281/zenodo.19848807.
Please cite the TopAneu 2026 challenge if you use this dataset in your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from urllib.parse import quote, unquote
from typing import Union, Tuple, List, Optional, Literal

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


SHARE_TOKEN = "O36U43RkChkNcHd"

URL = f"https://drive.switch.ch/index.php/s/{SHARE_TOKEN}"

WEBDAV_URL = "https://drive.switch.ch/public.php/webdav"

# The files are downloaded individually from the public share, so there is no checksum for a single archive.
CHECKSUM = None

MODALITIES = ["ct", "mr"]

LABEL_CHOICES = {"location": "location_masks", "type": "type_masks", "vessel": "vessel_masks"}


def _list_share_folder(folder):
    """List the file names in a folder of the public share via the WebDAV endpoint of Nextcloud.

    Public shares are accessed by using the share token as the user name and an empty password.
    """
    response = requests.request(
        "PROPFIND", f"{WEBDAV_URL}/{quote(folder)}/", headers={"Depth": "1"}, auth=(SHARE_TOKEN, "")
    )
    response.raise_for_status()

    fnames = []
    for href in re.findall(r"<d:href>(.*?)</d:href>", response.text):
        fname = unquote(href).rstrip("/").split("/")[-1]
        if fname.endswith(".nii.gz"):
            fnames.append(fname)
    return natsorted(fnames)


def _download_share_folder(folder, dst, download):
    """Download all nifti files of a folder of the public share into `dst`."""
    if os.path.exists(dst):
        return

    if not download:
        raise RuntimeError(f"Cannot find the data at {dst}, but download was set to False.")

    tmp_dir = f"{dst}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    fnames = _list_share_folder(folder)
    for fname in tqdm(fnames, desc=f"Download {len(fnames)} files from '{folder}'"):
        out_path = os.path.join(tmp_dir, fname)
        if os.path.exists(out_path):
            continue
        url = f"{URL}/download?path={quote('/' + folder)}&files={quote(fname)}"
        util.download_source(path=out_path, url=url, download=download, checksum=CHECKSUM)

    os.rename(tmp_dir, dst)


def get_topaneu_data(
    path: Union[os.PathLike, str], label_choice: Literal["location", "type", "vessel"] = "location",
    download: bool = False,
) -> str:
    """Download the TopAneu dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of segmentation target. One of 'location', 'type' or 'vessel'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if label_choice not in LABEL_CHOICES:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose one of {list(LABEL_CHOICES)}.")

    data_dir = os.path.join(path, "topaneu")
    os.makedirs(path, exist_ok=True)

    _download_share_folder("images", os.path.join(data_dir, "images"), download)
    _download_share_folder(LABEL_CHOICES[label_choice], os.path.join(data_dir, LABEL_CHOICES[label_choice]), download)

    return data_dir


def get_topaneu_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["location", "type", "vessel"] = "location",
    modality: Optional[Literal["ct", "mr"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TopAneu data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of segmentation target. One of 'location', 'type' or 'vessel'.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_topaneu_data(path, label_choice, download)

    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {MODALITIES}.")

    pattern = "topaneu_*.nii.gz" if modality is None else f"topaneu_*_{modality}_*.nii.gz"
    label_paths = natsorted(glob(os.path.join(data_dir, LABEL_CHOICES[label_choice], pattern)))
    # The images carry the channel suffix '_0000' of the nnU-Net format, the labels do not.
    raw_paths = [
        os.path.join(data_dir, "images", os.path.basename(p).replace(".nii.gz", "_0000.nii.gz"))
        for p in label_paths
    ]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_topaneu_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["location", "type", "vessel"] = "location",
    modality: Optional[Literal["ct", "mr"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TopAneu dataset for vessel-specific intracranial aneurysm segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of segmentation target. One of 'location', 'type' or 'vessel'.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_topaneu_paths(path, label_choice, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_topaneu_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["location", "type", "vessel"] = "location",
    modality: Optional[Literal["ct", "mr"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TopAneu dataloader for vessel-specific intracranial aneurysm segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of segmentation target. One of 'location', 'type' or 'vessel'.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_topaneu_dataset(path, patch_shape, label_choice, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
