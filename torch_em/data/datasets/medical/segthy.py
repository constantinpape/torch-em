"""The SegThy dataset contains annotations for thyroid segmentation in MRI and US scans,
and additional annotations for vein and artery segmentation in MRI.

NOTE: The label legends are described as following:
1: For thyroid-only labels: (at 'MRI_thyroid' or 'US_thyroid')
- background: 0 and thyroid: 1
2: For thyroid, jugular veins and carotid arteries (at 'MRI_thyroid+jugular+carotid_label')
- background: 0, thyroid: 1, jugular vein: 3 and 5, carotid artery: 2 and 4.

The dataset is located at https://www.cs.cit.tum.de/camp/publications/segthy-dataset/.

This dataset is from the publication https://doi.org/10.1371/journal.pone.0268550.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "MRI": "https://www.campar.in.tum.de/public_datasets/2022_plosone_eilers/MRI_data.zip",
    "US": "https://www.campar.in.tum.de/public_datasets/2022_plosone_eilers/US_data.zip",
}

CHECKSUMS = {
    "MRI": "e9d0599b305dfe36795c45282a8495d3bfb4a872851c221b321d59ed0b11e7eb",
    "US": "52c59ef4db08adfa0e6ea562c7fe747c612f2064e01f907a78b170b02fb459bb",
}


def get_segthy_data(path: Union[os.PathLike, str], source: Literal['MRI', 'US'], download: bool = False):
    """Download the SegThy dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, f"{source}_volunteer_dataset")
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{source}_data.zip")
    util.download_source(path=zip_path, url=URLS[source], download=download, checksum=CHECKSUMS[source])
    util.unzip(zip_path=zip_path, dst=path)

    # NOTE: There is one label with an empty channel.
    if source == "MRI":
        lpath = os.path.join(data_dir, "MRI_thyroid_label", "005_MRI_thyroid_label.nii.gz")

        import nibabel as nib
        # Load the label volume and remove the empty channel.
        label = nib.load(lpath).get_fdata()
        label = label[..., 0]

        # Store the updated label.
        label_nifti = nib.Nifti2Image(label, np.eye(4))
        nib.save(label_nifti, lpath)


def get_segthy_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    source: Literal['MRI', 'US'],
    region: Literal['thyroid', 'thyroid_and_vessels'] = "thyroid",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SegThy data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        source: The source of dataset. Either 'MRI' or 'US.
        region: The labeled regions for the corresponding volumes. Either 'thyroid' or 'thyroid_and_vessels'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_segthy_data(path, source, download)

    if source == "MRI":
        rdir, ldir = "MRI", "MRI_thyroid_label" if region == "thyroid" else "MRI_thyroid+jugular+carotid_label"
        fext = "*.nii.gz"
    else:  # US data
        assert region != "thyroid_and_vessels", "US source does not have labels for both thyroid and vessels."
        rdir, ldir = "ground_truth_data/US", "ground_truth_data/US_thyroid_label"
        fext = "*.nii"

    raw_paths = natsorted(glob(os.path.join(path, f"{source}_volunteer_dataset", rdir, fext)))
    label_paths = natsorted(glob(os.path.join(path, f"{source}_volunteer_dataset", ldir, fext)))

    print(len(raw_paths), len(label_paths))

    if split == "train":
        raw_paths, label_paths = raw_paths[:15], label_paths[:15]
    elif split == "train":
        raw_paths, label_paths = raw_paths[15:20], label_paths[15:20]
    elif split == "train":
        raw_paths, label_paths = raw_paths[20:], label_paths[20:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return raw_paths, label_paths


def get_segthy_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    source: Literal['MRI', 'US'],
    region: Literal['thyroid', 'thyroid_and_vessels'] = "thyroid",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegThy dataset for thyroid (and vessel) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        source: The source of dataset. Either 'MRI' or 'US.
        region: The labeled regions for the corresponding volumes. Either 'thyroid' or 'thyroid_and_vessels'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_segthy_paths(path, split, source, region, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_segthy_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    source: Literal['MRI', 'US'],
    region: Literal['thyroid', 'thyroid_and_vessels'] = "thyroid",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegThy dataloader for thyroid (and vessel) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        source: The source of dataset. Either 'MRI' or 'US.
        region: The labeled regions for the corresponding volumes. Either 'thyroid' or 'thyroid_and_vessels'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Args:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_segthy_dataset(path, patch_shape, split, source, region, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
