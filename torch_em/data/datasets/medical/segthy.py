"""

NOTE: The label legends are described as following:
For thyroid-only labels: (at 'MRI_thyroid' or 'US_thyroid')
- background: 0 and thyroid: 1

For thyroid, jugular veins and carotid arteries (at 'MRI_thyroid+jugular+carotid_label')
- background: 0, thyroid: 1, jugular vein: 3 and 5, carotid artery: 2 and 4.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

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


def get_segthy_data(
    path: Union[os.PathLike, str], source: Literal['MRI', 'US'], download: bool = False
):
    """
    """
    data_dir = os.path.join(path, f"{source}_volunteer_dataset")
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{source}_data.zip")
    util.download_source(path=zip_path, url=URLS[source], download=download, checksum=CHECKSUMS[source])
    util.unzip(zip_path=zip_path, dst=path)


def get_segthy_paths(
    path: Union[os.PathLike, str],
    source: Literal['MRI', 'US'],
    roi: Literal['thyroid', 'thyroid_and_vessels'] = "thyroid",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """
    """
    get_segthy_data(path, source, download)

    if source == "MRI":
        rdir, ldir = "MRI", "MRI_thyroid_label" if roi == "thyroid" else "MRI_thyroid+jugular+carotid_label"
        fext = "*.nii.gz"
    else:  # US data
        assert roi != "thyroid_and_vessels", "US source does not have labels for both thyroid and vessels."
        rdir, ldir = "ground_truth_data/US", "ground_truth_data/US_thyroid_label"
        fext = "*.nii"

    raw_paths = natsorted(glob(os.path.join(path, f"{source}_volunteer_dataset", rdir, fext)))
    label_paths = natsorted(glob(os.path.join(path, f"{source}_volunteer_dataset", ldir, fext)))

    return raw_paths, label_paths


def get_segthy_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Literal['MRI', 'US'],
    roi: Literal['thyroid', 'thyroid_and_vessels'] = "thyroid",
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_segthy_paths(path, source, roi, download)

    # HACK
    for rpath, lpath in zip(raw_paths, label_paths):
        from tukra.io import read_image
        print(read_image(rpath, ".nii").shape, read_image(lpath, ".nii").shape)

    breakpoint()

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
    source: Literal['MRI', 'US'],
    roi: Literal['thyroid', 'thyroid_and_vessels'] = "thyroid",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_segthy_dataset(path, patch_shape, source, roi, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
