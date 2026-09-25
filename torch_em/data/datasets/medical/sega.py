"""The SegA dataset contains annotations for aorta segmentation in CT scans.

The dataset is from the publication https://doi.org/10.1007/978-3-031-53241-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "kits": "https://figshare.com/ndownloader/files/30950821",
    "rider": "https://figshare.com/ndownloader/files/30950914",
    "dongyang": "https://figshare.com/ndownloader/files/30950971"
}

CHECKSUMS = {
    "kits": "6c9c2ea31e5998348acf1c4f6683ae07041bd6c8caf309dd049adc7f222de26e",
    "rider": "7244038a6a4f70ae70b9288a2ce874d32128181de2177c63a7612d9ab3c4f5fa",
    "dongyang": "0187e90038cba0564e6304ef0182969ff57a31b42c5969d2b9188a27219da541"
}

ZIPFILES = {
    "kits": "KiTS.zip",
    "rider": "Rider.zip",
    "dongyang": "Dongyang.zip"
}


def get_sega_data(
    path: Union[os.PathLike, str],
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    download: bool = False
) -> str:
    """Download the SegA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_choice = data_choice.lower()
    zip_fid = ZIPFILES[data_choice]
    data_dir = os.path.join(path, Path(zip_fid).stem)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, zip_fid)
    util.download_source(path=zip_path, url=URL[data_choice], download=download, checksum=CHECKSUMS[data_choice])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_sega_paths(
    path: Union[os.PathLike, str],
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SegA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if data_choice is None:
        data_choices = URL.keys()
    else:
        if isinstance(data_choice, str):
            data_choices = [data_choice]

    data_dirs = [get_sega_data(path=path, data_choice=data_choice, download=download) for data_choice in data_choices]

    image_paths, gt_paths = [], []
    for data_dir in data_dirs:
        all_volumes_paths = glob(os.path.join(data_dir, "*", "*.nrrd"))
        for volume_path in all_volumes_paths:
            if volume_path.endswith(".seg.nrrd"):
                gt_paths.append(volume_path)
            else:
                image_paths.append(volume_path)

    # now let's wrap the volumes to nifti format
    fimage_dir = os.path.join(path, "data", "images")
    fgt_dir = os.path.join(path, "data", "labels")

    os.makedirs(fimage_dir, exist_ok=True)
    os.makedirs(fgt_dir, exist_ok=True)

    fimage_paths, fgt_paths = [], []
    for image_path, gt_path in zip(natsorted(image_paths), natsorted(gt_paths)):
        fimage_path = os.path.join(fimage_dir, f"{Path(image_path).stem}.nii.gz")
        fgt_path = os.path.join(fgt_dir, f"{Path(image_path).stem}.nii.gz")

        fimage_paths.append(fimage_path)
        fgt_paths.append(fgt_path)

        if os.path.exists(fimage_path) and os.path.exists(fgt_path):
            continue

        import nrrd
        import numpy as np
        import nibabel as nib

        image = nrrd.read(image_path)[0]
        gt = nrrd.read(gt_path)[0]

        image_nifti = nib.Nifti2Image(image, np.eye(4))
        gt_nifti = nib.Nifti2Image(gt, np.eye(4))

        nib.save(image_nifti, fimage_path)
        nib.save(gt_nifti, fgt_path)

    return natsorted(fimage_paths), natsorted(fgt_paths)


def get_sega_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegA dataset for segmentation of aorta in computed tomography angiography (CTA) scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        data_choice: The choice of dataset.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_sega_paths(path, data_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs,
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_sega_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegA dataloader for segmentation of aorta in computed tomography angiography (CTA) scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        data_choice: The choice of dataset.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sega_dataset(path, patch_shape, data_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
