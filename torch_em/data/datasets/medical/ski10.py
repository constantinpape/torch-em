"""The SKI10 dataset contains annotations for knee bone and cartilage segmentation in MRI.

The data was curated for the SKI10 challenge (Segmentation of Knee Images 2010, https://ski10.grand-challenge.org),
which was held at MICCAI 2010. It consists of the 100 annotated training MRI of the challenge, which come from the
surgical planning program of Biomet Inc. The four annotated structures are the femoral bone and cartilage and the
tibial bone and cartilage, see `LABEL_IDS`. The 50 test MRI of the challenge were distributed without annotations
and are not included here.

NOTE: The official challenge is closed and its data cannot be downloaded from the challenge website anymore.
This module uses the redistribution at https://huggingface.co/datasets/YongchengYAO/SKI10 (CC BY-NC-SA 4.0),
which converted the original 'mhd' / 'raw' volumes to nifti without changing the image or mask values.

The volumes are stored with the slice axis last (the in-plane resolution is around 0.4 mm and the slice
thickness is 1 mm), so they are converted to hdf5 volumes with the slice axis first (the keys are 'raw' and
'labels') by this module.

This dataset is from the publication 'Segmentation of Knee Images: A Grand Challenge' by Heimann et al.,
MICCAI Workshop on Medical Image Analysis for the Clinic (2010), which has no DOI. The challenge papers are
collected at https://doi.org/10.5281/zenodo.4781231.
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


URL = "https://huggingface.co/datasets/YongchengYAO/SKI10/resolve/main/SKI10.zip"
CHECKSUM = "d367c1c68143f450e4cad92111afc064f116372442a4816718e20ea556507bc8"

LABEL_IDS = {
    "background": 0, "femur_bone": 1, "femur_cartilage": 2, "tibia_bone": 3, "tibia_cartilage": 4,
}

N_VOLUMES = 100


def _preprocess_inputs(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    image_paths = natsorted(glob(os.path.join(data_dir, "image", "image-*.nii")))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Preprocessing the SKI10 volumes"):
        case_id = os.path.basename(image_path)[len("image-"):-len(".nii")]
        volume_path = os.path.join(preprocessed_dir, f"ski10_{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        label_path = os.path.join(data_dir, "label", f"labels-{case_id}.nii")

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes,
        # so that the first axis is the slice axis of the acquisition.
        raw = np.asarray(nib.load(image_path).dataobj).T
        labels = np.asarray(nib.load(label_path).dataobj).T.astype("uint8")

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_ski10_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SKI10 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_VOLUMES:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "SKI10")
    if not os.path.exists(data_dir):
        zip_path = os.path.join(path, "SKI10.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_inputs(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_ski10_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the SKI10 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_ski10_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "ski10_*.h5")))
    assert len(volume_paths) > 0

    return volume_paths


def get_ski10_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SKI10 dataset for knee bone and cartilage segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_ski10_paths(path, download)

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


def get_ski10_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SKI10 dataloader for knee bone and cartilage segmentation.

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
    dataset = get_ski10_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
