"""The AortaSeg dataset contains annotations for multi-class segmentation of the aortic branches and zones
in computed tomography angiography (CTA) scans.

It comprises the training set of the AortaSeg24 challenge (https://aortaseg24.grand-challenge.org):
50 CTA volumes of patients with uncomplicated type B aortic dissection, resampled to an isotropic resolution
of 1mm, with 23 annotated aortic zones and branches.

NOTE: The label legend is as follows (see `AORTIC_SEGMENTS` and `LABEL_IDS`; 1 = Zone 0, 2 = Innominate Artery,
..., 23 = Zone 11 L). The ids were taken from the official evaluation code of the challenge
(https://github.com/ImranNust/AortaSeg24/blob/main/evaluation_docker_for_validation_phase/evaluate.py),
which one-hot encodes the labels with 24 classes and reports the per-class dice in this order.

NOTE: The dataset requires registration and cannot be downloaded automatically. Please follow these steps:
- Visit https://aortaseg24.grand-challenge.org/dataset-access-information/ and complete the dataset access
  agreement form via the DocuSign link given there. The approval may take up to 24 hours.
- Join the challenge at https://aortaseg24.grand-challenge.org/ and request access to the dataset on the
  dataset page. You will then receive the link to the Dropbox folder with the data.
- Download the training images and masks and place them at '<path>', such that
  '<path>/images/subject001_CTA.mha' and '<path>/masks/subject001_label.mha' exist.

The dataset is located at https://aortaseg24.grand-challenge.org/dataset-access-information/.

This dataset is from the publication https://doi.org/10.1016/j.media.2026.104188.
Please cite it if you use this dataset in your research.

NOTE: Reading the MetaImage (.mha) volumes requires 'SimpleITK'. Install it with 'pip install SimpleITK'.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


AORTIC_SEGMENTS = [
    "Zone_0", "Innominate_Artery", "Zone_1", "Left_Common_Carotid", "Zone_2", "Left_Subclavian_Artery", "Zone_3",
    "Zone_4", "Zone_5", "Zone_6", "Celiac_Artery", "Zone_7", "SMA", "Zone_8", "Right_Renal_Artery",
    "Left_Renal_Artery", "Zone_9", "Zone_10_R", "Zone_10_L", "Right_Internal_Iliac_Artery",
    "Left_Internal_Iliac_Artery", "Zone_11_R", "Zone_11_L",
]

LABEL_IDS = {"background": 0, **{name: i + 1 for i, name in enumerate(AORTIC_SEGMENTS)}}


def get_aortaseg24_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the AortaSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if download:
        msg = "Download is set to True, but 'torch_em' cannot download this dataset. "
        msg += "See 'torch_em.data.datasets.medical.aortaseg24' for the manual download instructions."
        raise NotImplementedError(msg)

    # The data is either placed directly in 'path' or in a subfolder, e.g. named after the downloaded archive.
    image_dirs = [p for p in glob(os.path.join(path, "**", "images"), recursive=True) if os.path.isdir(p)]
    if len(image_dirs) == 0:
        raise FileNotFoundError(
            f"It's expected to place the downloaded AortaSeg24 training data at '{path}'. "
            "See 'torch_em.data.datasets.medical.aortaseg24' for the manual download instructions."
        )

    return os.path.split(image_dirs[0])[0]


def get_aortaseg24_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the AortaSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_aortaseg24_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*_CTA.mha")))
    label_paths = [
        os.path.join(data_dir, "masks", f"{os.path.basename(p)[:-len('_CTA.mha')]}_label.mha") for p in raw_paths
    ]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_aortaseg24_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AortaSeg dataset for aortic branch and zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_aortaseg24_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_aortaseg24_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AortaSeg dataloader for aortic branch and zone segmentation.

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
    dataset = get_aortaseg24_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
