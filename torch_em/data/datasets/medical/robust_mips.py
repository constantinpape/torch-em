"""ROBUST-MIPS (Robust Minimally Invasive Pelvic Surgery) is a dataset for surgical
instrument instance segmentation and pose estimation in laparoscopic pelvic surgery.

The dataset contains 10,040 frames (5,983 for training, 4,057 for testing) sampled from
recordings of proctocolectomy, rectal resection and sigmoid resection surgeries. Each frame
has a raw endoscopy image, an instance segmentation mask for the surgical instruments, and
a JSON file with the instrument tool-tip / pose keypoints (not used by this module).

NOTE: The dataset is hosted on Synapse. Downloading it requires the 'synapseclient' python
library and a Synapse account with an authentication token stored in the '~/.synapseConfig'
file. See 'get_robust_mips_data' for details. The Synapse project 'syn64023381' is public
and has no access requirements (confirmed via the Synapse REST API).

The dataset is located at https://www.synapse.org/Synapse:syn64023381.
This dataset is from the publication https://doi.org/10.48550/arXiv.2508.21096.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


ENTITY = "syn68915165"


def get_robust_mips_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ROBUST-MIPS dataset.

    Follow the instructions below to get access to the dataset.
    - Create a free account at https://www.synapse.org.
    - Generate a personal access token and store it in a '~/.synapseConfig' file, see
      https://python-docs.synapse.org/tutorials/authentication/ for details.
    - Install the 'synapseclient' python library.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "RobustMIPS")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    zip_path = os.path.join(path, "RobustMIPS.zip")
    if not os.path.exists(zip_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {zip_path}, but download was set to False.")
        syn.get(ENTITY, downloadLocation=path, downloadFile=True)

    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return data_dir


def get_robust_mips_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"] = "train", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ROBUST-MIPS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ["train", "test"]:
        raise ValueError(f"'{split}' is not a valid split. Please choose from 'train' or 'test'.")

    data_dir = get_robust_mips_data(path, download)

    split_dir = "Training" if split == "train" else "Testing"
    image_paths = natsorted(glob(os.path.join(data_dir, split_dir, "**", "raw.png"), recursive=True))
    gt_paths = [os.path.join(os.path.dirname(p), "instrument_instances.png") for p in image_paths]

    assert len(image_paths) > 0, f"No images were found at '{os.path.join(data_dir, split_dir)}'."
    assert all(os.path.exists(p) for p in gt_paths), (
        "Some 'raw.png' frames do not have a matching 'instrument_instances.png' mask. The expected per-frame "
        f"folder layout may not match the actual structure of the downloaded data. Please inspect '{data_dir}'."
    )

    return image_paths, gt_paths


def get_robust_mips_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ROBUST-MIPS dataset for surgical instrument instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_robust_mips_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_robust_mips_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ROBUST-MIPS dataloader for surgical instrument instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_robust_mips_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
