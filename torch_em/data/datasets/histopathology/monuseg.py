"""This dataset contains annotations for nucleus segmentation in
H&E stained tissue images derived from different organs.

This dataset comes from https://monuseg.grand-challenge.org/Data/.

Please cite the relevant publications from the challenge
if you use this dataset in your research.
"""

import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
from typing import List, Optional, Union, Tuple, Literal

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "train": "https://drive.google.com/uc?export=download&id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
    "test": "https://drive.google.com/uc?export=download&id=1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw"
}

CHECKSUM = {
    "train": "25d3d3185bb2970b397cafa72eb664c9b4d24294aee382e7e3df9885affce742",
    "test": "13e522387ae8b1bcc0530e13ff9c7b4d91ec74959ef6f6e57747368d7ee6f88a"
}

# Here is the description: https://drive.google.com/file/d/1xYyQ31CHFRnvTCTuuHdconlJCMk2SK7Z/view?usp=sharing
ORGAN_SPLITS = {
    "breast": [
        "TCGA-A7-A13E-01Z-00-DX1", "TCGA-A7-A13F-01Z-00-DX1", "TCGA-AR-A1AK-01Z-00-DX1",
        "TCGA-AR-A1AS-01Z-00-DX1", "TCGA-E2-A1B5-01Z-00-DX1", "TCGA-E2-A14V-01Z-00-DX1"
    ],
    "kidney": [
        "TCGA-B0-5711-01Z-00-DX1", "TCGA-HE-7128-01Z-00-DX1", "TCGA-HE-7129-01Z-00-DX1",
        "TCGA-HE-7130-01Z-00-DX1", "TCGA-B0-5710-01Z-00-DX1", "TCGA-B0-5698-01Z-00-DX1"
    ],
    "liver": [
        "TCGA-18-5592-01Z-00-DX1", "TCGA-38-6178-01Z-00-DX1", "TCGA-49-4488-01Z-00-DX1",
        "TCGA-50-5931-01Z-00-DX1", "TCGA-21-5784-01Z-00-DX1", "TCGA-21-5786-01Z-00-DX1"
    ],
    "prostate": [
        "TCGA-G9-6336-01Z-00-DX1", "TCGA-G9-6348-01Z-00-DX1", "TCGA-G9-6356-01Z-00-DX1",
        "TCGA-G9-6363-01Z-00-DX1", "TCGA-CH-5767-01Z-00-DX1", "TCGA-G9-6362-01Z-00-DX1"
    ],
    "bladder": ["TCGA-DK-A2I6-01A-01-TS1", "TCGA-G2-A2EK-01A-02-TSB"],
    "colon": ["TCGA-AY-A8YK-01A-01-TS1", "TCGA-NH-A8F7-01A-01-TS1"],
    "stomach": ["TCGA-KB-A93J-01A-01-TS1", "TCGA-RD-A8N9-01A-01-TS1"]
}


def _process_monuseg(path, split):
    util.unzip(os.path.join(path, f"monuseg_{split}.zip"), path)

    # assorting the images into expected dir;
    # converting the label xml files to numpy arrays (of same dimension as input images) in the expected dir
    root_img_save_dir = os.path.join(path, "images", split)
    root_label_save_dir = os.path.join(path, "labels", split)

    os.makedirs(root_img_save_dir, exist_ok=True)
    os.makedirs(root_label_save_dir, exist_ok=True)

    if split == "train":
        all_img_dir = sorted(glob(os.path.join(path, "*", "Tissue*", "*")))
        all_xml_label_dir = sorted(glob(os.path.join(path, "*", "Annotations", "*")))
    else:
        all_img_dir = sorted(glob(os.path.join(path, "MoNuSegTestData", "*.tif")))
        all_xml_label_dir = sorted(glob(os.path.join(path, "MoNuSegTestData", "*.xml")))

    assert len(all_img_dir) == len(all_xml_label_dir)

    for img_path, xml_label_path in tqdm(
        zip(all_img_dir, all_xml_label_dir),
        desc=f"Converting {split} split to the expected format",
        total=len(all_img_dir)
    ):
        desired_label_shape = imageio.imread(img_path).shape[:-1]

        img_id = os.path.split(img_path)[-1]
        dst = os.path.join(root_img_save_dir, img_id)
        shutil.move(src=img_path, dst=dst)

        _label = util.generate_labeled_array_from_xml(shape=desired_label_shape, xml_file=xml_label_path)
        _fileid = img_id.split(".")[0]
        imageio.imwrite(os.path.join(root_label_save_dir, f"{_fileid}.tif"), _label, compression="zlib")

    shutil.rmtree(glob(os.path.join(path, "MoNuSeg*"))[0])
    if split == "train":
        shutil.rmtree(glob(os.path.join(path, "__MACOSX"))[0])


def get_monuseg_data(path: Union[os.PathLike, str], split: Literal['train', 'test'], download: bool = False):
    """Download the MoNuSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
    """
    assert split in ["train", "test"], "The split choices in MoNuSeg datset are train/test, please choose from them"

    # check if we have extracted the images and labels already
    im_path = os.path.join(path, "images", split)
    label_path = os.path.join(path, "labels", split)
    if os.path.exists(im_path) and os.path.exists(label_path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"monuseg_{split}.zip")
    util.download_source_gdrive(zip_path, URL[split], download=download, checksum=CHECKSUM[split])

    _process_monuseg(path, split)


def get_monuseg_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    organ_type: Optional[List[str]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the MoNuSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        organ_type: The choice of organ type.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    get_monuseg_data(path, split, download)

    image_paths = sorted(glob(os.path.join(path, "images", split, "*")))
    label_paths = sorted(glob(os.path.join(path, "labels", split, "*")))

    if split == "train" and organ_type is not None:
        # get all patients for multiple organ selection
        all_organ_splits = sum([ORGAN_SPLITS[_o] for _o in organ_type], [])

        image_paths = [_path for _path in image_paths if Path(_path).stem in all_organ_splits]
        label_paths = [_path for _path in label_paths if Path(_path).stem in all_organ_splits]

    elif split == "test" and organ_type is not None:
        # we don't have organ splits in the test dataset
        raise ValueError("The test split does not have any organ informations, please pass `organ_type=None`")

    return image_paths, label_paths


def get_monuseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'test'],
    organ_type: Optional[List[str]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the MoNuSeg dataset for nucleus segmentation in H&E stained tissue images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use for the dataset. Either 'train' or 'test'.
        organ_type: The choice of organ type.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_monuseg_paths(path, split, organ_type, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_monuseg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: Literal['train', 'test'],
    organ_type: Optional[List[str]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MoNuSeg dataloader for nucleus segmentation in H&E stained tissue images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The split to use for the dataset. Either 'train' or 'test'.
        organ_type: The choice of organ type.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_monuseg_dataset(
        path, patch_shape, split, organ_type, download, offsets, boundaries, binary, resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
