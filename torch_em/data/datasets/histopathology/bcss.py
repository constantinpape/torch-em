"""This dataset contains annotations for tissue region segmentation in
breast cancer histopathology images.

NOTE: There are multiple semantic instances in tissue labels. Below mentioned are their respective index details:
    - 0: outside_roi (~background)
    - 1: tumor
    - 2: stroma
    - 3: lymphocytic_infiltrate
    - 4: necrosis_or_debris
    - 5: glandular_secretions
    - 6: blood
    - 7: exclude
    - 8: metaplasia_NOS
    - 9: fat
    - 10: plasma_cells
    - 11: other_immune_infiltrate
    - 12: mucoid_material
    - 13: normal_acinus_or_duct
    - 14: lymphatics
    - 15: undetermined
    - 16: nerve
    - 17: skin_adnexa
    - 18: blood_vessel
    - 19: angioinvasion
    - 20: dcis
    - 21: other

This dataset is from https://bcsegmentation.grand-challenge.org/BCSS/.
Please cite this paper (https://doi.org/10.1093/bioinformatics/btz083) if you use this dataset for a publication.
"""

import os
import shutil
from glob import glob
from pathlib import Path
from typing import Union, Optional, List, Tuple

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss?usp=sharing"


# TODO
CHECKSUM = None


TEST_LIST = [
    "TCGA-A2-A0SX-DX1_xmin53791_ymin56683_MPP-0.2500", "TCGA-BH-A0BG-DX1_xmin64019_ymin24975_MPP-0.2500",
    "TCGA-AR-A1AI-DX1_xmin38671_ymin10616_MPP-0.2500", "TCGA-E2-A574-DX1_xmin54962_ymin47475_MPP-0.2500",
    "TCGA-GM-A3XL-DX1_xmin29910_ymin15820_MPP-0.2500", "TCGA-E2-A14X-DX1_xmin88836_ymin66393_MPP-0.2500",
    "TCGA-A2-A04P-DX1_xmin104246_ymin48517_MPP-0.2500", "TCGA-E2-A14N-DX1_xmin21383_ymin66838_MPP-0.2500",
    "TCGA-EW-A1OV-DX1_xmin126026_ymin65132_MPP-0.2500", "TCGA-S3-AA15-DX1_xmin55486_ymin28926_MPP-0.2500",
    "TCGA-LL-A5YO-DX1_xmin36631_ymin44396_MPP-0.2500", "TCGA-GI-A2C9-DX1_xmin20882_ymin11843_MPP-0.2500",
    "TCGA-BH-A0BW-DX1_xmin42346_ymin30843_MPP-0.2500", "TCGA-E2-A1B6-DX1_xmin16266_ymin50634_MPP-0.2500",
    "TCGA-AO-A0J2-DX1_xmin33561_ymin14515_MPP-0.2500"
]


def _download_bcss_dataset(path, download):
    """Current recommendation:
        - download the folder from URL manually
        - use the consortium's git repo to download the dataset (https://github.com/PathologyDataScience/BCSS)
    """
    raise NotImplementedError("Please download the dataset using the drive link / git repo directly")

    # FIXME: limitation for the installation below:
    #   - only downloads first 50 files - due to `gdown`'s download folder function
    #   - (optional) clone their git repo to download their data
    util.download_source_gdrive(path=path, url=URL, download=download, checksum=CHECKSUM, download_type="folder")


def _get_image_and_label_paths(path):
    # when downloading the files from `URL`, the input images are stored under `rgbs_colorNormalized`
    # when getting the files from the git repo's command line feature, the input images are stored under `images`
    if os.path.exists(os.path.join(path, "images")):
        image_paths = sorted(glob(os.path.join(path, "images", "*")))
        label_paths = sorted(glob(os.path.join(path, "masks", "*")))
    elif os.path.exists(os.path.join(path, "0_Public-data-Amgad2019_0.25MPP", "rgbs_colorNormalized")):
        image_paths = sorted(glob(os.path.join(path, "0_Public-data-Amgad2019_0.25MPP", "rgbs_colorNormalized", "*")))
        label_paths = sorted(glob(os.path.join(path, "0_Public-data-Amgad2019_0.25MPP", "masks", "*")))
    else:
        raise ValueError(
            "Please check the image directory. "
            "If downloaded from gdrive, it's named \"rgbs_colorNormalized\", if from github it's named \"images\""
        )

    return image_paths, label_paths


def get_bcss_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the BCSS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    if download:
        _download_bcss_dataset(path, download)

    if os.path.exists(os.path.join(path, "train")) and os.path.exists(os.path.join(path, "test")):
        return

    all_image_paths, all_label_paths = _get_image_and_label_paths(path)

    train_img_dir, train_lab_dir = os.path.join(path, "train", "images"), os.path.join(path, "train", "masks")
    test_img_dir, test_lab_dir = os.path.join(path, "test", "images"), os.path.join(path, "test", "masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lab_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lab_dir, exist_ok=True)

    for image_path, label_path in zip(all_image_paths, all_label_paths):
        img_idx, label_idx = os.path.split(image_path)[-1], os.path.split(label_path)[-1]
        if Path(image_path).stem in TEST_LIST:
            # move image and label to test
            dst_img_path, dst_lab_path = os.path.join(test_img_dir, img_idx), os.path.join(test_lab_dir, label_idx)
            shutil.copy(src=image_path, dst=dst_img_path)
            shutil.copy(src=label_path, dst=dst_lab_path)
        else:
            # move image and label to train
            dst_img_path, dst_lab_path = os.path.join(train_img_dir, img_idx), os.path.join(train_lab_dir, label_idx)
            shutil.copy(src=image_path, dst=dst_img_path)
            shutil.copy(src=label_path, dst=dst_lab_path)


def get_bcsss_paths(
    path: Union[os.PathLike, str], split: Optional[str] = None, val_fraction: float = 0.2, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BCSS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        val_fraction: The fraction of data to be considered for validation split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_bcss_data(path, download)

    if split is None:
        image_paths = sorted(glob(os.path.join(path, "*", "images", "*")))
        label_paths = sorted(glob(os.path.join(path, "*", "masks", "*")))
    else:
        assert split in ["train", "val", "test"], "Please choose from the available `train` / `val` / `test` splits"
        if split == "test":
            image_paths = sorted(glob(os.path.join(path, "test", "images", "*")))
            label_paths = sorted(glob(os.path.join(path, "test", "masks", "*")))
        else:
            image_paths = sorted(glob(os.path.join(path, "train", "images", "*")))
            label_paths = sorted(glob(os.path.join(path, "train", "masks", "*")))

            (train_image_paths, val_image_paths,
             train_label_paths, val_label_paths) = train_test_split(
                image_paths, label_paths, test_size=val_fraction, random_state=42
            )

            image_paths = train_image_paths if split == "train" else val_image_paths
            label_paths = train_label_paths if split == "train" else val_label_paths

    assert len(image_paths) == len(label_paths)

    return image_paths, label_paths


def get_bcss_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Optional[str] = None,
    val_fraction: float = 0.2,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    **kwargs
) -> Dataset:
    """Get the BCSS dataset for breast cancer tissue segmentation in histopathology.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        val_fraction: The fraction of data to be considered for validation split.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of labels.
        kwargs: kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_bcsss_paths(path, split, val_fraction, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=False,
        **kwargs
    )


def get_bcss_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: Optional[str] = None,
    val_fraction: float = 0.2,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    **kwargs
) -> DataLoader:
    """Get the BCSS dataloader for breast cancer tissue segmentation in histopathology.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        val_fraction: The fraction of data to be considered for validation split.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of labels.
        kwargs: kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bcss_dataset(
        path, patch_shape, split, val_fraction, download=download, label_dtype=label_dtype, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
