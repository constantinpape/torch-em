"""The OrgaExtractor dataset contains annotations for colon organoids in brightfield images.

NOTE: This dataset is kind of sparsely annotated (quite some organoids per image were missing when AA visualized).

This dataset is from the publication https://www.nature.com/articles/s41598-023-46485-2.
And the dataset is located at https://github.com/tpark16/orgaextractor, pointing to the
drive link at https://drive.google.com/drive/folders/17K4N7gEZUqAcwf9N2-I5DPbywwPvzAvo.

Please cite the publication if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# NOTE: The odd thing is, 'val' has no labels, but 'test' has labels.
# So, users are allowed to only request for 'train' and 'test' splits.
URLS = {
    "train": "https://drive.google.com/uc?export=download&id=1u987UNcZxWkEwe5gjLoR3-M0lBNicXQ1",
    "val": "https://drive.google.com/uc?export=download&id=1UsBrHOYY0Orkb4vsRP8SaDj-CeYfGpFG",
    "test": "https://drive.google.com/uc?export=download&id=1IXqu1MqMZzfw1_GzZauUhg1As_abbk6N",
}

CHECKSUMS = {
    "train": "279bcfbcbd2fba23bbdea362b23eedacc53193034f4d23eb94ef570896da4f60",
    "val": "3d2288a7be39a692af2eb86bea520e7db332191cd372a8c970679b5bede61b7e",
    "test": "8e110ad8543031ed61c61bee5e8b41492b746d0dc8c503b6f8d4869b29a308e6",
}


def _preprocess_data(data_dir):
    gt_paths = natsorted(glob(os.path.join(data_dir, "*.tif")))
    for gt_path in gt_paths:
        gt = imageio.imread(gt_path)[..., 0]  # labels are with 3 channels. choose one as all channels are same.
        gt = connected_components(gt).astype("uint16")  # convert semantic labels to instances
        imageio.imwrite(gt_path, gt, compression="zlib")


def get_orgaextractor_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False,
) -> str:
    """Download the OrgaExtractor dataset.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        split: The data split to use.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, f"{split}.zip")
    util.download_source_gdrive(
        path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split], download_type="zip",
    )
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    _preprocess_data(data_dir)

    return data_dir


def get_orgaextractor_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OrgaExtractor data.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        split: The data split to use.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_orgaextractor_data(path, split, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "*.jpg")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "*.tif")))

    assert image_paths and len(image_paths) == len(gt_paths)

    return image_paths, gt_paths


def get_orgaextractor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OrgaExtractor dataset for organoid segmentation in brightfield microscopy images.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_orgaextractor_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs
    )


def get_orgaextractor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OrgaExtractor dataloader for organoid segmentation in brightfield microscopy images.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orgaextractor_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
