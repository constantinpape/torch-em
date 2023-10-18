import os
from glob import glob
from typing import Tuple

import torch

import torch_em
from torch_em.data.datasets import util
from torch_em.data import ImageCollectionDataset


URL = "https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss?usp=sharing"


# TODO
CHECKSUM = None


def _download_bcss_dataset(path, download):
    """Current recommendation:
        - download the folder from URL manually
        - use the consortium's git repo to download the dataset (https://github.com/PathologyDataScience/BCSS)
    """
    raise NotImplementedError("Please download the dataset using the drive link / git repo directly")

    # FIXME: current limitation for the installation below:
    #   - only downloads first 50 files - due to `gdown`'s download folder function
    #   - (optional) clone their git repo to download their data
    util.download_source_gdrive(path=path, url=URL, download=download, checksum=CHECKSUM, download_type="folder")


def get_bcss_dataset(
        path: str,
        patch_shape: Tuple[int, int],
        download: bool = False,
        label_dtype: torch.dtype = torch.int64,
        **kwargs
):
    """Dataset for breast cancer tissue segmentation in histopathology.

    This dataset is from https://bcsegmentation.grand-challenge.org/BCSS/.
    Please cite this paper (https://doi.org/10.1093/bioinformatics/btz083) if you use this dataset for a publication.
    """
    if download:
        _download_bcss_dataset(path, download)

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
            "Please check the image directory. If downloaded from gdrive, it's named \"rgbs_colorNormalized\", if from github it's named \"images\""
        )

    assert len(image_paths) == len(label_paths)

    dataset = ImageCollectionDataset(
        image_paths, label_paths, patch_shape=patch_shape, label_dtype=label_dtype, **kwargs
    )
    return dataset


def get_bcss_loader(
        path, patch_shape, batch_size, download=False, label_dtype=torch.int64, **kwargs
):
    """Dataloader for breast cancer tissue segmentation in histopathology. See `get_bcss_dataset` for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bcss_dataset(
        path, patch_shape, download=download, label_dtype=label_dtype, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader


def main():
    path = "/scratch/usr/nimanwai/data/bcss/"
    download = False

    loader = get_bcss_loader(
        path=path, patch_shape=(256, 256), batch_size=2, download=download
    )

    print(len(loader))

    from torch_em.util.debug import check_loader
    check_loader(loader, 8, True, True, True, "./bcss.png")

    for x, y in loader:
        print(x.shape, y.shape)


main()
