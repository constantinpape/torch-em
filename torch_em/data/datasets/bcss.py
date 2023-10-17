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

    # FIXME: current limitation for the installation below:
    #   - only downloads first 50 files - due to `gdown`'s download folder function
    #   - (optional) clone their git repo to download their data
    util.download_source_gdrive(path=path, url=URL, download=download, checksum=CHECKSUM, download_type="folder")

    image_paths = sorted(glob(os.path.join(path, "*", "rgbs_colorNormalized", "*")))
    label_paths = sorted(glob(os.path.join(path, "*", "masks", "*")))
    print(len(image_paths), len(label_paths))
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
    path = "/scratch/usr/nimanwai/data/test_bcss/"
    download = True

    loader = get_bcss_loader(
        path=path, patch_shape=(256, 256), batch_size=2, download=download
    )

    print(len(loader))

    from torch_em.util.debug import check_loader
    check_loader(loader, 8, True, True, True, "./bcss.png")

    for x, y in loader:
        print(x.shape, y.shape)


main()
