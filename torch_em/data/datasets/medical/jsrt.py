import os
from glob import glob
from pathlib import Path
from typing import Optional, Union, Tuple

import torch_em

from .. import util
from ... import ImageCollectionDataset


URL = {
    "Segmentation01": "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2018/11/Segmentation01.zip",
    "Segmentation02": "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip"
}

CHECKSUM = {
    "Segmentation01": "ab1f26a910bc18eae170928e9f2d98512cc4dc8949bf6cd38b98a93398714fcf",
    "Segmentation02": "f1432af4fcbd69342cf1bf2ca3d0d43b9535cdc6b160b86191b5b67de2fdbf3c"
}

ZIP_PATH = {
    "Segmentation01": "Segmentation01.zip",
    "Segmentation02": "segmentation02.zip"
}

DATA_DIR = {
    "Segmentation01": "Segmentation01",
    "Segmentation02": "segmentation02"
}


def _download_jsrt_data(path, download, choice):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, DATA_DIR[choice])
    if os.path.exists(data_dir):
        return

    zip_path = os.path.join(path, ZIP_PATH[choice])

    util.download_source(path=zip_path, url=URL[choice], download=download, checksum=CHECKSUM[choice])
    util.unzip(zip_path=zip_path, dst=path)


def _get_jsrt_paths(path, split, download, choice=None):
    if choice is None:
        choice = list(URL.keys())
    else:
        if isinstance(choice, str):
            choice = [choice]

    image_paths, gt_paths = [], []
    for per_choice in choice:
        _download_jsrt_data(path=path, download=download, choice=per_choice)

        if per_choice == "Segmentation01":
            root_dir = os.path.join(path, Path(ZIP_PATH[per_choice]).stem, split)
            all_image_paths = sorted(glob(os.path.join(root_dir, "org", "*.png")))
            all_gt_paths = sorted(glob(os.path.join(root_dir, "label", "*.png")))

        elif per_choice == "Segmentation02":
            root_dir = os.path.join(path, Path(ZIP_PATH[per_choice]).stem, "segmentation")
            all_image_paths = sorted(glob(os.path.join(root_dir, f"org_{split}", "*.bmp")))
            all_gt_paths = sorted(glob(os.path.join(root_dir, f"label_{split}", "*.png")))

        else:
            raise ValueError(f"{per_choice} is not a valid segmentation dataset choice.")

        image_paths.extend(all_image_paths)
        gt_paths.extend(all_gt_paths)

        print(len(all_image_paths), len(all_gt_paths))

    print(len(image_paths), len(gt_paths))

    return image_paths, gt_paths


def get_jsrt_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    choice: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    """Dataset for the segmentation of lungs in x-ray.

    This dataset is from the publication https://doi.org/10.2214/ajr.174.1.1740071.
    The database is located at http://db.jsrt.or.jp/eng.php
    Please cite it if you use this dataset for a publication.
    """
    av_splits = ["train", "test"]
    assert split in av_splits, f"{split} isn't a valid split choice. Please choose from {av_splits}."

    image_paths, gt_paths = _get_jsrt_paths(path=path, split=split, download=download, choice=choice)

    dataset = ImageCollectionDataset(
        raw_image_paths=image_paths, label_image_paths=gt_paths, patch_shape=patch_shape, **kwargs
    )
    return dataset


def get_jsrt_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    choice: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    """Dataloader for the segmentation of lungs in x-ray. See 'get_jsrt_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_jsrt_dataset(
        path=path, split=split, patch_shape=patch_shape, choice=choice, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
