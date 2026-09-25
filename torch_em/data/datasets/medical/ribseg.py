"""The RibSeg dataset contains annotations for rib segmentation and labeling in chest-abdomen CT scans.

The annotations of RibSeg v2 cover all 660 CT scans of the RibFrac dataset. Each rib carries its own label id
from 1 to 24 (1-12 are the right ribs from top to bottom, 13-24 the left ribs), i.e. the segmentations are both
an instance segmentation of the individual ribs and a semantic labeling of the rib positions. Use the 'binary'
keyword argument of `torch_em.default_segmentation_dataset` to train on the binary rib mask instead.

The rib labels are downloaded from the RibSeg v2 release on google drive and the CT scans from the RibFrac
records on zenodo. The 'split' argument selects the official RibFrac split: 'train' (420 scans, ca. 51 GB),
'val' (80 scans, ca. 8.7 GB) or 'test' (160 scans, ca. 18 GB). The centerline annotations that are part of the
RibSeg v2 release (a 24 x 500 x 3 point array per scan) are not exposed by this module.

The dataset is located at https://github.com/M3DV/RibSeg and the CT scans at https://ribfrac.grand-challenge.org.

This dataset is from the publication https://doi.org/10.1109/TMI.2023.3313627.
The CT scans are from the publication https://doi.org/10.1016/j.ebiom.2020.103106.
Please cite them if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL_LABELS = "https://drive.google.com/uc?id=1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm"
CHECKSUM_LABELS = "6bf8a327f4a7f540a318caf66885e09d76fd8832a853806de4ad3d3ff4369714"

URLS = {
    "train": [
        "https://zenodo.org/records/3893508/files/ribfrac-train-images-1.zip",
        "https://zenodo.org/records/3893498/files/ribfrac-train-images-2.zip",
    ],
    "val": ["https://zenodo.org/records/3893496/files/ribfrac-val-images.zip"],
    "test": ["https://zenodo.org/records/3993380/files/ribfrac-test-images.zip"],
}

# NOTE: Only the checksum of the validation archive is known, the other archives are not verified.
CHECKSUMS = {
    "train": [None, None],
    "val": ["786cc14bf4ea55e93325d657d0e6490f4a1a7c8d073522404751b961dab95e24"],
    "test": [None],
}

N_VOLUMES = {"train": 420, "val": 80, "test": 160}

# The label ids are the individual ribs: 1-12 are the right ribs and 13-24 the left ribs, each counted
# from the first (top) to the twelfth (bottom) rib.
LABEL_IDS = {f"rib_{i}": i for i in range(1, 25)}


def get_ribseg_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[str, str]:
    """Download the RibSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the CT scans.
        Filepath to the folder with the rib labels.
    """
    if split not in URLS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(URLS.keys())}.")

    os.makedirs(path, exist_ok=True)

    label_dir = os.path.join(path, "ribseg_v2", "seg")
    if not os.path.exists(label_dir):
        zip_path = os.path.join(path, "ribseg_v2.zip")
        util.download_source_gdrive(path=zip_path, url=URL_LABELS, download=download, checksum=CHECKSUM_LABELS)
        util.unzip(zip_path=zip_path, dst=path)

    image_dir = os.path.join(path, "images", split)
    if not os.path.exists(image_dir):
        if not download:
            raise RuntimeError(f"Cannot find the data at {image_dir}, but download was set to False.")

        # The archives are extracted to a temporary folder, which is renamed once all parts are complete,
        # so that an interrupted download is not mistaken for a complete one.
        tmp_dir = f"{image_dir}.tmp"
        for url, checksum in zip(URLS[split], CHECKSUMS[split]):
            zip_path = os.path.join(path, os.path.basename(url))
            util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
            util.unzip(zip_path=zip_path, dst=tmp_dir)

        # The archives contain the volumes in a sub-folder, the name of which differs between the splits.
        for volume_path in glob(os.path.join(tmp_dir, "*", "*.nii.gz")):
            shutil.move(volume_path, os.path.join(tmp_dir, os.path.basename(volume_path)))
        for sub_dir in glob(os.path.join(tmp_dir, "*", "")):
            shutil.rmtree(sub_dir)
        os.rename(tmp_dir, image_dir)

    return image_dir, label_dir


def get_ribseg_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the RibSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, label_dir = get_ribseg_data(path, split, download)

    raw_paths = natsorted(glob(os.path.join(image_dir, "*-image.nii.gz")))
    label_paths = [
        os.path.join(label_dir, os.path.basename(p).replace("-image.nii.gz", "-rib-seg.nii.gz")) for p in raw_paths
    ]
    assert len(raw_paths) == N_VOLUMES[split] and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_ribseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RibSeg dataset for rib segmentation and labeling.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ribseg_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_ribseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RibSeg dataloader for rib segmentation and labeling.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ribseg_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
