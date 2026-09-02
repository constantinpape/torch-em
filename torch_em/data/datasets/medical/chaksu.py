"""The Chaksu dataset contains annotations for optic disc and optic cup
segmentation in Fundus images.

This dataset is located at https://doi.org/10.6084/m9.figshare.20123135.v2, under the CC BY 4.0 license.
The dataset is from the publication https://doi.org/10.1038/s41597-023-01943-4.
Please cite it if you use this dataset for your research.

NOTE: The full archive also ships four other expert annotation and fusion variants at full
uncompressed resolution, expanding to over 200GB. Only the raw fundus images and the
STAPLE-fused consensus masks are downloaded here, as those are the ones needed for training.
"""

import io
import os
import zipfile
from glob import glob
from typing import Union, Tuple, Literal, List

import requests
from tqdm import tqdm
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://ndownloader.figshare.com/files/37875672",
    "test": "https://ndownloader.figshare.com/files/37875687",
}
DEVICES = ["Bosch", "Forus", "Remidio"]


class _RemoteZipFile(io.RawIOBase):
    """File-like wrapper that reads a remote zip archive via HTTP range requests."""

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buffer):
        end = min(self.pos + len(buffer), self.size) - 1
        if end < self.pos:
            return 0
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
        data = response.content
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _download_split(url, dst, split):
    remote_file = io.BufferedReader(_RemoteZipFile(url), buffer_size=1 << 20)
    with zipfile.ZipFile(remote_file) as zf:
        members = [
            name for name in zf.namelist() if not name.endswith("/") and "__MACOSX" not in name
            and not name.endswith(".DS_Store")
            and ("1.0_Original_Fundus_Images/" in name
                 or ("5.0_OD_OC_Mean_Median_Majority_STAPLE/" in name and "/STAPLE/" in name))
        ]
        for member in tqdm(members, desc=f"Downloading Chaksu '{split}' split"):
            zf.extract(member, dst)


def get_chaksu_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Chaksu dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)
    for split, url in URLS.items():
        split_dir = os.path.join(path, split.capitalize())
        if os.path.exists(split_dir):
            continue
        if not download:
            raise RuntimeError(f"Chaksu data is not found at {split_dir} and 'download' is set to False.")
        _download_split(url, path, split)

    return path


def _binarize_mask(gt_path, gt_dir):
    dst_path = os.path.join(gt_dir, os.path.basename(gt_path))
    if os.path.exists(dst_path):
        return dst_path

    os.makedirs(gt_dir, exist_ok=True)
    mask = imageio.imread(gt_path)[..., 0] > 128  # the STAPLE masks are near-binary grayscale stored as RGBA
    imageio.imwrite(dst_path, mask.astype("uint8"))
    return dst_path


def get_chaksu_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    task: Literal["optic_disc", "optic_cup"] = "optic_disc",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Chaksu data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_chaksu_data(path=path, download=download)

    assert split in ["train", "test"], f"'{split}' is not a valid split."
    assert task in ["optic_disc", "optic_cup"], f"'{task}' is not a valid task."

    split_dir = split.capitalize()
    region = "Disc" if task == "optic_disc" else "Cup"

    image_paths, gt_paths = [], []
    for device in DEVICES:
        device_image_paths = sorted(glob(os.path.join(data_dir, split_dir, "1.0_Original_Fundus_Images", device, "*")))
        gt_dir = os.path.join(data_dir, split_dir, "segmentation_masks", device, region)
        for image_path in device_image_paths:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            staple_path = os.path.join(
                data_dir, split_dir, "5.0_OD_OC_Mean_Median_Majority_STAPLE", device, region, "STAPLE", f"{stem}.png"
            )
            if not os.path.exists(staple_path):
                continue

            image_paths.append(image_path)
            gt_paths.append(_binarize_mask(staple_path, gt_dir))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_chaksu_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    task: Literal["optic_disc", "optic_cup"] = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Chaksu dataset for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_chaksu_paths(path, split, task, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_chaksu_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    task: Literal["optic_disc", "optic_cup"] = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Chaksu dataloader for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chaksu_dataset(path, patch_shape, split, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
