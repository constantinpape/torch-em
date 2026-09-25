"""FusionX contains light microscopy images of mono- and multi-nucleated cells with
instance segmentation annotations, curated for quantifying cell-to-cell fusion.

The dataset is located at https://zenodo.org/records/18596465 and is licensed under
CC BY 4.0. A handful of annotated images are missing from the released archives; these
are skipped automatically when building the paths.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/18596465/files/{}?download=1"

CHECKSUMS = {
    "train_png.7z": "214988031e094b0aa9a181b52404a2eaef95bbe3f4921a6c92ac0c912faf7d70",
    "train_png.json": "216e4a33860b46f6712ddf1317db7d89cd77337477630af46f425cfb562c97d6",
    "test_png.7z": "f9769dc8a6587214d0f8919311fc13771def75830f04ba95b15289edfb8ed8d9",
    "test_png.json": "244063f5a5599af6983e8c601fea476b26e1b31defdd2acfb634d16aa86f5cc7",
}

SPLITS = {"train": "train_png.json", "test": "test_png.json"}


def _create_segmentations_from_coco_annotations(path, split):
    """Convert COCO mask annotations to instance segmentation masks."""
    import numpy as np
    import imageio.v3 as imageio
    from tqdm import tqdm

    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError(
            "'pycocotools' is required for processing the FusionX ground-truth. "
            "Install it with 'conda install -c conda-forge pycocotools'."
        )

    image_dir = os.path.join(path, f"{split}_png")
    label_dir = os.path.join(path, "labels", split)
    if os.path.exists(label_dir):
        label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))
        if len(label_paths) > 0:
            image_paths = [
                os.path.join(image_dir, f"{os.path.splitext(os.path.basename(p))[0]}.png") for p in label_paths
            ]
            return natsorted(image_paths), label_paths

    os.makedirs(label_dir, exist_ok=True)

    ann_file = os.path.join(path, SPLITS[split])
    coco = COCO(ann_file)

    image_paths, label_paths = [], []
    for image_id in tqdm(coco.getImgIds(), desc=f"Creating FusionX segmentations ({split})"):
        image_metadata = coco.loadImgs(image_id)[0]
        file_name = image_metadata["file_name"]

        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            continue  # A few annotated images are missing from the released archive.
        image_paths.append(image_path)

        label_path = os.path.join(label_dir, f"{os.path.splitext(file_name)[0]}.tif")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue

        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        shape = (image_metadata["height"], image_metadata["width"])
        seg = np.zeros(shape, dtype="uint32")

        # Paint the largest cells first, so smaller overlapping cells stay visible on top.
        masks = [coco.annToMask(a).astype(bool) for a in annotations]
        sorting = np.argsort([m.sum() for m in masks])[::-1]
        for seg_id, idx in enumerate(sorting, 1):
            seg[masks[idx]] = seg_id

        imageio.imwrite(label_path, seg.astype("uint16"), compression="zlib")

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0
    return natsorted(image_paths), natsorted(label_paths)


def get_fusionx_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FusionX dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the data is stored.
    """
    if os.path.exists(os.path.join(path, "train_png")) and os.path.exists(os.path.join(path, "test_png")):
        return path

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but 'download' is set to False.")

    os.makedirs(path, exist_ok=True)
    for fname, checksum in CHECKSUMS.items():
        fpath = os.path.join(path, fname)
        util.download_source(fpath, URL.format(fname), download, checksum=checksum)
        if fname.endswith(".7z"):
            util.unzip_7z(fpath, path, remove=True)

    return path


def get_fusionx_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FusionX data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in SPLITS, f"'{split}' is not a valid split. Choose from {list(SPLITS.keys())}."
    data_dir = get_fusionx_data(path, download)
    return _create_segmentations_from_coco_annotations(data_dir, split)


def get_fusionx_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FusionX dataset for cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_fusionx_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        with_channels=True,
        **kwargs,
    )


def get_fusionx_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the FusionX dataloader for cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fusionx_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
