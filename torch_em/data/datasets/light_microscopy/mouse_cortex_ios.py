"""The MouseCortex-IOS dataset contains annotations for the segmentation of intrinsic optical signals (IOS)
in videos of the cortex of awake mice.

The dataset consists of 5,722 annotated frames (512x512 pseudo-color TIFF images) from 14 mice, grouped into
194 video segments, each one covering a triggered IOS event. The paper reports 5,732 images, this loader
uses the image and annotation pairs that are present in the released archive. Each frame has one or two
annotated regions, stored as polygons in JSON files of the ISAT annotation tool. The annotations were created
with the help of SAM2 and are therefore not purely hand-drawn ground truth. `get_mouse_cortex_ios_data`
rasterizes them into instance label images, where each polygon gets its own id (1 and 2 at most, in
annotation order). Where two polygons overlap, the smaller one takes precedence, so that no annotated region
is lost. The category and group fields of the polygons are not documented in the publication and are not
used. The archive also contains ISAT mask images for a subset of the frames, which are not used.

The dataset is located at https://doi.org/10.6084/m9.figshare.28601813 and is released under a CC-BY-4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-026-06580-1.
Please cite it if you use this dataset for your research.
"""

import os
import json
import uuid
from glob import glob
from natsort import natsorted
from concurrent import futures
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/60099299"
CHECKSUM = "0edf1d19fa5356381dba0e236fb44f5067da03442487b3ac56c9da12c2d64f4d"


def _rasterize_annotation(json_path, out_path):
    if os.path.exists(out_path):
        return

    import tifffile
    from skimage.draw import polygon

    with open(json_path) as f:
        annotation = json.load(f)

    shape = (annotation["info"]["height"], annotation["info"]["width"])
    regions = []
    for instance_id, obj in enumerate(annotation["objects"], start=1):
        points = np.array(obj["segmentation"])
        regions.append((instance_id, polygon(points[:, 1], points[:, 0], shape)))

    # Larger regions are drawn first, so that a region nested in another one is not overwritten.
    labels = np.zeros(shape, dtype="uint8")
    for instance_id, (rr, cc) in sorted(regions, key=lambda region: -len(region[1][0])):
        labels[rr, cc] = instance_id

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete.tif"
    tifffile.imwrite(tmp_path, labels, compression="zlib")
    os.replace(tmp_path, out_path)


def get_mouse_cortex_ios_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MouseCortex-IOS dataset and rasterize its polygon annotations into label images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the label images.
    """
    data_dir = os.path.join(path, "MouseCortex-IOS_Dateset")
    label_dir = os.path.join(path, "labels")

    if not os.path.exists(data_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "MouseCortex-IOS.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    json_paths = natsorted(glob(os.path.join(data_dir, "*", "*", "*.json")))
    assert len(json_paths) > 0, f"No annotations were found in '{data_dir}'."

    def _out_path(json_path):
        return os.path.join(label_dir, os.path.relpath(json_path, data_dir)[:-len(".json")] + ".tif")

    missing = [p for p in json_paths if not os.path.exists(_out_path(p))]
    with futures.ThreadPoolExecutor(min(16, os.cpu_count() or 1)) as pool:
        list(pool.map(lambda p: _rasterize_annotation(p, _out_path(p)), missing))

    return label_dir


def get_mouse_cortex_ios_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the MouseCortex-IOS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    label_dir = get_mouse_cortex_ios_data(path, download)
    data_dir = os.path.join(path, "MouseCortex-IOS_Dateset")

    label_paths = natsorted(glob(os.path.join(label_dir, "*", "*", "*.tif")))
    raw_paths = [
        os.path.join(data_dir, os.path.relpath(p, label_dir)[:-len(".tif")] + ".tiff") for p in label_paths
    ]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_mouse_cortex_ios_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MouseCortex-IOS dataset for intrinsic optical signal segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mouse_cortex_ios_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_mouse_cortex_ios_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MouseCortex-IOS dataloader for intrinsic optical signal segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mouse_cortex_ios_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
