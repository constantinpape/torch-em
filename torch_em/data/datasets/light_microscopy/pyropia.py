"""The Pyropia dataset contains annotations for cell instance segmentation in high-resolution
microscopy images of the red alga Pyropia haitanensis (also called Porphyra haitanensis).

The dataset consists of 32 RGB images (2160 x 3840 pixels) of two strains, 'Sansha-5' and 'WO115-2'.
Each strain has 8 experimental subsets (`plant_1` to `plant_8` and `plant_9` to `plant_16`), which each
contain the same tissue imaged at 0 h and 24 h. Cells were annotated manually with LabelMe polygons,
which gives 3,359 cell instances in total. The annotation files also encode 1,255 tracking relationships
between the two time points and 283 cell divisions, which are not used by this loader.
This module rasterizes the polygons into instance label images (one id per cell, in the order of the
polygons in the annotation file). Overlaps between polygons are negligible, later polygons overwrite earlier ones.

The data is available on two Zenodo records with identical images and annotations:
https://zenodo.org/records/19571835 (used by this loader, includes a README and one folder per subset)
and https://zenodo.org/records/20301518 (the same files sorted into one folder per strain).
It is released under a CC-BY-4.0 license.

Please cite the publication associated with the Zenodo record if you use this dataset in your research.
"""

import os
import json
import uuid
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/19571835/files/data.zip"
CHECKSUM = "55aa506e6fe6a68582f85ef5ac084a4107c686ae231a89ee9c3ae08f35558ecb"

STRAINS = {"sansha-5": range(1, 9), "wo115-2": range(9, 17)}


def _rasterize_annotation(json_path, out_path):
    import numpy as np
    import tifffile
    from skimage.draw import polygon

    if os.path.exists(out_path):
        return

    with open(json_path) as f:
        annotation = json.load(f)

    height, width = annotation["imageHeight"], annotation["imageWidth"]
    labels = np.zeros((height, width), dtype="uint16")
    for instance_id, shape in enumerate(annotation["shapes"], start=1):
        points = np.array(shape["points"])
        rr, cc = polygon(points[:, 1], points[:, 0], (height, width))
        labels[rr, cc] = instance_id

    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete.tif"
    tifffile.imwrite(tmp_path, labels, compression="zlib")
    os.replace(tmp_path, out_path)


def get_pyropia_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Pyropia dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "pyropia.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create the expected folder in '{path}'."

    return data_dir


def get_pyropia_paths(
    path: Union[os.PathLike, str], strain: Optional[Literal["sansha-5", "wo115-2"]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Pyropia data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        strain: The choice of strain. Either 'sansha-5' or 'wo115-2'. By default both strains are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if strain is not None and strain not in STRAINS:
        raise ValueError(f"'{strain}' is not a valid strain. Choose one of {list(STRAINS)}.")

    data_dir = get_pyropia_data(path, download)
    label_dir = os.path.join(path, "labels")
    os.makedirs(label_dir, exist_ok=True)

    plants = [i for name, ids in STRAINS.items() if strain in (None, name) for i in ids]

    raw_paths, label_paths = [], []
    for plant in plants:
        json_paths = natsorted(glob(os.path.join(data_dir, f"plant_{plant}", "json", "*.json")))
        for json_path in json_paths:
            stem = os.path.splitext(os.path.basename(json_path))[0]
            raw_path = glob(os.path.join(data_dir, f"plant_{plant}", "img", f"{stem}.tif*"))
            assert len(raw_path) == 1, f"Expected one image for '{json_path}', found {len(raw_path)}."

            label_path = os.path.join(label_dir, f"plant_{plant}_{stem}.tif")
            _rasterize_annotation(json_path, label_path)

            raw_paths.append(raw_path[0])
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_pyropia_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    strain: Optional[Literal["sansha-5", "wo115-2"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Pyropia dataset for cell instance segmentation in plant microscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        strain: The choice of strain. Either 'sansha-5' or 'wo115-2'. By default both strains are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_pyropia_paths(path, strain, download)

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


def get_pyropia_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    strain: Optional[Literal["sansha-5", "wo115-2"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Pyropia dataloader for cell instance segmentation in plant microscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        strain: The choice of strain. Either 'sansha-5' or 'wo115-2'. By default both strains are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pyropia_dataset(path, patch_shape, strain, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
