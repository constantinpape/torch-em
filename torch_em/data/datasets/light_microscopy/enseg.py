"""The ENSeg dataset contains expert-annotated enteric neuron cells
for instance segmentation in microscopy images.

The dataset is located at https://www.kaggle.com/datasets/gustavozanonifelipe/enseg-dataset.
This dataset is from the publication https://doi.org/10.3390/app15031046.
Please cite it if you use this dataset in your research.
"""

import os
import json
import base64
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Optional, Sequence, List

import numpy as np
import imageio.v3 as imageio

from skimage.draw import polygon as draw_polygon

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "gustavozanonifelipe/enseg-dataset"

ANIMAL_TAGS = ["2C", "4C", "5C", "22TW", "23TW", "28TW"]


def _process_json(json_path, image_dir, seg_dir):
    """Extract image and instance segmentation mask from a LabelMe JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    animal_tag = data["animalTag"]
    stem = f"{animal_tag}_{os.path.basename(json_path).replace('.json', '')}"

    image_path = os.path.join(image_dir, f"{stem}.png")
    seg_path = os.path.join(seg_dir, f"{stem}.tif")

    # Extract image from base64 data.
    if not os.path.exists(image_path):
        image_bytes = base64.b64decode(data["imageData"])
        image = imageio.imread(image_bytes, extension=".jpg")
        imageio.imwrite(image_path, image)

    # Create instance segmentation mask from polygon annotations.
    if not os.path.exists(seg_path):
        shape = (data["imageHeight"], data["imageWidth"])
        seg = np.zeros(shape, dtype="uint16")
        for seg_id, obj in enumerate(data["shapes"], 1):
            points = np.array(obj["points"])
            rr, cc = draw_polygon(points[:, 1], points[:, 0], shape=shape)
            seg[rr, cc] = seg_id

        imageio.imwrite(seg_path, seg, compression="zlib")

    return image_path, seg_path, animal_tag


def _preprocess_data(data_dir, image_dir, seg_dir):
    """Extract all images and create instance masks from LabelMe JSON files."""
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    json_paths = natsorted(glob(os.path.join(data_dir, "*", "*.json")))
    assert len(json_paths) > 0, f"No JSON annotation files found in {data_dir}"

    for json_path in tqdm(json_paths, desc="Processing ENSeg data"):
        _process_json(json_path, image_dir, seg_dir)


def get_enseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ENSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)
    util.unzip(zip_path=os.path.join(path, "enseg-dataset.zip"), dst=data_dir)

    return data_dir


def get_enseg_paths(
    path: Union[os.PathLike, str],
    animal_tags: Optional[Sequence[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ENSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        animal_tags: Filter images by animal tags (e.g. ['2C', '4C']).
            Valid tags: '2C', '4C', '5C' (Control) and '22TW', '23TW', '28TW' (Tumor).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_enseg_data(path, download)

    image_dir = os.path.join(path, "images")
    seg_dir = os.path.join(path, "segmentations")

    # Preprocess: extract images from JSON and create instance masks.
    if not os.path.exists(image_dir) or not os.path.exists(seg_dir):
        _preprocess_data(data_dir, image_dir, seg_dir)

    seg_paths = natsorted(glob(os.path.join(seg_dir, "*.tif")))
    image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
    assert len(image_paths) == len(seg_paths) and len(image_paths) > 0

    if animal_tags is not None:
        assert isinstance(animal_tags, (list, tuple)), \
            f"'animal_tags' must be a list or tuple, got {type(animal_tags)}"
        for tag in animal_tags:
            assert tag in ANIMAL_TAGS, f"'{tag}' is not a valid animal tag. Choose from {ANIMAL_TAGS}."

        # Filter by animal tag using the filename prefix (<animal_tag>_<number>).
        filtered_image_paths, filtered_seg_paths = [], []
        for image_path, seg_path in zip(image_paths, seg_paths):
            fname = os.path.basename(image_path)
            # The tag is everything before the last underscore-number part.
            tag = fname.rsplit("_", 1)[0]
            if tag in animal_tags:
                filtered_image_paths.append(image_path)
                filtered_seg_paths.append(seg_path)

        image_paths, seg_paths = filtered_image_paths, filtered_seg_paths
        assert len(image_paths) > 0, f"No images found for animal tags {animal_tags}."

    return image_paths, seg_paths


def get_enseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    animal_tags: Optional[Sequence[str]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ENSeg dataset for enteric neuron cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        animal_tags: Filter images by animal tags (e.g. ['2C', '4C']).
            Valid tags: '2C', '4C', '5C' (Control) and '22TW', '23TW', '28TW' (Tumor).
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, seg_paths = get_enseg_paths(path, animal_tags, download)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=seg_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_enseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    animal_tags: Optional[Sequence[str]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ENSeg dataloader for enteric neuron cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        animal_tags: Filter images by animal tags (e.g. ['2C', '4C']).
            Valid tags: '2C', '4C', '5C' (Control) and '22TW', '23TW', '28TW' (Tumor).
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_enseg_dataset(
        path=path,
        patch_shape=patch_shape,
        animal_tags=animal_tags,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
