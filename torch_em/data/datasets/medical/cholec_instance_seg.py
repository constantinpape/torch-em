"""CholecInstanceSeg contains instance segmentation annotations for surgical instruments
in laparoscopic cholecystectomy video frames.

The full CholecInstanceSeg release has four subsets, annotated on top of the frames of four
existing datasets (CholecSeg8k, T50, T80 and the CholecInstanceSeg-specific full set). This
module only covers the 'Instance-CholecSeg8k' subset: 8,080 frames from 17 sequences, which
is the identical frame set already integrated in 'torch_em/data/datasets/medical/cholecseg8k.py',
now additionally annotated with per-instance polygons for two instrument classes ('grasper'
and 'hook'). The raw frames themselves are downloaded via 'get_cholecseg8k_data' (from Kaggle);
this module only downloads the instance annotations and rasterizes them into instance masks.

NOTE: The annotations are hosted on Synapse (project 'syn60239970'). The Synapse wiki for this
project states the license as CC BY 4.0 (not CC BY-NC-ND, as an earlier, unverified note about
this dataset had assumed - the Synapse project has no access requirements and its wiki's
'License' section links to the standard CC BY 4.0 license). Downloading it requires the
'synapseclient' python library and a Synapse account with an authentication token stored in the
'~/.synapseConfig' file. See 'get_cholec_instance_seg_data' for details.

The dataset is located at https://www.synapse.org/Synapse:syn60239970.
This dataset is from the publication https://doi.org/10.1038/s41597-025-05163-w.
Please cite it if you use this dataset in your research.
"""

import os
import re
import json
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Tuple, Union, Literal, List

import numpy as np
import imageio.v3 as imageio
from skimage.draw import polygon as sk_polygon

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .cholecseg8k import get_cholecseg8k_data


ENTITY = "syn66477541"

FRAME_PATTERN = re.compile(r"seg8k_video(\d+)_(\d+)\.json")


def get_cholec_instance_seg_data(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Download the CholecInstanceSeg annotations (Instance-CholecSeg8k subset) and the raw CholecSeg8k frames.

    Follow the instructions below to get access to the Synapse-hosted annotations.
    - Create a free account at https://www.synapse.org.
    - Generate a personal access token and store it in a '~/.synapseConfig' file, see
      https://python-docs.synapse.org/tutorials/authentication/ for details.
    - Install the 'synapseclient' python library.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the extracted CholecInstanceSeg annotations.
        Filepath to the raw CholecSeg8k frames.
    """
    ann_dir = os.path.join(path, "cholecinstanceseg")
    if not os.path.exists(ann_dir):
        os.makedirs(path, exist_ok=True)

        import synapseclient

        syn = synapseclient.Synapse()
        syn.login()
        zip_path = os.path.join(path, "cholecinstanceseg.zip")
        if not os.path.exists(zip_path):
            if not download:
                raise RuntimeError(f"Cannot find the data at {zip_path}, but download was set to False.")
            syn.get(ENTITY, downloadLocation=path, downloadFile=True)

        util.unzip(zip_path=zip_path, dst=path, remove=False)

    raw_data_dir = get_cholecseg8k_data(path, download)

    return ann_dir, raw_data_dir


def _rasterize_instances(annotation, shape):
    instances = np.zeros(shape, dtype="uint16")
    for i, shape_ann in enumerate(annotation["shapes"], start=1):
        points = np.array(shape_ann["points"])
        rr, cc = sk_polygon(points[:, 1], points[:, 0], shape)
        instances[rr, cc] = i
    return instances


def get_cholec_instance_seg_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths for the CholecInstanceSeg (Instance-CholecSeg8k subset) dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"'{split}' is not a valid split. Please choose from 'train', 'val' or 'test'.")

    ann_dir, raw_data_dir = get_cholec_instance_seg_data(path, download)

    json_paths = natsorted(glob(os.path.join(ann_dir, split, "VID*_seg8k", "ann_dir", "*.json")))
    assert len(json_paths) > 0, f"No annotations were found at '{os.path.join(ann_dir, split)}'."

    ppdir = os.path.join(ann_dir, "preprocessed", split)
    os.makedirs(os.path.join(ppdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(ppdir, "masks"), exist_ok=True)

    image_paths, gt_paths = [], []
    for json_path in tqdm(json_paths, desc=f"Preprocessing CholecInstanceSeg '{split}' split"):
        match = FRAME_PATTERN.match(os.path.basename(json_path))
        assert match is not None, f"Unexpected annotation filename: '{json_path}'."
        video_id, frame_id = match.group(1), int(match.group(2))

        org_image_paths = glob(
            os.path.join(raw_data_dir, f"video{int(video_id):02d}", "video*", f"frame_{frame_id}_endo.png")
        )
        assert len(org_image_paths) == 1, (
            f"Expected exactly one matching CholecSeg8k frame for video {video_id}, frame {frame_id}, "
            f"found {len(org_image_paths)}."
        )
        org_image_path = org_image_paths[0]

        image_id = os.path.basename(org_image_path)
        image_path = os.path.join(ppdir, "images", image_id)
        gt_path = os.path.join(ppdir, "masks", Path(image_id).with_suffix(".tif"))

        image_paths.append(image_path)
        gt_paths.append(gt_path)

        if os.path.exists(image_path) and os.path.exists(gt_path):
            continue

        if not os.path.exists(image_path):
            os.symlink(os.path.abspath(org_image_path), image_path)

        with open(json_path) as f:
            annotation = json.load(f)

        # Frames without any annotated instrument instances lack the 'imageHeight' / 'imageWidth' keys,
        # so the raw image shape is used as a fallback for the rasterized mask shape.
        if "imageHeight" in annotation and "imageWidth" in annotation:
            shape = (annotation["imageHeight"], annotation["imageWidth"])
        else:
            shape = imageio.imread(org_image_path).shape[:2]

        instances = _rasterize_instances(annotation, shape)
        imageio.imwrite(gt_path, instances, compression="zlib")

    return image_paths, gt_paths


def get_cholec_instance_seg_dataset(
    path: Union[str, os.PathLike],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CholecInstanceSeg (Instance-CholecSeg8k subset) dataset for instrument instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_cholec_instance_seg_paths(path, split, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_cholec_instance_seg_loader(
    path: Union[str, os.PathLike],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CholecInstanceSeg (Instance-CholecSeg8k subset) dataloader for instrument instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cholec_instance_seg_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
