"""The CytoNuke dataset contains annotations for nucleus and whole-cell segmentation
in H&E stained bright-field histopathology images of head and neck squamous cell
carcinoma. The images are re-distributed crops from the CPTAC-HNSCC collection.

This dataset is located at https://zenodo.org/records/10560728.
This dataset is from the publication https://doi.org/10.1016/j.cmpb.2024.108215.
Please cite it if you use this dataset for your research.
"""

import os
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import json
import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.draw import polygon as sk_polygon
from skimage.segmentation import relabel_sequential
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10560728/files/CytoNuke%20Dataset.zip"
CHECKSUM = "834836444fe749cc48daa7557458ba578bc00306a59286896e3d29666e98777a"

CATEGORY_IDS = {"nuclei": 0, "cell": 1}


def _annotations_to_instances(coco, image_metadata, category_id):
    height, width = image_metadata["height"], image_metadata["width"]
    seg = np.zeros((height, width), dtype="uint32")

    annotations = [a for a in coco["annotations"] if a["image_id"] == image_metadata["id"]]
    annotations = [a for a in annotations if a["category_id"] == category_id]

    for seg_id, annotation in enumerate(annotations, 1):
        for polygon in annotation["segmentation"]:
            xs, ys = polygon[0::2], polygon[1::2]
            rr, cc = sk_polygon(ys, xs, shape=(height, width))
            seg[rr, cc] = seg_id

    seg, _, _ = relabel_sequential(seg)
    return seg.astype("uint16")


def _create_segmentations_from_annotations(data_dir, annotations):
    image_dir = os.path.join(data_dir, "images")
    seg_dir = os.path.join(data_dir, "labels", annotations)
    os.makedirs(seg_dir, exist_ok=True)

    with open(os.path.join(data_dir, "coco.json")) as f:
        coco = json.load(f)

    category_id = CATEGORY_IDS[annotations]

    image_paths, seg_paths = [], []
    for image_metadata in tqdm(coco["images"], desc=f"Creating '{annotations}' segmentations from coco annotations"):
        file_name = image_metadata["file_name"]
        image_path = os.path.join(image_dir, file_name)
        assert os.path.exists(image_path), image_path
        image_paths.append(image_path)

        seg_path = os.path.join(seg_dir, file_name.replace(".png", ".tif"))
        seg_paths.append(seg_path)
        if os.path.exists(seg_path):
            continue

        seg = _annotations_to_instances(coco, image_metadata, category_id)
        imageio.imwrite(seg_path, seg, compression="zlib")

    return natsorted(image_paths), natsorted(seg_paths)


def _create_split_csv(path, image_paths):
    csv_path = os.path.join(path, "cytonuke_split.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {split: json.loads(df.iloc[0][split].replace("'", '"')) for split in ("train", "val", "test")}

    print(f"Creating a new split file at '{csv_path}'.")
    image_ids = natsorted(os.path.basename(p) for p in image_paths)

    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    df = pd.DataFrame.from_dict([split_ids])
    df.to_csv(csv_path, index=False)

    return split_ids


def get_cytonuke_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CytoNuke data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded and stored for further preprocessing.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "cytonuke.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_cytonuke_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    annotations: Literal["nuclei", "cell"] = "cell",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CytoNuke data.

    NOTE: The source publishes no official split, so this function creates and stores a
    deterministic split (65% train, 15% val, 20% test) the first time it is called.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    if annotations not in CATEGORY_IDS:
        raise ValueError(f"'{annotations}' is not a valid annotation choice.")

    data_dir = get_cytonuke_data(path, download)
    image_paths, seg_paths = _create_segmentations_from_annotations(data_dir, annotations)

    split_ids = _create_split_csv(path, image_paths)[split]
    raw_paths = [p for p in image_paths if os.path.basename(p) in split_ids]
    label_paths = [p for p in seg_paths if os.path.basename(p).replace(".tif", ".png") in split_ids]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_cytonuke_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    annotations: Literal["nuclei", "cell"] = "cell",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CytoNuke dataset for nucleus and whole-cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotations: The choice of annotations.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cytonuke_paths(path, split, annotations, download)

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
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_cytonuke_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    annotations: Literal["nuclei", "cell"] = "cell",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CytoNuke dataloader for nucleus and whole-cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotations: The choice of annotations.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cytonuke_dataset(path, patch_shape, split, annotations, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
