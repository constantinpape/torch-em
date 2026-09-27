"""The AIR-LEISH dataset contains annotations for the segmentation of Leishmania amastigotes, host cells
(macrophages) and nuclei in microscopy images of Giemsa-stained Leishmania-infected macrophages.

The dataset consists of 180 RGB images of 1844 x 2709 pixels, in two sets of 90 images (`Set1`, file names
'20250328_CC*', and `Set2`, file names '20250203_CF*'). Every image is annotated with three classes,
'AM' (amastigotes), 'HC' (host cells) and 'NU' (nuclei), as COCO polygons (one polygon per instance) and,
with one exception, as a pixel-wise class mask (0: background, 1: amastigote, 2: host cell, 3: nucleus).
In the class masks the nuclei are cut out of the host cells (the polygon of a host cell covers its nucleus).

The `target` argument selects the labels:
- 'semantic': the class masks shipped with the dataset. Image '20250328_CCimage49' of `Set1` has no class mask
  and is not part of this target (179 images).
- 'amastigotes', 'host_cells' or 'nuclei': instance labels rasterized from the COCO polygons of this class
  (all 180 images, one id per instance in the order of the annotation file, later instances overwrite earlier
  ones where they overlap, which is rare). 26 images do not contain any amastigote and have an empty label.

NOTE: The images are stored as RGBA and are converted to RGB, and the labels of the instance targets are
rasterized once when the paths are first requested. 54 file names in the annotations of `Set2` have a doubled
'.png' extension, which is handled by this module.

The data is located at https://doi.org/10.5281/zenodo.17384855 and released under a CC-BY-4.0 license.
The Zenodo record also contains a second, smaller archive ('AIR-Leish_dataset.zip'), which holds 67 of the
same images and is not used here.

Please cite the publication associated with the Zenodo record if you use this dataset in your research.
"""

import os
import json
import uuid
from tqdm import tqdm
from concurrent import futures
from typing import Union, Tuple, Optional, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/17384855/files/AIR_LEISH_dataset_v1.zip"
CHECKSUM = "36cfeeecac27cc84266c40c8676c957065f4ddca2dfe9b231e86ad84d7451d2e"

SETS = {"set1": "Set1", "set2": "Set2"}
CLASS_IDS = {"amastigotes": 1, "host_cells": 2, "nuclei": 3}
TARGETS = ("semantic", *CLASS_IDS)


def _image_stem(file_name):
    stem = os.path.basename(file_name)
    while stem.lower().endswith(".png"):
        stem = stem[:-len(".png")]
    return stem


def _write_atomic(path, array):
    import imageio.v3 as imageio

    extension = os.path.splitext(path)[1]
    tmp_path = f"{os.path.splitext(path)[0]}.{uuid.uuid4().hex}.incomplete{extension}"
    imageio.imwrite(tmp_path, array)
    os.replace(tmp_path, path)


def _prepare_item(image_path, rgb_path, label_path, polygons, size):
    import numpy as np
    from PIL import Image, ImageDraw

    if not os.path.exists(rgb_path):
        with Image.open(image_path) as image:
            _write_atomic(rgb_path, np.asarray(image.convert("RGB")))

    if label_path is None or os.path.exists(label_path):
        return

    canvas = Image.new("I", size, 0)
    draw = ImageDraw.Draw(canvas)
    for instance_id, parts in enumerate(polygons, start=1):
        for part in parts:
            draw.polygon(list(zip(part[0::2], part[1::2])), fill=instance_id)

    _write_atomic(label_path, np.asarray(canvas).astype("uint16"))


def get_air_leish_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the AIR-LEISH dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "AIR LEISH dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "AIR_LEISH_dataset_v1.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create the expected folder in '{path}'."

    return data_dir


def get_air_leish_paths(
    path: Union[os.PathLike, str],
    target: Literal["semantic", "amastigotes", "host_cells", "nuclei"] = "semantic",
    subset: Optional[Literal["set1", "set2"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AIR-LEISH data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        target: The choice of labels. Either 'semantic' (class masks with 1: amastigote, 2: host cell, 3: nucleus)
            or the instances of one class: 'amastigotes', 'host_cells' or 'nuclei'.
        subset: The choice of image set. Either 'set1' or 'set2'. By default both sets are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if target not in TARGETS:
        raise ValueError(f"'{target}' is not a valid target. Choose one of {list(TARGETS)}.")
    if subset is not None and subset not in SETS:
        raise ValueError(f"'{subset}' is not a valid subset. Choose one of {list(SETS)}.")

    data_dir = get_air_leish_data(path, download)
    rgb_dir = os.path.join(path, "images_rgb")
    label_dir = os.path.join(path, "labels", target)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    jobs, raw_paths, label_paths = [], [], []
    for set_name in [SETS[subset]] if subset is not None else SETS.values():
        with open(os.path.join(data_dir, set_name, "_annotations.coco.json")) as f:
            coco = json.load(f)

        instances = {}
        for annotation in coco["annotations"]:
            if target != "semantic" and annotation["category_id"] == CLASS_IDS[target]:
                instances.setdefault(annotation["image_id"], []).append(annotation["segmentation"])

        for image in sorted(coco["images"], key=lambda entry: _image_stem(entry["file_name"])):
            stem = _image_stem(image["file_name"])
            image_path = os.path.join(data_dir, set_name, "Images", f"{stem}.png")
            assert os.path.exists(image_path), f"Cannot find the image for '{image['file_name']}'."

            rgb_path = os.path.join(rgb_dir, f"{set_name}_{stem}.png")
            if target == "semantic":
                mask_path = os.path.join(data_dir, set_name, "Masks", f"{stem}.png")
                if not os.path.exists(mask_path):
                    continue
                label_path, polygons = None, None
            else:
                label_path = os.path.join(label_dir, f"{set_name}_{stem}.tif")
                polygons = instances.get(image["id"], [])
                mask_path = label_path

            jobs.append((image_path, rgb_path, label_path, polygons, (image["width"], image["height"])))
            raw_paths.append(rgb_path)
            label_paths.append(mask_path)

    with futures.ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
        tasks = [pool.submit(_prepare_item, *job) for job in jobs]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Prepare AIR-LEISH"):
            task.result()

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_air_leish_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    target: Literal["semantic", "amastigotes", "host_cells", "nuclei"] = "semantic",
    subset: Optional[Literal["set1", "set2"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AIR-LEISH dataset for the segmentation of Leishmania amastigotes, host cells and nuclei.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        target: The choice of labels. Either 'semantic' (class masks with 1: amastigote, 2: host cell, 3: nucleus)
            or the instances of one class: 'amastigotes', 'host_cells' or 'nuclei'.
        subset: The choice of image set. Either 'set1' or 'set2'. By default both sets are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_air_leish_paths(path, target, subset, download)

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


def get_air_leish_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    target: Literal["semantic", "amastigotes", "host_cells", "nuclei"] = "semantic",
    subset: Optional[Literal["set1", "set2"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AIR-LEISH dataloader for the segmentation of Leishmania amastigotes, host cells and nuclei.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        target: The choice of labels. Either 'semantic' (class masks with 1: amastigote, 2: host cell, 3: nucleus)
            or the instances of one class: 'amastigotes', 'host_cells' or 'nuclei'.
        subset: The choice of image set. Either 'set1' or 'set2'. By default both sets are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_air_leish_dataset(path, patch_shape, target, subset, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
