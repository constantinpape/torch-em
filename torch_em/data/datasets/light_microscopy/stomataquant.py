"""The StomataQuant dataset contains annotations for stomata, stomatal pore and pavement cell segmentation
in bright-field microscopy images of leaf epidermis.

Two segmentation tasks are available, selected with the 'task' argument:
- 'pores': stomata (class 0) and stomatal pores (class 1), from 500 development images.
- 'pavement_cells': stomata (class 0) and pavement cells (class 1), from 613 development images.

The development images are split into 'train' and 'val'. The independent test images of the publication
(45 per task, from other species and imaging conditions) are available as the 'test' split. The annotations are
distributed as YOLO polygons (class id and normalized polygon coordinates), which are rasterized to label images
by this module. The class ids are not documented by the authors. The order above was inferred from the data:
pores are small polygons lying inside stomata polygons, and pavement cells are the more numerous class.

Label images can be one of two types, selected with the 'label_type' argument:
- 'instances': every polygon gets its own id. Where polygons overlap, stomata are drawn over pavement cells and
  pores are drawn over stomata (see `INSTANCE_DRAW_ORDER`).
- 'semantic': the background is 0 and every polygon is labeled with its class id plus one, using the same
  drawing order.

The images differ in size and some are stored as grayscale, palette or RGBA images. These are converted to RGB.
The large detection data (Supplementary Dataset S1-1) contains bounding boxes only and is not used.

The data is located at https://doi.org/10.5281/zenodo.18934358 and released under a CC-BY-4.0 license.
This dataset is from the publication https://doi.org/10.1093/jpe/rtag063.
Please cite it if you use this dataset in your research.
"""

import os
import uuid
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from concurrent import futures
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/api/records/18934358/files"

FILES = {
    "pores": "Supplementary Dataset S1-2_Stomata_and_pores_segmentation_model.zip",
    "pavement_cells": "Supplementary Dataset S1-3_Stomata_and_pavement_cells_segmentation_model.zip",
    "test": "Supplementary Dataset S2.zip",
}

CHECKSUMS = {
    "pores": "d6c06377cf21e5f9c42fc2f65df6cd1f5c18b9c36e4fd3edca5eef4fac618eef",
    "pavement_cells": "8db4ccb60ba4f8153a7f915b6691fa9841eb70ed9462f2cf5e0387b39d02eb7f",
    "test": "4378b7a6ffd125e7ef036b7ca9baa8aa03ccd771c6009407e9450b6eeb0c705c",
}

TASKS = ("pores", "pavement_cells")
SPLITS = ("train", "val", "test")
LABEL_TYPES = ("instances", "semantic")

TEST_FOLDERS = {
    "pores": "Test_stomata_and_pores_segmentation_model",
    "pavement_cells": "Test_stomata_and_pavement_cells_segmentation_model",
}

INSTANCE_DRAW_ORDER = {"pores": (0, 1), "pavement_cells": (1, 0)}
"""The order in which the classes are drawn for each task, later classes overwrite earlier ones."""


def _read_polygons(txt_path):
    import numpy as np

    polygons = {}
    with open(txt_path) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
                continue
            polygons.setdefault(int(tokens[0]), []).append(np.array(tokens[1:], dtype="float64").reshape(-1, 2))
    return polygons


def _write_atomic(path, array):
    import imageio.v3 as imageio

    extension = os.path.splitext(path)[1]
    tmp_path = f"{os.path.splitext(path)[0]}.{uuid.uuid4().hex}.incomplete{extension}"
    imageio.imwrite(tmp_path, array)
    os.replace(tmp_path, path)


def _process_item(image_path, txt_path, rgb_path, label_path, task, label_type):
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(image_path) as image:
        width, height = image.size
        needs_conversion = image.mode != "RGB"

    if needs_conversion and not os.path.exists(rgb_path):
        with Image.open(image_path) as image:
            _write_atomic(rgb_path, np.asarray(image.convert("RGB")))

    if os.path.exists(label_path):
        return

    polygons = _read_polygons(txt_path)
    # skimage.draw.polygon needs about 0.5 s per polygon on these large images, the PIL scanline fill takes ms.
    canvas = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    next_id = 1
    for class_id in INSTANCE_DRAW_ORDER[task]:
        for points in polygons.get(class_id, []):
            vertices = [(x * width, y * height) for x, y in points]
            draw.polygon(vertices, fill=next_id if label_type == "instances" else class_id + 1)
            next_id += 1

    _write_atomic(label_path, np.asarray(canvas).astype("uint16"))


def _list_items(data_dir, task, split):
    if split == "test":
        folder = os.path.join(data_dir, "test", TEST_FOLDERS[task])
        image_paths = natsorted(glob(os.path.join(folder, "imgs", "*")))
        label_dir = os.path.join(folder, "groundtruth_labels")
    else:
        folder = os.path.join(data_dir, task)
        image_paths = natsorted(glob(os.path.join(folder, "images", split, "*")))
        label_dir = os.path.join(folder, "labels", split)

    items = []
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(label_dir, f"{stem}.txt")
        if os.path.exists(txt_path):
            items.append((image_path, txt_path, stem))

    return items


def get_stomataquant_data(
    path: Union[os.PathLike, str], task: Literal["pores", "pavement_cells"], split: Literal["train", "val", "test"],
    download: bool = False,
) -> str:
    """Download the StomataQuant dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        task: The segmentation task. Either 'pores' or 'pavement_cells'.
        split: The data split. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if task not in TASKS:
        raise ValueError(f"'{task}' is not a valid task. Choose one of {TASKS}.")
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    archive = "test" if split == "test" else task
    dst = os.path.join(path, archive)
    if os.path.exists(dst) and os.listdir(dst):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{archive}.zip")
    url = f"{BASE_URL}/{FILES[archive].replace(' ', '%20')}/content"
    util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[archive])
    util.unzip(zip_path=zip_path, dst=dst, remove=False)

    return path


def get_stomataquant_paths(
    path: Union[os.PathLike, str],
    task: Literal["pores", "pavement_cells"],
    split: Literal["train", "val", "test"],
    label_type: Literal["instances", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the StomataQuant data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        task: The segmentation task. Either 'pores' or 'pavement_cells'.
        split: The data split. One of 'train', 'val' or 'test'.
        label_type: The type of label image. Either 'instances' or 'semantic'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_type not in LABEL_TYPES:
        raise ValueError(f"'{label_type}' is not a valid label type. Choose one of {LABEL_TYPES}.")

    data_dir = get_stomataquant_data(path, task, split, download)
    items = _list_items(data_dir, task, split)
    assert len(items) > 0, f"No images with annotations were found for task '{task}' and split '{split}'."

    rgb_dir = os.path.join(path, "images_rgb", task, split)
    label_dir = os.path.join(path, "labels", task, label_type, split)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    jobs = [
        (image_path, txt_path, os.path.join(rgb_dir, f"{stem}.png"), os.path.join(label_dir, f"{stem}.tif"))
        for image_path, txt_path, stem in items
    ]
    todo = [job for job in jobs if not os.path.exists(job[3])]
    if todo:
        with futures.ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
            tasks = [pool.submit(_process_item, *job, task, label_type) for job in todo]
            for job in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Preprocess StomataQuant"):
                job.result()

    image_paths = [job[2] if os.path.exists(job[2]) else job[0] for job in jobs]
    label_paths = [job[3] for job in jobs]
    return image_paths, label_paths


def get_stomataquant_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: Literal["pores", "pavement_cells"],
    split: Literal["train", "val", "test"],
    label_type: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the StomataQuant dataset for stomata, pore and pavement cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        task: The segmentation task. Either 'pores' or 'pavement_cells'.
        split: The data split. One of 'train', 'val' or 'test'.
        label_type: The type of label image. Either 'instances' or 'semantic'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        resize_inputs: Whether to resize the inputs to the patch shape. The images differ in size.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_stomataquant_paths(path, task, split, label_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    if label_type == "instances":
        kwargs, _ = util.add_instance_label_transform(
            kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_stomataquant_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    task: Literal["pores", "pavement_cells"],
    split: Literal["train", "val", "test"],
    label_type: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the StomataQuant dataloader for stomata, pore and pavement cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The segmentation task. Either 'pores' or 'pavement_cells'.
        split: The data split. One of 'train', 'val' or 'test'.
        label_type: The type of label image. Either 'instances' or 'semantic'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        resize_inputs: Whether to resize the inputs to the patch shape. The images differ in size.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_stomataquant_dataset(
        path, patch_shape, task, split, label_type, offsets, boundaries, binary, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
