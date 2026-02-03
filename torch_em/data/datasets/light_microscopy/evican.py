"""The EVICAN dataset contains phase contrast and brightfield microscopy images
with cell and nucleus segmentations.

NOTE: The data is sparsely annotated.

The dataset is located at https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.AJBV1S.
This dataset is from the publication https://doi.org/10.1093/bioinformatics/btaa225.
Please cite it if you use this dataset in your research.
"""

import os
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, Optional, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from .. import util


URLS = {
    # Image archives
    "images_train": "https://edmond.mpg.de/api/access/datafile/102910",
    "images_val": "https://edmond.mpg.de/api/access/datafile/102911",
    "images_eval": "https://edmond.mpg.de/api/access/datafile/102912",

    # EVICAN2 annotations
    "annotations_evican2_train": "https://edmond.mpg.de/api/access/datafile/102915",
    "annotations_evican2_val": "https://edmond.mpg.de/api/access/datafile/102916",
    "annotations_evican2_eval_easy": "https://edmond.mpg.de/api/access/datafile/102918",
    "annotations_evican2_eval_medium": "https://edmond.mpg.de/api/access/datafile/102919",
    "annotations_evican2_eval_difficult": "https://edmond.mpg.de/api/access/datafile/102917",

    # EVICAN60 annotations
    "annotations_evican60_train": "https://edmond.mpg.de/api/access/datafile/102921",
    "annotations_evican60_val": "https://edmond.mpg.de/api/access/datafile/102922",
    "annotations_evican60_eval_easy": "https://edmond.mpg.de/api/access/datafile/102924",
    "annotations_evican60_eval_medium": "https://edmond.mpg.de/api/access/datafile/102920",
    "annotations_evican60_eval_difficult": "https://edmond.mpg.de/api/access/datafile/102923",
}
CHECKSUMS = None

ANNOTATION_TYPES = ["evican2", "evican60"]
SEGMENTATION_TYPES = ["cell", "nucleus"]
SPLITS = ["train", "val", "eval_easy", "eval_medium", "eval_difficult"]

# Map from segmentation type to COCO category name.
_CATEGORY_NAMES = {"cell": "Cell", "nucleus": "Nucleus"}


def _annotations_to_instances(coco, image_metadata, category_ids):
    """Convert COCO annotations to an instance segmentation mask.

    NOTE: The EVICAN annotations lack the 'area' field, so we compute it from the masks.
    """
    import vigra

    annotation_ids = coco.getAnnIds(imgIds=image_metadata["id"], catIds=category_ids)
    annotations = coco.loadAnns(annotation_ids)
    assert len(annotations) <= np.iinfo("uint16").max

    shape = (image_metadata["height"], image_metadata["width"])
    seg = np.zeros(shape, dtype="uint32")

    # Compute masks and their areas for sorting.
    masks = []
    areas = []
    for annotation in annotations:
        mask = coco.annToMask(annotation).astype("bool")
        masks.append(mask)
        areas.append(mask.sum())

    # Sort by area (smallest last so they are not occluded by larger objects).
    sorting = np.argsort(areas)[::-1]

    for seg_id, idx in enumerate(sorting, 1):
        seg[masks[idx]] = seg_id

    # Filter small noise objects from overlapping annotations.
    min_size = 50
    seg_ids, sizes = np.unique(seg, return_counts=True)
    seg[np.isin(seg, seg_ids[sizes < min_size])] = 0

    vigra.analysis.relabelConsecutive(seg, out=seg)

    return seg.astype("uint16")


def _download_images(path, split, download):
    """Download and extract image archives for the given split."""
    # eval_easy, eval_medium, eval_difficult all share the same eval images.
    image_split = "eval" if split.startswith("eval") else split
    image_dir = os.path.join(path, "images", image_split)
    if os.path.exists(image_dir):
        return image_dir

    os.makedirs(image_dir, exist_ok=True)
    url_key = f"images_{image_split}"
    zip_path = os.path.join(path, f"EVICAN_{image_split}.zip")
    util.download_source(zip_path, URLS[url_key], download, checksum=None)
    util.unzip(zip_path, image_dir, remove=True)

    return image_dir


def _download_annotations(path, split, annotation_type, download):
    """Download the COCO annotation JSON for the given split and annotation type."""
    ann_dir = os.path.join(path, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    ann_file = os.path.join(ann_dir, f"instances_{split}_{annotation_type}.json")
    if os.path.exists(ann_file):
        return ann_file

    url_key = f"annotations_{annotation_type}_{split}"
    util.download_source(ann_file, URLS[url_key], download, checksum=None)

    return ann_file


def _create_segmentations_from_coco_annotations(path, split, annotation_type, segmentation_type):
    """Convert COCO annotations to instance segmentation masks."""
    assert COCO is not None, (
        "'pycocotools' is required for processing the EVICAN ground-truth. "
        "Install it with 'conda install -c conda-forge pycocotools'."
    )

    image_split = "eval" if split.startswith("eval") else split
    image_dir = os.path.join(path, "images", image_split)
    seg_dir = os.path.join(path, "segmentations", annotation_type, segmentation_type, split)

    if os.path.exists(seg_dir):
        # Check that segmentation files exist.
        seg_paths = glob(os.path.join(seg_dir, "*.tif"))
        if len(seg_paths) > 0:
            image_paths = [os.path.join(image_dir, f"{Path(sp).stem}.jpg") for sp in seg_paths]
            return natsorted(image_paths), natsorted(seg_paths)

    os.makedirs(seg_dir, exist_ok=True)

    ann_file = os.path.join(path, "annotations", f"instances_{split}_{annotation_type}.json")
    assert os.path.exists(ann_file), f"Annotation file not found: {ann_file}"

    coco = COCO(ann_file)
    category_name = _CATEGORY_NAMES[segmentation_type]
    category_ids = coco.getCatIds(catNms=[category_name])
    image_ids = coco.getImgIds(catIds=category_ids)

    image_paths, seg_paths = [], []
    for image_id in tqdm(
        image_ids, desc=f"Creating EVICAN {segmentation_type} segmentations from COCO annotations ({split})"
    ):
        image_metadata = coco.loadImgs(image_id)[0]
        file_name = image_metadata["file_name"]

        image_path = os.path.join(image_dir, file_name)
        assert os.path.exists(image_path), image_path
        image_paths.append(image_path)

        seg_path = os.path.join(seg_dir, f"{Path(file_name).stem}.tif")
        seg_paths.append(seg_path)
        if os.path.exists(seg_path):
            continue

        # Suppress numpy 2.0 deprecation warning from pycocotools.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pycocotools")
            seg = _annotations_to_instances(coco, image_metadata, category_ids)

        imageio.imwrite(seg_path, seg, compression="zlib")

    assert len(image_paths) == len(seg_paths) and len(image_paths) > 0
    return natsorted(image_paths), natsorted(seg_paths)


def get_evican_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval_easy", "eval_medium", "eval_difficult"],
    annotation_type: Literal["evican2", "evican60"] = "evican2",
    download: bool = False,
) -> str:
    """Download the EVICAN dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val', 'eval_easy', 'eval_medium' or 'eval_difficult'.
        annotation_type: The annotation type to use. Either 'evican2' or 'evican60'.
        download: Whether to download the data if it is not present.

    Returns:
        The path where the dataset is stored.
    """
    os.makedirs(path, exist_ok=True)
    _download_images(path, split, download)
    _download_annotations(path, split, annotation_type, download)
    return path


def get_evican_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval_easy", "eval_medium", "eval_difficult"],
    annotation_type: Literal["evican2", "evican60"] = "evican2",
    segmentation_type: Literal["cell", "nucleus"] = "cell",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the EVICAN data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val', 'eval_easy', 'eval_medium' or 'eval_difficult'.
        annotation_type: The annotation type to use. Either 'evican2' or 'evican60'.
        segmentation_type: The segmentation target. Either 'cell' or 'nucleus'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_evican_data(path, split, annotation_type, download)
    image_paths, seg_paths = _create_segmentations_from_coco_annotations(
        path, split, annotation_type, segmentation_type
    )
    return image_paths, seg_paths


def get_evican_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "eval_easy", "eval_medium", "eval_difficult"],
    annotation_type: Literal["evican2", "evican60"] = "evican2",
    segmentation_type: Literal["cell", "nucleus"] = "cell",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EVICAN dataset for cell and nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'eval_easy', 'eval_medium' or 'eval_difficult'.
        annotation_type: The annotation type to use. Either 'evican2' or 'evican60'.
        segmentation_type: The segmentation target. Either 'cell' or 'nucleus'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in SPLITS, f"'{split}' is not a valid split. Choose from {SPLITS}."
    assert annotation_type in ANNOTATION_TYPES, f"'{annotation_type}' is not valid. Choose from {ANNOTATION_TYPES}."
    assert segmentation_type in SEGMENTATION_TYPES, \
        f"'{segmentation_type}' is not valid. Choose from {SEGMENTATION_TYPES}."

    image_paths, seg_paths = get_evican_paths(path, split, annotation_type, segmentation_type, download)

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
        **kwargs
    )


def get_evican_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "eval_easy", "eval_medium", "eval_difficult"],
    annotation_type: Literal["evican2", "evican60"] = "evican2",
    segmentation_type: Literal["cell", "nucleus"] = "cell",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EVICAN dataloader for cell and nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'eval_easy', 'eval_medium' or 'eval_difficult'.
        annotation_type: The annotation type to use. Either 'evican2' or 'evican60'.
        segmentation_type: The segmentation target. Either 'cell' or 'nucleus'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_evican_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        annotation_type=annotation_type,
        segmentation_type=segmentation_type,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
