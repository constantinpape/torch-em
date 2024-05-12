import os
from shutil import copyfileobj

import imageio
import numpy as np
import requests
import vigra
from tqdm import tqdm

import torch_em
import torch.utils.data
from .. import util

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

URLS = {
    "images": "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip",
    "train": ("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/"
              "LIVECell/livecell_coco_train.json"),
    "val": ("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/"
            "LIVECell/livecell_coco_val.json"),
    "test": ("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/"
             "LIVECell/livecell_coco_test.json")
}
# TODO
CHECKSUM = None


def _download_livecell_images(path, download):
    os.makedirs(path, exist_ok=True)
    image_path = os.path.join(path, "images")

    if os.path.exists(image_path):
        return

    url = URLS["images"]
    checksum = CHECKSUM
    zip_path = os.path.join(path, "livecell.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)


# TODO use download flag
def _download_annotation_file(path, split, download):
    annotation_file = os.path.join(path, f"{split}.json")
    if not os.path.exists(annotation_file):
        url = URLS[split]
        print("Downloading livecell annotation file from", url)
        with requests.get(url, stream=True) as r:
            with open(annotation_file, 'wb') as f:
                copyfileobj(r.raw, f)
    return annotation_file


def _annotations_to_instances(coco, image_metadata, category_ids):
    # create and save the segmentation
    annotation_ids = coco.getAnnIds(imgIds=image_metadata["id"], catIds=category_ids)
    annotations = coco.loadAnns(annotation_ids)
    assert len(annotations) <= np.iinfo("uint16").max
    shape = (image_metadata["height"], image_metadata["width"])
    seg = np.zeros(shape, dtype="uint32")

    # sort annotations by size, except for iscrowd which go first
    # we do this to minimize small noise from overlapping multi annotations
    # (see below)
    sizes = [ann["area"] if ann["iscrowd"] == 0 else 1 for ann in annotations]
    sorting = np.argsort(sizes)
    annotations = [annotations[i] for i in sorting]

    for seg_id, annotation in enumerate(annotations, 1):
        mask = coco.annToMask(annotation).astype("bool")
        assert mask.shape == seg.shape
        seg[mask] = seg_id

    # some images have multiple masks per object with slightly different foreground
    # this causes small noise objects we need to filter
    min_size = 50
    seg_ids, sizes = np.unique(seg, return_counts=True)
    seg[np.isin(seg, seg_ids[sizes < min_size])] = 0

    vigra.analysis.relabelConsecutive(seg, out=seg)

    return seg.astype("uint16")


def _create_segmentations_from_annotations(annotation_file, image_folder, seg_folder, cell_types):
    assert COCO is not None, "pycocotools is required for processing the LiveCELL ground-truth."

    coco = COCO(annotation_file)
    category_ids = coco.getCatIds(catNms=["cell"])
    image_ids = coco.getImgIds(catIds=category_ids)

    image_paths, seg_paths = [], []
    for image_id in tqdm(image_ids, desc="creating livecell segmentations from coco-style annotations"):
        # get the path for the image data and make sure the corresponding image exists
        image_metadata = coco.loadImgs(image_id)[0]
        file_name = image_metadata["file_name"]

        # if cell_type names are given we only select file names that match a cell_type
        if cell_types is not None and (not any([cell_type in file_name for cell_type in cell_types])):
            continue

        sub_folder = file_name.split("_")[0]
        image_path = os.path.join(image_folder, sub_folder, file_name)
        # something changed in the image layout? we keep the old version around in case this changes back...
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder, file_name)
        assert os.path.exists(image_path), image_path
        image_paths.append(image_path)

        # get the output path
        out_folder = os.path.join(seg_folder, sub_folder)
        os.makedirs(out_folder, exist_ok=True)
        seg_path = os.path.join(out_folder, file_name)
        seg_paths.append(seg_path)
        if os.path.exists(seg_path):
            continue

        seg = _annotations_to_instances(coco, image_metadata, category_ids)
        imageio.imwrite(seg_path, seg)

    assert len(image_paths) == len(seg_paths)
    assert len(image_paths) > 0, \
        f"No matching image paths were found. Did you pass invalid cell type naems ({cell_types})?"
    return image_paths, seg_paths


def _download_livecell_annotations(path, split, download, cell_types, label_path):
    annotation_file = _download_annotation_file(path, split, download)
    if split == "test":
        split_name = "livecell_test_images"
    else:
        split_name = "livecell_train_val_images"

    image_folder = os.path.join(path, "images", split_name)
    seg_folder = os.path.join(path, "annotations", split_name) if label_path is None\
        else os.path.join(label_path, "annotations", split_name)

    assert os.path.exists(image_folder), image_folder

    return _create_segmentations_from_annotations(annotation_file, image_folder, seg_folder, cell_types)


def get_livecell_dataset(
    path, split, patch_shape, download=False,
    offsets=None, boundaries=False, binary=False,
    cell_types=None, label_path=None, label_dtype=torch.int64, **kwargs
):
    """Dataset for the segmentation of cells in phase-contrast microscopy.

    This dataset is from the publication https://doi.org/10.1038/s41592-021-01249-6.
    Please cite it if you use this dataset for a publication.
    """
    assert split in ("train", "val", "test")
    if cell_types is not None:
        assert isinstance(cell_types, (list, tuple)), \
            f"cell_types must be passed as a list or tuple instead of {cell_types}"

    _download_livecell_images(path, download)
    image_paths, seg_paths = _download_livecell_annotations(path, split, download, cell_types, label_path)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, label_dtype = util.add_instance_label_transform(
        kwargs, add_binary_target=True, label_dtype=label_dtype,
        offsets=offsets, boundaries=boundaries, binary=binary
    )

    dataset = torch_em.data.ImageCollectionDataset(
        image_paths, seg_paths, patch_shape=patch_shape, label_dtype=label_dtype, **kwargs
    )
    return dataset


def get_livecell_loader(
    path, split, patch_shape, batch_size, download=False,
    offsets=None, boundaries=False, binary=False,
    cell_types=None, label_path=None, label_dtype=torch.int64, **kwargs
):
    """Dataloader for the segmentation of cells in phase-contrast microscopy. See 'get_livecell_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_livecell_dataset(
        path, split, patch_shape, download=download, offsets=offsets, boundaries=boundaries, binary=binary,
        cell_types=cell_types, label_path=label_path, label_dtype=label_dtype, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
