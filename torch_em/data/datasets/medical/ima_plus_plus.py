"""The IMA++ (ISIC Archive Multi-Annotator) dataset contains multiple independent human
lesion segmentation masks for dermoscopy images from the ISIC Archive, for the task of
studying inter-annotator variability in skin lesion segmentation.

This dataset is located at https://doi.org/10.5281/zenodo.14201693, under the
CC BY-NC-ND 4.0 license (non-commercial use, no derivatives). The dataset is from the
publication https://doi.org/10.48550/arXiv.2512.21472. Please cite it if you use this
dataset for your research.

The dataset contains 17,684+ segmentation masks (single-annotator, majority-vote consensus
'MV', and STAPLE consensus 'ST') spanning 14,967 dermoscopic images, of which 2,394 images
have 2-5 independent masks annotated by up to 16 distinct annotators. Zenodo only hosts the
masks and metadata; the corresponding raw images are distributed separately via the ISIC
Archive (as the dedicated "IMA++" collection, id 482) and are downloaded here from the
public ISIC S3 bucket, using the image identifiers listed in the metadata.

NOTE: This is a much larger dataset than the ISIC 2018 challenge data already available in
`torch_em.data.datasets.medical.isic`: it ships multiple independent masks per image (instead
of a single ground truth) and its images are not bundled in a single archive, so downloading
them requires a separate, per-image acquisition step.
"""

import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Tuple, Optional, List

import requests
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


URLS = {
    "segs": "https://zenodo.org/api/records/14201693/files/segs.zip/content",
    "seg_metadata": "https://zenodo.org/api/records/14201693/files/seg_metadata.csv/content",
    "img_metadata": "https://zenodo.org/api/records/14201693/files/img_metadata.csv/content",
}
CHECKSUMS = {
    "segs": "4141feef60a168d5c699599e0a2089e6c689661ba6630459a138721cbf74ddc2",
    "seg_metadata": None,
    "img_metadata": None,
}
IMAGE_URL = "https://isic-archive.s3.amazonaws.com/images/{}.jpg"

CONSENSUS_ANNOTATORS = ["MV", "ST"]


def get_ima_plus_plus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the segmentation masks and metadata for the IMA++ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    seg_dir = os.path.join(path, "segs")
    if not os.path.exists(seg_dir):
        zip_path = os.path.join(path, "segs.zip")
        util.download_source(path=zip_path, url=URLS["segs"], download=download, checksum=CHECKSUMS["segs"])
        util.unzip(zip_path=zip_path, dst=seg_dir, remove=False)

    for name in ["seg_metadata", "img_metadata"]:
        csv_path = os.path.join(path, f"{name}.csv")
        util.download_source(path=csv_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])

    return path


def _read_seg_metadata(path):
    with open(os.path.join(path, "seg_metadata.csv")) as f:
        return list(csv.DictReader(f))


def _select_rows(rows, annotator):
    if annotator is None:
        # Default to a single mask per image: the STAPLE consensus mask for images that have
        # one, and the (unique) single-annotator mask otherwise.
        by_image = {}
        for row in rows:
            by_image.setdefault(row["ISIC_id"], []).append(row)

        selected = []
        for isic_id, image_rows in by_image.items():
            consensus_rows = [row for row in image_rows if row["annotator"] == "ST"]
            selected.append(consensus_rows[0] if consensus_rows else image_rows[0])

        return selected

    elif annotator == "all":
        return rows

    else:
        selected = [row for row in rows if row["annotator"] == annotator]
        assert len(selected) > 0, f"'{annotator}' did not match any masks. See 'seg_metadata.csv' for valid values."
        return selected


def _download_images(isic_ids, image_dir, download):
    os.makedirs(image_dir, exist_ok=True)

    missing = [isic_id for isic_id in isic_ids if not os.path.exists(os.path.join(image_dir, f"{isic_id}.jpg"))]
    if len(missing) == 0:
        return

    if not download:
        raise RuntimeError(f"Cannot find {len(missing)} image(s) at {image_dir}, but download was set to False")

    def _download_one(isic_id):
        dst = os.path.join(image_dir, f"{isic_id}.jpg")
        tmp = f"{dst}.incomplete"
        response = requests.get(IMAGE_URL.format(isic_id), timeout=60)
        response.raise_for_status()
        with open(tmp, "wb") as f:
            f.write(response.content)
        os.replace(tmp, dst)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_download_one, isic_id) for isic_id in missing]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading IMA++ images"):
            future.result()


def get_ima_plus_plus_paths(
    path: Union[os.PathLike, str], annotator: Optional[str] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the IMA++ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotator: The choice of annotator whose masks are used. By default (`None`), a single
            mask per image is returned: the 'ST' (STAPLE) consensus mask for the 2,394 images
            that have multiple annotations, and the unique single-annotator mask otherwise. Pass
            `'all'` to get every mask (multiple entries per multi-annotator image), or a specific
            annotator id (e.g. `'A04'`, `'MV'`, `'ST'`) to filter to masks from that source only.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ima_plus_plus_data(path=path, download=download)

    rows = _select_rows(_read_seg_metadata(data_dir), annotator)

    isic_ids = sorted(set(row["ISIC_id"] for row in rows))
    image_dir = os.path.join(data_dir, "images")
    _download_images(isic_ids, image_dir, download)

    image_paths = [os.path.join(image_dir, f"{row['ISIC_id']}.jpg") for row in rows]
    gt_paths = [os.path.join(data_dir, "segs", row["seg_filename"]) for row in rows]

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0
    for gt_path in gt_paths:
        assert os.path.exists(gt_path), gt_path

    return image_paths, gt_paths


def get_ima_plus_plus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotator: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the IMA++ dataset for multi-annotator skin lesion segmentation in dermoscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        annotator: The choice of annotator whose masks are used. See `get_ima_plus_plus_paths`.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ima_plus_plus_paths(path=path, annotator=annotator, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs,
            patch_shape=patch_shape,
            resize_inputs=resize_inputs,
            resize_kwargs=resize_kwargs,
            ensure_rgb=to_rgb,
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_ima_plus_plus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotator: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the IMA++ dataloader for multi-annotator skin lesion segmentation in dermoscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotator: The choice of annotator whose masks are used. See `get_ima_plus_plus_paths`.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ima_plus_plus_dataset(path, patch_shape, annotator, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
