"""The PMCanalSeg dataset contains annotations for segmentation of the maxillary pterygopalatine
canal and the mandibular canal in 3D CBCT images.

The dataset is located at https://doi.org/10.7910/DVN/RTIGTP, hosted on Harvard Dataverse under
a CC0 1.0 license.

The dataset is from the publication https://doi.org/10.1038/s41597-026-06620-w.
Please cite it if you use this dataset for your research.

The dataset comprises 191 patients, each with an 'upper' scan (maxilla) annotated for the
pterygopalatine canal and a 'lower' scan (mandible) annotated for the mandibular canal, plus an
unannotated full 'skull' scan. This module only exposes the annotated 'upper' and 'lower' volumes.
"""

import os
import hashlib
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import requests
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


PERSISTENT_ID = "doi:10.7910/DVN/RTIGTP"
BASE_URL = "https://dataverse.harvard.edu"

# The Dataverse API rejects requests with the default 'python-requests' user agent (403 Forbidden).
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get_manifest(path):
    import json

    manifest_path = os.path.join(path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)

    url = f"{BASE_URL}/api/datasets/:persistentId/versions/:latest?persistentId={PERSISTENT_ID}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    files = r.json()["data"]["files"]

    manifest = []
    for f in files:
        directory_label = f.get("directoryLabel", "")
        if not directory_label.startswith(("upper/", "lower/")):
            continue

        data_file = f["dataFile"]
        manifest.append({
            "directory": directory_label,
            "filename": data_file["filename"],
            "id": data_file["id"],
            "md5": data_file.get("md5"),
        })

    os.makedirs(path, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest


def _download_file(url, path, md5=None):
    if os.path.exists(path):
        return

    tmp_path = f"{path}.incomplete"
    with requests.get(url, stream=True, headers=HEADERS) as r:
        r.raise_for_status()
        file_size = int(r.headers.get("Content-Length", 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=f"Download {url} to {path}") as r_raw:
            with open(tmp_path, "wb") as f:
                for chunk in iter(lambda: r_raw.read(1 << 20), b""):
                    f.write(chunk)

    if md5 is not None:
        hasher = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != md5:
            raise RuntimeError(f"The checksum of {url} does not match the expected checksum.")

    os.replace(tmp_path, path)


def get_pmcanalseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PMCanalSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)
    manifest = _get_manifest(path)

    missing = [entry for entry in manifest if not os.path.exists(os.path.join(path, entry["directory"], entry["filename"]))]  # noqa
    if missing and not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    for entry in tqdm(missing, desc="Downloading PMCanalSeg"):
        out_dir = os.path.join(path, entry["directory"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, entry["filename"])
        url = f"{BASE_URL}/api/access/datafile/{entry['id']}"
        _download_file(url, out_path, entry["md5"])

    return path


def get_pmcanalseg_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["mandibular", "pterygopalatine"] = "mandibular",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PMCanalSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of canal to segment. Either 'mandibular' (from the 'lower'
            mandible CBCT scans) or 'pterygopalatine' (from the 'upper' maxillary CBCT scans).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in ("mandibular", "pterygopalatine"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose 'mandibular' or 'pterygopalatine'.")  # noqa

    data_dir = get_pmcanalseg_data(path, download)
    subdir = "lower" if label_choice == "mandibular" else "upper"

    image_paths = natsorted(glob(os.path.join(data_dir, subdir, "Patient_*", "image.nii.gz")))
    gt_paths = [p.replace("image.nii.gz", "label.nii.gz") for p in image_paths]

    image_paths = [p for p, g in zip(image_paths, gt_paths) if os.path.exists(g)]
    gt_paths = [g for g in gt_paths if os.path.exists(g)]

    return image_paths, gt_paths


def get_pmcanalseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["mandibular", "pterygopalatine"] = "mandibular",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PMCanalSeg dataset for segmentation of the mandibular or pterygopalatine canal in CBCT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of canal to segment. Either 'mandibular' or 'pterygopalatine'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_pmcanalseg_paths(path, label_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_pmcanalseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["mandibular", "pterygopalatine"] = "mandibular",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PMCanalSeg dataloader for segmentation of the mandibular or pterygopalatine canal in CBCT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of canal to segment. Either 'mandibular' or 'pterygopalatine'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pmcanalseg_dataset(path, patch_shape, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
