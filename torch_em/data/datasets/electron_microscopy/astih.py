"""ASTIH is a dataset for axon and myelin segmentation in microscopy images.

It contains diverse microscopy datasets (TEM, SEM, BF) designed to benchmark
and train axon and myelin segmentation models. It provides over 60,000 manually
segmented fibers across three microscopy modalities.

The dataset is described at https://axondeepseg.github.io/ASTIH/.
Please cite the corresponding publication if you use the dataset in your research.
"""

import os
import io
from glob import glob
from typing import List, Literal, Optional, Sequence, Tuple, Union

import imageio
import numpy as np
import requests
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DANDI_API = "https://api.dandiarchive.org/api"

DATASETS = {
    "TEM1": {
        "dandi_id": "001436",
        "version": "0.250512.1625",
        "description": "TEM Images of Corpus Callosum in Control and Cuprizone-Intoxicated Mice",
        "test_subjects": ["sub-nyuMouse26"],
        "file_ext": "png",
    },
    "TEM2": {
        "dandi_id": "001350",
        "version": "0.250511.1527",
        "description": "TEM Images of Corpus Callosum in Flox/SRF-cKO Mice",
        "test_subjects": None,  # External test set.
        "test_url": "https://github.com/axondeepseg/data_axondeepseg_srf_testing/archive/refs/tags/r20250513-neurips2025.zip",  # noqa
        "file_ext": "png",
    },
    "SEM1": {
        "dandi_id": "001442",
        "version": "0.250512.1626",
        "description": "SEM Images of Rat Spinal Cord",
        "test_subjects": ["sub-rat6"],
        "file_ext": "png",
    },
    "BF1": {
        "dandi_id": "001440",
        "version": "0.250509.1913",
        "description": "BF Images of Rat Nerves at Different Regeneration Stages",
        "test_subjects": ["sub-uoftRat02", "sub-uoftRat07"],
        "file_ext": "png",
    },
    "BF2": {
        "dandi_id": "001630",
        "version": "0.251127.1424",
        "description": "Bright-Field Images of Rabbit Nerves",
        "test_subjects": ["sub-22G132040x3"],
        "file_ext": "tif",
    },
}

DATASET_NAMES = list(DATASETS.keys())

LABEL_CLASSES = {"background": 0, "myelin": 1, "axon": 2}


def _list_dandi_assets(dandi_id, version):
    """List all assets in a DANDI dataset via the REST API."""
    all_assets = []
    url = f"{DANDI_API}/dandisets/{dandi_id}/versions/{version}/assets/?page_size=200"
    while url:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        all_assets.extend(data["results"])
        url = data.get("next")
    return all_assets


def _download_dandi_asset(asset_id, out_path):
    """Download a single DANDI asset by its ID."""
    url = f"{DANDI_API}/assets/{asset_id}/download/"
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        file_size = int(r.headers.get("Content-Length", 0))
        desc = f"Download {os.path.basename(out_path)}"
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw, open(out_path, "wb") as f:
            from shutil import copyfileobj
            copyfileobj(r_raw, f)


def _find_image_label_pairs(assets, file_ext):
    """Find matching image and axonmyelin label pairs from the DANDI asset list."""
    # Index label assets by their stem.
    label_map = {}
    for a in assets:
        p = a["path"]
        if "axonmyelin-manual.png" in p:
            # Extract the image stem: remove the _seg-axonmyelin-manual.png suffix
            stem = os.path.basename(p).replace("_seg-axonmyelin-manual.png", "")
            label_map[stem] = a

    # Find images that have a matching label.
    pairs = []
    for a in assets:
        p = a["path"]
        if "/micr/" in p and not p.startswith("derivatives") and p.endswith(f".{file_ext}"):
            stem = os.path.basename(p).rsplit(".", 1)[0]
            if stem in label_map:
                subject = p.split("/")[0]
                pairs.append({
                    "subject": subject,
                    "image_asset": a,
                    "label_asset": label_map[stem],
                    "stem": stem,
                })
    return pairs


def _preprocess_label(label):
    """Map label values to: 0=background, 1=myelin, 2=axon."""
    if label.ndim == 3:
        label = label[..., 0]
    new_label = np.zeros_like(label)
    new_label[(label == 127) | (label == 128)] = 1
    new_label[label == 255] = 2
    return new_label


def _download_and_preprocess(out_path, dataset_info, split, download):
    """Download data from DANDI, pair images with labels, and save as h5 files."""
    import h5py

    if not download:
        raise RuntimeError(f"Cannot find the data at {out_path}, but download was set to False")

    os.makedirs(out_path, exist_ok=True)

    dandi_id = dataset_info["dandi_id"]
    version = dataset_info["version"]
    file_ext = dataset_info["file_ext"]
    test_subjects = dataset_info["test_subjects"]

    # List and pair assets.
    assets = _list_dandi_assets(dandi_id, version)
    pairs = _find_image_label_pairs(assets, file_ext)

    if len(pairs) == 0:
        raise RuntimeError(f"No image-label pairs found for DANDI:{dandi_id}")

    # Filter by split.
    if test_subjects is not None:
        if split == "train":
            pairs = [p for p in pairs if p["subject"] not in test_subjects]
        else:
            pairs = [p for p in pairs if p["subject"] in test_subjects]
    else:
        # For datasets with external test sets (TEM2), all DANDI data is training.
        if split == "test":
            raise NotImplementedError(
                "The test set for this dataset is hosted externally. "
                "Please use the ASTIH repository's get_data.py script for the test split."
            )

    # Download and preprocess each pair.
    for pair in tqdm(pairs, desc=f"Processing {split} data"):
        h5_path = os.path.join(out_path, f"{pair['stem']}.h5")
        if os.path.exists(h5_path):
            continue

        # Download image.
        img_data = requests.get(f"{DANDI_API}/assets/{pair['image_asset']['asset_id']}/download/").content
        raw = imageio.imread(io.BytesIO(img_data))
        if raw.ndim == 3:
            raw = raw[..., 0]

        # Download label.
        lbl_data = requests.get(f"{DANDI_API}/assets/{pair['label_asset']['asset_id']}/download/").content
        label = imageio.imread(io.BytesIO(lbl_data))
        label = _preprocess_label(label)

        assert raw.shape == label.shape, f"Shape mismatch: {raw.shape} vs {label.shape}"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=label, compression="gzip")


def get_astih_data(
    path: Union[os.PathLike, str],
    name: str,
    split: Literal["train", "test"],
    download: bool = False,
) -> str:
    """Download the ASTIH data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the dataset. Available names are 'TEM1', 'TEM2', 'SEM1', 'BF1', 'BF2'.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    assert name in DATASETS, f"Invalid dataset name: {name}. Choose from {DATASET_NAMES}."

    out_path = os.path.join(path, name, split)
    if os.path.exists(out_path) and len(glob(os.path.join(out_path, "*.h5"))) > 0:
        return out_path

    _download_and_preprocess(out_path, DATASETS[name], split, download)
    return out_path


def get_astih_paths(
    path: Union[os.PathLike, str],
    name: Optional[Union[str, Sequence[str]]] = None,
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the ASTIH data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the dataset. Available names are 'TEM1', 'TEM2', 'SEM1', 'BF1', 'BF2'.
            Can be a single name, a list of names, or None to load all datasets.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths for the stored data.
    """
    if name is None:
        name = DATASET_NAMES
    elif isinstance(name, str):
        name = [name]

    all_paths = []
    for nn in name:
        data_root = get_astih_data(path, nn, split, download)
        paths = glob(os.path.join(data_root, "*.h5"))
        paths.sort()
        all_paths.extend(paths)

    return all_paths


def get_astih_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    name: Optional[Union[str, Sequence[str]]] = None,
    split: Literal["train", "test"] = "train",
    download: bool = False,
    label_classes: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dataset:
    """Get the ASTIH dataset for axon and myelin segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        name: The name of the dataset. Can be one of 'TEM1', 'TEM2', 'SEM1', 'BF1', 'BF2',
            a list of these names to combine datasets, or None to load all datasets.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        label_classes: The label classes to use for one-hot encoding. Available classes are
            'background', 'myelin', and 'axon'. By default set to None, which returns
            the label map with all classes (0=background, 1=myelin, 2=axon).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    all_paths = get_astih_paths(path, name, split, download)

    if label_classes is not None:
        class_ids = []
        for cls_name in label_classes:
            if cls_name not in LABEL_CLASSES:
                raise ValueError(
                    f"Invalid class name: '{cls_name}'. Choose from {list(LABEL_CLASSES.keys())}."
                )
            class_ids.append(LABEL_CLASSES[cls_name])
        label_transform = torch_em.transform.label.OneHotTransform(class_ids=class_ids)
        msg = "'label_classes' is set, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    return torch_em.default_segmentation_dataset(
        raw_paths=all_paths,
        raw_key="raw",
        label_paths=all_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_astih_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    name: Optional[Union[str, Sequence[str]]] = None,
    split: Literal["train", "test"] = "train",
    download: bool = False,
    label_classes: Optional[Sequence[str]] = None,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for axon and myelin segmentation in the ASTIH dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        name: The name of the dataset. Can be one of 'TEM1', 'TEM2', 'SEM1', 'BF1', 'BF2',
            a list of these names to combine datasets, or None to load all datasets.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        label_classes: The label classes to use for one-hot encoding. Available classes are
            'background', 'myelin', and 'axon'. By default set to None, which returns
            the label map with all classes (0=background, 1=myelin, 2=axon).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The PyTorch DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_astih_dataset(
        path, patch_shape, name=name, split=split, download=download,
        label_classes=label_classes, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
