"""DeepContact dataset for organelle segmentation in 2D EM.

The dataset contains 2D SEM and TEM images of cultured cells and tissue with
manual polygon annotations for three organelle classes:
- mito: mitochondria
- er: endoplasmic reticulum
- ld: lipid droplets

Data is provided as LabelMe JSON annotations paired with TIFF images.
During preprocessing, polygon annotations are rasterized to binary label masks
and stored alongside the raw images in HDF5 files.

Three image sources are available:
- cell: SEM images of U-2 OS cultured cells at 5 nm/px
- sem: SEM images of Sertoli tissue cells at 10 nm/px
- tem: TEM images at 4.68 nm/px

This dataset is from the publication https://doi.org/10.1083/jcb.202106190.
Please cite it if you use this dataset in your research.

The data is available at https://figshare.com/articles/dataset/DeepContact_Training_Data/19898404.
"""

import os
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch_em
from torch.utils.data import Dataset, DataLoader
from .. import util


DEEPCONTACT_URLS = {
    "cell": ("https://ndownloader.figshare.com/files/35317564", "cell_data.zip"),
    "sem": ("https://ndownloader.figshare.com/files/35317573", "sem_data.zip"),
    "tem": ("https://ndownloader.figshare.com/files/35317576", "tem_data.zip"),
}

DEEPCONTACT_CHECKSUMS = {
    "cell": None,
    "sem": None,
    "tem": None,
}

DEEPCONTACT_LABEL_NAMES = {
    "mito": ["Mito", "mito", "Mitochondria", "mitochondria"],
    "er": ["ER", "er"],
    "ld": ["Lipid Droplets", "lipid droplets", "LipidDroplets", "LD", "ld"],
}


def _rasterize_labelme_json(json_path, label_choice):
    import json
    from skimage.draw import polygon as sk_polygon

    with open(json_path) as f:
        data = json.load(f)

    h = data.get("imageHeight") or data.get("image_height")
    w = data.get("imageWidth") or data.get("image_width")
    mask = np.zeros((h, w), dtype=np.uint8)

    target_names = DEEPCONTACT_LABEL_NAMES[label_choice]
    for shape in data.get("shapes", []):
        if shape.get("label") not in target_names:
            continue
        pts = np.array(shape["points"])
        rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
        mask[rr, cc] = 1

    return mask


def _find_image_for_json(json_path):
    from imageio import imread

    base = os.path.splitext(json_path)[0]
    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        img_path = base + ext
        if os.path.exists(img_path):
            return imread(img_path)

    # Check imagePath field in JSON
    import json
    with open(json_path) as f:
        data = json.load(f)
    img_name = data.get("imagePath", "")
    img_path = os.path.join(os.path.dirname(json_path), img_name)
    if os.path.exists(img_path):
        return imread(img_path)

    return None


def _preprocess_source(extract_dir, output_dir, source):
    from elf.io import open_file

    os.makedirs(output_dir, exist_ok=True)
    json_files = sorted(glob(os.path.join(extract_dir, "**", "*.json"), recursive=True))

    for json_path in tqdm(json_files, desc=f"Processing {source}"):
        name = os.path.splitext(os.path.relpath(json_path, extract_dir))[0].replace(os.sep, "_")
        h5_path = os.path.join(output_dir, f"{name}.h5")
        if os.path.exists(h5_path):
            continue

        raw = _find_image_for_json(json_path)
        if raw is None:
            continue

        if raw.ndim == 3:
            raw = raw[..., 0]

        with open_file(h5_path, "a") as f:
            f.create_dataset("raw", data=raw.astype(np.uint8), compression="gzip")
            for label_choice in DEEPCONTACT_LABEL_NAMES:
                mask = _rasterize_labelme_json(json_path, label_choice)
                f.create_dataset(f"labels/{label_choice}", data=mask, compression="gzip")


def get_deepcontact_data(
    path: Union[os.PathLike, str],
    sources: Optional[List[Literal["cell", "sem", "tem"]]] = None,
    download: bool = False,
) -> str:
    """Download and preprocess the DeepContact dataset.

    Args:
        path: Filepath to a folder where the data will be saved.
        sources: Which image sources to use. Defaults to all ("cell", "sem", "tem").
        download: Whether to download the data if not present.

    Returns:
        Path to the folder containing preprocessed HDF5 files.
    """
    if sources is None:
        sources = ["cell", "sem", "tem"]

    processed_dir = os.path.join(str(path), "processed")
    os.makedirs(str(path), exist_ok=True)

    for source in sources:
        source_dir = os.path.join(processed_dir, source)
        if os.path.isdir(source_dir) and len(glob(os.path.join(source_dir, "*.h5"))) > 0:
            continue

        url, fname = DEEPCONTACT_URLS[source]
        zip_path = os.path.join(str(path), fname)

        if not os.path.exists(zip_path):
            if not download:
                raise RuntimeError(
                    f"Data for source '{source}' not found at '{zip_path}'. "
                    "Set download=True or download manually from "
                    "https://figshare.com/articles/dataset/DeepContact_Training_Data/19898404."
                )
            util.download_source(zip_path, url, download, checksum=DEEPCONTACT_CHECKSUMS[source])

        extract_dir = os.path.join(str(path), f"{source}_raw")
        if not os.path.isdir(extract_dir):
            util.unzip(zip_path, extract_dir, remove=False)

        _preprocess_source(extract_dir, source_dir, source)

    return processed_dir


def get_deepcontact_paths(
    path: Union[os.PathLike, str],
    sources: Optional[List[Literal["cell", "sem", "tem"]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to DeepContact HDF5 files.

    Args:
        path: Filepath to a folder where the data will be saved.
        sources: Which image sources to use. Defaults to all ("cell", "sem", "tem").
        download: Whether to download the data if not present.

    Returns:
        List of paths to HDF5 files.
    """
    if sources is None:
        sources = ["cell", "sem", "tem"]
    processed_dir = get_deepcontact_data(path, sources, download)
    paths = []
    for source in sources:
        paths.extend(sorted(glob(os.path.join(processed_dir, source, "*.h5"))))
    return paths


def get_deepcontact_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    label_choice: Literal["mito", "er", "ld"] = "mito",
    sources: Optional[List[Literal["cell", "sem", "tem"]]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the DeepContact dataset for organelle segmentation in 2D EM.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape (H, W) for training.
        label_choice: Which organelle to segment. One of "mito", "er", or "ld".
        sources: Which image sources to use. Defaults to all ("cell", "sem", "tem").
        download: Whether to download the data if not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 2
    data_paths = get_deepcontact_paths(path, sources, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_deepcontact_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    label_choice: Literal["mito", "er", "ld"] = "mito",
    sources: Optional[List[Literal["cell", "sem", "tem"]]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in the DeepContact dataset.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (H, W) for training.
        label_choice: Which organelle to segment. One of "mito", "er", or "ld".
        sources: Which image sources to use. Defaults to all ("cell", "sem", "tem").
        download: Whether to download the data if not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_deepcontact_dataset(
        path=path,
        patch_shape=patch_shape,
        label_choice=label_choice,
        sources=sources,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
