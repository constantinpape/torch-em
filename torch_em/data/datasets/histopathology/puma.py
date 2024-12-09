"""The PUMA dataset contains annotations for nucleus and tissue segmentation
in melanoma H&E stained histopathology images.

This dataset is located at https://zenodo.org/records/13859989.
This is part of the PUMA Grand Challenge: https://puma.grand-challenge.org/
- Preprint with details about the data: https://doi.org/10.1101/2024.10.07.24315039

Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, List, Tuple

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "data": "https://zenodo.org/records/13859989/files/01_training_dataset_tif_ROIs.zip",
    "annotations": {
        "nuclei": "https://zenodo.org/records/13859989/files/01_training_dataset_geojson_nuclei.zip",
        "tissue": "https://zenodo.org/records/13859989/files/01_training_dataset_geojson_tissue.zip",
    }
}

CHECKSUM = {
    "data": "a69fd0d8443da29233df103ece5674fb50e8f0cc4b448dc60508cfe883881993",
    "annotations": {
        "nuclei": "17f77ca83fb8fccd918ce723a7b3e5cb5a1730b342ad486628f8885d14a1acbd",
        "tissue": "3b7d6697dd728e3481df0b779ad1e76962f36fc8c871c50edd9aa56ec44c4cc9",
    }
}


def _preprocess_inputs(path, annotations):
    import h5py
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    annotation_paths = glob(os.path.join(path, "annotations", annotations, "*.geojson"))
    roi_dir = os.path.join(path, "data")
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for ann_path in tqdm(annotation_paths, desc=f"Preprocessing '{annotations}'"):
        fname = os.path.basename(ann_path).replace(f"_{annotations}.geojson", ".tif")
        volume_path = os.path.join(preprocessed_dir, Path(fname).with_suffix(".h5"))

        image_path = os.path.join(roi_dir, fname)

        gdf = gpd.read_file(ann_path)
        minx, miny, maxx, maxy = gdf.total_bounds

        width, height = 1024, 1024  # roi shape
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        gdf['id'] = range(1, len(gdf) + 1)
        shapes = ((geom, unique_id) for geom, unique_id in zip(gdf.geometry, gdf['id']))
        mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.int32)

        # Transform labels to match expected orientation
        mask = np.flip(mask)
        mask = np.fliplr(mask)

        image = imageio.imread(image_path)
        image = image[..., :-1].transpose(2, 0, 1)

        with h5py.File(volume_path, "a") as f:
            if "raw" not in f.keys():
                f.create_dataset("raw", data=image, compression="gzip")

            if f"labels/{annotations}" not in f.keys():
                f.create_dataset(f"labels/{annotations}", data=mask, compression="gzip")


def get_puma_data(
    path: Union[os.PathLike, str],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    download: bool = False
):
    """Download the PUMA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.
    """
    if annotations not in ["nuclei", "tissue"]:
        raise ValueError(f"'{annotations}' is not a valid annotation for the data.")

    data_dir = os.path.join(path, "annotations", annotations)
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    if not os.path.exists(os.path.join(path, "data")):
        # Download the data.
        zip_path = os.path.join(path, "roi.zip")
        util.download_source(path=zip_path, url=URL["data"], download=download, checksum=CHECKSUM["data"])
        util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))

    # Download the annotations.
    zip_path = os.path.join(path, "annotations.zip")
    util.download_source(
        path=zip_path,
        url=URL["annotations"][annotations],
        download=download,
        checksum=CHECKSUM["annotations"][annotations]
    )
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "annotations", annotations))

    _preprocess_inputs(path, annotations)


def get_puma_paths(
    path: Union[os.PathLike, str],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    download: bool = False
) -> List[str]:
    """Get paths to the PUMA dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    get_puma_data(path, annotations, download)
    volume_paths = natsorted(glob(os.path.join(path, "preprocessed", "*.h5")))
    return volume_paths


def get_puma_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PUMA dataset for nuclei and tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_puma_paths(path, annotations, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{annotations}",
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_puma_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PUMA dataloader for nuclei and tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_puma_dataset(path, patch_shape, annotations, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
