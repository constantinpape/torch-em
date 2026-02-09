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

import json
import numpy as np
import pandas as pd
import imageio.v3 as imageio
from sklearn.model_selection import train_test_split

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

NUCLEI_CLASS_DICT = {
    "nuclei_stroma": 1,
    "nuclei_tumor": 2,
    "nuclei_plasma_cell": 3,
    "nuclei_histiocyte": 4,
    "nuclei_lymphocyte": 5,
    "nuclei_melanophage": 6,
    "nuclei_neutrophil": 7,
    "nuclei_endothelium": 8,
    "nuclei_epithelium": 9,
    "nuclei_apoptosis": 10,
}

TISSUE_CLASS_DICT = {
    "tissue_stroma": 1,
    "tissue_tumor": 2,
    "tissue_epidermis": 3,
    "tissue_blood_vessel": 4,
    "tissue_necrosis": 5,
    "tissue_white_background": 6,
}

CLASS_DICT = {
    "nuclei": NUCLEI_CLASS_DICT,
    "tissue": TISSUE_CLASS_DICT,
}


def _create_split_csv(path, split):
    "This creates a split saved to a .csv file in the dataset directory"
    csv_path = os.path.join(path, "puma_split.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[split] = df[split].apply(lambda x: json.loads(x.replace("'", '"')))  # ensures all items from column in list.
        split_list = df.iloc[0][split]
    else:
        print(f"Creating a new split file at '{csv_path}'.")
        metastatic_ids = [
            os.path.basename(image).split(".")[0] for image in glob(os.path.join(path, "data", "*metastatic*"))
        ]
        primary_ids = [
            os.path.basename(image).split(".")[0] for image in glob(os.path.join(path, "data", "*primary*"))
        ]

        # Create random splits per dataset.
        train_ids, test_ids = train_test_split(metastatic_ids, test_size=0.2)  # 20% for test.
        train_ids, val_ids = train_test_split(train_ids, test_size=0.15)  # 15% of the train set for val.
        ptrain_ids, ptest_ids = train_test_split(primary_ids, test_size=0.2)  # do same as above for 'primary' samples.
        ptrain_ids, pval_ids = train_test_split(ptrain_ids, test_size=0.15)  # do same as above for 'primary' samples.
        train_ids.extend(ptrain_ids)
        val_ids.extend(pval_ids)
        test_ids.extend(ptest_ids)

        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        df = pd.DataFrame.from_dict([split_ids])
        df.to_csv(csv_path, index=False)

        split_list = split_ids[split]

    return split_list


def _preprocess_inputs(path, annotations, split):
    import h5py
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        raise RuntimeError("Please install 'geopandas': 'conda install -c conda-forge geopandas'.")

    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
    except ModuleNotFoundError:
        raise RuntimeError("Please install 'rasterio': 'conda install -c conda-forge rasterio'.")

    annotation_paths = glob(os.path.join(path, "annotations", annotations, "*.geojson"))
    roi_dir = os.path.join(path, "data")
    preprocessed_dir = os.path.join(path, split, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    split_list = _create_split_csv(path, split)
    print(f"The data split '{split}' has '{len(split_list)}' samples!")

    for ann_path in tqdm(annotation_paths, desc=f"Preprocessing '{annotations}'"):
        fname = os.path.basename(ann_path).replace(f"_{annotations}.geojson", ".tif")
        image_path = os.path.join(roi_dir, fname)

        if os.path.basename(image_path).split(".")[0] not in split_list:
            continue

        volume_path = os.path.join(preprocessed_dir, Path(fname).with_suffix(".h5"))
        gdf = gpd.read_file(ann_path)
        minx, miny, maxx, maxy = gdf.total_bounds

        width, height = 1024, 1024  # roi shape
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Extract class ids mapped to each class name.
        class_dict = CLASS_DICT[annotations]
        class_ids = [class_dict[cls_entry["name"]] for cls_entry in gdf["classification"]]
        semantic_shapes = ((geom, unique_id) for geom, unique_id in zip(gdf.geometry, class_ids))
        semantic_mask = rasterize(
            semantic_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8
        )

        gdf['id'] = range(1, len(gdf) + 1)
        instance_shapes = ((geom, unique_id) for geom, unique_id in zip(gdf.geometry, gdf['id']))
        instance_mask = rasterize(
            instance_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.int32
        )

        # Transform labels to match expected orientation
        instance_mask = np.flip(instance_mask)
        instance_mask = np.fliplr(instance_mask)

        semantic_mask = np.flip(semantic_mask)
        semantic_mask = np.fliplr(semantic_mask)

        image = imageio.imread(image_path)
        image = image[..., :-1].transpose(2, 0, 1)

        with h5py.File(volume_path, "a") as f:
            if "raw" not in f.keys():
                f.create_dataset("raw", data=image, compression="gzip")

            if f"labels/instances/{annotations}" not in f.keys():
                f.create_dataset(f"labels/instances/{annotations}", data=instance_mask, compression="gzip")

            if f"labels/semantic/{annotations}" not in f.keys():
                f.create_dataset(f"labels/semantic/{annotations}", data=semantic_mask, compression="gzip")


def _annotations_are_stored(data_dir, annotations):
    import h5py
    volume_path = glob(os.path.join(data_dir, "preprocessed", "*.h5"))[0]
    f = h5py.File(volume_path, "r")
    return f"labels/instances/{annotations}" in f.keys()


def get_puma_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    download: bool = False,
) -> str:
    """Download the PUMA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded and stored for further preprocessing.
    """
    if annotations not in ["nuclei", "tissue"]:
        raise ValueError(f"'{annotations}' is not a valid annotation for the data.")

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir) and _annotations_are_stored(data_dir, annotations):
        return data_dir

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

    _preprocess_inputs(path, annotations, split)

    return data_dir


def get_puma_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    download: bool = False
) -> List[str]:
    """Get paths to the PUMA dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        annotations: The choice of annotations.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_puma_data(path, split, annotations, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "preprocessed", "*.h5")))
    return volume_paths


def get_puma_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PUMA dataset for nuclei and tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotations: The choice of annotations.
        label_choice: The choice of segmentation type.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_puma_paths(path, split, annotations, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{label_choice}/{annotations}",
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
    split: Literal["train", "val", "test"],
    annotations: Literal['nuclei', 'tissue'] = "nuclei",
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PUMA dataloader for nuclei and tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotations: The choice of annotations.
        label_choice: The choice of segmentation type.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_puma_dataset(
        path, patch_shape, split, annotations, label_choice, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
