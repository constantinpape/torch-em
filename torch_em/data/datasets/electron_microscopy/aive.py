"""The AIVE dataset contains FIB-SEM volumes of cells and tissue with dense organelle annotations.

It was released with the AI-directed voxel extraction (AIVE) approach for volume EM. The overview volumes have a
voxel size of 6.5 x 6.5 x 10 nm and contain semantic annotations of several organelles:
- Dataset 1: 724 x 720 x 2960 voxels (z, y, x), with the classes of `CLASS_IDS` except 'chromatin'.
- Dataset 2: 858 x 806 x 2930 voxels, with the classes of `CLASS_IDS` except 'golgi'.
- Dataset 3: 618 x 830 x 2948 voxels, with all classes of `CLASS_IDS`.
- Dataset 4 (muscle tissue): 311 x 500 x 500 voxels, with mitochondria annotations only.

The per-dataset class ids of the original annotations differ. They are harmonized to the ids in `CLASS_IDS`
(0 is background) when the data is converted to hdf5 files after downloading.
Note that class 11 of dataset 3 is not described in the EMPIAR entry, it is assumed to be 'chromatin' (as in
dataset 2). Also, 'plasma_membrane' covers a large area, i.e. it is not a thin membrane annotation only.

The data is available at https://www.ebi.ac.uk/empiar/EMPIAR-12885/ (doi: 10.6019/EMPIAR-12885).
The licence of the EMPIAR entry could not be confirmed (EMPIAR entries are CC0 by default).
The corresponding publication is https://doi.org/10.1083/jcb.202411138. Please cite it if you use this dataset
in your research.
"""

import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/12885/data"
URLS = {
    1: {
        "raw": f"{BASE_URL}/Set1_-_AIVE_SOURCE_DATA/Dataset1_overview.tif",
        "labels": f"{BASE_URL}/Set1_-_AIVE_SOURCE_DATA/Dataset1_overview%20-%20ORGANELLE%20CLASS%20LABELS.tif",
    },
    2: {
        "raw": f"{BASE_URL}/Set2_-_AIVE_SOURCE_DATA/Dataset2_overview.tif",
        "labels": f"{BASE_URL}/Set2_-_AIVE_SOURCE_DATA/Dataset2_overview%20-%20ORGANELLE%20CLASS%20LABELS.tif",
    },
    3: {
        "raw": f"{BASE_URL}/Set3_-_AIVE_SOURCE_DATA/Dataset3_overview.tif",
        "labels": f"{BASE_URL}/Set3_-_AIVE_SOURCE_DATA/Dataset3_overview%20-%20ORGANELLE%20CLASS%20LABELS.tif",
    },
    4: {
        "raw": f"{BASE_URL}/Set4_-_AIVE_SOURCE_DATA/Dataset4_overview.tif",
        "labels": f"{BASE_URL}/Set4_-_AIVE_SOURCE_DATA/Dataset4_MitoClasss.tif",
    },
}
CHECKSUMS = {
    1: {
        "raw": "925d6e36fdc268408b41f85563f78627c26a6c1dffabb8eec1d2c05acffe8b41",
        "labels": "5d343e525757143ac92db47f1be2c70b5c51e5bba3120ca1b83c732dba909eba",
    },
    2: {
        "raw": "71d39cc101f5d1754ab6e8950561e1cb322750c7af89e0c540be7b74bd2ac3e5",
        "labels": "b805978d995b7fad3a172a225f4553ceb5957fbd1493390d9162d9d5a32caf62",
    },
    3: {
        "raw": "682c54ef1f945f565a50577d94c334cde36ff71e520ebb22d5360a2d2b76e58a",
        "labels": "1546d16b4057ee6c409e52bf1a49beface2a2af8780f7a2dda8997c68b0cfa4a",
    },
    4: {
        "raw": "565a74783f8f9769c0054359728b37eea333fc6c05fe148ebeed4818c2454355",
        "labels": "be7dc8749c33a32f0d9f5a64853268e07d37418f458edc0b021aedeff32fb962",
    },
}

# The harmonized class ids, which are used for the labels after the conversion.
CLASS_IDS = {
    "mitochondria": 1,
    "lipid_droplet": 2,
    "early_endosome": 3,
    "golgi": 4,
    "late_endosome": 5,
    "nucleus": 6,
    "cytoskeleton": 7,
    "er": 8,
    "plasma_membrane": 9,
    "vesicle": 10,
    "chromatin": 11,
}
# The class ids in the original annotations of the datasets.
ORIGINAL_CLASS_IDS = {
    1: {
        "mitochondria": 1, "lipid_droplet": 2, "early_endosome": 3, "golgi": 4, "late_endosome": 5, "nucleus": 6,
        "cytoskeleton": 7, "er": 8, "plasma_membrane": 9, "vesicle": 10,
    },
    2: {
        "nucleus": 1, "mitochondria": 2, "lipid_droplet": 3, "plasma_membrane": 4, "chromatin": 5, "er": 6,
        "cytoskeleton": 7, "late_endosome": 8, "early_endosome": 9, "vesicle": 10,
    },
    3: {
        "mitochondria": 1, "lipid_droplet": 2, "late_endosome": 3, "early_endosome": 4, "nucleus": 5,
        "cytoskeleton": 6, "golgi": 7, "er": 8, "plasma_membrane": 9, "vesicle": 10, "chromatin": 11,
    },
    4: {"mitochondria": 255},
}
DATASET_IDS = [1, 2, 3, 4]
Organelle = Literal[
    "mitochondria", "lipid_droplet", "early_endosome", "golgi", "late_endosome", "nucleus", "cytoskeleton", "er",
    "plasma_membrane", "vesicle", "chromatin"
]


class _OrganelleTransform:
    """Convert the harmonized labels to a binary mask for one organelle (picklable for multiprocessing)."""
    def __init__(self, class_id: int):
        self.class_id = class_id

    def __call__(self, labels):
        return (labels == self.class_id).astype("uint8")


def _convert_to_hdf5(raw_path, label_path, out_path, dataset_id):
    import h5py
    import tifffile

    # Lookup table from the original to the harmonized class ids.
    lut = np.zeros(256, dtype="uint8")
    for name, original_id in ORIGINAL_CLASS_IDS[dataset_id].items():
        lut[original_id] = CLASS_IDS[name]

    chunks = (32, 256, 256)
    with tifffile.TiffFile(raw_path) as f_raw, tifffile.TiffFile(label_path) as f_label:
        shape = f_raw.series[0].shape
        assert shape == f_label.series[0].shape, f"Raw and label shapes differ: {shape}, {f_label.series[0].shape}"
        n_slices = shape[0]
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as f:
            kw = dict(shape=shape, dtype="uint8", chunks=chunks, compression="gzip", compression_opts=1)
            ds_raw = f.create_dataset("raw", **kw)
            ds_label = f.create_dataset("labels", **kw)
            for start in range(0, n_slices, chunks[0]):
                stop = min(start + chunks[0], n_slices)
                ds_raw[start:stop] = np.stack([f_raw.pages[i].asarray() for i in range(start, stop)])
                ds_label[start:stop] = lut[np.stack([f_label.pages[i].asarray() for i in range(start, stop)])]
    os.rename(tmp_path, out_path)


def get_aive_data(
    path: Union[os.PathLike, str], dataset_ids: Optional[List[int]] = None, download: bool = False
) -> List[str]:
    """Download the AIVE dataset.

    The volumes are converted to hdf5 files with the keys 'raw' and 'labels' after downloading.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_ids: The ids of the datasets to download. Choose from 1, 2, 3 and 4. By default all are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the hdf5 files with the data.
    """
    dataset_ids = DATASET_IDS if dataset_ids is None else dataset_ids
    for dataset_id in dataset_ids:
        if dataset_id not in DATASET_IDS:
            raise ValueError(f"'{dataset_id}' is not a valid dataset id. Choose from {DATASET_IDS}.")

    os.makedirs(path, exist_ok=True)
    h5_paths = []
    for dataset_id in dataset_ids:
        h5_path = os.path.join(path, f"aive_dataset{dataset_id}.h5")
        h5_paths.append(h5_path)
        if os.path.exists(h5_path):
            continue

        tif_paths = {}
        for key in ("raw", "labels"):
            tif_paths[key] = os.path.join(path, f"dataset{dataset_id}_{key}.tif")
            util.download_source(
                tif_paths[key], URLS[dataset_id][key], download, checksum=CHECKSUMS[dataset_id][key]
            )
        _convert_to_hdf5(tif_paths["raw"], tif_paths["labels"], h5_path, dataset_id)
        for tif_path in tif_paths.values():
            os.remove(tif_path)

    return h5_paths


def get_aive_paths(
    path: Union[os.PathLike, str],
    dataset_ids: Optional[List[int]] = None,
    organelle: Optional[Organelle] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the AIVE data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_ids: The ids of the datasets to use. Choose from 1, 2, 3 and 4. By default all datasets
            that contain annotations for the organelle (or all datasets if no organelle is given) are used.
        organelle: The organelle that needs to be annotated in the datasets. Used to select the default datasets.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the hdf5 files with the data.
    """
    if organelle is not None and organelle not in CLASS_IDS:
        raise ValueError(f"'{organelle}' is not a valid organelle. Choose from {list(CLASS_IDS)}.")

    if dataset_ids is None:
        dataset_ids = DATASET_IDS if organelle is None else [
            i for i in DATASET_IDS if organelle in ORIGINAL_CLASS_IDS[i]
        ]
    elif organelle is not None:
        missing = [i for i in dataset_ids if organelle not in ORIGINAL_CLASS_IDS[i]]
        if missing:
            raise ValueError(f"The datasets {missing} do not contain annotations for '{organelle}'.")

    return get_aive_data(path, dataset_ids, download)


def get_aive_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    dataset_ids: Optional[List[int]] = None,
    organelle: Optional[Organelle] = "mitochondria",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AIVE dataset for organelle segmentation in FIB-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        dataset_ids: The ids of the datasets to use. Choose from 1, 2, 3 and 4. By default all datasets
            that contain annotations for the organelle are used.
        organelle: The organelle to segment, which gives a binary segmentation task. One of the names in `CLASS_IDS`.
            If None, the multi-class labels with the harmonized class ids of `CLASS_IDS` are returned.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_aive_paths(path, dataset_ids, organelle, download)

    if organelle is not None:
        kwargs = util.update_kwargs(kwargs, "label_transform", _OrganelleTransform(CLASS_IDS[organelle]))
    kwargs = util.update_kwargs(kwargs, "ndim", 3)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_aive_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    dataset_ids: Optional[List[int]] = None,
    organelle: Optional[Organelle] = "mitochondria",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in the AIVE dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        dataset_ids: The ids of the datasets to use. Choose from 1, 2, 3 and 4. By default all datasets
            that contain annotations for the organelle are used.
        organelle: The organelle to segment, which gives a binary segmentation task. One of the names in `CLASS_IDS`.
            If None, the multi-class labels with the harmonized class ids of `CLASS_IDS` are returned.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_aive_dataset(path, patch_shape, dataset_ids, organelle, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
