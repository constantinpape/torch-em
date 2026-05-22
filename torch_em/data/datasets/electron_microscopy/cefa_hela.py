"""The Cefa-HeLa dataset contains SBF-SEM images of HeLa cells with semantic segmentation
annotations for nuclear envelope, nucleus, cell, and other cells.

The dataset is from the publications:
- https://doi.org/10.3390/jimaging5090075
- https://doi.org/10.1371/journal.pone.0230605

Please cite these if you use this dataset in your research.

The ground truth labels are available at https://zenodo.org/records/4590903.
The raw electron microscopy images are available via EMPIAR-10094
(https://www.ebi.ac.uk/empiar/EMPIAR-10094/).

Downloading from EMPIAR requires the aspera CLI. Install it via:
    conda install -c hcc aspera-cli
"""

import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


ZENODO_URL = "https://zenodo.org/api/records/4590903/files-archive"
ZENODO_CHECKSUM = None

LABEL_CLASSES = {
    "nuclear_envelope": 1,
    "nucleus": 2,
    "other_cells": 3,
    "background": 4,
    "cell": 5,
}


def _preprocess_data(empiar_path, gt_path, output_path):
    import h5py

    try:
        import ncempy.io.dm as dm
    except ImportError:
        raise ImportError(
            "ncempy is required to read DM4 files from EMPIAR. "
            "Install it via 'pip install ncempy'."
        )

    import scipy.io

    gt_files = sorted(glob(os.path.join(gt_path, "GT_Slice_*.mat")))
    dm4_files = sorted(glob(os.path.join(empiar_path, "*.dm4")))

    if not gt_files:
        raise RuntimeError(f"No GT_Slice_*.mat files found at {gt_path}.")
    if not dm4_files:
        raise RuntimeError(f"No DM4 files found at {empiar_path}.")

    n_files = min(len(gt_files), len(dm4_files))
    os.makedirs(output_path, exist_ok=True)

    for i in tqdm(range(n_files), desc="Preprocessing Cefa-HeLa"):
        h5_path = os.path.join(output_path, f"slice_{i + 1:03d}.h5")
        if os.path.exists(h5_path):
            continue

        label = scipy.io.loadmat(gt_files[i])["groundTruth"]

        with dm.fileDM(dm4_files[i]) as f:
            raw = np.array(f.getDataset(0)["data"], dtype=np.float32)

        # Center-crop raw to match label size if the DM4 image is larger.
        if raw.shape[:2] != label.shape[:2]:
            h, w = label.shape[:2]
            cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
            raw = raw[cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w]

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=label, compression="gzip")


def get_cefa_hela_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and prepare the Cefa-HeLa dataset.

    Ground truth labels are downloaded from Zenodo and raw SBF-SEM images from
    EMPIAR-10094. Both are paired by slice index and stored as H5 files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
            Downloading from EMPIAR requires the aspera CLI.

    Returns:
        The filepath for the preprocessed H5 data.
    """
    output_path = os.path.join(path, "data")
    if os.path.exists(output_path) and len(glob(os.path.join(output_path, "*.h5"))) > 0:
        return output_path

    # Download GT labels from Zenodo.
    gt_dir = os.path.join(path, "gt_labels")
    if not os.path.exists(gt_dir) or len(glob(os.path.join(gt_dir, "GT_Slice_*.mat"))) == 0:
        zip_path = os.path.join(path, "cefa_hela_gt.zip")
        util.download_source(zip_path, ZENODO_URL, download, ZENODO_CHECKSUM)
        util.unzip(zip_path, gt_dir, remove=True)

    # Download raw images from EMPIAR-10094.
    empiar_root = util.download_source_empiar(path, "10094", download)
    empiar_path = os.path.join(empiar_root, "data", "Micrographs_ROI_00")
    if not os.path.exists(empiar_path):
        raise RuntimeError(
            f"Expected EMPIAR-10094 data at {empiar_path}. "
            "Please ensure the download completed successfully."
        )

    _preprocess_data(empiar_path, gt_dir, output_path)
    return output_path


def get_cefa_hela_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the Cefa-HeLa data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the stored data.
    """
    data_root = get_cefa_hela_data(path, download)
    paths = sorted(glob(os.path.join(data_root, "*.h5")))
    return paths


def get_cefa_hela_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    label_classes: Optional[Sequence[str]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Cefa-HeLa dataset for semantic segmentation of HeLa cell structures in SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        label_classes: The label classes to use for one-hot encoding.
            Available classes are 'nuclear_envelope', 'nucleus', 'other_cells',
            'background', 'cell'. If None, returns the full label map
            (1=nuclear_envelope, 2=nucleus, 3=other_cells, 4=background, 5=cell).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_cefa_hela_paths(path, download)

    if label_classes is not None:
        class_ids = []
        for cls_name in label_classes:
            if cls_name not in LABEL_CLASSES:
                raise ValueError(
                    f"Invalid class name: '{cls_name}'. Choose from {list(LABEL_CLASSES.keys())}."
                )
            class_ids.append(LABEL_CLASSES[cls_name])
        label_transform = torch_em.transform.label.OneHotTransform(class_ids=class_ids)
        msg = "'label_classes' is set, but 'label_transform' is in kwargs. It will be overridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_cefa_hela_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    label_classes: Optional[Sequence[str]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for the Cefa-HeLa dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        label_classes: The label classes to use for one-hot encoding.
            Available classes are 'nuclear_envelope', 'nucleus', 'other_cells',
            'background', 'cell'. If None, returns the full label map.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cefa_hela_dataset(path, patch_shape, label_classes, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
