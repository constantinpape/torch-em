"""MyoFuse contains annotations for myonuclei in fluorescence microscopy images of muscle cell cultures.

The images show the myotube (myosin) channel of mouse C2C12 and human primary myotube cultures.
Every nucleus has a class. The class states if the nucleus is inside or outside a myotube.
MyoFuse uses these classes to measure the fusion index.
NOTE: A custom cellpose model created the nuclei instance masks. Nobody curated them by hand.
NOTE: This dataset does not contain the nuclei (DAPI) channel. Only the myotube channel has annotations.

The dataset is located at https://doi.org/10.5281/zenodo.14731491.
This dataset is from the publication https://github.com/BenLair/MyoFuse.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import tifffile

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/14731491/files/Training%20Images%20MyoFuse%20v1.0.0.zip"
CHECKSUM = "247725509cf5dd785ba919438d2eba7f544dc588880396f245b0700664bdd0b8"

SUBSET_PREFIX = {"human": "H", "mouse": "M"}


def _get_semantic_labels(instances, regions, classes):
    """Map the class of every nucleus onto the instance mask.

    MyoFuse stores the regions in a shuffled order. This function matches each region
    to its instance id with the coordinates of the region.
    """
    semantic = np.zeros(instances.shape, dtype="uint8")
    for region, class_id in zip(regions, classes):
        coords = region["coords"]
        instance_ids = np.unique(instances[coords[:, 0], coords[:, 1]])
        assert len(instance_ids) == 1, f"The region maps to more than one instance: {instance_ids}."
        semantic[instances == instance_ids[0]] = class_id + 1

    return semantic


def _preprocess_data(input_dir, data_dir):
    import h5py
    import torch

    os.makedirs(data_dir, exist_ok=True)

    annotations = torch.load(os.path.join(input_dir, "labels"), weights_only=False)
    for i, image_path in enumerate(annotations["image_path"]):
        fname = os.path.basename(image_path)
        image = tifffile.imread(os.path.join(input_dir, "Images", fname))
        instances = tifffile.imread(os.path.join(input_dir, "Masks", fname)).astype("uint16")
        semantic = _get_semantic_labels(instances, annotations["regionprops"][i], annotations["labels_list"][i])

        with h5py.File(os.path.join(data_dir, os.path.splitext(fname)[0] + ".h5"), "a") as f:
            f.create_dataset("raw", data=image, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")


def get_myofuse_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MyoFuse dataset.

    Args:
        path: The folder where the function stores the data.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the prepared data.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Training_Images_MyoFuse.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    _preprocess_data(os.path.join(path, "Training Images MyoFuse v1.0.0"), data_dir)

    return data_dir


def get_myofuse_paths(
    path: Union[os.PathLike, str],
    subset: Optional[Literal["human", "mouse"]] = None,
    download: bool = False,
) -> List[str]:
    """Get the paths to the MyoFuse data.

    Args:
        path: The folder where the function stores the data.
        subset: The cell type. Use 'human' for human primary myotubes or 'mouse' for mouse C2C12 myotubes.
            The function uses both cell types by default.
        download: Whether to download the data if it is not present.

    Returns:
        The list of filepaths to the input data.
    """
    data_dir = get_myofuse_data(path, download)

    if subset is None:
        pattern = "*.h5"
    else:
        if subset not in SUBSET_PREFIX:
            raise ValueError(f"'{subset}' is not a valid subset. Choose one of {list(SUBSET_PREFIX.keys())}.")
        pattern = f"{SUBSET_PREFIX[subset]}_*.h5"

    volume_paths = natsorted(glob(os.path.join(data_dir, pattern)))
    assert len(volume_paths) > 0, f"Could not find data for the subset '{subset}'."
    return volume_paths


def get_myofuse_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Optional[Literal["human", "mouse"]] = None,
    label_choice: Literal["semantic", "instances"] = "semantic",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MyoFuse dataset for segmentation of myonuclei.

    Args:
        path: The folder where the function stores the data.
        patch_shape: The patch shape to use for training.
        subset: The cell type. Use 'human' for human primary myotubes or 'mouse' for mouse C2C12 myotubes.
            The function uses both cell types by default.
        label_choice: The label type. Use 'semantic' for the class of every nucleus,
            or 'instances' for the masks of the nuclei.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_choice not in ("semantic", "instances"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'semantic' or 'instances'.")

    volume_paths = get_myofuse_paths(path, subset, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_myofuse_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Optional[Literal["human", "mouse"]] = None,
    label_choice: Literal["semantic", "instances"] = "semantic",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MyoFuse dataloader for segmentation of myonuclei.

    Args:
        path: The folder where the function stores the data.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: The cell type. Use 'human' for human primary myotubes or 'mouse' for mouse C2C12 myotubes.
            The function uses both cell types by default.
        label_choice: The label type. Use 'semantic' for the class of every nucleus,
            or 'instances' for the masks of the nuclei.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_myofuse_dataset(path, patch_shape, subset, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
