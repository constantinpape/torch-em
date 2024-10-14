"""The PanNuke datasets contains annotations for nucleus segmentation
in histopathology images across different tissue types.

This dataset is from the publication https://doi.org/10.48550/arXiv.2003.10778.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from typing import List, Union, Dict, Tuple

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# PanNuke Dataset - https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
URLS = {
    "fold_1": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
    "fold_2": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
    "fold_3": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip"
}

CHECKSUM = {
    "fold_1": "6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
    "fold_2": "5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
    "fold_3": "c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d"
}


def get_pannuke_data(path, download, folds):
    """Download the PanNuke data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
        folds: The data fold(s) of choice to be used.
    """
    os.makedirs(path, exist_ok=True)
    for tmp_fold in folds:
        if os.path.exists(os.path.join(path, f"pannuke_{tmp_fold}.h5")):
            return

        util.download_source(os.path.join(path, f"{tmp_fold}.zip"), URLS[tmp_fold], download, CHECKSUM[tmp_fold])

        print(f"Unzipping the PanNuke dataset in {tmp_fold} directories...")
        util.unzip(os.path.join(path, f"{tmp_fold}.zip"), os.path.join(path, f"{tmp_fold}"), True)

        _convert_to_hdf5(path, tmp_fold)


def _convert_to_hdf5(path, fold):
    """Here, we create the h5 files from the input data into 4 essentials (keys):
        - "images" - the raw input images (transposed into the expected format) (S x 3 x H x W)
        - "labels/masks" - the raw input masks (transposed as above) (S x 6 x H x W)
        - "labels/instances" - the converted all-instance labels (S x H x W)
        - "labels/semantic" - the converted semantic labels (S x H x W)
            - where, the semantic instance representation is as follows:
                (0: Background, 1: Neoplastic cells, 2: Inflammatory,
                 3: Connective/Soft tissue cells, 4: Dead Cells, 5: Epithelial)
    """
    import h5py

    if os.path.exists(os.path.join(path, f"pannuke_{fold}.h5")):
        return

    print(f"Converting {fold} into h5 file format...")
    img_paths = glob(os.path.join(path, "**", "images.npy"), recursive=True)
    gt_paths = glob(os.path.join(path, "**", "masks.npy"), recursive=True)

    for img_path, gt_path in zip(img_paths, gt_paths):
        # original (raw) shape : S x H x W x C -> transposed shape (expected) : C x S x H x W
        img = np.load(img_path)
        labels = np.load(gt_path)

        instances = _channels_to_instances(labels)
        semantic = _channels_to_semantics(labels)

        img = img.transpose(3, 0, 1, 2)
        labels = labels.transpose(3, 0, 1, 2)

        # img.shape -> (3, 2656, 256, 256) --- img_chunks -> (3, 1, 256, 256)
        # (same logic as above for labels)
        img_chunks = (img.shape[0], 1) + img.shape[2:]
        label_chunks = (labels.shape[0], 1) + labels.shape[2:]
        other_label_chunks = (1,) + labels.shape[2:]  # for instance and semantic labels

        with h5py.File(os.path.join(path, f"pannuke_{fold}.h5"), "w") as f:
            f.create_dataset("images", data=img, compression="gzip", chunks=img_chunks)
            f.create_dataset("labels/masks", data=labels, compression="gzip", chunks=label_chunks)
            f.create_dataset("labels/instances", data=instances, compression="gzip", chunks=other_label_chunks)
            f.create_dataset("labels/semantic", data=semantic, compression="gzip", chunks=other_label_chunks)

    dir_to_rm = glob(os.path.join(path, "*[!.h5]"))
    for tmp_dir in dir_to_rm:
        shutil.rmtree(tmp_dir)


def _channels_to_instances(labels):
    """Converting the ground-truth of 6 (instance) channels into 1 label with instances from all channels
    channel info -
    (0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 6: Background)

    Returns:
        - instance labels of dimensions -> (C x H x W)
    """
    import vigra

    labels = labels.transpose(0, 3, 1, 2)  # to access with the shape S x 6 x H x W
    list_of_instances = []

    for label_slice in labels:  # access the slices (each with 6 channels of H x W labels)
        segmentation = np.zeros(labels.shape[2:])
        max_ids = []
        for label_channel in label_slice[:-1]:  # access the channels
            # the 'start_label' takes care of where to start allocating the instance ids from
            this_labels, max_id, _ = vigra.analysis.relabelConsecutive(
                label_channel.astype("uint64"),
                start_label=max_ids[-1] + 1 if len(max_ids) > 0 else 1)

            # some trailing channels might not have labels, hence appending only for elements with RoIs
            if max_id > 0:
                max_ids.append(max_id)

            segmentation[this_labels > 0] = this_labels[this_labels > 0]

        list_of_instances.append(segmentation)

    f_segmentation = np.stack(list_of_instances)

    return f_segmentation


def _channels_to_semantics(labels):
    """Converting the ground-truth of 6 (instance) channels  into semantic labels, ollowing below the id info as:
    (1 -> Neoplastic cells, 2 -> Inflammatory, 3 -> Connective/Soft tissue cells,
    4 -> Dead Cells, 5 -> Epithelial, 0 -> Background)

    Returns:
        - semantic labels of dimensions -> (C x H x W)
    """
    labels = labels.transpose(0, 3, 1, 2)
    list_of_semantic = []

    for label_slice in labels:
        segmentation = np.zeros(labels.shape[2:])
        for i, label_channel in enumerate(label_slice[:-1]):
            segmentation[label_channel > 0] = i + 1
        list_of_semantic.append(segmentation)

    f_segmentation = np.stack(list_of_semantic)

    return f_segmentation


def get_pannuke_paths(
    path: Union[os.PathLike, str], folds: List[str] = ["fold_1", "fold_2", "fold_3"], download: bool = False,
) -> List[str]:
    """Get paths to the PanNuke data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        folds: The data fold(s) of choice to be used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the stored data.
    """
    get_pannuke_data(path, download, folds)

    data_paths = [os.path.join(path, f"pannuke_{fold}.h5") for fold in folds]
    return data_paths


def get_pannuke_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    folds: List[str] = ["fold_1", "fold_2", "fold_3"],
    rois: Dict = {},
    download: bool = False,
    custom_label_choice: str = "instances",
    with_channels: bool = True,
    with_label_channels: bool = False,
    **kwargs
) -> Dataset:
    """Get the PanNuke dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        folds: The data fold(s) of choice to be used.
        download: Whether to download the data if it is not present.
        rois: The choice of rois per fold to create the dataloader for training.
        custom_label_choice: The choice of labels to be used for training.
        with_channels: Whether the inputs have channels.
        with_label_channels: Whether the labels have channels.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset
    """
    assert custom_label_choice in [
        "masks", "instances", "semantic"
    ], "Select the type of labels you want from [masks/instances/semantic] (See `_convert_to_hdf5` for details)"

    if rois is not None:
        assert isinstance(rois, dict)

    data_paths = get_pannuke_paths(path, folds, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="images",
        label_paths=data_paths,
        label_key=f"labels/{custom_label_choice}",
        patch_shape=patch_shape,
        rois=[rois.get(fold, np.s_[:, :, :]) for fold in folds],
        with_channels=with_channels,
        with_label_channels=with_label_channels,
        **kwargs
    )


def get_pannuke_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: str,
    folds: List[str] = ["fold_1", "fold_2", "fold_3"],
    download: bool = False,
    rois: Dict = {},
    custom_label_choice: str = "instances",
    **kwargs
) -> DataLoader:
    """Get the PanNuke dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        folds: The data fold(s) of choice to be used.
        download: Whether to download the data if it is not present.
        rois: The choice of rois per fold to create the dataloader for training.
        custom_label_choice: The choice of labels to be used for training.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader
    """
    dataset_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_pannuke_dataset(
        path=path,
        patch_shape=patch_shape,
        folds=folds,
        rois=rois,
        download=download,
        custom_label_choice=custom_label_choice,
        **dataset_kwargs
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
