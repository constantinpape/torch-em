"""Dataset for segmentation of structures in Cryo ET.
The DeePict dataset contains annotations for several structures in CryoET.
The dataset implemented here currently only provides access to the actin annotations.

The dataset is part of the publication https://doi.org/10.1038/s41592-022-01746-2.
Plase cite it if you use this dataset in your research.
"""

import os
from glob import glob
from shutil import rmtree
from typing import Tuple, Union, List

from torch.utils.data import Dataset, DataLoader

try:
    import mrcfile
except ImportError:
    mrcfile = None

import torch_em

from .. import util


ACTIN_ID = 10002


def _process_deepict_actin(input_path, output_path):
    from elf.io import open_file

    os.makedirs(output_path, exist_ok=True)

    # datasets = ["00004", "00011", "00012"]
    # There are issues with the 00011 dataset
    datasets = ["00004", "00012"]
    for dataset in datasets:
        ds_folder = os.path.join(input_path, dataset)
        assert os.path.exists(ds_folder)
        ds_out = os.path.join(output_path, f"{dataset}.h5")
        if os.path.exists(ds_out):
            continue

        assert mrcfile is not None, "Plese install mrcfile"

        tomo_folder = glob(os.path.join(ds_folder, "Tomograms", "VoxelSpacing*"))
        assert len(tomo_folder) == 1
        tomo_folder = tomo_folder[0]

        annotation_folder = os.path.join(tomo_folder, "Annotations")
        annotion_files = glob(os.path.join(annotation_folder, "*.zarr"))

        tomo_path = os.path.join(tomo_folder, "CanonicalTomogram", f"{dataset}.mrc")
        with mrcfile.open(tomo_path, "r") as f:
            data = f.data[:]

        annotations = {}
        for annotation in annotion_files:
            with open_file(annotation, "r") as f:
                annotation_data = f["0"][:].astype("uint8")
            assert annotation_data.shape == data.shape
            annotation_name = os.path.basename(annotation).split("-")[1]
            annotations[annotation_name] = annotation_data

        with open_file(ds_out, "a") as f:
            f.create_dataset("raw", data=data, compression="gzip")
            for name, annotation in annotations.items():
                f.create_dataset(f"labels/original/{name}", data=annotation, compression="gzip")

            # Create combined annotations for actin
            actin_seg = annotations["actin_deepict_training_prediction"]
            actin_seg2 = annotations["actin_ground_truth"]
            actin_seg[actin_seg2 == 1] = 1
            f.create_dataset("labels/actin", data=actin_seg, compression="gzip")


def get_deepict_actin_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the DeePict actin dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
    """
    # Check if the processed data is already present.
    dataset_path = os.path.join(path, "deepict_actin")
    if os.path.exists(dataset_path):
        return dataset_path

    # Otherwise download the data.
    dl_path = util.download_from_cryo_et_portal(path, ACTIN_ID, download)

    # And then process it.
    _process_deepict_actin(dl_path, dataset_path)

    # Clean up the original data after processing.
    rmtree(dl_path)

    return dataset_path


def get_deepict_actin_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to DeePict actin data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the stored data.
    """
    get_deepict_actin_data(path, download)
    data_paths = sorted(glob(os.path.join(path, "deepict_actin", "*.h5")))
    return data_paths


def get_deepict_actin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    label_key: str = "labels/actin",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for actin segmentation in Cryo ET data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        label_key: The key for the labels to load. By default this uses 'labels/actin',
            which holds the best version of actin ground-truth images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3

    data_paths = get_deepict_actin_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_deepict_actin_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    label_key: str = "labels/actin",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for actin segmentation in CryoET data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        label_key: The key for the labels to load. By default this uses 'labels/actin',
            which holds the best version of actin ground-truth images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepict_actin_dataset(path, patch_shape, label_key=label_key, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
