"""FIB-25 is a dataset for neuron segmentation in EM.

It contains FIB-SEM data and segmentation ground truth from the Drosophila medulla,
as part of the FlyEM project at Janelia Research Campus.

The dataset is from the publication https://doi.org/10.1073/pnas.1509820112.
Please cite this publication if you use the dataset in your research.

The data is hosted at https://github.com/google/ffn via Google Cloud Storage.
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


GCS_BUCKET = "https://storage.googleapis.com/ffn-flyem-fib25"

URLS = {
    "training_sample2": {
        "raw": f"{GCS_BUCKET}/training_sample2/grayscale_maps.h5",
        "labels": f"{GCS_BUCKET}/training_sample2/groundtruth.h5",
    },
    "validation_sample": {
        "raw": f"{GCS_BUCKET}/validation_sample/grayscale_maps.h5",
        "labels": f"{GCS_BUCKET}/validation_sample/groundtruth.h5",
    },
    "tstvol-520-1": {
        "raw": f"{GCS_BUCKET}/tstvol-520-1/raw.h5",
        "labels": f"{GCS_BUCKET}/tstvol-520-1/groundtruth.h5",
    },
}

CHECKSUMS = {
    "training_sample2": {
        "raw": "ea031c98ee2de778a9a3a1e6d410df5de73e4ac28022df8e7255d84e3394cafa",
        "labels": "fd508e7aee1fe51ac9ae0460db4a841d275236f013c1f2552314b4f21b1010ea",
    },
    "validation_sample": {
        "raw": "400ccb2a7268a3880c63656e0d794f8e6252e62031869455cc8caeef245b2a83",
        "labels": "2c5e31af0af5476bc9669b88d01a4570a26eb020799eaf6131aa75f2f7d92e98",
    },
    "tstvol-520-1": {
        "raw": "0667e701c8b4464003d8a6cb0cf9deb2aa79fb415ec51deeac92e5f9c67a5a66",
        "labels": "ae61ae78a9874eb35ae8e5ed29b4cbfe7bbd07a61789ddb70aef4deb2532eb4e",
    },
}

SAMPLES = list(URLS.keys())


def _apply_transforms(groundtruth_path):
    """Apply the supervoxel-to-neuron mapping from the 'transforms' dataset.

    The groundtruth h5 files contain a 'stack' dataset with supervoxel IDs
    and a 'transforms' dataset that maps supervoxels to neuron body IDs.
    This function applies the mapping and saves the result as 'neuron_ids'.
    """
    import h5py

    with h5py.File(groundtruth_path, "a") as f:
        if "neuron_ids" in f:
            return

        stack = f["stack"][:]
        transforms = f["transforms"][:]

        # Build the mapping from supervoxel IDs to neuron body IDs.
        mapping = np.zeros(stack.max() + 1, dtype=stack.dtype)
        for src, dst in transforms:
            mapping[src] = dst
        neuron_ids = mapping[stack]

        f.create_dataset("neuron_ids", data=neuron_ids, compression="gzip")


def get_fib25_data(
    path: Union[os.PathLike, str], samples: Tuple[str, ...], download: bool = False
):
    """Download the FIB-25 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The samples to download. Available samples are
            'training_sample2', 'validation_sample', and 'tstvol-520-1'.
        download: Whether to download the data if it is not present.
    """
    os.makedirs(path, exist_ok=True)
    for sample in samples:
        assert sample in URLS, f"Invalid sample: {sample}. Choose from {SAMPLES}."
        urls = URLS[sample]
        checksums = CHECKSUMS[sample]

        sample_dir = os.path.join(path, sample)
        os.makedirs(sample_dir, exist_ok=True)

        raw_path = os.path.join(sample_dir, "raw.h5")
        labels_path = os.path.join(sample_dir, "groundtruth.h5")

        util.download_source(raw_path, urls["raw"], download, checksum=checksums["raw"])
        util.download_source(labels_path, urls["labels"], download, checksum=checksums["labels"])

        # Apply the supervoxel-to-neuron mapping.
        _apply_transforms(labels_path)


def get_fib25_paths(
    path: Union[os.PathLike, str],
    samples: Tuple[str, ...] = ("training_sample2",),
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FIB-25 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The samples to use. Available samples are
            'training_sample2', 'validation_sample', and 'tstvol-520-1'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the raw data and the label data.
    """
    get_fib25_data(path, samples, download)
    raw_paths = [os.path.join(path, sample, "raw.h5") for sample in samples]
    label_paths = [os.path.join(path, sample, "groundtruth.h5") for sample in samples]
    return raw_paths, label_paths


def get_fib25_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    samples: Tuple[str, ...] = ("training_sample2",),
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FIB-25 dataset for the segmentation of neurons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        samples: The samples to use. Available samples are
            'training_sample2', 'validation_sample', and 'tstvol-520-1'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    raw_paths, label_paths = get_fib25_paths(path, samples, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="neuron_ids",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_fib25_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    samples: Tuple[str, ...] = ("training_sample2",),
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for EM neuron segmentation in the FIB-25 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        samples: The samples to use. Available samples are
            'training_sample2', 'validation_sample', and 'tstvol-520-1'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_fib25_dataset(
        path=path,
        patch_shape=patch_shape,
        samples=samples,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
