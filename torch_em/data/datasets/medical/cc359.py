"""The Calgary-Campinas-359 (CC-359) dataset contains annotations for brain extraction (skull-stripping)
in T1-weighted brain MRI.

The dataset comprises 359 volumetric T1-weighted scans of healthy older adults (29-80 years), acquired on
scanners from three vendors (Siemens, Philips, General Electric) at both 1.5T and 3T field strengths, with
approximately 60 subjects per vendor / field strength combination. The scans are described at
https://www.ccdataset.com; this module downloads the data from its mirror on the Canada Open Neuroscience
Platform (CONP, https://github.com/conpdatasets/calgary-campinas), which serves the files directly over
https with no signup or click-through form (unlike the ccdataset.com website, which asks for a filled-in
usage form before handing out the same files).

NOTE ON LABEL QUALITY: The brain masks are "silver-standard" labels, not manual ground truth. They were
created by STAPLE consensus fusion of several automated skull-stripping methods (Souza et al., NeuroImage
2018, https://doi.org/10.1016/j.neuroimage.2017.08.021). The raw STAPLE output is a per-voxel consensus
probability in [0, 1] rather than a hard label; this module binarizes it at a probability of 0.5 to obtain
the 0/1 mask used for training. Do not treat these labels as expert manual annotations; they are best used
for pretraining, benchmarking segmentation agreement, or as weak supervision.

NOTE: The dataset also ships silver-standard hippocampus masks (also STAPLE-derived), but those are stored
on a cropped bounding box with a different image origin/shape than the corresponding raw scan (i.e. they
are not defined on the same voxel grid), so they cannot be paired directly as same-shape raw/label volumes
and are not exposed by this module.

The data is distributed under the CC BY-ND 4.0 license.

This dataset is from the publication https://doi.org/10.1016/j.neuroimage.2017.08.021.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://sftp.conp.ca/users/calgary-campinas/CC359/Reconstructed"

URLS = {"raw": f"{BASE_URL}/Original.zip", "brain": f"{BASE_URL}/Silver-standard-STAPLE.zip"}

CHECKSUMS = {
    "raw": "c3108b56e4d2c950537b04b7b3f597c5511111dbfe02cc629610ae6538c91211",
    "brain": "8779fb6cd4b403d2cafe9bb8bcf72d14ed59f6b73dc8c1d39be9675fffc5a03e",
}

LABEL_IDS = {"background": 0, "brain": 1}

STAPLE_THRESHOLD = 0.5


def _preprocess_inputs(path, raw_dir, label_dir):
    import h5py
    import nibabel as nib

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    raw_paths = natsorted(glob(os.path.join(raw_dir, "*.nii.gz")))
    for raw_path in tqdm(raw_paths, desc="Preprocessing the CC-359 volumes"):
        subject_id = os.path.basename(raw_path)[:-len(".nii.gz")]
        volume_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(volume_path):
            continue

        label_path = os.path.join(label_dir, f"{subject_id}_staple.nii.gz")
        if not os.path.exists(label_path):
            continue

        raw = np.asarray(nib.load(raw_path).dataobj)
        label = np.asarray(nib.load(label_path).dataobj)
        assert raw.shape == label.shape, f"Shape mismatch for {subject_id}: {raw.shape} vs. {label.shape}."
        label = (label > STAPLE_THRESHOLD).astype("uint8")

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=label, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)

    # Marks that preprocessing has finished for all subjects, so that a partially preprocessed
    # directory (e.g. from an interrupted previous run) is not mistaken for a complete one.
    open(os.path.join(preprocessed_dir, ".preprocessing_done"), "w").close()

    return preprocessed_dir


def get_cc359_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CC-359 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(os.path.join(preprocessed_dir, ".preprocessing_done")):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    raw_dir = os.path.join(path, "Original")
    if not os.path.exists(raw_dir):
        zip_path = os.path.join(path, "raw.zip")
        util.download_source(path=zip_path, url=URLS["raw"], download=download, checksum=CHECKSUMS["raw"])
        util.unzip(zip_path=zip_path, dst=path)

    label_dir = os.path.join(path, "STAPLE")
    if not os.path.exists(label_dir):
        zip_path = os.path.join(path, "brain.zip")
        util.download_source(path=zip_path, url=URLS["brain"], download=download, checksum=CHECKSUMS["brain"])
        util.unzip(zip_path=zip_path, dst=path)

    return _preprocess_inputs(path, raw_dir, label_dir)


def get_cc359_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the CC-359 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    preprocessed_dir = get_cc359_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed CC-359 volumes at '{path}'."
    return volume_paths, volume_paths


def get_cc359_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CC-359 dataset for brain extraction.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cc359_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_cc359_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CC-359 dataloader for brain extraction.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cc359_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
