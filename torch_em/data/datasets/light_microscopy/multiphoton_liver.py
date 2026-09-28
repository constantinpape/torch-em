"""The Multiphoton Liver dataset contains annotations for 3D segmentation of cell borders, bile canaliculi (BC),
sinusoids and nuclei in multiphoton microscopy volumes of liver tissue.

The dataset is a resource for benchmarking 3D microscopy image-analysis methods, published on the BioImage Archive
as accession S-BIAD2705. It contains four real microscopy samples ('G1', 'G2', 'Ch2', 'Ch3'), each a five-channel
volume (raw fluorescence channels, meaning undocumented) paired with a four-channel segmentation mask. Per the
dataset's own 'Naming_Convention.md', the mask channels are, in order: BC, Sinusoids, Nuclei, Sinusoid Fill.
Nuclei are already instance-labeled by the authors (verified: up to several thousand distinct ids per volume);
the other three channels are binary semantic masks (verified: only two distinct values per channel). Use `target`
to select which of the four to expose as the label ('bc', 'sinusoids', 'nuclei' or 'sinusoid_fill').

NOTE: The full BioImage Archive resource is about 500 GB across roughly 23,000 files, including simulated data,
restoration and segmentation baselines. This module only downloads the four raw 'Preprocessed' microscopy volumes
and their four 'Microscopy_segmented_mask' files (about 7-8 GB per sample). Use `samples` to further restrict to
a subset. Only the checksum of the 'G2' sample's two files is known (computed from a verified download); the
'G1', 'Ch2' and 'Ch3' files have no published checksum and are downloaded without one.

The data is located at https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2705, released under a CC-BY-4.0
license (per the BioStudies study page).
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, Sequence, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/705/S-BIAD2705/Files"

SAMPLES = {
    "G1": ("Dataset/Image_sets/Microscopy_images/Preprocessed/20221014_6_Control(G1).tif",
           "Dataset/Mask_sets/Microscopy_segmented_masks/Microscopy_G1_segmented_mask.tif", None, None),
    "G2": ("Dataset/Image_sets/Microscopy_images/Preprocessed/20221025_29_Control (G2).tif",
           "Dataset/Mask_sets/Microscopy_segmented_masks/Microscopy_G2_segmented_mask.tif",
           "531e2f00db607760bc7587c6391db3833ea2de9cbf930597c9ef2b97c942bcd5",
           "60c1a9fd43a9f99550ce994dc825cb3e498c514a667116977b1a9c04d70f6a93"),
    "Ch2": ("Dataset/Image_sets/Microscopy_images/Preprocessed/20240913_control_c1 (Ch2).tif",
            "Dataset/Mask_sets/Microscopy_segmented_masks/Microscopy_Ch2_segmented_mask.tif", None, None),
    "Ch3": ("Dataset/Image_sets/Microscopy_images/Preprocessed/20241029_control (ch3).tif",
            "Dataset/Mask_sets/Microscopy_segmented_masks/Microscopy_Ch3_segmented_mask.tif", None, None),
}
"""Mapping from sample name to (image path, mask path, image sha256, mask sha256) relative to `BASE_URL`."""

MASK_CHANNELS = {"bc": 0, "sinusoids": 1, "nuclei": 2, "sinusoid_fill": 3}


def _preprocess_sample(name, path, preprocessed_dir):
    import h5py
    import tifffile

    h5_path = os.path.join(preprocessed_dir, f"{name}.h5")
    if os.path.exists(h5_path):
        return

    image_rel, mask_rel, _, _ = SAMPLES[name]
    raw = tifffile.imread(os.path.join(path, f"{name}_image.tif"))
    mask = tifffile.imread(os.path.join(path, f"{name}_mask.tif"))
    assert raw.shape[1:] == mask.shape[1:], f"Shape mismatch for {name}: raw={raw.shape}, mask={mask.shape}"

    tmp_path = f"{h5_path}.{os.getpid()}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(1,) + raw.shape[1:])
        for target, channel in MASK_CHANNELS.items():
            data = mask[channel] if target == "nuclei" else (mask[channel] > 0).astype("uint8")
            f.create_dataset(f"labels/{target}", data=data, compression="gzip", chunks=raw.shape[1:])
    os.replace(tmp_path, h5_path)


def get_multiphoton_liver_data(
    path: Union[os.PathLike, str], samples: Optional[Sequence[str]] = None, download: bool = False,
) -> str:
    """Download the Multiphoton Liver dataset and convert it to hdf5 files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The sample names to use, see `SAMPLES`. By default all four samples are used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the preprocessed data.
    """
    samples = list(SAMPLES) if samples is None else samples
    invalid = set(samples) - set(SAMPLES)
    if invalid:
        raise ValueError(f"{invalid} are not valid samples. Choose from {list(SAMPLES)}.")

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for name in samples:
        if os.path.exists(os.path.join(preprocessed_dir, f"{name}.h5")):
            continue

        image_rel, mask_rel, image_checksum, mask_checksum = SAMPLES[name]
        image_path = os.path.join(path, f"{name}_image.tif")
        mask_path = os.path.join(path, f"{name}_mask.tif")
        util.download_source(
            path=image_path, url=f"{BASE_URL}/{image_rel}", download=download, checksum=image_checksum
        )
        util.download_source(
            path=mask_path, url=f"{BASE_URL}/{mask_rel}", download=download, checksum=mask_checksum
        )
        _preprocess_sample(name, path, preprocessed_dir)

    return preprocessed_dir


def get_multiphoton_liver_paths(
    path: Union[os.PathLike, str], samples: Optional[Sequence[str]] = None, download: bool = False,
) -> List[str]:
    """Get paths to the Multiphoton Liver data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The sample names to use, see `SAMPLES`. By default all four samples are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the raw data ('raw') and the label data
        ('labels/bc', 'labels/sinusoids', 'labels/nuclei', 'labels/sinusoid_fill').
    """
    data_dir = get_multiphoton_liver_data(path, samples, download)
    paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    if samples is not None:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in samples]
    assert len(paths) > 0
    return paths


def get_multiphoton_liver_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    target: Literal["bc", "sinusoids", "nuclei", "sinusoid_fill"] = "nuclei",
    samples: Optional[Sequence[str]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Multiphoton Liver dataset for 3D segmentation of liver tissue structures.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        target: The choice of segmentation target. One of 'bc', 'sinusoids', 'nuclei' or 'sinusoid_fill'.
            Only 'nuclei' is an instance segmentation target, the others are binary.
        samples: The sample names to use, see `SAMPLES`. By default all four samples are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if target not in MASK_CHANNELS:
        raise ValueError(f"'{target}' is not a valid target. Choose one of {list(MASK_CHANNELS)}.")

    data_paths = get_multiphoton_liver_paths(path, samples, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key=f"labels/{target}",
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=True,
        ndim=3,
        **kwargs
    )


def get_multiphoton_liver_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    target: Literal["bc", "sinusoids", "nuclei", "sinusoid_fill"] = "nuclei",
    samples: Optional[Sequence[str]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Multiphoton Liver dataloader for 3D segmentation of liver tissue structures.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        target: The choice of segmentation target. One of 'bc', 'sinusoids', 'nuclei' or 'sinusoid_fill'.
        samples: The sample names to use, see `SAMPLES`. By default all four samples are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_multiphoton_liver_dataset(path, patch_shape, target, samples, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
