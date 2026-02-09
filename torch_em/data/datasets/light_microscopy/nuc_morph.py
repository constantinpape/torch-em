"""The NucMorph dataset contains 3D fluorescence microscopy images of hiPSC nuclei
with instance segmentation annotations.

The dataset provides 410 paired 100x 3D images and watershed-based nuclear instance
segmentation masks from human induced pluripotent stem cells (hiPSCs). It includes
train (372), validation (20), and test (18) splits.

The dataset is located at https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/.
This dataset is from the publication https://doi.org/10.1016/j.cels.2025.101265.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


S3_BASE = (
    "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/"
    "hipsc_nuclei_image_datasets_for_training_deep_learning_models/"
    "segmentation_decoder_training_fov_dataset"
)

NUM_FILES = 410
VALID_SPLITS = ["train", "val", "test"]


def _download_manifest(path):
    """Download the training data manifest CSV."""
    manifest_path = os.path.join(path, "training_data_manifest.csv")
    if not os.path.exists(manifest_path):
        url = f"{S3_BASE}/training_data_manifest.csv"
        util.download_source(path=manifest_path, url=url, download=True, checksum=None)
    return manifest_path


def _get_split_indices(path, split):
    """Get file indices for a given split from the manifest."""
    import pandas as pd

    manifest_path = _download_manifest(path)
    df = pd.read_csv(manifest_path)

    # Map split names: manifest uses "valid" but we expose "val".
    manifest_split = "valid" if split == "val" else split
    indices = df[df["mode"] == manifest_split].iloc[:, 0].tolist()
    return sorted(indices)


def _download_files(path, split, download):
    """Download raw and segmentation files for a given split."""
    from tqdm import tqdm

    raw_dir = os.path.join(path, "high_res_100x")
    seg_dir = os.path.join(path, "watershed_segmentation_100x")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    indices = _get_split_indices(path, split)

    for idx in tqdm(indices, desc=f"Downloading {split} data"):
        fname = f"IMG_{idx:04d}.tif"

        raw_path = os.path.join(raw_dir, fname)
        if not os.path.exists(raw_path):
            url = f"{S3_BASE}/high_res_100x/{fname}"
            util.download_source(path=raw_path, url=url, download=download, checksum=None)

        seg_path = os.path.join(seg_dir, fname)
        if not os.path.exists(seg_path):
            url = f"{S3_BASE}/watershed_segmentation_100x/{fname}"
            util.download_source(path=seg_path, url=url, download=download, checksum=None)


def _create_h5_data(path, split):
    """Create h5 files with raw images and nuclear instance labels."""
    import h5py
    import imageio.v3 as imageio
    from tqdm import tqdm

    h5_dir = os.path.join(path, "h5_data", split)
    os.makedirs(h5_dir, exist_ok=True)

    indices = _get_split_indices(path, split)

    for idx in tqdm(indices, desc=f"Creating h5 for '{split}'"):
        fname = f"IMG_{idx:04d}"
        h5_path = os.path.join(h5_dir, f"{fname}.h5")

        if os.path.exists(h5_path):
            continue

        raw_path = os.path.join(path, "high_res_100x", f"{fname}.tif")
        seg_path = os.path.join(path, "watershed_segmentation_100x", f"{fname}.tif")

        raw = imageio.imread(raw_path)
        seg = imageio.imread(seg_path)

        # Crop to the minimum shape along each axis to handle off-by-one mismatches
        # (weird one-pixel interpolation shifts across one axis)
        min_shape = tuple(min(r, s) for r, s in zip(raw.shape, seg.shape))
        raw = raw[:min_shape[0], :min_shape[1], :min_shape[2]]
        seg = seg[:min_shape[0], :min_shape[1], :min_shape[2]]

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=seg.astype("int64"), compression="gzip")

    return h5_dir


def get_nuc_morph_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> str:
    """Download the NucMorph dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    assert split in VALID_SPLITS, f"'{split}' is not a valid split. Choose from {VALID_SPLITS}."
    _download_files(path, split, download)
    return path


def get_nuc_morph_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the NucMorph data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    assert split in VALID_SPLITS, f"'{split}' is not a valid split. Choose from {VALID_SPLITS}."

    get_nuc_morph_data(path, split, download)

    h5_dir = os.path.join(path, "h5_data", split)
    if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
        _create_h5_data(path, split)

    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) > 0, f"No data found for split '{split}'"

    return h5_paths


def get_nuc_morph_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NucMorph dataset for 3D nuclear instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_nuc_morph_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_nuc_morph_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NucMorph dataloader for 3D nuclear instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nuc_morph_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
