"""The BMGD (Breast Mammary Gland Dataset) contains DAPI-stained fluorescent microscopy
images for nuclei segmentation in mammary gland tissue.

The dataset includes 819 image patches with over 9,500 manually segmented nuclei
from mammary epithelial cells cultured under different microenvironmental stiffness conditions.

The dataset is from: https://github.com/zt089/Breast-Mammary-Gland-Dataset-BMGD
Please cite the following paper if you use this dataset in your research:
https://doi.org/10.21203/rs.3.rs-8263420/v1
"""

import os
from glob import glob
from typing import Union, Tuple, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "250pa": "https://github.com/zt089/Breast-Mammary-Gland-Dataset-BMGD/raw/main/250%20Pa.7z",
    "950pa": "https://github.com/zt089/Breast-Mammary-Gland-Dataset-BMGD/raw/main/950%20Pa.7z",
    "1200pa": "https://github.com/zt089/Breast-Mammary-Gland-Dataset-BMGD/raw/main/1200%20Pa.7z",
    "1800pa": "https://github.com/zt089/Breast-Mammary-Gland-Dataset-BMGD/raw/main/1800%20Pa.7z",
}

# Folder names inside the archives (with spaces)
_FOLDER_NAMES = {
    "250pa": "250 Pa",
    "950pa": "950 Pa",
    "1200pa": "1200 Pa",
    "1800pa": "1800 Pa",
}

STIFFNESS_LEVELS = list(URLS.keys())


def get_bmgd_data(
    path: Union[os.PathLike, str],
    stiffness: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> str:
    """Download the BMGD dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        stiffness: The stiffness level(s) to download. One of '250pa', '950pa', '1200pa', '1800pa'.
            If None, downloads all stiffness levels.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the dataset directory.
    """
    if stiffness is None:
        stiffness = STIFFNESS_LEVELS
    elif isinstance(stiffness, str):
        stiffness = [stiffness]

    for s in stiffness:
        assert s in STIFFNESS_LEVELS, f"'{s}' is not valid. Choose from {STIFFNESS_LEVELS}."

        folder_name = _FOLDER_NAMES[s]
        data_dir = os.path.join(path, folder_name)

        if os.path.exists(data_dir) and len(glob(os.path.join(data_dir, "image", "*.tif"))) > 0:
            continue

        os.makedirs(path, exist_ok=True)

        archive_path = os.path.join(path, f"{s}.7z")
        util.download_source(path=archive_path, url=URLS[s], download=download, checksum=None)

        # Extract 7z archive
        util.unzip(zip_path=archive_path, dst=path, remove=False)

    return path


def _create_bmgd_h5(path, stiffness):
    """Create processed h5 files with instance labels from semantic masks."""
    import h5py
    from skimage.measure import label
    from tqdm import tqdm
    import tifffile

    folder_name = _FOLDER_NAMES[stiffness]
    data_dir = os.path.join(path, folder_name)
    h5_out_dir = os.path.join(path, "processed", stiffness)
    os.makedirs(h5_out_dir, exist_ok=True)

    images_dir = os.path.join(data_dir, "image")
    masks_dir = os.path.join(data_dir, "mask")

    # Find all image files
    image_files = sorted(glob(os.path.join(images_dir, "*.tif")))

    for img_path in tqdm(image_files, desc=f"Processing BMGD {stiffness}"):
        fname = os.path.basename(img_path)
        mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(mask_path):
            continue

        out_fname = f"bmgd_{stiffness}_{fname.replace('.tif', '.h5')}"
        out_path = os.path.join(h5_out_dir, out_fname)

        if os.path.exists(out_path):
            continue

        raw = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        # Convert semantic mask to instance labels using connected components
        instances = label(mask > 0).astype("int64")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=(mask > 0).astype("uint8"), compression="gzip")

    return h5_out_dir


def get_bmgd_paths(
    path: Union[os.PathLike, str],
    stiffness: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the BMGD data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        stiffness: The stiffness level(s). If None, uses all levels.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the processed h5 data.
    """
    from natsort import natsorted

    get_bmgd_data(path, stiffness, download)

    if stiffness is None:
        stiffness = STIFFNESS_LEVELS
    elif isinstance(stiffness, str):
        stiffness = [stiffness]

    all_h5_paths = []
    for s in stiffness:
        h5_out_dir = os.path.join(path, "processed", s)

        # Process data if not already done
        if not os.path.exists(h5_out_dir) or len(glob(os.path.join(h5_out_dir, "*.h5"))) == 0:
            _create_bmgd_h5(path, s)

        h5_paths = glob(os.path.join(h5_out_dir, "*.h5"))
        all_h5_paths.extend(h5_paths)

    assert len(all_h5_paths) > 0, f"No data found for stiffness '{stiffness}'"

    return natsorted(all_h5_paths)


def get_bmgd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    stiffness: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BMGD dataset for nuclei segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        stiffness: The stiffness level(s). One of '250pa', '950pa', '1200pa', '1800pa'.
            If None, uses all stiffness levels.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_bmgd_paths(path, stiffness, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, label_dtype=np.int64,
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels/instances",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_bmgd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    stiffness: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BMGD dataloader for nuclei segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        stiffness: The stiffness level(s). One of '250pa', '950pa', '1200pa', '1800pa'.
            If None, uses all stiffness levels.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bmgd_dataset(
        path=path,
        patch_shape=patch_shape,
        stiffness=stiffness,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
