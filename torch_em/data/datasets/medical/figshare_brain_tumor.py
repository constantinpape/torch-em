"""The Figshare Brain Tumor dataset contains annotations for meningioma, glioma and pituitary tumor
segmentation in T1-weighted contrast-enhanced brain MRI.

The dataset consists of 3064 slices from 233 patients, acquired at Nanfang Hospital, Guangzhou, China
and General Hospital, Tianjin Medical University, China from 2005 to 2010, with an in-plane resolution
of 512 x 512 and pixel size of 0.49 x 0.49 mm^2. Each slice comes with a tumor type label (meningioma:
708 slices, glioma: 1426 slices, pituitary tumor: 930 slices) and a binary tumor mask, both distributed
as v7.3 matlab '.mat' files (readable with `h5py`, since `scipy.io.loadmat` cannot read this format).

The dataset is located at https://doi.org/10.6084/m9.figshare.1512427 and is distributed under the
CC BY 4.0 license.

This dataset is from the publications https://doi.org/10.1371/journal.pone.0140381 and
https://doi.org/10.1371/journal.pone.0157112. Please cite them if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "brainTumorDataPublic_1-766.zip": "https://ndownloader.figshare.com/files/3381290",
    "brainTumorDataPublic_767-1532.zip": "https://ndownloader.figshare.com/files/3381296",
    "brainTumorDataPublic_1533-2298.zip": "https://ndownloader.figshare.com/files/3381293",
    "brainTumorDataPublic_2299-3064.zip": "https://ndownloader.figshare.com/files/3381302",
}

CHECKSUMS = {
    "brainTumorDataPublic_1-766.zip": "cdac0a7f4152cb34dd79b13543da96c37a6928ad8cb823e2662150b1a73ade54",
    "brainTumorDataPublic_767-1532.zip": "d18e896d14f7af791cdd3d9b9342f9f974742a06132effde5a5219209862b4ce",
    "brainTumorDataPublic_1533-2298.zip": "612d9f506c55c54db5f2f38b6d50741ef97a6d7c32b3e1be76e789674a97aefb",
    "brainTumorDataPublic_2299-3064.zip": "9a673c55d139133cfbc57097bf155b5ac9f33684ee97f40bf2afba332d416a3a",
}

TUMOR_TYPES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def _preprocess_figshare_brain_tumor(mat_dir, preprocessed_dir):
    import h5py

    os.makedirs(preprocessed_dir, exist_ok=True)
    mat_paths = natsorted(glob(os.path.join(mat_dir, "*.mat")))
    for mat_path in tqdm(mat_paths, desc="Preprocessing inputs"):
        sample_id = os.path.splitext(os.path.basename(mat_path))[0]

        with h5py.File(mat_path, "r") as f:
            cjdata = f["cjdata"]
            image = cjdata["image"][:]
            mask = cjdata["tumorMask"][:]
            label = int(cjdata["label"][()].item())

        out_path = os.path.join(preprocessed_dir, f"{sample_id}_{TUMOR_TYPES[label]}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=image, compression="gzip")
            f.create_dataset("labels", data=mask, compression="gzip")

    shutil.rmtree(mat_dir)


def get_figshare_brain_tumor_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Figshare Brain Tumor data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and glob(os.path.join(preprocessed_dir, "*.h5")):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    mat_dir = os.path.join(path, "mat_files")
    os.makedirs(mat_dir, exist_ok=True)
    for fname, url in URLS.items():
        zip_path = os.path.join(path, fname)
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[fname])
        util.unzip(zip_path=zip_path, dst=mat_dir)

    _preprocess_figshare_brain_tumor(mat_dir, preprocessed_dir)
    return preprocessed_dir


def get_figshare_brain_tumor_paths(
    path: Union[os.PathLike, str], tumor_type: Optional[str] = None, download: bool = False
) -> List[str]:
    """Get paths to the Figshare Brain Tumor data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        tumor_type: The choice of tumor type. One of 'meningioma', 'glioma', 'pituitary'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_figshare_brain_tumor_data(path, download)

    if tumor_type is None:
        pattern = "*.h5"
    elif tumor_type in TUMOR_TYPES.values():
        pattern = f"*_{tumor_type}.h5"
    else:
        raise ValueError(f"'{tumor_type}' is not a valid tumor type. Choose from {list(TUMOR_TYPES.values())}.")

    sample_paths = natsorted(glob(os.path.join(preprocessed_dir, pattern)))
    assert len(sample_paths) > 0, f"Could not find any preprocessed samples in '{preprocessed_dir}'."
    return sample_paths


def get_figshare_brain_tumor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    tumor_type: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Figshare Brain Tumor dataset for meningioma, glioma and pituitary tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        tumor_type: The choice of tumor type. One of 'meningioma', 'glioma', 'pituitary'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    sample_paths = get_figshare_brain_tumor_paths(path, tumor_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=sample_paths,
        raw_key="raw",
        label_paths=sample_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_figshare_brain_tumor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    tumor_type: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Figshare Brain Tumor dataloader for meningioma, glioma and pituitary tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        tumor_type: The choice of tumor type. One of 'meningioma', 'glioma', 'pituitary'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_figshare_brain_tumor_dataset(path, patch_shape, tumor_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
