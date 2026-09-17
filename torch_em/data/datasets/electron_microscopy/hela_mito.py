"""The HeLa-Mito dataset contains mitochondria instance segmentation for 2D electron
microscopy slices of HeLa cells.

The raw slices are from the same acquisition (EMPIAR-10094) already used by the
`cefa_hela` module, but that module only provides nuclear-envelope annotations; this
module adds a separate, complementary layer of mitochondria instance masks on 9 of the
24 raw slices released alongside it.

Annotation coverage is sparse, not exhaustive: only 9 of the 24 available slices carry
labels, and some mitochondria visible in the raw data may be left unannotated even on
labeled slices.

The dataset is available at https://github.com/reyesaldasoro/MitoEM.
The dataset was published in https://doi.org/10.1101/2023.11.14.567016.
Please cite this publication if you use the dataset in your research.
"""

import os
from typing import List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://raw.githubusercontent.com/reyesaldasoro/MitoEM/main/CODE"
RAW_URL = BASE_URL + "/OriginalImages/ROI_6005_4739_81_z{slice_id}.tif"
LABEL_URL = BASE_URL + "/{prefix}_ROI_6005_4739_81_z{slice_id}.mat"

# The annotated slice ids and the ground-truth file prefix to use for each. GT and GT2
# both annotate z0001 (0.94 IoU between them); GT is used there and GT2 only for the
# 4 additional slices it uniquely covers.
SLICES = {
    "0001": "GT",
    "0002": "GT2",
    "0004": "GT2",
    "0026": "GT2",
    "0028": "GT2",
    "0030": "GT",
    "0060": "GT",
    "0116": "GT",
    "0150": "GT",
}


def _convert_slice(raw_path, mat_path, out_path):
    import h5py
    import tifffile
    import scipy.io as sio

    raw = tifffile.imread(raw_path)
    labels = sio.loadmat(mat_path)["groundTruthM"].max(axis=2)
    assert raw.shape == labels.shape, f"{raw.shape} != {labels.shape}"

    tmp_path = out_path + ".incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    os.rename(tmp_path, out_path)


def get_hela_mito_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HeLa-Mito dataset.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the per-slice HDF5 files.
    """
    os.makedirs(path, exist_ok=True)

    for slice_id, prefix in SLICES.items():
        out_path = os.path.join(path, f"z{slice_id}.h5")
        if os.path.exists(out_path):
            continue
        if not download:
            raise RuntimeError(f"Cannot find the data at {out_path}, but download was set to False.")

        raw_path = os.path.join(path, f"raw_z{slice_id}.tif")
        mat_path = os.path.join(path, f"{prefix}_z{slice_id}.mat")
        util.download_source(raw_path, RAW_URL.format(slice_id=slice_id), download)
        util.download_source(mat_path, LABEL_URL.format(prefix=prefix, slice_id=slice_id), download)

        _convert_slice(raw_path, mat_path, out_path)
        os.remove(raw_path)
        os.remove(mat_path)

    return path


def get_hela_mito_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the HeLa-Mito data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the per-slice HDF5 files.
    """
    data_dir = get_hela_mito_data(path, download)
    return [os.path.join(data_dir, f"z{slice_id}.h5") for slice_id in SLICES]


def get_hela_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for mitochondria instance segmentation in HeLa cell EM slices.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 2

    paths = get_hela_mito_paths(path, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_hela_mito_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for mitochondria instance segmentation in HeLa cell EM slices.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hela_mito_dataset(path, patch_shape, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
