"""The hiPSC single-cell image dataset contains 3D confocal fluorescence microscopy images of
human induced pluripotent stem cells (hiPSC), with segmentation masks for the nucleus, cell
membrane, and one fluorescently-tagged subcellular structure per cell line (e.g. mitochondria,
Golgi, microtubules).

The dataset provides per-cell crops (binary masks) and full field-of-view images (instance masks
for nucleus and cell, binary mask for the structure). It is located at
https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset under the
Allen Institute for Cell Science Terms of Use (https://www.allencell.org/terms-of-use.html).
This dataset is from the publication https://doi.org/10.1038/s41586-022-05563-7.
Please cite it if you use this dataset in your research.

NOTE: The full dataset covers almost 32,000 cells and is roughly a terabyte in size. Use
`structure_names` and/or `n_samples` to bound the download to a manageable subset.
"""

import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset"

# Crop-level segmentation channels, as documented in the package README's `name_dict`:
# ['dna_segmentation', 'membrane_segmentation', 'membrane_segmentation_roof',
#  'struct_segmentation', 'struct_segmentation_roof'].
CROP_SEG_CHANNELS = {"nucleus": 0, "cell": 1, "structure": 3}
# FOV-level segmentation channels, as documented in the package README's `fov_seg_path`:
# nuclear segmentation, cell segmentation, contour of nuclei, contour of cell.
FOV_SEG_CHANNELS = {"nucleus": 0, "cell": 1}

VALID_TARGETS = ["nucleus", "cell", "structure"]
VALID_SAMPLE_TYPES = ["cell", "fov"]


def _get_metadata(path, download):
    import pandas as pd

    csv_path = os.path.join(path, "metadata.csv")
    util.download_source(path=csv_path, url=f"{BASE_URL}/metadata.csv", download=download, checksum=None)
    return pd.read_csv(csv_path)


def _select_rows(path, structure_names, sample_type, n_samples, download):
    df = _get_metadata(path, download)
    if structure_names is not None:
        df = df[df["structure_name"].isin(structure_names)]
    if n_samples is not None:
        df = df.groupby("structure_name", group_keys=False).head(n_samples)
    if sample_type == "fov":
        df = df.groupby("FOVId", as_index=False).first()
    return df.reset_index(drop=True)


def _download_cell_files(path, row, download):
    raw_path = os.path.join(path, row["crop_raw"])
    seg_path = os.path.join(path, row["crop_seg"])
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(seg_path), exist_ok=True)
    util.download_source(path=raw_path, url=f"{BASE_URL}/{row['crop_raw']}", download=download, checksum=None)
    util.download_source(path=seg_path, url=f"{BASE_URL}/{row['crop_seg']}", download=download, checksum=None)


def _download_fov_files(path, row, download):
    raw_path = os.path.join(path, row["fov_path"])
    seg_path = os.path.join(path, row["fov_seg_path"])
    struct_path = os.path.join(path, row["struct_seg_path"])
    for out_path in (raw_path, seg_path, struct_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    util.download_source(path=raw_path, url=f"{BASE_URL}/{row['fov_path']}", download=download, checksum=None)
    util.download_source(path=seg_path, url=f"{BASE_URL}/{row['fov_seg_path']}", download=download, checksum=None)
    util.download_source(
        path=struct_path, url=f"{BASE_URL}/{row['struct_seg_path']}", download=download, checksum=None
    )


def _create_h5(path, df, sample_type, target):
    import h5py
    import tifffile
    from tqdm import tqdm

    h5_dir = os.path.join(path, "h5_data", sample_type, target)
    os.makedirs(h5_dir, exist_ok=True)

    h5_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing '{sample_type}/{target}' data"):
        h5_path = os.path.join(h5_dir, f"{row['CellId']}.h5")
        h5_paths.append(h5_path)
        if os.path.exists(h5_path):
            continue

        if sample_type == "cell":
            raw = tifffile.imread(os.path.join(path, row["crop_raw"]))  # (Z, 3, Y, X): dna, membrane, structure
            seg = tifffile.imread(os.path.join(path, row["crop_seg"]))  # (Z, 5, Y, X), see CROP_SEG_CHANNELS
            raw = np.moveaxis(raw, 1, 0)
            labels = (seg[:, CROP_SEG_CHANNELS[target]] > 0).astype("uint8")
        else:
            raw_full = tifffile.imread(os.path.join(path, row["fov_path"]))  # (Z, 7, Y, X)
            channel_idx = {
                "dna": row["ChannelNumber405"],
                "membrane": row["ChannelNumber638"],
                "structure": row["ChannelNumberStruct"],
            }
            raw = np.stack([raw_full[:, channel_idx[name]] for name in ("dna", "membrane", "structure")], axis=0)
            if target == "structure":
                labels = (tifffile.imread(os.path.join(path, row["struct_seg_path"])) > 0).astype("uint8")
            else:
                seg = tifffile.imread(os.path.join(path, row["fov_seg_path"]))  # (Z, 4, Y, X), see FOV_SEG_CHANNELS
                labels = seg[:, FOV_SEG_CHANNELS[target]].astype("uint32")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

    return h5_paths


def get_hipsc_single_cell_data(
    path: Union[os.PathLike, str],
    structure_names: Optional[List[str]] = None,
    sample_type: Literal["cell", "fov"] = "cell",
    n_samples: Optional[int] = None,
    download: bool = False,
) -> str:
    """Download the hiPSC single-cell image dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        structure_names: The tagged structures to restrict the data to, e.g. ['TOMM20']. By default all are used.
        sample_type: Whether to download per-cell crops or full field-of-view (FOV) images.
        n_samples: The maximum number of cells to use per structure. By default all are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the downloaded data.
    """
    assert sample_type in VALID_SAMPLE_TYPES, f"'{sample_type}' is not a valid sample type: {VALID_SAMPLE_TYPES}."
    os.makedirs(path, exist_ok=True)
    df = _select_rows(path, structure_names, sample_type, n_samples, download)
    for _, row in df.iterrows():
        if sample_type == "cell":
            _download_cell_files(path, row, download)
        else:
            _download_fov_files(path, row, download)
    return path


def get_hipsc_single_cell_paths(
    path: Union[os.PathLike, str],
    structure_names: Optional[List[str]] = None,
    sample_type: Literal["cell", "fov"] = "cell",
    target: Literal["nucleus", "cell", "structure"] = "nucleus",
    n_samples: Optional[int] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the hiPSC single-cell image data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        structure_names: The tagged structures to restrict the data to, e.g. ['TOMM20']. By default all are used.
        sample_type: Whether to use per-cell crops or full field-of-view (FOV) images.
        target: The segmentation target. One of 'nucleus', 'cell' or 'structure'. For `sample_type='cell'`, all
            targets are binary masks. For `sample_type='fov'`, 'nucleus' and 'cell' are instance masks and
            'structure' is a binary mask.
        n_samples: The maximum number of cells to use per structure. By default all are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the h5 files with the raw and label data.
    """
    assert target in VALID_TARGETS, f"'{target}' is not a valid target. Choose from {VALID_TARGETS}."

    get_hipsc_single_cell_data(path, structure_names, sample_type, n_samples, download)
    df = _select_rows(path, structure_names, sample_type, n_samples, download=False)
    h5_paths = _create_h5(path, df, sample_type, target)

    return h5_paths


def get_hipsc_single_cell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    structure_names: Optional[List[str]] = None,
    sample_type: Literal["cell", "fov"] = "cell",
    target: Literal["nucleus", "cell", "structure"] = "nucleus",
    n_samples: Optional[int] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the hiPSC single-cell image dataset for 3D segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        structure_names: The tagged structures to restrict the data to, e.g. ['TOMM20']. By default all are used.
        sample_type: Whether to use per-cell crops or full field-of-view (FOV) images.
        target: The segmentation target. One of 'nucleus', 'cell' or 'structure'.
        n_samples: The maximum number of cells to use per structure. By default all are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_hipsc_single_cell_paths(path, structure_names, sample_type, target, n_samples, download)

    kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True)
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        with_channels=True,
        ndim=3,
        **kwargs,
    )


def get_hipsc_single_cell_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    structure_names: Optional[List[str]] = None,
    sample_type: Literal["cell", "fov"] = "cell",
    target: Literal["nucleus", "cell", "structure"] = "nucleus",
    n_samples: Optional[int] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the hiPSC single-cell image dataloader for 3D segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        structure_names: The tagged structures to restrict the data to, e.g. ['TOMM20']. By default all are used.
        sample_type: Whether to use per-cell crops or full field-of-view (FOV) images.
        target: The segmentation target. One of 'nucleus', 'cell' or 'structure'.
        n_samples: The maximum number of cells to use per structure. By default all are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hipsc_single_cell_dataset(
        path, patch_shape, structure_names, sample_type, target, n_samples, download, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
