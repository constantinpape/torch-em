"""The SynapseWeb hippocampus dataset contains three volumes of hippocampal CA1 neuropil
from adult rat, imaged with serial section TEM at ~2x2x50 nm resolution. All axons,
dendrites, glia, and synapses are reconstructed as instance segmentations.

The dataset is described in Harris et al. (2015):
"A resource from 3D electron microscopy of hippocampal neuropil for user training and tool development"
https://doi.org/10.1038/sdata.2015.46

Please cite this publication if you use this dataset in your research.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REGIONS = ("spine", "oblique", "apical")

# Bounding boxes (x0, x1, y0, y1, z0, z1) of the annotated sub-regions within each CloudVolume.
ANNO_BBOXES = {
    "spine": (3072, 6144, 1536, 3840, 30, 80),
    "oblique": (512, 4608, 768, 4608, 6, 91),
    "apical": (2048, 6144, 2048, 6400, 55, 167),
}

# ROIs covering only the densely annotated cube within each downloaded volume,
# determined by finding contiguous slices with >5% label coverage per axis.
DENSE_ROIS = {
    "spine": np.s_[0:42, 784:1665, 1007:1944],
    "oblique": np.s_[5:75, 1243:3505, 1385:3215],
    "apical": np.s_[5:106, 217:3681, 477:3936],
}


def _download_volume(region, out_path):
    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError(
            "cloudvolume is required to download this data. Install it with: pip install cloud-volume"
        )

    x0, x1, y0, y1, z0, z1 = ANNO_BBOXES[region]
    nx, ny, nz = x1 - x0, y1 - y0, z1 - z0

    vol_em = CloudVolume(
        f"s3://open-neurodata/kharris15/{region}/em", mip=0, use_https=True, fill_missing=True
    )
    vol_anno = CloudVolume(
        f"s3://open-neurodata/kharris15/{region}/anno", mip=0, use_https=True, fill_missing=True
    )

    # Download in z-slabs and write incrementally to avoid loading the full volume into memory.
    z_slab = 16
    with h5py.File(out_path, "w") as f:
        ds_raw = f.create_dataset("raw", shape=(nz, ny, nx), dtype="uint8", compression="gzip")
        ds_labels = f.create_dataset("labels", shape=(nz, ny, nx), dtype="uint64", compression="gzip")
        for z in range(z0, z1, z_slab):
            ze = min(z + z_slab, z1)
            # CloudVolume returns (x, y, z, channel); squeeze and transpose to (z, y, x).
            slab_raw = np.array(vol_em[x0:x1, y0:y1, z:ze]).squeeze().transpose(2, 1, 0)
            slab_labels = np.array(vol_anno[x0:x1, y0:y1, z:ze]).squeeze().transpose(2, 1, 0)
            zi = z - z0
            ds_raw[zi:zi + ze - z] = slab_raw
            ds_labels[zi:zi + ze - z] = slab_labels


def get_synapseweb_hippocampus_data(path: Union[os.PathLike, str], region: str, download: bool):
    """Download the SynapseWeb hippocampus data for a given region and store it as an HDF5 file.

    Args:
        path: Filepath to a folder where the data will be saved.
        region: The region to download. One of 'spine', 'oblique', 'apical'.
        download: Whether to download the data if it is not present.
    """
    if region not in REGIONS:
        raise ValueError(f"'{region}' is not a valid region. Choose from {REGIONS}.")

    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"synapseweb_hippocampus_{region}.h5")
    if os.path.exists(out_path):
        return

    if not download:
        raise RuntimeError(
            f"SynapseWeb hippocampus data for region '{region}' not found at {out_path}. "
            "Pass download=True to download it."
        )

    _download_volume(region, out_path)


def get_synapseweb_hippocampus_paths(
    path: Union[os.PathLike, str],
    regions: Tuple[str, ...] = ("spine", "oblique", "apical"),
    download: bool = False,
) -> List[str]:
    """Get paths to the SynapseWeb hippocampus HDF5 files.

    Args:
        path: Filepath to a folder where the data will be saved.
        regions: The regions to use. Subset of 'spine', 'oblique', 'apical'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the HDF5 files.
    """
    for region in regions:
        get_synapseweb_hippocampus_data(path, region, download)
    return [os.path.join(path, f"synapseweb_hippocampus_{region}.h5") for region in regions]


def get_synapseweb_hippocampus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    regions: Tuple[str, ...] = ("spine", "oblique", "apical"),
    rois: Dict[str, Any] = {},
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SynapseWeb hippocampus dataset for neuron segmentation in serial section TEM.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        regions: The regions to use. Subset of 'spine', 'oblique', 'apical'.
        rois: Dict mapping region name to a region of interest slice. Defaults to the
            densely annotated sub-cube per region.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    data_paths = get_synapseweb_hippocampus_paths(path, regions, download)
    data_rois = [rois.get(region, DENSE_ROIS[region]) for region in regions]

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        rois=data_rois,
        **kwargs
    )


def get_synapseweb_hippocampus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    regions: Tuple[str, ...] = ("spine", "oblique", "apical"),
    rois: Dict[str, Any] = {},
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron segmentation in the SynapseWeb hippocampus dataset.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        regions: The regions to use. Subset of 'spine', 'oblique', 'apical'.
        rois: Dict mapping region name to a region of interest slice. Defaults to the
            densely annotated sub-cube per region.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_synapseweb_hippocampus_dataset(
        path, patch_shape, regions, rois, download, offsets, boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
