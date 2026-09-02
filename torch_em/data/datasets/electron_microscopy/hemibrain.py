"""The Hemibrain dataset contains a FIB-SEM volume of the Drosophila central brain
with dense neuron instance segmentation and additional volumetric annotations.

It covers approximately half of the central brain at 8 nm isotropic resolution,
with ~25,000 neurons reconstructed and proofread across regions including the
mushroom body, central complex, lateral horn, and portions of the optic lobe.

Three label types are available via the `label_choice` parameter:

- "neurons": dense neuron instance segmentation (uint64, 8 nm). Nearly fully dense
  (~99% of voxels labeled), each neuron/glial cell gets a unique ID. Background (0)
  is only extracellular space.

- "mito": mitochondria detection (uint64, 16 nm, upsampled to 8 nm for storage).
  Four classes - in practice ID=1 ("special") is the mitochondria class (~10% of
  voxels), ID=4 ("unlabeled") is background (~88%), ID=3 ("background") ~2%,
  ID=2 ("interior") ~0.1%. For binary training use ID=1 as foreground.

- "tissue": coarse tissue type semantic map (uint64, 16 nm, upsampled to 8 nm).
  Seven classes covering the full volume:
  1=broken white tissue (<1%), 2=trachea (<1%), 3=cell bodies (~2%),
  4=dark matter (~1%), 5=large dendrites (~14%), 6=neuropil (~83%),
  7=out of bounds. Useful as a region prior or multi-class semantic target.

Bounding boxes are always specified in 8 nm voxel coordinates. For mito and tissue
(native 16 nm), coordinates are halved during download and labels are upsampled back
to 8 nm using nearest-neighbour rescaling (skimage.transform.rescale, order=0).

This dataset is from the publication https://doi.org/10.7554/eLife.57443.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://www.janelia.org/project-team/flyem/hemibrain.
Requires cloud-volume: pip install cloud-volume.

NOTE (on data size): the full EM volume is (34427, 39725, 41394) voxels at 8 nm isotropic
resolution. Downloading the entire volume is not feasible. Data is instead accessed by
specifying bounding boxes (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel
coordinates, streamed from GCS and cached locally as HDF5 files.
"""

import hashlib
import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


EM_URL = "gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg"
SEG_URL = "gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
MITO_URL = "gs://neuroglancer-janelia-flyem-hemibrain/mito_20190717.27250582"
TISSUE_URL = "gs://neuroglancer-janelia-flyem-hemibrain/mask_normalized_round6"

LABEL_URLS = {
    "neurons": SEG_URL,
    "mito": MITO_URL,
    "tissue": TISSUE_URL,
}
# Mito and tissue are at 16 nm (factor 2 coarser than the 8 nm EM).
LABEL_RESOLUTION_FACTOR = {"neurons": 1, "mito": 2, "tissue": 2}

# A representative 1024³-voxel subvolume near the centre of the reconstructed region.
# Units are 8 nm voxels in (x, y, z) order, matching the CloudVolume coordinate space.
DEFAULT_BOUNDING_BOX = (15000, 16024, 18000, 19024, 18000, 19024)


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def get_hemibrain_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int] = DEFAULT_BOUNDING_BOX,
    label_choice: Literal["neurons", "mito", "tissue"] = "neurons",
    download: bool = False,
) -> str:
    """Stream a subvolume from the Hemibrain dataset and cache it as an HDF5 file.

    Args:
        path: Filepath to a folder where the cached HDF5 file will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in 8 nm voxel coordinates. Defaults to a central 1024³ training region.
        label_choice: Which labels to cache alongside the EM. One of "neurons", "mito",
            or "tissue". Mito and tissue are at 16 nm and are resampled to match the EM.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached HDF5 file.
    """
    import h5py

    os.makedirs(str(path), exist_ok=True)
    h5_path = os.path.join(str(path), f"{label_choice}_{_bbox_to_str(bounding_box)}.h5")
    if os.path.exists(h5_path):
        return h5_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{h5_path}'. Set download=True to stream it from GCS."
        )

    try:
        import cloudvolume
    except ImportError:
        raise ImportError("The 'cloud-volume' package is required: pip install cloud-volume")

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    print(f"Streaming Hemibrain EM + {label_choice} for bbox {bounding_box} ...")

    em_vol = cloudvolume.CloudVolume(EM_URL, use_https=True, mip=0, progress=True)
    raw = np.array(em_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)

    factor = LABEL_RESOLUTION_FACTOR[label_choice]
    lx0, lx1 = x_min // factor, x_max // factor
    ly0, ly1 = y_min // factor, y_max // factor
    lz0, lz1 = z_min // factor, z_max // factor
    lbl_vol = cloudvolume.CloudVolume(LABEL_URLS[label_choice], use_https=True, mip=0, progress=True)
    labels = np.array(lbl_vol[lx0:lx1, ly0:ly1, lz0:lz1])[..., 0].transpose(2, 1, 0)
    if factor > 1:
        from skimage.transform import rescale
        labels = rescale(labels, factor, order=0, anti_aliasing=False, preserve_range=True).astype(labels.dtype)
        labels = labels[:raw.shape[0], :raw.shape[1], :raw.shape[2]]

    with h5py.File(h5_path, "w", locking=False) as f:
        f.attrs["bounding_box"] = bounding_box
        f.attrs["label_choice"] = label_choice
        f.attrs["resolution_nm"] = em_vol.resolution.tolist()
        f.create_dataset("raw", data=raw.astype("uint8"), compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels.astype("uint64"), compression="gzip", chunks=True)

    print(f"Cached to {h5_path} (raw {raw.shape}, labels {labels.shape})")
    return h5_path


def get_hemibrain_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    label_choice: Literal["neurons", "mito", "tissue"] = "neurons",
    download: bool = False,
) -> List[str]:
    """Get paths to Hemibrain HDF5 cache files.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX].
        label_choice: Which labels to use. One of "neurons", "mito", or "tissue".
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached HDF5 files.
    """
    if bounding_boxes is None:
        bounding_boxes = [DEFAULT_BOUNDING_BOX]
    return [get_hemibrain_data(path, bbox, label_choice, download) for bbox in bounding_boxes]


def get_hemibrain_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    label_choice: Literal["neurons", "mito", "tissue"] = "neurons",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Hemibrain dataset for neuron or organelle segmentation.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] - a central 1024³ region.
        label_choice: Which labels to use. One of "neurons", "mito", or "tissue".
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_hemibrain_paths(path, bounding_boxes, label_choice, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_hemibrain_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    label_choice: Literal["neurons", "mito", "tissue"] = "neurons",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron or organelle segmentation in the Hemibrain dataset.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] - a central 1024³ region.
        label_choice: Which labels to use. One of "neurons", "mito", or "tissue".
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hemibrain_dataset(
        path, patch_shape, bounding_boxes=bounding_boxes, label_choice=label_choice,
        download=download, offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
