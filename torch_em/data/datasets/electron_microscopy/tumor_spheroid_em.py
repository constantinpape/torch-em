"""The tumor spheroid EM dataset contains SBF-SEM imaging of tumor spheroids with gold nanoparticles.

Two data sources are available, selected via the `source` parameter:

**"2d_manual"** - Manually annotated 2D TIFF slices at two isotropic resolutions
(50 x 50 x 50 nm and 100 x 100 x 100 nm). Each slice has paired instance
segmentation labels for cells and nuclei. Slices span all three orthogonal
planes (XY, XZ, YZ). Available targets: "cells", "nuclei".

**"3d_automatic"** - Full 3D volume with automated instance segmentation for cells,
nuclei, and gold nanoparticles (nps). Raw data available at four resolutions:
native 50 x 10 x 10 nm ("50-10-10"), and downsampled "50-25-25", "50-50-50",
"100-100-100". Labels at "50-50-50" and "100-100-100" for cells/nuclei, and
"50-50-50" only for nps. Requires downloading the full 67 GB zarr archive.
Available targets: "cells", "nuclei", "nps".

The volume covers approximately 102.4 x 102.4 x 35 um at native voxel size.

This dataset is from the publication https://doi.org/10.64898/2026.04.17.719153.
Please cite it if you use this dataset for a publication.

The data is available at https://doi.org/10.6019/S-BIAD3263.
"""

import os
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


# The BioStudies file service; the FTP mirror only carries the directory skeleton of this study.
FTP_BASE = "https://www.ebi.ac.uk/biostudies/files/S-BIAD3263"
ZARR_URL = f"{FTP_BASE}/Au_01-vol_01.zarr.zip"
ZARR_ROOT = "Au_01-vol_01.zarr"

SLICE_IDS = {
    "50-50-50": {
        "x": ["0277", "0336", "0390", "0653", "1300"],
        "y": ["0288", "0488", "0889", "1272", "1606"],
        "z": ["0016", "0034", "0073", "0075", "0169", "0173", "0180", "0192", "0212", "0274"],
    },
    "100-100-100": {
        "x": ["0138", "0168", "0195", "0326", "0650"],
        "y": ["0144", "0244", "0444", "0636", "0803"],
        "z": ["0008", "0017", "0036", "0038", "0084", "0086", "0090", "0096", "0106", "0137"],
    },
}

LABEL_RESOLUTIONS_3D = {
    "cells": ("50-50-50", "100-100-100"),
    "nuclei": ("50-50-50", "100-100-100"),
    "nps": ("50-50-50",),
}

SourceChoice = Literal["2d_manual", "3d_automatic"]
Resolution2DChoice = Literal["50-50-50", "100-100-100"]
Resolution3DChoice = Literal["50-10-10", "50-25-25", "50-50-50", "100-100-100"]
TargetChoice = Literal["cells", "nuclei", "nps"]
OrientationChoice = Literal["x", "y", "z"]


def _download_2d_slice(axis, coord, resolution, out_dir):
    import h5py

    stem = f"Au_01-vol_01-{axis}_{coord}"
    h5_path = os.path.join(out_dir, f"{stem}.h5")
    if os.path.exists(h5_path):
        return

    base_url = f"{FTP_BASE}/ground_truths/{resolution}"
    raw_tmp = os.path.join(out_dir, f"{stem}_raw.tif")
    cells_tmp = os.path.join(out_dir, f"{stem}_cells.tif")
    nuclei_tmp = os.path.join(out_dir, f"{stem}_nuclei.tif")

    util.download_source(raw_tmp, f"{base_url}/{stem}.tif", download=True)
    util.download_source(cells_tmp, f"{base_url}/labels/{stem}-cells.tif", download=True)
    util.download_source(nuclei_tmp, f"{base_url}/labels/{stem}-nuclei.tif", download=True)

    raw = imageio.imread(raw_tmp)
    cells = imageio.imread(cells_tmp)
    nuclei = imageio.imread(nuclei_tmp)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/cells", data=cells.astype("uint32"), compression="gzip")
        f.create_dataset("labels/nuclei", data=nuclei.astype("uint32"), compression="gzip")

    os.remove(raw_tmp)
    os.remove(cells_tmp)
    os.remove(nuclei_tmp)


def get_tumor_spheroid_data(
    path: Union[os.PathLike, str],
    source: SourceChoice = "2d_manual",
    resolution: str = "50-50-50",
    download: bool = False,
) -> str:
    """Download the tumor spheroid EM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: Data source. "2d_manual" downloads sparse 2D annotated TIFF slices
            (cells + nuclei, ~50 MB). "3d_automatic" downloads the full 3D zarr
            archive with automated segmentation for cells, nuclei, and nanoparticles
            (~67 GB).
        resolution: The voxel resolution to use. For "2d_manual": "50-50-50" or
            "100-100-100". For "3d_automatic": "50-10-10", "50-25-25", "50-50-50",
            or "100-100-100" (all in nm, ZYX order).
        download: Whether to download the data if it is not present.

    Returns:
        Path to the downloaded data (folder for "2d_manual", zip file for "3d_automatic").
    """
    if source == "2d_manual":
        assert resolution in SLICE_IDS, \
            f"Invalid resolution '{resolution}' for 2d_manual, expected one of {list(SLICE_IDS)}."
        out_dir = os.path.join(str(path), "2d_manual", resolution)
        expected = sum(len(v) for v in SLICE_IDS[resolution].values())
        if len(glob(os.path.join(out_dir, "*.h5"))) >= expected:
            return out_dir
        if not download:
            raise RuntimeError(
                f"No cached data found at '{out_dir}'. Set download=True to download from BioImage Archive."
            )
        os.makedirs(out_dir, exist_ok=True)
        for axis, ids in SLICE_IDS[resolution].items():
            for coord in ids:
                _download_2d_slice(axis, coord, resolution, out_dir)
        return out_dir

    elif source == "3d_automatic":
        zarr_path = os.path.join(str(path), "3d_automatic", "Au_01-vol_01.zarr.zip")
        if os.path.exists(zarr_path):
            return zarr_path
        if not download:
            raise RuntimeError(
                f"Zarr archive not found at '{zarr_path}'. Set download=True to download (~67 GB)."
            )
        os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
        util.download_source(zarr_path, ZARR_URL, download=True)
        return zarr_path

    else:
        raise ValueError(f"Invalid source '{source}', expected '2d_manual' or '3d_automatic'.")


def get_tumor_spheroid_paths(
    path: Union[os.PathLike, str],
    source: SourceChoice = "2d_manual",
    resolution: str = "50-50-50",
    target: TargetChoice = "cells",
    orientations: Optional[List[OrientationChoice]] = None,
    download: bool = False,
) -> Tuple[List[str], str, str]:
    """Get paths and array keys for the tumor spheroid EM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: Data source, either "2d_manual" or "3d_automatic".
        resolution: The voxel resolution to use. For "2d_manual": "50-50-50" or
            "100-100-100". For "3d_automatic": "50-10-10", "50-25-25", "50-50-50",
            or "100-100-100".
        target: The segmentation target. "cells" and "nuclei" are available for
            both sources. "nps" (gold nanoparticles) is only available for
            "3d_automatic" at "50-50-50" resolution.
        orientations: Slice orientations to include ("x", "y", "z"). Defaults to
            all three. Only relevant for "2d_manual".
        download: Whether to download the data if it is not present.

    Returns:
        Tuple of (file paths, raw key, label key).
    """
    if source == "2d_manual":
        assert target in ("cells", "nuclei"), \
            f"Target '{target}' is not available for '2d_manual'. Choose 'cells' or 'nuclei'."
        if orientations is None:
            orientations = ["x", "y", "z"]
        out_dir = get_tumor_spheroid_data(path, source, resolution, download)
        file_paths = []
        for axis in orientations:
            for coord in SLICE_IDS[resolution][axis]:
                file_paths.append(os.path.join(out_dir, f"Au_01-vol_01-{axis}_{coord}.h5"))
        file_paths.sort()
        return file_paths, "raw", f"labels/{target}"

    elif source == "3d_automatic":
        assert target in LABEL_RESOLUTIONS_3D, \
            f"Invalid target '{target}', expected one of {list(LABEL_RESOLUTIONS_3D)}."
        valid_resolutions = LABEL_RESOLUTIONS_3D[target]
        assert resolution in valid_resolutions, (
            f"Resolution '{resolution}' is not available for target '{target}'. "
            f"Valid options: {valid_resolutions}."
        )
        if orientations is not None:
            raise ValueError("The 'orientations' parameter is only valid for source='2d_manual'.")
        zarr_path = get_tumor_spheroid_data(path, source, resolution, download)
        raw_key = f"{ZARR_ROOT}/images/{resolution}"
        label_key = f"{ZARR_ROOT}/labels/{target}/masks/{resolution}"
        return [zarr_path], raw_key, label_key

    else:
        raise ValueError(f"Invalid source '{source}', expected '2d_manual' or '3d_automatic'.")


def get_tumor_spheroid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: SourceChoice = "2d_manual",
    resolution: str = "50-50-50",
    target: TargetChoice = "cells",
    orientations: Optional[List[OrientationChoice]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the tumor spheroid EM dataset for cell/nucleus/nanoparticle segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training. Use (H, W) for "2d_manual"
            and (D, H, W) for "3d_automatic".
        source: Data source. "2d_manual" uses sparse manually annotated 2D slices
            (cells + nuclei). "3d_automatic" uses the full 3D volume with automated
            segmentation (cells, nuclei, nps). Requires ~67 GB download.
        resolution: The voxel resolution. For "2d_manual": "50-50-50" or
            "100-100-100". For "3d_automatic": "50-10-10", "50-25-25", "50-50-50",
            or "100-100-100".
        target: The segmentation target ("cells", "nuclei", or "nps").
            "nps" is only available for "3d_automatic" at "50-50-50".
        orientations: Slice orientations to include. Only for "2d_manual".
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert sum((offsets is not None, boundaries, binary)) <= 1, f"{offsets}, {boundaries}, {binary}"

    file_paths, raw_key, label_key = get_tumor_spheroid_paths(
        path, source, resolution, target, orientations, download
    )

    if offsets is not None:
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, ignore_label=None, add_binary_target=True, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to True, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to True, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    return torch_em.default_segmentation_dataset(
        raw_paths=file_paths,
        raw_key=raw_key,
        label_paths=file_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_tumor_spheroid_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    source: SourceChoice = "2d_manual",
    resolution: str = "50-50-50",
    target: TargetChoice = "cells",
    orientations: Optional[List[OrientationChoice]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for segmentation in the tumor spheroid EM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training. Use (H, W) for "2d_manual"
            and (D, H, W) for "3d_automatic".
        batch_size: The batch size for training.
        source: Data source. "2d_manual" uses sparse manually annotated 2D slices
            (cells + nuclei). "3d_automatic" uses the full 3D volume with automated
            segmentation (cells, nuclei, nps). Requires ~67 GB download.
        resolution: The voxel resolution. For "2d_manual": "50-50-50" or
            "100-100-100". For "3d_automatic": "50-10-10", "50-25-25", "50-50-50",
            or "100-100-100".
        target: The segmentation target ("cells", "nuclei", or "nps").
            "nps" is only available for "3d_automatic" at "50-50-50".
        orientations: Slice orientations to include. Only for "2d_manual".
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tumor_spheroid_dataset(
        path, patch_shape, source=source, resolution=resolution, target=target,
        orientations=orientations, download=download, offsets=offsets, boundaries=boundaries,
        binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
