"""The DERMA-OCTA dataset contains annotations for dermatological vessel segmentation in optical
coherence tomography angiography (OCTA) images, comprising 330 volumetric acquisitions from 74
subjects (chronic venous disease, healthy skin and skin lesions).

NOTE: The dataset abstract mentions that segmentation labels shown in the paper's figures were
generated with a U-Net. The archive itself, however, ships a dedicated 'Manual segmentations' folder
(both for the 2D en-face projections and the 3D volumes), which are the real reference annotations
created in Amira and manually refined/verified by four annotators plus a technician and a clinician.
This loader only uses these manual segmentations, never any U-Net-generated files.

The raw acquisitions are only distributed after one of five preprocessing pipelines has been applied
to them (the archive does not ship the untouched raw scans separately). This loader uses the 'Norm'
(min-max normalized) version, paired 1:1 with the manual segmentations by filename.

Each case provides three mask variants: the full-depth projection ('all'), the superficial vascular
plexus ('sup') and the deep vascular plexus ('deep'), selected with the 'plexus' argument (2D only;
the 3D volumes only ship the full-depth variant).

NOTE: A single case in the 3D archive ('13801') ships a label volume with one extra frame relative to
its raw volume; this loader drops that mismatched pair defensively.

The data is located at https://doi.org/10.5281/zenodo.15088516, released under a CC-BY-NC-4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-025-05763-6.
Please cite it if you use this dataset for your research.
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "2d": {
        "raw": "https://zenodo.org/api/records/15088516/files/Norm%20-%202D.zip/content",
        "labels": "https://zenodo.org/api/records/15088516/files/Manual%20segmentations%20-%202D.zip/content",
    },
    "3d": {
        "raw": "https://zenodo.org/api/records/15088516/files/Norm%20-%203D.zip/content",
        "labels": "https://zenodo.org/api/records/15088516/files/Manual%20segmentations%20-%203D.zip/content",
    },
}

CHECKSUMS = {
    "2d": {
        "raw": "f481652c2753d0d23f56cba1c52cc980c9075dff2a144805ff30f7003f6e1c62",
        "labels": "b4971dba5dc80d0322396300a9f40cc554205bd68b5734b2f5f4c449baa7b1b4",
    },
    "3d": {
        "raw": "5c226bc7dd2f625463473f8cc6b448719f7efa05909b9fc664acc17ce9e218bb",
        "labels": "6155e49943dbe591e3b112e0a2db643a622f13a2fd4857d559b699be9724ac1a",
    },
}

DIR_NAMES = {"2d": ("Norm - 2D", "Manual segmentations - 2D"), "3d": ("Norm - 3D", "Manual segmentations - 3D")}


def get_derma_octa_data(path: Union[os.PathLike, str], dim: Literal["2d", "3d"] = "2d", download: bool = False) -> str:
    """Download the DERMA-OCTA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        dim: The choice of data dimensionality. Either '2d' (en-face projections) or '3d' (volumes).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if dim not in URLS:
        raise ValueError(f"'{dim}' is not a valid choice of dimensionality. Choose one of {list(URLS.keys())}.")

    os.makedirs(path, exist_ok=True)

    raw_dirname, label_dirname = DIR_NAMES[dim]
    raw_dir = os.path.join(path, raw_dirname)
    label_dir = os.path.join(path, label_dirname)

    if not os.path.exists(raw_dir):
        raw_zip = os.path.join(path, f"Norm_{dim}.zip")
        util.download_source(path=raw_zip, url=URLS[dim]["raw"], download=download, checksum=CHECKSUMS[dim]["raw"])
        util.unzip(zip_path=raw_zip, dst=path)

    if not os.path.exists(label_dir):
        label_zip = os.path.join(path, f"Manual_segmentations_{dim}.zip")
        util.download_source(
            path=label_zip, url=URLS[dim]["labels"], download=download, checksum=CHECKSUMS[dim]["labels"]
        )
        util.unzip(zip_path=label_zip, dst=path)

    return path


def get_derma_octa_paths(
    path: Union[os.PathLike, str],
    dim: Literal["2d", "3d"] = "2d",
    plexus: Literal["all", "sup", "deep"] = "all",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DERMA-OCTA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        dim: The choice of data dimensionality. Either '2d' (en-face projections) or '3d' (volumes).
        plexus: The choice of vascular plexus mask. One of 'all' (full-depth), 'sup' (superficial) or
            'deep' (deep). Only relevant for 'dim=2d': the 3D volumes only ship the 'all' variant.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_derma_octa_data(path, dim, download)
    raw_dirname, label_dirname = DIR_NAMES[dim]

    if dim == "2d":
        label_paths = natsorted(glob(os.path.join(data_dir, label_dirname, "*", f"*_{plexus}.png")))
        raw_paths = [
            os.path.join(data_dir, raw_dirname, os.path.basename(os.path.dirname(p)), os.path.basename(p))
            for p in label_paths
        ]
    else:
        label_paths = natsorted(
            p for p in glob(os.path.join(data_dir, label_dirname, "*", "*.tif")) if re.match(r"^\d+\.tif$", os.path.basename(p))  # noqa
        )
        raw_paths = [
            os.path.join(
                data_dir, raw_dirname, os.path.basename(os.path.dirname(p)),
                f"norm_{os.path.splitext(os.path.basename(p))[0]}.tiff"
            )
            for p in label_paths
        ]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    if dim == "3d":
        # A single case in the archive ('13801') ships a label volume with one extra frame relative
        # to its raw volume (90 vs. 91 slices). Drop any such mismatched pairs defensively.
        import tifffile

        def _shape(p):
            pages = tifffile.TiffFile(p).pages
            return (len(pages),) + pages[0].shape

        filtered = [(r, lb) for r, lb in zip(raw_paths, label_paths) if _shape(r) == _shape(lb)]
        raw_paths, label_paths = map(list, zip(*filtered))

    return raw_paths, label_paths


def get_derma_octa_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    dim: Literal["2d", "3d"] = "2d",
    plexus: Literal["all", "sup", "deep"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DERMA-OCTA dataset for dermatological vessel segmentation in OCTA images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        dim: The choice of data dimensionality. Either '2d' (en-face projections) or '3d' (volumes).
        plexus: The choice of vascular plexus mask. See `get_derma_octa_paths` for details.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_derma_octa_paths(path, dim, plexus, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": dim == "2d"}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=(dim == "3d"),
        patch_shape=patch_shape,
        ndim=2 if dim == "2d" else 3,
        **kwargs
    )


def get_derma_octa_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    dim: Literal["2d", "3d"] = "2d",
    plexus: Literal["all", "sup", "deep"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DERMA-OCTA dataloader for dermatological vessel segmentation in OCTA images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        dim: The choice of data dimensionality. Either '2d' (en-face projections) or '3d' (volumes).
        plexus: The choice of vascular plexus mask. See `get_derma_octa_paths` for details.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_derma_octa_dataset(path, patch_shape, dim, plexus, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
