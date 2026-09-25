"""This dataset contains cell instance segmentation annotations for imaging mass cytometry (IMC)
of formalin-fixed paraffin-embedded (FFPE) mouse lung sections, imaged to study the relationship
between cellular organisation and extracellular matrix (ECM) composition during allergic airway
inflammation.

The data is from the publication "Extracellular matrix phenotyping by imaging mass cytometry
defines distinct cellular matrix environments associated with allergic airway inflammation" and
hosted on the BioImage Archive at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3184.
It is available under the CC BY 4.0 license. Please cite it if you use this dataset in your
research.

This loader covers the "Hyperion Imaging of Stained Allergic Mouse Lung" study component, which
provides 36 acquisitions (3 slides, 12 ROIs each) as processed multichannel TIFF stacks (76
channels, covering a 42-marker antibody panel plus derived distance-transform channels) with
matching per-cell instance segmentation masks generated with DeepCell. Note that, despite the
"ECM phenotyping" framing of the study, the ECM channels themselves are only provided as
continuous per-pixel distance-transform maps (see the "dist" folder on the BioImage Archive),
not as discrete class-labeled ECM masks; the only raster segmentation masks in this study are
the generic per-cell instance masks used here as the label target. The study also hosts two
further "removable" components (immunofluorescence slide scans and confocal imaging of precision
cut lung slices) with only geometrical (QuPath / arivis) region annotations rather than instance
or semantic masks; these are out of scope for this loader.
"""

import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/184/S-BIAD3184/Files/data_upload/processed_data"

SLIDES = ("slide1", "slide2", "slide3")
ROIS = tuple(f"{i:03d}" for i in range(1, 13))


def get_ecm_phenotyping_data(
    path: Union[os.PathLike, str],
    slides: Optional[Sequence[str]] = None,
    download: bool = False,
) -> str:
    """Download the ECM phenotyping (allergic mouse lung IMC) data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        slides: The slide names to prepare. By default all three slides (36 acquisitions in
            total) are prepared, which requires downloading several gigabytes of data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the data is stored.
    """
    if slides is None:
        slides = SLIDES
    else:
        invalid = sorted(set(slides) - set(SLIDES))
        if invalid:
            raise ValueError(f"Invalid slide name(s) {invalid}. Choose from {SLIDES}.")

    raw_dir = os.path.join(path, "img")
    label_dir = os.path.join(path, "masks")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for slide in slides:
        for roi in ROIS:
            fname = f"m16263_{slide}_{roi}.tiff"
            raw_path = os.path.join(raw_dir, fname)
            label_path = os.path.join(label_dir, fname)
            util.download_source(raw_path, f"{BASE_URL}/img/{fname}", download, checksum=None)
            util.download_source(label_path, f"{BASE_URL}/masks/{fname}", download, checksum=None)

    return path


def get_ecm_phenotyping_paths(
    path: Union[os.PathLike, str],
    slides: Optional[Sequence[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ECM phenotyping images and cell instance segmentation masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        slides: The slide names to load. By default all three slides are loaded.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding mask paths.
    """
    root = get_ecm_phenotyping_data(path, slides, download)

    raw_paths = sorted(glob(os.path.join(root, "img", "*.tiff")))
    label_paths = sorted(glob(os.path.join(root, "masks", "*.tiff")))

    missing_paths = [p for p in raw_paths + label_paths if not os.path.exists(p)]
    if missing_paths:
        raise RuntimeError(f"Could not find {len(missing_paths)} ECM phenotyping files.")

    return raw_paths, label_paths


def get_ecm_phenotyping_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    slides: Optional[Sequence[str]] = None,
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the ECM phenotyping dataset for cell instance segmentation in imaging mass cytometry.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        slides: The slide names to load. By default all three slides are loaded.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The ECM phenotyping patch shape must be two-dimensional, got {patch_shape}.")

    raw_paths, label_paths = get_ecm_phenotyping_paths(path, slides, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_ecm_phenotyping_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    slides: Optional[Sequence[str]] = None,
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ECM phenotyping dataloader for cell instance segmentation in imaging mass cytometry.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        batch_size: The batch size for training.
        slides: The slide names to load. By default all three slides are loaded.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for
            the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ecm_phenotyping_dataset(
        path, patch_shape, slides=slides, download=download, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
