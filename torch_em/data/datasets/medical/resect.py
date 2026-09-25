"""The RESECT dataset contains pre-operative MRI and intra-operative 3D ultrasound of 23 patients
with low-grade gliomas, together with annotations from the RESECT-SEG extension.

For every patient, three intra-operative ultrasound (US) volumes were acquired: before, during and after
tumor resection. The annotations (RESECT-SEG, an extension of the CuRIOUS 2022 challenge labels) are:
- 'tumor': the tumor in the US volume before resection and in the pre-operative FLAIR MRI (23 cases each),
- 'resection': the resection cavity in the US volumes during (21 cases) and after (22 cases) resection,
- 'sulci': the cerebral sulci in all three US phases (23 cases each),
- 'falx': the cerebral falx in the US volumes (7 to 8 cases per phase).
All labels are binary (0: background, 1: structure).

The image volumes are located at https://doi.org/10.11582/2017.00004 (NIRD research data archive, CC BY 4.0)
and the annotations at https://osf.io/jv8bk/ (CC BY-NC-SA 4.0). Only the volumes needed for the chosen
source, phase and structure are downloaded.

This dataset is from the publications https://doi.org/10.1002/mp.12268 (RESECT) and
https://doi.org/10.1002/mp.17317 (RESECT-SEG annotations).
Please cite them if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


IMAGE_URL = "https://data.archive.sigma2.no/dataset/5686d8fa-2003-4837-8e66-8e887fabe21e/download/RESECT/NIFTI"
# The folder zip is generated on-the-fly by OSF, hence the checksum of the archive is not reliable.
LABEL_URL = "https://files.osf.io/v1/resources/jv8bk/providers/osfstorage/64cd20819cbf033b051e46c8/?zip="

CASE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27]

STRUCTURES = {
    "before": ["tumor", "sulci", "falx"],
    "during": ["resection", "sulci", "falx"],
    "after": ["resection", "sulci", "falx"],
    "MRI": ["tumor"],
}


def _get_structure(source, phase, structure):
    if source not in ["US", "MRI"]:
        raise ValueError(f"'{source}' is not a valid source. Choose either 'US' or 'MRI'.")
    if source == "US" and phase not in ["before", "during", "after"]:
        raise ValueError(f"'{phase}' is not a valid phase. Choose one of 'before', 'during' or 'after'.")

    valid_structures = STRUCTURES["MRI" if source == "MRI" else phase]
    if structure is None:
        structure = valid_structures[0]
    if structure not in valid_structures:
        raise ValueError(
            f"'{structure}' is not a valid structure for source '{source}' and phase '{phase}'. "
            f"Choose one of {valid_structures}."
        )
    return structure


def get_resect_data(
    path: Union[os.PathLike, str],
    source: Literal["US", "MRI"] = "US",
    phase: Literal["before", "during", "after"] = "before",
    download: bool = False,
) -> Tuple[str, str]:
    """Download the RESECT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The imaging source. Either 'US' (intra-operative ultrasound) or 'MRI' (pre-operative FLAIR).
        phase: The surgical phase of the ultrasound volumes. Either 'before', 'during' or 'after' resection.
            Ignored for the 'MRI' source.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the image volumes.
        Filepath to the folder with the label volumes.
    """
    _get_structure(source, phase, None)

    label_dir = os.path.join(path, "labels")
    if not os.path.exists(label_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "RESECT-Segmentation.zip")
        util.download_source(path=zip_path, url=LABEL_URL, download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=label_dir)

    image_dir = os.path.join(path, "images", source)
    os.makedirs(image_dir, exist_ok=True)
    for case_id in CASE_IDS:
        fname = f"Case{case_id}-US-{phase}.nii.gz" if source == "US" else f"Case{case_id}-FLAIR.nii.gz"
        url = f"{IMAGE_URL}/Case{case_id}/{source}/{fname}"
        util.download_source(path=os.path.join(image_dir, fname), url=url, download=download, checksum=None)

    return image_dir, label_dir


def get_resect_paths(
    path: Union[os.PathLike, str],
    source: Literal["US", "MRI"] = "US",
    phase: Literal["before", "during", "after"] = "before",
    structure: Optional[Literal["tumor", "resection", "sulci", "falx"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RESECT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The imaging source. Either 'US' (intra-operative ultrasound) or 'MRI' (pre-operative FLAIR).
        phase: The surgical phase of the ultrasound volumes. Either 'before', 'during' or 'after' resection.
            Ignored for the 'MRI' source.
        structure: The annotated structure. One of 'tumor' (US before resection and MRI), 'resection'
            (resection cavity, US during and after resection), 'sulci' or 'falx' (US, all phases).
            By default, 'tumor' for 'before' and 'MRI' and 'resection' for 'during' and 'after'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    structure = _get_structure(source, phase, structure)
    image_dir, label_dir = get_resect_data(path, source, phase, download)

    prefix = f"US-{phase}" if source == "US" else "FLAIR"
    label_paths = natsorted(glob(os.path.join(label_dir, "Case*", f"Case*-{prefix}-{structure}.nii.gz")))
    raw_paths = [
        os.path.join(image_dir, os.path.basename(p).replace(f"-{structure}.nii.gz", ".nii.gz")) for p in label_paths
    ]
    assert all(os.path.exists(p) for p in raw_paths), "Some image volumes are missing."
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_resect_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Literal["US", "MRI"] = "US",
    phase: Literal["before", "during", "after"] = "before",
    structure: Optional[Literal["tumor", "resection", "sulci", "falx"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RESECT dataset for brain tumor, resection cavity, sulci and falx segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        source: The imaging source. Either 'US' (intra-operative ultrasound) or 'MRI' (pre-operative FLAIR).
        phase: The surgical phase of the ultrasound volumes. Either 'before', 'during' or 'after' resection.
            Ignored for the 'MRI' source.
        structure: The annotated structure. One of 'tumor' (US before resection and MRI), 'resection'
            (resection cavity, US during and after resection), 'sulci' or 'falx' (US, all phases).
            By default, 'tumor' for 'before' and 'MRI' and 'resection' for 'during' and 'after'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_resect_paths(path, source, phase, structure, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_resect_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    source: Literal["US", "MRI"] = "US",
    phase: Literal["before", "during", "after"] = "before",
    structure: Optional[Literal["tumor", "resection", "sulci", "falx"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RESECT dataloader for brain tumor, resection cavity, sulci and falx segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The imaging source. Either 'US' (intra-operative ultrasound) or 'MRI' (pre-operative FLAIR).
        phase: The surgical phase of the ultrasound volumes. Either 'before', 'during' or 'after' resection.
            Ignored for the 'MRI' source.
        structure: The annotated structure. One of 'tumor' (US before resection and MRI), 'resection'
            (resection cavity, US during and after resection), 'sulci' or 'falx' (US, all phases).
            By default, 'tumor' for 'before' and 'MRI' and 'resection' for 'during' and 'after'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_resect_dataset(path, patch_shape, source, phase, structure, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
