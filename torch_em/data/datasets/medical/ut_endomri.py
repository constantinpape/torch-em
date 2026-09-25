"""The UT-EndoMRI dataset contains annotations for pelvic organ segmentation in multi-sequence
MRI of endometriosis patients.

The dataset was collected at two clinical institutions. The 'D1' cohort (51 patients, T2-weighted
and T1-weighted fat-suppressed sequences) has uterus, ovary, endometrioma, cyst and cul-de-sac
structures independently contoured by up to 3 raters. The 'D2' cohort (82 patients, T1-weighted,
T1-weighted fat-suppressed, T2-weighted and T2-weighted fat-suppressed sequences) has uterus,
ovary and endometrioma structures contoured by a single rater (no rater argument applies to it).
Not every sequence or structure is available for every patient, and a label is not tied to a
specific sequence in its filename (it may have been contoured on a different sequence than the
one requested); `get_ut_endomri_paths` only returns pairs where the requested sequence and
structure annotation both exist and have matching volume shapes.

The data is located at https://doi.org/10.5281/zenodo.13749613. There is no structured license;
the record's user agreement states: "The UT-EndoMRI dataset is available for free use exclusively
in non-commercial scientific research."

This dataset is from the publication "A Multi-Modal Pelvic MRI Dataset for Deep Learning-Based
Pelvic Organ Segmentation in Endometriosis" (Liang et al., submitted).
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/13749613/files/UT-EndoMRI.zip"
CHECKSUM = "7ab4f9d758c5a2692d78ddf19c35c20b58d77f670a1ec3abb782d6969e278096"

DATASETS = {"D1": "D1_MHS", "D2": "D2_TCPW"}
SEQUENCES = ("T1", "T1FS", "T2", "T2FS")
STRUCTURES = ("ut", "ov", "em", "cy", "cds")


def get_ut_endomri_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the UT-EndoMRI dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "UT-EndoMRI")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "UT-EndoMRI.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_ut_endomri_paths(
    path: Union[os.PathLike, str],
    dataset: Literal["D1", "D2"] = "D2",
    sequence: str = "T2",
    structure: str = "ut",
    rater: Optional[Literal[1, 2, 3]] = 3,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the UT-EndoMRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        dataset: The choice of cohort. Either 'D1' (Memorial Hermann Hospital System) or
            'D2' (Texas Children's Hospital Pavilion for Women).
        sequence: The choice of MRI sequence. One of 'T1', 'T1FS', 'T2', 'T2FS'.
        structure: The choice of anatomical structure. One of 'ut' (uterus), 'ov' (ovary),
            'em' (endometrioma), 'cy' (cyst), 'cds' (cul-de-sac). Not every structure is
            annotated for every patient of either cohort.
        rater: The choice of annotator for the 'D1' cohort. One of 1, 2 or 3. Ignored for
            the 'D2' cohort, which has a single rater.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if dataset not in DATASETS:
        raise ValueError(f"'{dataset}' is not a valid dataset. Choose one of {list(DATASETS)}.")
    if sequence not in SEQUENCES:
        raise ValueError(f"'{sequence}' is not a valid sequence. Choose one of {SEQUENCES}.")
    if structure not in STRUCTURES:
        raise ValueError(f"'{structure}' is not a valid structure. Choose one of {STRUCTURES}.")

    import nibabel as nib

    data_dir = get_ut_endomri_data(path, download)
    cohort_dir = os.path.join(data_dir, DATASETS[dataset])

    label_suffix = f"_{structure}_r{rater}.nii.gz" if dataset == "D1" else f"_{structure}.nii.gz"
    label_paths = natsorted(glob(os.path.join(cohort_dir, "*", f"*{label_suffix}")))

    raw_paths = []
    matched_label_paths = []
    for label_path in label_paths:
        patient_dir = os.path.dirname(label_path)
        patient_id = os.path.basename(label_path)[: -len(label_suffix)]
        raw_path = os.path.join(patient_dir, f"{patient_id}_{sequence}.nii.gz")
        if not os.path.exists(raw_path):
            continue
        # A label is not tied to a specific sequence in its filename, so it may have been
        # contoured on a different sequence than the one requested here; skip such mismatches.
        if nib.load(raw_path).shape != nib.load(label_path).shape:
            continue
        raw_paths.append(raw_path)
        matched_label_paths.append(label_path)

    assert len(raw_paths) == len(matched_label_paths) and len(raw_paths) > 0
    return raw_paths, matched_label_paths


def get_ut_endomri_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    dataset: Literal["D1", "D2"] = "D2",
    sequence: str = "T2",
    structure: str = "ut",
    rater: Optional[Literal[1, 2, 3]] = 3,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the UT-EndoMRI dataset for pelvic organ segmentation in endometriosis MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        dataset: The choice of cohort. Either 'D1' or 'D2'.
        sequence: The choice of MRI sequence. One of 'T1', 'T1FS', 'T2', 'T2FS'.
        structure: The choice of anatomical structure. One of 'ut', 'ov', 'em', 'cy', 'cds'.
        rater: The choice of annotator for the 'D1' cohort. Ignored for the 'D2' cohort.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ut_endomri_paths(path, dataset, sequence, structure, rater, download)

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


def get_ut_endomri_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    dataset: Literal["D1", "D2"] = "D2",
    sequence: str = "T2",
    structure: str = "ut",
    rater: Optional[Literal[1, 2, 3]] = 3,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the UT-EndoMRI dataloader for pelvic organ segmentation in endometriosis MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        dataset: The choice of cohort. Either 'D1' or 'D2'.
        sequence: The choice of MRI sequence. One of 'T1', 'T1FS', 'T2', 'T2FS'.
        structure: The choice of anatomical structure. One of 'ut', 'ov', 'em', 'cy', 'cds'.
        rater: The choice of annotator for the 'D1' cohort. Ignored for the 'D2' cohort.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset_ = get_ut_endomri_dataset(
        path, patch_shape, dataset, sequence, structure, rater, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset_, batch_size, **loader_kwargs)
