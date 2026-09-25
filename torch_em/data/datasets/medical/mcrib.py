"""The M-CRIB dataset contains cortical and subcortical parcellations of neonatal brain MRI.

The dataset consists of 10 healthy term-born neonates (scanned at 40-43 weeks gestational age) with T2-weighted
and T1-weighted MRI (the T1 volumes are provided registered to the T2 volumes) and manual parcellations
following the M-CRIB 2.0 protocol, which is compatible with the adult Desikan-Killiany cortical atlas
and the FreeSurfer subcortical labels.

NOTE: The label volumes are semantic parcellations with FreeSurfer-style ids (94 structures per volume):
- subcortical structures use the FreeSurfer ids (e.g. 2: Left-Cerebral-White-Matter, 4: Left-Lateral-Ventricle,
  9: Left-Thalamus, 17: Left-Hippocampus, 24: CSF, 41: Right-Cerebral-White-Matter, 170: brainstem,
  192: Corpus_Callosum, and the cerebellar labels 75, 76, 90, 91, 93),
- the left hemisphere cortical regions use the ids 1000-1035 (e.g. 1002: ctx-lh-caudalanteriorcingulate),
- the right hemisphere cortical regions use the ids 2000-2035 (e.g. 2002: ctx-rh-caudalanteriorcingulate).
The complete lookup table (id, RGB color, name) is shipped with the data in
'M-CRIB_2-0_labels_itk_format.txt', which is downloaded next to the volumes.

The dataset is located at https://osf.io/4vthr/.

This dataset is from the publications https://doi.org/10.1038/sdata.2017.57 (M-CRIB) and
https://doi.org/10.3389/fnins.2019.00034 (M-CRIB 2.0).
Please cite them if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# The folder zips are generated on-the-fly by OSF, hence the checksums of the archives are not reliable.
URLS = {
    "T2": "https://files.osf.io/v1/resources/4vthr/providers/osfstorage/5d36b7efa667db0019f9dbc9/?zip=",
    "T1": "https://files.osf.io/v1/resources/4vthr/providers/osfstorage/5d37bbe0a667db0018fc7ab0/?zip=",
    "labels": "https://files.osf.io/v1/resources/4vthr/providers/osfstorage/5d36b267251f0e0017091695/?zip=",
    "lookup_table": "https://osf.io/download/9u42m/",
}

CHECKSUMS = {
    "T2": None,
    "T1": None,
    "labels": None,
    "lookup_table": "bcd373c68399220a8884606b4abfd0df474edcb7522c1b292fa1b60f378b0505",
}


def get_mcrib_data(path: Union[os.PathLike, str], modality: Literal["T2", "T1"] = "T2", download: bool = False):
    """Download the M-CRIB dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The MRI modality. Either 'T2' or 'T1' (the T1 volumes registered to the T2 volumes).
        download: Whether to download the data if it is not present.
    """
    if modality not in ["T2", "T1"]:
        raise ValueError(f"'{modality}' is not a valid modality. Choose either 'T2' or 'T1'.")

    os.makedirs(path, exist_ok=True)

    for name in [modality, "labels"]:
        data_dir = os.path.join(path, name)
        if os.path.exists(data_dir):
            continue

        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=data_dir)

    lut_path = os.path.join(path, "M-CRIB_2-0_labels_itk_format.txt")
    util.download_source(path=lut_path, url=URLS["lookup_table"], download=download, checksum=CHECKSUMS["lookup_table"])


def get_mcrib_paths(
    path: Union[os.PathLike, str], modality: Literal["T2", "T1"] = "T2", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the M-CRIB data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The MRI modality. Either 'T2' or 'T1' (the T1 volumes registered to the T2 volumes).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_mcrib_data(path, modality, download)

    label_paths = natsorted(glob(os.path.join(path, "labels", "M-CRIB_2-0_P*_parc.nii.gz")))
    suffix = "T2" if modality == "T2" else "T1_registered_to_T2"
    raw_paths = [
        os.path.join(path, modality, os.path.basename(p).replace("_2-0", "").replace("parc", suffix))
        for p in label_paths
    ]
    assert all(os.path.exists(p) for p in raw_paths), "Some image volumes are missing."
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_mcrib_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["T2", "T1"] = "T2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the M-CRIB dataset for neonatal brain parcellation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI modality. Either 'T2' or 'T1' (the T1 volumes registered to the T2 volumes).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mcrib_paths(path, modality, download)

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


def get_mcrib_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["T2", "T1"] = "T2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the M-CRIB dataloader for neonatal brain parcellation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI modality. Either 'T2' or 'T1' (the T1 volumes registered to the T2 volumes).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mcrib_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
