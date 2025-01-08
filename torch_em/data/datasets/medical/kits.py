"""The KiTS dataset contains annotations for kidney, tumor and cyst segmentation in CT scans.
NOTE: All patients have kidney and tumor annotations (however, not always have cysts annotated).

The label ids are - kidney: 1, tumor: 2, cyst: 3

This dataset is from the KiTS2 Challenge: https://kits-challenge.org/kits23/.
Please cite it if you use this dataset for your research.
"""

import os
import subprocess
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/neheller/kits23"


def get_kits_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    if not download:
        raise RuntimeError("The dataset is not found and download is set to False.")

    # We clone the environment.
    subprocess.run(["git", "clone", URL, os.path.join(path, "kits23")])

    # # # We install the package-only (with the assumption that the other necessary packages already exists).
    subprocess.run(["pip", "install", "-e", os.path.join(path, "kits23"), "--no-deps"])

    print("The download might take several hours. Make sure you have consistent internet connection.")

    # # Now, we run the CLI.
    subprocess.run(["kits23_download_data"])

    # Preprocess the images.
    _preprocess_inputs(path)

    return data_dir


def _preprocess_inputs(path):
    patient_dirs = glob(os.path.join(path, "kits23", "dataset", "case*"))

    breakpoint()

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for patient_dir in tqdm(patient_dirs, desc="Preprocessing inputs"):
        patient_id = os.path.basename(patient_dir)
        patient_path = os.path.join(preprocessed_dir, Path(patient_id).with_suffix(".h5"))

        # Next, we find all rater annotations.
        kidney_anns = natsorted(glob(os.path.join(patient_dir, "instances", "kidney_instance-1*")))
        tumor_anns = natsorted(glob(os.path.join(patient_dir, "instances", "tumor_instance*")))
        cyst_anns = natsorted(glob(os.path.join(patient_dir, "instances", "cyst_instance*")))

        import h5py
        import nibabel as nib

        with h5py.File(patient_path, "w") as f:
            # Input image.
            raw = nib.load(os.path.join(patient_dir, "imaging.nii.gz")).get_fdata()
            f.create_dataset("raw", data=raw.get_fdata())

            # Valid segmentation masks for all classes.
            labels = nib.load(os.path.join(patient_dir, "segmentation.nii.gz")).get_fdata()
            assert raw.shape == labels.shape, "The shape of inputs and corresponding segmentation does not match."

            f.create_dataset("labels/all", data=labels)

            # Add annotations for kidneys per rater.
            if len(kidney_anns) > 0:
                for p in kidney_anns:
                    masks = np.zeros_like(raw)
                    rater_id = p[-8]  # The rater count

                    # Get the other kidney instance.
                    other_p = p.replace("instance-1", "instance-2")
                    assert os.path.exists(other_p), "The kidney instance does not exist. That is unexpected."

                    # Merge both left and right kidney as one semantic id.
                    masks[nib.load(p).get_fdata() > 0] = 1
                    masks[nib.load(other_p).get_fdata() > 0] = 1

                    # Create a hierarchy for the particular rater's kidney annotations.
                    f.create_dataset(f"labels/kidney/rater_{rater_id}", data=masks)

            # Add annotations for tumor per rater.
            if len(tumor_anns) > 0:
                for p in tumor_anns:
                    masks = np.zeros_like(raw)
                    rater_id = p[-8]
                    masks[nib.load(p).get_fdata() > 0] = 1
                    f.create_dataset(f"labels/tumor/rater_{rater_id}", data=masks)

            # Add annotations for cysts per rater.
            if len(cyst_anns) > 0:
                for p in cyst_anns:
                    masks = np.zeros_like(raw)
                    rater_id = p[-8]
                    masks[nib.load(p).get_fdata() > 0] = 1
                    f.create_dataset(f"labels/cyst/rater_{rater_id}", data=masks)


def get_kits_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """
    """
    data_dir = get_kits_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_kits_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    rater: Optional[Literal[1, 2, 3]] = None,
    annotation_choice: Optional[Literal["kidney", "tumor", "cyst"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    volume_paths = get_kits_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    # TODO: simplify the design below later, to allow:
    # - multi-rater label loading.
    # - multi-annotation label loading.
    # (for now, only 1v1 annotation-rater loading is supported).
    if rater is None and annotation_choice is None:
        label_key = "labels/all"
    else:
        assert rater is not None and annotation_choice is not None, \
            "Both rater and annotation_choice must be specified together."

        label_key = f"labels/{annotation_choice}/rater_{rater}"

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs
    )


def get_kits_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    rater: Optional[Literal[1, 2, 3]] = None,
    annotation_choice: Optional[Literal["kidney", "tumor", "cyst"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_kits_dataset(path, patch_shape, rater, annotation_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
