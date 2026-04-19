"""The KiTS dataset contains annotations for kidney, tumor and cyst segmentation in CT scans.
NOTE: All patients have kidney and tumor annotations (however, not always have cysts annotated).

The label ids are - kidney: 1, tumor: 2, cyst: 3

This dataset is from the KiTS2 Challenge: https://kits-challenge.org/kits23/.
Please cite it if you use this dataset for your research.
"""

import os
import json
import subprocess
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/neheller/kits23"
VALID_SPLITS = ("train", "val", "test")


def get_kits_data(path: Union[os.PathLike, str], download: bool = False, overwrite_splits: bool = False) -> str:
    """Download the KiTS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
        overwrite_splits: Whether to recompute the tumor-size-stratified splits even if split metadata exists.

    Returns:
        The folder where the dataset is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, "preprocessed")
    json_path = os.path.join(path, "splits_kits.json")
    if (
        not overwrite_splits
        and os.path.exists(json_path)
        and os.path.exists(data_dir)
        and all(os.path.exists(os.path.join(data_dir, s)) for s in VALID_SPLITS)
    ):
        return data_dir

    os.makedirs(path, exist_ok=True)

    kits_root = os.path.join(path, "kits23")
    dataset_root = os.path.join(kits_root, "dataset")
    if not os.path.exists(kits_root):
        if not download:
            raise RuntimeError("The dataset is not found and download is set to False.")
        subprocess.run(["git", "clone", URL, kits_root])

    # We install the package-only (with the assumption that the other necessary packages already exists).
    patient_dirs = natsorted(glob(os.path.join(dataset_root, "case*")))
    if not patient_dirs:
        raise RuntimeError(f"No KiTS case folders found in '{dataset_root}'.")

    chosen_patient_dir = patient_dirs[-1]
    if not os.path.exists(os.path.join(chosen_patient_dir, "imaging.nii.gz")):
        if not download:
            raise RuntimeError("The KiTS images are not found and download is set to False.")
        subprocess.run(["pip", "install", "-e", kits_root, "--no-deps"])

        print("The download might take several hours. Make sure you have consistent internet connection.")

        # Run the CLI to download the input images.
        subprocess.run(["kits23_download_data"])

    # Preprocess the images.
    _preprocess_inputs(path, overwrite_splits=overwrite_splits)

    return data_dir


def _make_tumor_size_stratified_splits(patient_dirs, seed=42):
    import nibabel as nib

    patient_dirs = natsorted(patient_dirs)
    cases, tumor_sizes = [], []
    for patient_dir in tqdm(patient_dirs, desc="Computing tumor sizes"):
        labels = np.asarray(nib.load(os.path.join(patient_dir, "segmentation.nii.gz")).dataobj)
        cases.append(str(Path(os.path.basename(patient_dir)).with_suffix(".h5")))
        tumor_sizes.append(int((labels == 2).sum()))

    tumor_sizes = np.asarray(tumor_sizes, dtype="float64")
    if np.all(tumor_sizes == tumor_sizes[0]):
        size_bins = np.zeros(len(tumor_sizes), dtype="int64")
    else:
        edges = np.unique(np.quantile(tumor_sizes, np.linspace(0, 1, 6)[1:-1]))
        size_bins = np.searchsorted(edges, tumor_sizes, side="right").astype("int64")

    records = [
        {"case": case, "tumor_voxels": int(size), "size_bin": int(size_bin)}
        for case, size, size_bin in zip(cases, tumor_sizes, size_bins)
    ]

    def _strata(subset):
        labels = [record["size_bin"] for record in subset]
        counts = {label: labels.count(label) for label in set(labels)}
        return labels if min(counts.values()) >= 2 else None

    train_val, test = train_test_split(
        records, test_size=0.25, random_state=seed, stratify=_strata(records)
    )
    train, val = train_test_split(
        train_val, test_size=0.1, random_state=seed, stratify=_strata(train_val)
    )
    split_records = {"train": train, "val": val, "test": test}
    split_info = {
        split: natsorted(record["case"] for record in records)
        for split, records in split_records.items()
    }

    report = {"seed": seed, "strategy": "tumor_size_stratified", "summary": {}}
    for split, records in split_records.items():
        values = np.asarray([record["tumor_voxels"] for record in records], dtype="float64")
        bins = {}
        for record in records:
            bins[str(record["size_bin"])] = bins.get(str(record["size_bin"]), 0) + 1
        report["summary"][split] = {
            "n_cases": len(records),
            "tumor_voxels_min": int(values.min()),
            "tumor_voxels_median": float(np.median(values)),
            "tumor_voxels_mean": float(values.mean()),
            "tumor_voxels_max": int(values.max()),
            "size_bins": dict(sorted(bins.items())),
        }

    return split_info, report


def _preprocess_inputs(path, overwrite_splits=False):
    patient_dirs = glob(os.path.join(path, "kits23", "dataset", "case*"))

    preprocessed_dir = os.path.join(path, "preprocessed")

    for split in VALID_SPLITS:
        os.makedirs(os.path.join(preprocessed_dir, split), exist_ok=True)

    json_path = os.path.join(path, "splits_kits.json")

    created_new_splits = overwrite_splits or not os.path.exists(json_path)

    if not created_new_splits:
        with open(json_path) as f:
            split_info = json.load(f)
    else:
        split_info, split_report = _make_tumor_size_stratified_splits(patient_dirs, seed=42)
        with open(os.path.join(path, "splits_kits_morphology.json"), "w") as f:
            json.dump(split_report, f, indent=2)

    split_map = {
        os.path.join(path, "kits23", "dataset", Path(fname).stem): split
        for split, fnames in split_info.items()
        for fname in fnames
    }

    for patient_dir in tqdm(patient_dirs, desc="Preprocessing inputs"):
        patient_id = os.path.basename(patient_dir)
        split = split_map[patient_dir]
        patient_fname = Path(patient_id).with_suffix(".h5")
        patient_path = os.path.join(preprocessed_dir, split, patient_fname)

        if not os.path.exists(patient_path):
            for old_split in VALID_SPLITS:
                old_patient_path = os.path.join(preprocessed_dir, old_split, patient_fname)
                if old_patient_path != patient_path and os.path.exists(old_patient_path):
                    os.replace(old_patient_path, patient_path)
                    break

        if os.path.exists(patient_path):
            continue

        # Next, we find all rater annotations.
        kidney_anns = natsorted(glob(os.path.join(patient_dir, "instances", "kidney_instance-1*")))
        tumor_anns = natsorted(glob(os.path.join(patient_dir, "instances", "tumor_instance*")))
        cyst_anns = natsorted(glob(os.path.join(patient_dir, "instances", "cyst_instance*")))

        import h5py
        import nibabel as nib

        with h5py.File(patient_path, "w") as f:
            # Input image.
            raw = nib.load(os.path.join(patient_dir, "imaging.nii.gz")).get_fdata()
            f.create_dataset("raw", data=raw, compression="gzip")

            # Valid segmentation masks for all classes.
            labels = nib.load(os.path.join(patient_dir, "segmentation.nii.gz")).get_fdata()
            assert raw.shape == labels.shape, "The shape of inputs and corresponding segmentation does not match."
            f.create_dataset("labels/all", data=labels, compression="gzip")

            # Add annotations for kidneys per rater.
            _k_exclusive = False
            if not kidney_anns:
                _k_exclusive = True
                kidney_anns = natsorted(glob(os.path.join(patient_dir, "instances", "kidney_instance-2*")))

            assert kidney_anns, f"There must be kidney annotations for '{patient_id}'."
            for p in kidney_anns:
                masks = np.zeros_like(raw)
                rater_id = p[-8]  # The rater count

                # Get the other kidney instance.
                if _k_exclusive:
                    print("The kidney annotations are numbered strangely.")
                    other_p = p.replace("instance-2", "instance-3")
                else:
                    other_p = p.replace("instance-1", "instance-2")

                # Merge both left and right kidney as one semantic id.
                masks[nib.load(p).get_fdata() > 0] = 1
                if os.path.exists(other_p):
                    masks[nib.load(other_p).get_fdata() > 0] = 1
                else:
                    print(f"The second kidney instance does not exist for patient: '{patient_id}'.")

                # Create a hierarchy for the particular rater's kidney annotations.
                f.create_dataset(f"labels/kidney/rater_{rater_id}", data=masks, compression="gzip")

            # Add annotations for tumor per rater.
            assert tumor_anns, f"There must be tumor annotations for '{patient_id}'."
            # Find the raters.
            raters = [p[-8] for p in tumor_anns]
            # Get masks per rater
            unique_raters = np.unique(raters)
            for rater in unique_raters:
                masks = np.zeros_like(raw)
                for p in glob(os.path.join(patient_dir, "instances", f"tumor_instance*-{rater}.nii.gz")):
                    masks[nib.load(p).get_fdata() > 0] = 1

                f.create_dataset(f"labels/tumor/rater_{rater}", data=masks, compression="gzip")

            # Add annotations for cysts per rater.
            if cyst_anns:
                # Find the raters first
                raters = [p[-8] for p in cyst_anns]
                # Get masks per rater
                unique_raters = np.unique(raters)
                for rater in unique_raters:
                    masks = np.zeros_like(raw)
                    for p in glob(os.path.join(patient_dir, "instances", f"cyst_instance*-{rater}.nii.gz")):
                        masks[nib.load(p).get_fdata() > 0] = 1

                    f.create_dataset(f"labels/cyst/rater_{rater}", data=masks, compression="gzip")

    if created_new_splits:
        with open(json_path, "w") as f:
            json.dump(split_info, f, indent=2)


def get_kits_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    download: bool = False,
    overwrite_splits: bool = False,
) -> List[str]:
    """Get paths to the KiTS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: Which data split to use.
        download: Whether to download the data if it is not present.
        overwrite_splits: Whether to recompute the tumor-size-stratified splits even if split metadata exists.

    Returns:
        List of filepaths for the input data.
    """

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}.")

    get_kits_data(path, download, overwrite_splits)

    split_dir = os.path.join(path, "preprocessed", split)
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Split folder '{split_dir}' does not exist.")

    volume_paths = natsorted(glob(os.path.join(split_dir, "*.h5")))
    if not volume_paths:
        raise RuntimeError(f"No .h5 files found in split folder '{split_dir}'.")

    return volume_paths


def get_kits_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    rater: Optional[Literal[1, 2, 3]] = None,
    annotation_choice: Optional[Literal["kidney", "tumor", "cyst"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    overwrite_splits: bool = False,
    **kwargs
) -> Dataset:
    """Get the KiTS dataset for kidney, tumor and cyst segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: Which data split to use.
        rater: The choice of rater.
        annotation_choice: The choice of annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        overwrite_splits: Whether to recompute the tumor-size-stratified splits even if split metadata exists.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_kits_paths(path, split, download, overwrite_splits)

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
    split: Literal["train", "val", "test"],
    rater: Optional[Literal[1, 2, 3]] = None,
    annotation_choice: Optional[Literal["kidney", "tumor", "cyst"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    overwrite_splits: bool = False,
    **kwargs
) -> DataLoader:
    """Get the KiTS dataloader for kidney, tumor and cyst segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: Which data split to use.
        rater: The choice of rater.
        annotation_choice: The choice of annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        overwrite_splits: Whether to recompute the tumor-size-stratified splits even if split metadata exists.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_kits_dataset(
        path, patch_shape, split, rater, annotation_choice, resize_inputs, download, overwrite_splits, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
