"""The WAW-TACE dataset contains annotations for liver tumors in multiphase CT scans.

The dataset consists of multiphase abdominal CT scans of 233 treatment-naive patients with hepatocellular
carcinoma that were treated with transarterial chemoembolization. The tumors of one phase per patient are
delineated by hand, giving 378 masks. The masks are stored as instance labels, so that the tumors of a
scan get the ids 1 to n.

NOTE: The release also holds masks of several internal organs, but those were generated with
TotalSegmentator rather than drawn by hand, so they are predictions and not annotations and are not
provided here. `medical.totalsegmentator` provides the data that model was trained on.

NOTE: This requires the pynrrd python package to read the tumor masks.

The dataset is located at https://doi.org/10.5281/zenodo.12741586 and is distributed under the
CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1148/ryai.240296.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/12741586/files/{filename}?download=1"

SCAN_ARCHIVES = [f"ct_scans_{index}_4_wawtace_09_05_24.zip" for index in range(1, 5)]

TUMOR_ARCHIVE = "tumor_masks_wawtace_v1_08_05_2024.zip"

CHECKSUMS = {TUMOR_ARCHIVE: "39268ef2899cb5ccd422950d7f352a173c0f54071a5e2775157501e4756dff1e"}


def _mask_geometry(mask_path):
    """The shape, origin and spacing of a mask, with the signs of its LPS origin dropped."""
    import nrrd

    header = nrrd.read_header(mask_path)
    origin = np.abs(np.array([float(value) for value in header["space origin"]]))
    spacing = np.abs(np.diag(np.array(header["space directions"], dtype="float64")))
    return tuple(int(size) for size in header["sizes"]), tuple(np.round(origin, 2)), tuple(np.round(spacing, 3))


def _scan_geometry(image_path):
    """The same geometry for a scan. Its nifti axes are LAS, so the signs of the origin differ from the mask."""
    import nibabel as nib

    image = nib.load(image_path)
    affine = image.affine
    return (
        tuple(image.shape),
        tuple(np.round(np.abs(affine[:3, 3]), 2)),
        tuple(np.round(np.abs(np.diag(affine[:3, :3])), 3)),
    )


def _find_scan(data_dir, patient_id, phase, mask_path):
    """Find the scan a mask was drawn on.

    The phase in the name of a mask is not always the phase of the scan it belongs to, so the geometry
    decides. The named phase is preferred, because the phases of a study often share their geometry and
    the match would otherwise be ambiguous.
    """
    geometry = _mask_geometry(mask_path)
    named = os.path.join(data_dir, patient_id, f"{patient_id}_{phase}_scan.nii.gz")
    if os.path.exists(named) and _scan_geometry(named) == geometry:
        return named

    matches = [
        path for path in natsorted(glob(os.path.join(data_dir, patient_id, f"{patient_id}_*_scan.nii.gz")))
        if _scan_geometry(path) == geometry
    ]
    return matches[0] if len(matches) == 1 else None


def _preprocess_waw_tace(data_dir, tumor_dir, preprocessed_dir):
    import h5py
    import nrrd
    import nibabel as nib

    # The tumors of a scan are stored one per file, as '<patient>_<phase>_<tumor>_tumor_seg.nrrd'.
    tumors = defaultdict(list)
    for mask_path in natsorted(glob(os.path.join(tumor_dir, "*", "*_tumor_seg.nrrd"))):
        patient_id, phase = os.path.basename(mask_path).split("_")[:2]
        tumors[(patient_id, phase)].append(mask_path)

    os.makedirs(preprocessed_dir, exist_ok=True)
    for (patient_id, phase), mask_paths in tqdm(sorted(tumors.items()), desc="Preprocess WAW-TACE"):
        out_path = os.path.join(preprocessed_dir, f"{patient_id}_{phase}.h5")
        if os.path.exists(out_path):
            continue

        image_path = _find_scan(data_dir, patient_id, phase, mask_paths[0])
        if image_path is None:
            continue

        image = nib.load(image_path)
        # The scans and their masks are stored with axis order (x, y, z) and are transposed to (z, y, x).
        volume = np.asarray(image.dataobj).transpose(2, 1, 0)
        labels = np.zeros(volume.shape, dtype="uint8")
        for instance_id, mask_path in enumerate(mask_paths, start=1):
            mask, _ = nrrd.read(mask_path)
            mask = mask.transpose(2, 1, 0)
            if mask.shape != volume.shape:
                continue
            labels[mask > 0] = instance_id

        if labels.max() == 0:
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_waw_tace_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the WAW-TACE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and glob(os.path.join(preprocessed_dir, "*.h5")):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)
    data_dir = os.path.join(path, "scans")
    for filename in SCAN_ARCHIVES:
        zip_path = os.path.join(path, filename)
        util.download_source(path=zip_path, url=URL.format(filename=filename), download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    tumor_dir = os.path.join(path, TUMOR_ARCHIVE[:-len(".zip")])
    if not os.path.exists(tumor_dir):
        zip_path = os.path.join(path, TUMOR_ARCHIVE)
        util.download_source(
            path=zip_path, url=URL.format(filename=TUMOR_ARCHIVE), download=download,
            checksum=CHECKSUMS[TUMOR_ARCHIVE],
        )
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    _preprocess_waw_tace(data_dir, tumor_dir, preprocessed_dir)
    return preprocessed_dir


def get_waw_tace_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the WAW-TACE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_waw_tace_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_waw_tace_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the WAW-TACE dataset for liver tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_waw_tace_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_waw_tace_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the WAW-TACE dataloader for liver tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_waw_tace_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
