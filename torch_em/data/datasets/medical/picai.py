"""The PI-CAI dataset contains annotations for clinically significant prostate cancer and the prostate
gland in biparametric MRI.

The dataset consists of 1500 biparametric MRI scans of 1476 patients from three centers. Three sets of
annotations are provided and selected with the 'annotation' argument: 'lesion_expert' are the csPCa
lesions delineated by human experts for 1295 scans, with the ISUP grade as the label id, 'lesion_ai' are
the AI derived csPCa lesions for all 1500 scans, with binary labels, and 'whole_gland' is the AI derived
prostate mask for all 1500 scans. See also `ANNOTATIONS`.

NOTE: Only the axial T2 weighted scan is used. The annotations are resampled to its grid, while the
diffusion weighted scans of the same study are acquired on a much coarser grid and do not align with
them.

NOTE: This requires the SimpleITK python package to read the MetaImage (.mha) scans.

The images are located at https://doi.org/10.5281/zenodo.6624726 and the annotations at
https://github.com/DIAGNijmegen/picai_labels. Both are distributed under the CC BY-NC 4.0 license.
This dataset is from the publication https://doi.org/10.1016/S1470-2045(24)00220-1.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/6624726/files/picai_public_images_fold{fold}.zip?download=1"

CHECKSUMS = {
    0: "1c9683b436bedbe4384bb4a8133ebb4213349e5af2296ba62cd26fafa36366e0",
    1: "eb31d43a949e9245385cea5e66f4584197bbdb7df4ee99af6e47da980ddbb565",
    2: "5116355bc0cbf152122467405a0e9636c268e46a144f0f5de51abb48753e9d29",
    3: "070f172b5668e9a07900d9f07a48e0a313067e63280140ca0c43a6070487e765",
    4: "feabdbf8bb2d28c091dc41fcd66cbd4c2a898aaee05ca27401ccfc343ccec016",
}

LABELS_URL = "https://github.com/DIAGNijmegen/picai_labels/archive/refs/heads/main.zip"

ANNOTATIONS = {
    "lesion_expert": "csPCa_lesion_delineations/human_expert/resampled",
    "lesion_ai": "csPCa_lesion_delineations/AI/Bosma22a",
    "whole_gland": "anatomical_delineations/whole_gland/AI/Bosma22b",
}
"""Mapping from the annotation choice to its folder in the annotation release."""

FOLDS = (0, 1, 2, 3, 4)


def _preprocess_picai(image_dir, label_root, preprocessed_dir):
    import h5py
    import nibabel as nib
    import SimpleITK as sitk

    os.makedirs(preprocessed_dir, exist_ok=True)
    image_paths = natsorted(glob(os.path.join(image_dir, "*", "*_t2w.mha")))
    for image_path in tqdm(image_paths, desc="Preprocess PI-CAI"):
        case_id = os.path.basename(image_path)[:-len("_t2w.mha")]
        out_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(out_path):
            continue

        # SimpleITK returns the volume with axis order (z, y, x).
        volume = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            for name, rel_path in ANNOTATIONS.items():
                label_path = os.path.join(label_root, rel_path, f"{case_id}.nii.gz")
                if not os.path.exists(label_path):
                    continue
                # The annotations are stored as nifti with axis order (x, y, z).
                labels = np.asarray(nib.load(label_path).dataobj).T
                assert labels.shape == volume.shape, f"The '{name}' mask of '{case_id}' does not match its scan."
                f.create_dataset(f"labels/{name}", data=labels.astype("uint8"), compression="gzip")


def get_picai_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PI-CAI dataset.

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

    image_dir = os.path.join(path, "images")
    for fold in FOLDS:
        zip_path = os.path.join(path, f"picai_public_images_fold{fold}.zip")
        util.download_source(
            path=zip_path, url=URL.format(fold=fold), download=download, checksum=CHECKSUMS[fold]
        )
        util.unzip(zip_path=zip_path, dst=image_dir, remove=False)

    label_root = os.path.join(path, "picai_labels-main")
    if not os.path.exists(label_root):
        zip_path = os.path.join(path, "picai_labels.zip")
        util.download_source(path=zip_path, url=LABELS_URL, download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    _preprocess_picai(image_dir, label_root, preprocessed_dir)
    return preprocessed_dir


def get_picai_paths(
    path: Union[os.PathLike, str],
    annotation: Literal["lesion_expert", "lesion_ai", "whole_gland"] = "lesion_ai",
    download: bool = False,
) -> List[str]:
    """Get paths to the PI-CAI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotations. Either 'lesion_expert', 'lesion_ai' or 'whole_gland'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    import h5py

    if annotation not in ANNOTATIONS:
        raise ValueError(f"'{annotation}' is not a valid annotation. Choose from {list(ANNOTATIONS.keys())}.")

    preprocessed_dir = get_picai_data(path, download)

    # The human expert lesions are only delineated for a subset of the scans.
    volume_paths = []
    for volume_path in natsorted(glob(os.path.join(preprocessed_dir, "*.h5"))):
        with h5py.File(volume_path, "r") as f:
            if f"labels/{annotation}" in f:
                volume_paths.append(volume_path)

    assert len(volume_paths) > 0, f"Could not find any volume with '{annotation}' labels in '{preprocessed_dir}'."
    return volume_paths


def get_picai_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation: Literal["lesion_expert", "lesion_ai", "whole_gland"] = "lesion_ai",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PI-CAI dataset for prostate cancer and prostate gland segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. Either 'lesion_expert', 'lesion_ai' or 'whole_gland'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_picai_paths(path, annotation, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{annotation}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_picai_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation: Literal["lesion_expert", "lesion_ai", "whole_gland"] = "lesion_ai",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PI-CAI dataloader for prostate cancer and prostate gland segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. Either 'lesion_expert', 'lesion_ai' or 'whole_gland'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_picai_dataset(path, patch_shape, annotation, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
