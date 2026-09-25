"""The MVAA dataset contains annotations for mitral valve anatomy analysis across three imaging
modalities that reflect the clinical workflow from diagnosis to intervention: preoperative cardiac
CT (annular geometry), intraoperative 3D transesophageal echocardiography (3D TEE, leaflet
morphology) and surgical video (leaflet and instrument segmentation).

The data was curated for the MVAA 2026 challenge (Mitral Valve Anatomy Analysis Using Multimodal
Imaging Data), held together with MICCAI 2026 (https://www.codabench.org/competitions/15662/). The
released training data consists of three genuinely separate distributions, exposed by this module
as three independent sets of functions:
- CT (`get_mvaa_ct_*`): 27 annotated volumes with a binary annulus mask, plus 1040 additional
  unlabeled volumes that are not exposed by this module.
- 3D TEE (`get_mvaa_tee_*`): 105 annotated ultrasound volumes with a 3-class label map (the exact
  semantics of the two foreground classes are not documented in the release).
- Surgical video (`get_mvaa_video_*`): 180 annotated RGB frames (30 frames each from 6 recordings),
  with polygon-derived instance masks for up to 17 anatomy and instrument classes, see
  `VIDEO_LABEL_IDS`. The mitral valve itself is class 10.

NOTE: the validation split of each modality is released without ground truth (held out for the
official challenge evaluation), so this module only exposes the labeled training splits.

The training data is distributed via Google Drive at
https://drive.google.com/file/d/14WneBUBZ1X4p69tRdRzximNb0IsWuh2B/view, as announced by the
organizers, see
https://communities.springernature.com/posts/miccai-2026-challenge-on-mitral-valve-multimodal-anatomical-analysis-challenge.

This dataset is from https://doi.org/10.5281/zenodo.19726755.
Please cite it if you use this dataset in your research.
"""

import os
import tarfile
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://drive.google.com/uc?id=14WneBUBZ1X4p69tRdRzximNb0IsWuh2B"
CHECKSUM = "62695c24b269ae10962bd73bc6062ccb5a56bea0125e18325943b32f9e3a9bbf"

CT_LABEL_IDS = {"background": 0, "annulus": 1}

N_CT_VOLUMES = 27
N_TEE_VOLUMES = 105
N_VIDEO_FRAMES = 180

VIDEO_LABEL_IDS = {
    "background": 0, "atrial_retractor": 1, "dissecting_forceps": 2, "scissors": 3, "needle_holder": 4,
    "sharp_knife": 5, "suture_organizer": 6, "suture": 7, "needle": 8, "atrial_inner_surface": 9,
    "mitral_valve": 10, "ventricle": 11, "blood": 12, "irrelevant": 13, "prosthetic_valve": 14,
    "annuloplasty_ring": 15, "gasket": 16, "valve_sizer": 17,
}


def get_mvaa_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MVAA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the raw 'reference_data' release.
    """
    data_dir = os.path.join(path, "reference_data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "mvaa_train.zip")
    util.download_source_gdrive(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _preprocess_ct(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    image_dir = os.path.join(data_dir, "t1_ct", "train", "labeled", "images")
    label_dir = os.path.join(data_dir, "t1_ct", "train", "labeled", "labels")
    image_paths = natsorted(glob(os.path.join(image_dir, "*.nii.gz")))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Preprocessing the MVAA CT volumes"):
        case_id = os.path.basename(image_path).split(".")[0]
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        label_path = os.path.join(label_dir, f"{case_id}-seg.nii.gz")

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        raw = np.asarray(nib.load(image_path).dataobj).T
        labels = np.asarray(nib.load(label_path).dataobj).T

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_mvaa_ct_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and preprocess the MVAA cardiac CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed", "ct")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_CT_VOLUMES:
        return preprocessed_dir

    data_dir = get_mvaa_data(path, download)
    _preprocess_ct(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_mvaa_ct_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the MVAA cardiac CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_mvaa_ct_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."
    return volume_paths


def get_mvaa_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MVAA dataset for mitral annulus segmentation in cardiac CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_mvaa_ct_paths(path, download)

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


def get_mvaa_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MVAA dataloader for mitral annulus segmentation in cardiac CT.

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
    dataset = get_mvaa_ct_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)


def _preprocess_tee(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    train_dir = os.path.join(data_dir, "t2_tee", "train")
    image_paths = natsorted(glob(os.path.join(train_dir, "*-US.nii.gz")))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Preprocessing the MVAA TEE volumes"):
        case_id = os.path.basename(image_path).split("-US.nii.gz")[0]
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        label_path = os.path.join(train_dir, f"{case_id}-label.nii.gz")

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        raw = np.asarray(nib.load(image_path).dataobj).T
        labels = np.asarray(nib.load(label_path).dataobj).T

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_mvaa_tee_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and preprocess the MVAA 3D TEE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed", "tee")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_TEE_VOLUMES:
        return preprocessed_dir

    data_dir = get_mvaa_data(path, download)
    _preprocess_tee(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_mvaa_tee_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the MVAA 3D TEE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_mvaa_tee_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."
    return volume_paths


def get_mvaa_tee_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MVAA dataset for mitral valve leaflet segmentation in 3D TEE.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_mvaa_tee_paths(path, download)

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


def get_mvaa_tee_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MVAA dataloader for mitral valve leaflet segmentation in 3D TEE.

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
    dataset = get_mvaa_tee_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)


def _preprocess_video(data_dir, preprocessed_dir):
    import nibabel as nib
    import imageio.v3 as imageio

    image_dir = os.path.join(preprocessed_dir, "images")
    label_dir = os.path.join(preprocessed_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    video_dirs = natsorted(glob(os.path.join(data_dir, "t3_vid", "train", "REC_*")))
    for video_dir in tqdm(video_dirs, desc="Preprocessing the MVAA surgical video frames"):
        tar_paths = natsorted(glob(os.path.join(video_dir, "*_png_Label.tar")))
        for tar_path in tar_paths:
            frame_name = os.path.basename(tar_path).split("_png_Label.tar")[0]
            image_path = os.path.join(image_dir, f"{frame_name}.png")
            label_path = os.path.join(label_dir, f"{frame_name}.tif")
            if os.path.exists(image_path) and os.path.exists(label_path):
                continue

            with tarfile.open(tar_path) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith(".nii.gz"))
                tar.extract(member, path=video_dir, filter="data")
                nii_path = os.path.join(video_dir, member.name)

            # The transpose maps the nifti axis order (X, Y) to the (Y, X) order used by the RGB frame.
            label = np.asarray(nib.load(nii_path).dataobj).T
            os.remove(nii_path)

            frame_path = os.path.join(video_dir, f"{frame_name}.png")
            imageio.imwrite(image_path, imageio.imread(frame_path))
            imageio.imwrite(label_path, label.astype("uint8"))


def get_mvaa_video_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and preprocess the MVAA surgical video data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed", "video")
    if len(glob(os.path.join(preprocessed_dir, "images", "*.png"))) == N_VIDEO_FRAMES:
        return preprocessed_dir

    data_dir = get_mvaa_data(path, download)
    _preprocess_video(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_mvaa_video_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the MVAA surgical video data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    preprocessed_dir = get_mvaa_video_data(path, download)
    image_paths = natsorted(glob(os.path.join(preprocessed_dir, "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(preprocessed_dir, "labels", "*.tif")))
    assert image_paths and len(image_paths) == len(label_paths), \
        f"The images and labels for '{preprocessed_dir}' do not match."
    return image_paths, label_paths


def get_mvaa_video_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MVAA dataset for mitral valve and instrument segmentation in surgical video.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_mvaa_video_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_mvaa_video_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MVAA dataloader for mitral valve and instrument segmentation in surgical video.

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
    dataset = get_mvaa_video_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
