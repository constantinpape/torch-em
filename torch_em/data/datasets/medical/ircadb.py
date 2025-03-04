"""
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np

import torch_em

from .. import util


URL = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
CHECKSUM = None  # NOTE: checksums are mismatching for some reason with every new download instance :/


def _preprocess_inputs(path):
    data_dir = os.path.join(path, "3Dircadb1")
    patient_dirs = glob(os.path.join(data_dir, "*"))

    # Store all preprocessed images in one place
    preprocessed_dir = os.path.join(path, "data")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Let's extract all files per patient, preprocess them, store the final version and remove the zip files.
    for pdir in tqdm(patient_dirs, desc="Preprocessing files"):

        patient_name = os.path.basename(pdir)

        # Get all zipfiles
        label_file = os.path.join(pdir, "LABELLED_DICOM.zip")
        masks_file = os.path.join(pdir, "MASKS_DICOM.zip")
        meshes_file = os.path.join(pdir, "MESHES_VTK.zip")
        patient_file = os.path.join(pdir, "PATIENT_DICOM.zip")

        # Unzip all.
        util.unzip(label_file, pdir, remove=False)
        util.unzip(masks_file, pdir, remove=False)
        util.unzip(meshes_file, pdir, remove=False)
        util.unzip(patient_file, pdir, remove=False)

        # Get all files and stack each slice together.
        import pydicom as dicom
        images = [dicom.dcmread(p).pixel_array for p in natsorted(glob(os.path.join(pdir, "PATIENT_DICOM", "*")))]
        images = np.stack(images, axis=0)

        # Get masks per slice per class.
        masks, mask_names = [], []
        for mask_dir in glob(os.path.join(pdir, "MASKS_DICOM", "*")):
            mask_names.append(os.path.basename(mask_dir))
            curr_mask = np.stack(
                [dicom.dcmread(p).pixel_array for p in natsorted(glob(os.path.join(mask_dir, "*")))], axis=0,
            )
            assert curr_mask.shape == images.shape, "The shapes for images and labels don't match."
            masks.append(curr_mask)

        breakpoint()

        # Store them in one place
        import h5py
        with h5py.File(os.path.join(preprocessed_dir, f"{patient_name}.h5"), "a") as f:
            f.create_dataset("raw", shape=images.shape, dtype=images.dtype, compression="gzip")
            # Add labels one by one
            for name, _mask in zip(mask_names, masks):
                f.create_dataset(f"labels/{name}", shape=_mask.shape, dtype=_mask.dtype, compression="gzip")


def get_ircadb_data(path, download):
    """
    """
    # data_dir = os.path.join(path, "data")
    # if os.path.exists(data_dir):
    #     return data_dir

    # os.makedirs(path, exist_ok=True)

    # zip_path = os.path.join(path, "data.zip")
    # util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    # util.unzip(zip_path=zip_path, dst=path, remove=True)

    _preprocess_inputs(path)

    return data_dir


def get_ircadb_paths(path, split, download):
    """
    """

    data_dir = get_ircadb_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))

    # Create splits on-the-fly.
    if split == "train":
        volume_paths = volume_paths[:12]
    elif split == "val":
        volume_paths = volume_paths[12:15]
    elif split == "test":
        volume_paths = volume_paths[15:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return volume_paths


def get_ircadb_dataset(
    path, patch_shape, split, label_choice, resize_inputs=False, download=False, **kwargs
):
    """
    """
    volume_paths = get_ircadb_paths(path, split, download)

    # Get the labels in the expected hierarchy name.
    if isinstance(label_choice, str):
        label_choice = [label_choice]

    label_choice = [f"labels/{choice}" for choice in label_choice if not choice.startswith("labels")]

    # Get the parameters for resizing inputs
    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_choice,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_ircadb_loader(
    path, batch_size, patch_shape, split, label_choice, resize_inputs=False, download=False, **kwargs
):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ircadb_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
