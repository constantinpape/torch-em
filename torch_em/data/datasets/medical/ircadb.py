"""
"""

import os
from glob import glob
from natsort import natsorted

import numpy as np

import torch_em

from .. import util


URL = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"


def _preprocess_inputs(path):
    data_dir = os.path.join(path, "3Dircadb1")
    patient_dirs = glob(os.path.join(data_dir, "*"))

    # Let's extract all files per patient, preprocess them, store the final version and remove the zip files.
    for pdir in patient_dirs:
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
        labels = [dicom.dcmread(p).pixel_array for p in natsorted(glob(os.path.join(pdir, "LABELLED_DICOM", "*")))]
        # masks = [p for p in natsorted(glob(os.path.join(pdir, "MASKS_DICOM", "*")))]
        images = [dicom.dcmread(p).pixel_array for p in natsorted(glob(os.path.join(pdir, "PATIENT_DICOM", "*")))]

        labels = np.stack(labels, axis=0)
        images = np.stack(images, axis=0)

        import napari
        v = napari.Viewer()
        v.add_image(images)
        v.add_labels(labels)
        napari.run()

        breakpoint()


def get_ircadb_data(path, download):
    """
    """
    # data_dir = os.path.join(path, "3Dircadb1")
    # if os.path.exists(data_dir):
    #     return data_dir

    # os.makedirs(path, exist_ok=True)

    # zip_path = os.path.join(path, "data.zip")
    # util.download_source(path=zip_path, url=URL, download=download, checksum=None)  # checksums don't match.
    # util.unzip(zip_path=zip_path, dst=path, remove=True)

    _preprocess_inputs(path)

    # return data_dir


def get_ircadb_paths(path, split, download):
    """
    """

    get_ircadb_data(path, download)

    breakpoint()

    raw_paths = ...
    gt_paths = ...

    return raw_paths, gt_paths


def get_ircadb_dataset(path, patch_shape, split, resize_inputs=False, download=False, **kwargs):
    """
    """
    raw_paths, gt_paths = get_ircadb_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_ircadb_loader(path, batch_size, patch_shape, split, resize_inputs=False, download=False, **kwargs):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ircadb_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
