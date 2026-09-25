"""The CAP LV dataset contains annotations for left ventricular myocardium segmentation in cine cardiac MRI.

The data was curated for the LV Segmentation Challenge of the Cardiac Atlas Project
(https://www.cardiacatlas.org/lv-segmentation-challenge/), which was held at the STACOM 2011 workshop.
It consists of 200 cine MR studies of patients with myocardial infarction and impaired left ventricular
contraction, randomly drawn from the DETERMINE cohort, and split into 100 studies with reference
segmentations of the left ventricular myocardium and 100 studies without them. The images are
steady-state free precession acquisitions in a short-axis and a long-axis view, distributed as
de-identified DICOM, and the reference segmentations are binary mask images. Only the short-axis view is
used here, since the masks are defined on it.

NOTE: The challenge calls the 100 annotated studies the 'test set' and the 100 studies without reference
segmentations the 'validation set', which is the opposite of the usual machine learning convention. This
module uses the annotated studies and ignores the others, so it does not expose a split argument.

Unlike most cardiac benchmarks, the masks cover every frame of the cardiac cycle instead of only the end
diastole and the end systole phase. Each frame of the short-axis stack therefore becomes its own volume,
so that the 100 annotated studies amount to roughly 1600 annotated volumes.

The labels are binary, see `LABEL_IDS`: 1 = left ventricular myocardium. The papillary muscles are
excluded from the myocardium.

NOTE: The data is only available to users who signed the CAP Data Use Agreement, so it cannot be downloaded
automatically. To obtain the data, please follow these steps:
- Read the Terms and Conditions and the CAP Data Use Agreement linked at
  https://www.cardiacatlas.org/lv-segmentation-challenge/ and submit the request form at
  https://www.cardiacatlas.org/lv-segmentation-challenge/request-lv-segmentation-challenge/.
- After approval you receive login credentials for the download. Extract the downloaded archives into the
  folder passed as 'path', so that one folder per study (e.g. 'DET0000101') holds the DICOM images
  ('<study>_SA<slice>_ph<frame>.dcm', plus the long-axis images that are not used here) and the binary mask
  images of the annotated studies ('<study>_SA<slice>_ph<frame>.png').

NOTE: The file naming convention is documented by the organizers. The study ids are the prefix 'DET' of the
DETERMINE cohort followed by 7 digits (https://www.cardiacatlas.org/determine/). The image filenames are of
the form 'DET0011201_SA3_ph0.dcm', see the annotation file page of the CAP LV Landmark Detection Challenge
(http://stacom.cardiacatlas.org/lv-landmark-detection-challenge/annotation-file/, which is distributed from
the same 200 studies), and the masks are PNG files with exactly the same stem, e.g.
'DET0000101_SA1_ph10.png', see the challenge FAQ (http://www.cardiacatlas.org/web/guest/faq). The slice and
the frame index are neither zero-padded nor 1-based.

NOTE: The slice number in the filename is taken from the DICOM database and, as the challenge FAQ states
explicitly, does not always follow the anatomical slice order. The slices are therefore sorted by their
position along the slice normal, which is computed from the DICOM header, and not by the filename.

NOTE: This module was written against the documented layout of the challenge distribution and could not be
validated on the data itself, because no openly published copy of it exists. It requires the pydicom
python package.

The DICOM slices are stacked into volumes and stored in hdf5 files (the keys are 'raw' and 'labels').

The data is only shared under the CAP Data Use Agreement for the DETERMINE cohort and the Terms and
Conditions of the consensus segmentation project, which restrict the use to that project and do not permit
redistribution, so please make sure that you are allowed to use the data for your purpose. NOTE: The CAP
states on https://www.cardiacatlas.org/determine/ that the DETERMINE data is currently unavailable for
download while the data sharing agreement with the data contributor is being renewed.

This dataset is from the publication https://doi.org/10.1016/j.media.2013.09.001.
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {"background": 0, "myocardium": 1}

SLICE_PATTERN = re.compile(r"^(?P<study>.+)_SA(?P<slice>\d+)_ph(?P<frame>\d+)$")


def _index_masks(path):
    """Index the binary masks by their stem, which is the same as the stem of the DICOM slice they belong to."""
    masks = {}
    for extension in ["png", "tif", "dcm"]:
        for mask_path in glob(os.path.join(path, "**", f"*_SA*_ph*.{extension}"), recursive=True):
            stem = os.path.splitext(os.path.basename(mask_path))[0]
            if extension == "dcm" and stem in masks:  # The images are DICOM as well, so PNG masks take precedence.
                continue
            masks.setdefault(stem, mask_path)

    return masks


def _load_mask(mask_path):
    if mask_path.endswith(".dcm"):
        import pydicom
        return np.asarray(pydicom.dcmread(mask_path).pixel_array)

    import imageio.v3 as imageio
    mask = np.asarray(imageio.imread(mask_path))
    return mask[..., 0] if mask.ndim == 3 else mask


def _slice_position(dcm):
    """Compute the position of a slice along the slice normal, which orders the slices from apex to base."""
    orientation = np.asarray(dcm.ImageOrientationPatient, dtype="float64")
    normal = np.cross(orientation[:3], orientation[3:])
    return float(np.dot(np.asarray(dcm.ImagePositionPatient, dtype="float64"), normal))


def _group_slices(path):
    """Group the short-axis DICOM slices of the studies into one volume per cardiac frame."""
    volumes = {}
    for image_path in natsorted(glob(os.path.join(path, "**", "*_SA*_ph*.dcm"), recursive=True)):
        match = SLICE_PATTERN.match(os.path.splitext(os.path.basename(image_path))[0])
        if match is None:
            continue

        key = (match.group("study"), int(match.group("frame")))
        volumes.setdefault(key, []).append(image_path)

    return volumes


def _preprocess_inputs(path, preprocessed_dir):
    import h5py
    import pydicom

    os.makedirs(preprocessed_dir, exist_ok=True)
    volumes = _group_slices(path)
    masks = _index_masks(path)

    for (study, frame), image_paths in tqdm(sorted(volumes.items()), desc="Preprocessing the CAP LV studies"):
        volume_path = os.path.join(preprocessed_dir, f"{study}_ph{frame:02}.h5")
        if os.path.exists(volume_path):
            continue

        stems = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
        if any(stem not in masks for stem in stems):  # The unannotated studies come without reference segmentations.
            continue

        # The slice number in the filename does not always follow the anatomical order, so the slices are
        # sorted by their position along the slice normal instead, as recommended by the challenge FAQ.
        slices = [(pydicom.dcmread(p), stem) for p, stem in zip(image_paths, stems)]
        slices = sorted(slices, key=lambda item: _slice_position(item[0]))

        raw = np.stack([np.asarray(dcm.pixel_array) for dcm, _ in slices])
        labels = np.stack([_load_mask(masks[stem]) for _, stem in slices]) > 0

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_cap_lv_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the CAP LV dataset.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present. The data cannot be downloaded
            automatically, so this raises if the data has not been downloaded manually.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if glob(os.path.join(preprocessed_dir, "*.h5")):
        return preprocessed_dir

    if not glob(os.path.join(path, "**", "*_SA*_ph*.dcm"), recursive=True):
        msg = "'torch_em' cannot download this dataset, because the LV Segmentation Challenge data is only "
        msg += "available to users who signed the CAP Data Use Agreement. Please submit a request at "
        msg += "'https://www.cardiacatlas.org/lv-segmentation-challenge/request-lv-segmentation-challenge/' and "
        msg += f"extract the data you receive into '{path}', one folder per study."
        raise NotImplementedError(msg)

    _preprocess_inputs(path, preprocessed_dir)
    return preprocessed_dir


def get_cap_lv_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the CAP LV data.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_cap_lv_data(path, download)
    return natsorted(glob(os.path.join(data_dir, "*.h5")))


def get_cap_lv_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CAP LV dataset for left ventricular myocardium segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_cap_lv_paths(path, download)

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


def get_cap_lv_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CAP LV dataloader for left ventricular myocardium segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cap_lv_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
