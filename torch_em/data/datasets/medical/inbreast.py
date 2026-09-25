"""The INbreast dataset contains annotations for lesion segmentation in full-field digital mammograms.

INbreast was acquired at the Breast Research Group, INESC Porto / Hospital de Sao Joao (Portugal) with a
full-field digital mammography (FFDM) system, which distinguishes it from CBIS-DDSM (a re-scan of older
film mammograms). It comprises 410 images from 115 cases (90 cases with both breasts, 4 images each; 25
mastectomy cases, 2 images each), each distributed as a DICOM file. Specialists outlined masses, calcifications,
calcification clusters, spiculated regions, asymmetries and architectural distortions with the OsiriX viewer,
and the resulting contours are exported per case as an OsiriX property list ('.xml', parseable with 'plistlib').
Each ROI lists its pixel coordinates under 'Point_px'; ROIs with 3 or more points are closed polygons
(rasterized here with 'skimage.draw.polygon'), ROIs with 1 or 2 points are isolated pixel-level markers
(usually individual, non-clustered calcifications).

The label ids used here are (see `LABEL_IDS`): 0 = background, 1 = mass, 2 = calcification, 3 = cluster
(of calcifications), 4 = spiculated region, 5 = asymmetry, 6 = distortion. The 'Name' field of a few ROIs
in the original xml files is misspelled or inconsistently capitalized (e.g. 'Assymetry', 'Espiculated Region',
'Calcifications'); these are normalized to the canonical names above. A handful of ROIs with an empty or
placeholder name (e.g. 'Unnamed', 'Point 1') carry no lesion information and are skipped.

NOTE: The original INbreast distribution required a request form to the Breast Research Group and is not
publicly downloadable anymore. This module downloads the 'INbreast Release 1.0' mirror hosted on Kaggle
(https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset), which reproduces the original release
folder structure ('AllDICOMs', 'AllXML', 'AllROI', 'MedicalReports', 'INbreast.xls', 'README.txt') including
the per-lesion xml contours. Other INbreast mirrors on Kaggle (e.g. 'martholi/inbreast') only redistribute
the DICOM images without the xml annotations and are not suitable for segmentation.

NOTE: Extracting the mirrored zip file with python's 'zipfile' can fail with a 'bad zipfile offset' /
'bad magic number' error, because the archive that Kaggle serves for this dataset has extra bytes prepended
to it. This module extracts it with the 'unzip' command line tool instead, which recovers from this offset
issue. Please make sure 'unzip' is available (it ships with most Linux distributions).

This dataset is from the publication https://doi.org/10.1016/j.acra.2011.09.014. Please cite it if you use
this dataset in your research.

NOTE: The DICOM loading requires the 'pydicom' python package.
"""

import os
from glob import glob
from shutil import which
from subprocess import run
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {
    "background": 0,
    "mass": 1,
    "calcification": 2,
    "cluster": 3,
    "spiculated_region": 4,
    "asymmetry": 5,
    "distortion": 6,
}

# The 'Name' field of the xml ROIs is not fully standardized (typos, capitalization). This maps every
# variant that was found in the release to a canonical label id. Names that are not listed here (e.g. the
# empty string, 'Unnamed', 'Point 1') do not carry lesion information and are skipped.
ROI_NAME_TO_LABEL_ID = {
    "mass": LABEL_IDS["mass"],
    "calcification": LABEL_IDS["calcification"],
    "calcifications": LABEL_IDS["calcification"],
    "cluster": LABEL_IDS["cluster"],
    "spiculated region": LABEL_IDS["spiculated_region"],
    "espiculated region": LABEL_IDS["spiculated_region"],
    "asymmetry": LABEL_IDS["asymmetry"],
    "assymetry": LABEL_IDS["asymmetry"],
    "distortion": LABEL_IDS["distortion"],
}


def _unzip_inbreast(zip_path, dst):
    # 'zipfile' fails on this archive with a 'bad zipfile offset' error, because Kaggle serves it with
    # extra bytes prepended. The 'unzip' CLI recovers from this offset issue, so it is used instead of
    # 'torch_em.data.datasets.util.unzip'.
    if which("unzip") is None:
        raise RuntimeError("Need the 'unzip' CLI to extract the INbreast archive.")
    run(["unzip", "-q", "-o", zip_path, "-d", dst])
    os.remove(zip_path)


def _parse_xml_rois(xml_path):
    import plistlib

    with open(xml_path, "rb") as f:
        annotations = plistlib.load(f)

    rois = []
    for image in annotations["Images"]:
        for roi in image["ROIs"]:
            label_id = ROI_NAME_TO_LABEL_ID.get(roi["Name"].strip().lower())
            if label_id is None:
                continue
            points = np.array([
                [float(v) for v in point.strip("()").split(",")] for point in roi["Point_px"]
            ])
            rois.append((label_id, points))
    return rois


def _rasterize_rois(rois, shape):
    from skimage.draw import polygon

    labels = np.zeros(shape, dtype="uint8")
    for label_id, points in rois:
        x, y = points[:, 0], points[:, 1]
        if len(points) >= 3:
            rr, cc = polygon(y, x, shape=shape)
        else:
            rr, cc = np.round(y).astype(int), np.round(x).astype(int)
            valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            rr, cc = rr[valid], cc[valid]
        labels[rr, cc] = label_id
    return labels


def _preprocess_inbreast(release_dir, preprocessed_dir):
    import h5py
    import pydicom

    os.makedirs(preprocessed_dir, exist_ok=True)

    dcm_paths = natsorted(glob(os.path.join(release_dir, "AllDICOMs", "*.dcm")))
    for dcm_path in tqdm(dcm_paths, desc="Preprocess INbreast"):
        fname = os.path.basename(dcm_path)
        case_id = fname.split("_")[0]

        out_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(out_path):
            continue

        raw = pydicom.dcmread(dcm_path).pixel_array

        xml_path = os.path.join(release_dir, "AllXML", f"{case_id}.xml")
        if os.path.exists(xml_path):
            labels = _rasterize_rois(_parse_xml_rois(xml_path), raw.shape)
        else:  # A few images have no annotated lesions.
            labels = np.zeros(raw.shape, dtype="uint8")

        tmp_path = f"{out_path}.tmp"
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        os.rename(tmp_path, out_path)

    return preprocessed_dir


def get_inbreast_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the INbreast dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    release_dir = os.path.join(path, "INbreast Release 1.0")
    if not os.path.exists(release_dir):
        zip_path = os.path.join(path, "inbreast-dataset.zip")
        util.download_source_kaggle(path=path, dataset_name="ramanathansp20/inbreast-dataset", download=download)
        _unzip_inbreast(zip_path, path)

    return _preprocess_inbreast(release_dir, preprocessed_dir)


def get_inbreast_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the INbreast data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_inbreast_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed files in '{data_dir}'."
    return volume_paths, volume_paths


def get_inbreast_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the INbreast dataset for lesion segmentation in mammograms.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_inbreast_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_inbreast_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the INbreast dataloader for lesion segmentation in mammograms.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_inbreast_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
