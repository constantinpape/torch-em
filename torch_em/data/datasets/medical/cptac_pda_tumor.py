"""The CPTAC-PDA-Tumor-Annotations dataset contains annotations for pancreatic ductal adenocarcinoma
and its metastatic lesions in CT.

The dataset consists of contrast-enhanced CT or MR series of pancreatic cancer patients, with expert
contours for the primary tumor and, where present, metastatic lesions (lymph nodes, liver, lung and
other sites), each drawn as a separate RTSTRUCT series and paired here with the exact series it
references. Every lesion is assigned label id 1, regardless of its anatomical site.

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.7937/BW9V-BX61 and is distributed under the
CC BY 4.0 license.
This dataset is from the CPTAC-PDA collection on The Cancer Imaging Archive; please cite the DOI
above if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


COLLECTION = "CPTAC-PDA"


def _get_series_metadata(path, download):
    """Get the metadata of all series in the collection from the NBIA REST API."""
    import requests

    metadata_path = os.path.join(path, "cptac_pda_tumor_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(f"{util.NBIA_API_URL}getSeries", params={"Collection": COLLECTION})
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _is_lesion_rtstruct(series):
    description = (series.get("SeriesDescription") or "").upper()
    return series.get("Modality") == "RTSTRUCT" and "SEED POINT" not in description


def _referenced_series_uid(rtstruct):
    referenced_study = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
    return str(referenced_study.RTReferencedSeriesSequence[0].SeriesInstanceUID)


def _preprocess_cptac_pda_tumor(dicom_dir, series_metadata, preprocessed_dir):
    import h5py
    import pydicom

    rtstruct_series = [series for series in series_metadata if _is_lesion_rtstruct(series)]

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series in tqdm(rtstruct_series, desc="Preprocess CPTAC-PDA-Tumor-Annotations"):
        rtstruct_dir = os.path.join(dicom_dir, series["SeriesInstanceUID"])
        rtstruct_paths = glob(os.path.join(rtstruct_dir, "*.dcm"))
        if not rtstruct_paths:
            continue

        rtstruct = pydicom.dcmread(rtstruct_paths[0], stop_before_pixels=True)
        image_uid = _referenced_series_uid(rtstruct)
        image_dir = os.path.join(dicom_dir, image_uid)
        if not glob(os.path.join(image_dir, "*.dcm")):
            continue

        out_path = os.path.join(preprocessed_dir, f"{series['SeriesInstanceUID']}.h5")
        if os.path.exists(out_path):
            continue

        volume, geometry = util.load_dicom_series(image_dir)
        labels = util.rasterize_rtstruct(rtstruct_paths[0], geometry, volume.shape, roi_labels=lambda num, name: 1)
        if labels.max() == 0:  # A few RTSTRUCT files carry no findings.
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_cptac_pda_tumor_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CPTAC-PDA-Tumor-Annotations dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)
    series_metadata = _get_series_metadata(path, download)

    rtstruct_uids = [series["SeriesInstanceUID"] for series in series_metadata if _is_lesion_rtstruct(series)]

    dicom_dir = os.path.join(path, "dicom")
    if download:  # The RTSTRUCT series are downloaded first, so the image series they reference can be found.
        util.download_tcia_series(
            rtstruct_uids, dst=dicom_dir, csv_filename=os.path.join(path, "cptac_pda_tumor_rtstruct")
        )

        import pydicom
        image_uids = set()
        for uid in rtstruct_uids:
            rtstruct_paths = glob(os.path.join(dicom_dir, uid, "*.dcm"))
            if rtstruct_paths:
                rtstruct = pydicom.dcmread(rtstruct_paths[0], stop_before_pixels=True)
                image_uids.add(_referenced_series_uid(rtstruct))
        util.download_tcia_series(
            sorted(image_uids), dst=dicom_dir, csv_filename=os.path.join(path, "cptac_pda_tumor_image")
        )
    elif not glob(os.path.join(dicom_dir, "*", "*.dcm")):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_cptac_pda_tumor(dicom_dir, series_metadata, preprocessed_dir)
    return preprocessed_dir


def get_cptac_pda_tumor_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the CPTAC-PDA-Tumor-Annotations data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_cptac_pda_tumor_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_cptac_pda_tumor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CPTAC-PDA-Tumor-Annotations dataset for pancreatic tumor and metastasis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_cptac_pda_tumor_paths(path, download)

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


def get_cptac_pda_tumor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CPTAC-PDA-Tumor-Annotations dataloader for pancreatic tumor and metastasis segmentation.

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
    dataset = get_cptac_pda_tumor_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
