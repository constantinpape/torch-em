"""The NasalSeg dataset contains annotations for nasal cavity and paranasal sinus segmentation in 3D CT scans.

The dataset consists of 130 CT scans (NRRD, 'images/<case>_img.nrrd' and 'labels/<case>_seg.nrrd') with voxel-wise
annotations of five structures. The label ids are described in `LABEL_IDS`. The record does not document the id
order; it was determined from the label positions in the scans (the images use the left-posterior-superior
convention): 1 and 2 are the lateral pair (maxillary sinuses), 3 and 4 the medial pair (nasal cavities) and 5 is
the posterior midline structure (nasopharynx), with the lower id of each pair on the right side of the patient.
The scans are converted to hdf5 files (keys 'raw' and 'labels') by this module.

The dataset is located at https://doi.org/10.5281/zenodo.13893419, released under a CC-BY-4.0 license.
Please cite the corresponding publication if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/13893419/files/NasalSeg.zip/content"
CHECKSUM = "60c6facf843685802c39e4adff4a05c081c1c4b6175c9cb573745c55abb0fa6a"

LABEL_IDS = {
    "right maxillary sinus": 1,
    "left maxillary sinus": 2,
    "right nasal cavity": 3,
    "left nasal cavity": 4,
    "nasopharynx": 5,
}


def _convert_case(image_path, label_path, out_path):
    import h5py
    import SimpleITK as sitk

    if os.path.exists(out_path):
        return

    raw = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    labels = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype("uint8")

    tmp_path = f"{out_path}.{os.getpid()}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    os.replace(tmp_path, out_path)


def get_nasalseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NasalSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the converted data is stored.
    """
    image_dir = os.path.join(path, "images")
    label_dir = os.path.join(path, "labels")
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "NasalSeg.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(image_dir) and os.path.exists(label_dir), \
        f"The extraction of the NasalSeg archive did not create the expected folders in '{path}'."

    converted_dir = os.path.join(path, "converted")
    os.makedirs(converted_dir, exist_ok=True)
    for image_path in natsorted(glob(os.path.join(image_dir, "*_img.nrrd"))):
        case_id = os.path.basename(image_path)[:-len("_img.nrrd")]
        label_path = os.path.join(label_dir, f"{case_id}_seg.nrrd")
        _convert_case(image_path, label_path, os.path.join(converted_dir, f"{case_id}.h5"))

    return converted_dir


def get_nasalseg_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the NasalSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    converted_dir = get_nasalseg_data(path, download)
    volume_paths = natsorted(glob(os.path.join(converted_dir, "*.h5")))
    assert len(volume_paths) > 0
    return volume_paths


def get_nasalseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NasalSeg dataset for nasal cavity and paranasal sinus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_nasalseg_paths(path, download)

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
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_nasalseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NasalSeg dataloader for nasal cavity and paranasal sinus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nasalseg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
