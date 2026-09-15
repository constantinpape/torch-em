"""The CETUS dataset contains annotations for left ventricle segmentation in
3D echocardiography of the heart.

The data was curated for the CETUS challenge (Challenge on Endocardial Three-dimensional Ultrasound
Segmentation, https://www.creatis.insa-lyon.fr/Challenge/CETUS/), which was held at MICCAI 2014. The public
release consists of the 3D echocardiographic sequences of 45 patients. For each patient the end-diastolic (ED)
and the end-systolic (ES) frame are extracted and annotated, which gives 90 annotated volumes. The annotation
is a binary mask of the left ventricle lumen (the endocardial surface), see `LABEL_IDS`. The volumes of a
single phase can be selected with the 'phase' argument.

The volumes are stored as nifti files with the slice axis last and the masks use the foreground value 255,
so they are converted to hdf5 volumes with the slice axis first (the keys are 'raw' and 'labels') and the
masks are binarized by this module. The image intensities are 8 bit and the voxels are isotropic.

The data is located at https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62eb991b73e9f0048c3a6c45
and is distributed under the CC BY-NC-SA 4.0 license.

This dataset is from the publication https://doi.org/10.1109/TMI.2015.2503890.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/62eb9a3e73e9f0048c3a6c46/download"

# NOTE: The archive is created on the fly by the girder server, so its checksum changes with every download.
CHECKSUM = None

LABEL_IDS = {"background": 0, "left_ventricle": 1}

PHASES = ["ED", "ES"]

N_VOLUMES = 90


def _preprocess_inputs(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    case_dirs = natsorted(glob(os.path.join(data_dir, "patient*")))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for case_dir in tqdm(case_dirs, desc="Preprocessing the CETUS volumes"):
        case_id = os.path.basename(case_dir)
        for phase in PHASES:
            volume_path = os.path.join(preprocessed_dir, f"{case_id}_{phase}.h5")
            if os.path.exists(volume_path):
                continue

            # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
            raw = np.asarray(nib.load(os.path.join(case_dir, f"{case_id}_{phase}.nii.gz")).dataobj).T
            mask = np.asarray(nib.load(os.path.join(case_dir, f"{case_id}_{phase}_gt.nii.gz")).dataobj).T

            # The intensities are 8 bit values stored as floats and the mask uses the foreground value 255.
            raw = np.round(raw).astype("uint8")
            labels = (mask > 0).astype("uint8") * LABEL_IDS["left_ventricle"]

            # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
            with h5py.File(f"{volume_path}.tmp", "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels", data=labels, compression="gzip")

            os.rename(f"{volume_path}.tmp", volume_path)


def get_cetus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CETUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_VOLUMES:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "dataset")
    if not os.path.exists(data_dir):
        zip_path = os.path.join(path, "CETUS.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_inputs(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_cetus_paths(
    path: Union[os.PathLike, str], phase: Optional[Literal["ED", "ES"]] = None, download: bool = False
) -> List[str]:
    """Get paths to the CETUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        phase: The choice of cardiac phase. Either 'ED' or 'ES'. By default both phases are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    if phase is not None and phase not in PHASES:
        raise ValueError(f"'{phase}' is not a valid cardiac phase. Please choose one of {PHASES}.")

    data_dir = get_cetus_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, f"patient*_{'*' if phase is None else phase}.h5")))
    assert len(volume_paths) > 0

    return volume_paths


def get_cetus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    phase: Optional[Literal["ED", "ES"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CETUS dataset for left ventricle segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        phase: The choice of cardiac phase. Either 'ED' or 'ES'. By default both phases are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_cetus_paths(path, phase, download)

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


def get_cetus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    phase: Optional[Literal["ED", "ES"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CETUS dataloader for left ventricle segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        phase: The choice of cardiac phase. Either 'ED' or 'ES'. By default both phases are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cetus_dataset(path, patch_shape, phase, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
