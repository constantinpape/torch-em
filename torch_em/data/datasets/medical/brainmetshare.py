"""The BrainMetShare dataset contains annotations for brain metastasis segmentation in multi-sequence brain MRI.

It comprises 156 whole brain MRI studies (105 with radiologist-drawn metastasis segmentations, 51 unlabeled)
with four co-registered, skull-stripped 3D sequences resampled to 256 x 256 pixels: T1 gradient-echo
post-contrast, T1 spin-echo pre-contrast, T1 spin-echo post-contrast and T2 FLAIR post-contrast.
Only the 105 labeled studies are provided by this dataset.

NOTE: The label legend is as follows:
- background: 0, metastasis: 1
Verified on the data: the label volumes only contain the ids 0 and 1.

The data is a redistribution of the official Stanford release at
https://www.kaggle.com/datasets/kapilesha/brainmetshare-nii, which stores the four sequences and the
segmentation of each study as nifti files (uploaded under the MIT license; the underlying data is subject
to the Stanford University Dataset Research Use Agreement). The official release at
https://aimi.stanford.edu/brainmetshare requires registration, so please make sure that you are allowed
to use the data for your purpose.

NOTE: The official release can also be used. Download it as described below and this dataset will use it
instead of the redistribution:
- Visit https://aimi.stanford.edu/brainmetshare and follow the link to the dataset on Stanford's Redivis
  platform (https://stanford.redivis.com/datasets/1emj-bjxt3p6s0).
- Register / log in, fill in your contact details and accept the research use agreement.
- Download the data (e.g. with 'azcopy' as described on the website) and place it such that the labeled cases
  are located at '<path>/mets_stanford_release_train/<case>/{0,1,2,3,seg}'. Each modality folder holds the
  slices of one sequence ('0': T1 gradient-echo post, '1': T1 spin-echo pre, '2': T1 spin-echo post,
  '3': T2 FLAIR post) and 'seg' holds the binary metastasis mask (0, 255).

This dataset is from the publication https://doi.org/10.1002/jmri.26766.
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


KAGGLE_DATASET = "kapilesha/brainmetshare-nii"

LABEL_IDS = {"background": 0, "metastasis": 1}

MODALITIES = {"t1_gre_post": "0", "t1_se_pre": "1", "t1_se_post": "2", "flair": "3"}

# The names used for the sequences in the redistributed nifti files.
NIFTI_NAMES = {"t1_gre_post": "bravo", "t1_se_pre": "t1_pre", "t1_se_post": "t1_gd", "flair": "flair"}


def _load_volume(case_dir, nifti_name, folder_name):
    """Load a volume stored either as a nifti file or as a stack of 2d slice images."""
    nifti_path = os.path.join(case_dir, f"{nifti_name}.nii")
    if os.path.exists(nifti_path):
        import nibabel as nib
        return np.asarray(nib.load(nifti_path).dataobj).T  # (Z, Y, X)

    import imageio.v3 as imageio
    folder = os.path.join(case_dir, folder_name)
    slice_paths = natsorted([p for p in glob(os.path.join(folder, "*")) if os.path.isfile(p)])
    assert len(slice_paths) > 0, f"Could not find any image files in {folder}."
    volume = np.stack([imageio.imread(p) for p in slice_paths])
    if volume.ndim == 4:  # Multi-channel slice images (e.g. RGB pngs) are reduced to a single channel.
        volume = volume[..., 0]
    return volume


def _preprocess_inputs(path, case_dirs):
    import h5py

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for case_dir in tqdm(case_dirs, desc="Preprocessing the BrainMetShare cases"):
        case_id = os.path.basename(case_dir)
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        labels = (_load_volume(case_dir, "seg", "seg") > 0).astype(np.uint8)

        # The file is written to a temporary path first, so that an interrupted run does not leave a corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("labels", data=labels, compression="gzip")
            for modality in MODALITIES:
                raw = _load_volume(case_dir, NIFTI_NAMES[modality], MODALITIES[modality])
                assert raw.shape == labels.shape, f"Shape mismatch for {case_id}: {raw.shape} vs. {labels.shape}."
                f.create_dataset(f"raw/{modality}", data=raw, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)

    return preprocessed_dir


def _find_case_dirs(path):
    # The labeled cases of the official release.
    case_dirs = glob(os.path.join(path, "mets_stanford_release_train", "*"))
    if len(case_dirs) == 0:  # The labeled cases of the redistribution.
        case_dirs = glob(os.path.join(path, "train", "Mets_*"))
    return natsorted([p for p in case_dirs if os.path.isdir(p)])


def get_brainmetshare_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BrainMetShare dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is preprocessed.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    case_dirs = _find_case_dirs(path)
    if len(case_dirs) == 0:
        os.makedirs(path, exist_ok=True)
        util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET, download=download)
        util.unzip(zip_path=os.path.join(path, "brainmetshare-nii.zip"), dst=path)
        case_dirs = _find_case_dirs(path)

    assert len(case_dirs) > 0, f"Could not find any BrainMetShare cases at '{path}'."

    return _preprocess_inputs(path, case_dirs)


def get_brainmetshare_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BrainMetShare data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_brainmetshare_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths, volume_paths


def get_brainmetshare_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["t1_gre_post", "t1_se_pre", "t1_se_post", "flair"] = "t1_gre_post",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BrainMetShare dataset for brain metastasis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. One of 't1_gre_post', 't1_se_pre', 't1_se_post' or 'flair'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose one of {list(MODALITIES)}.")

    raw_paths, label_paths = get_brainmetshare_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=f"raw/{modality}",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_brainmetshare_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["t1_gre_post", "t1_se_pre", "t1_se_post", "flair"] = "t1_gre_post",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BrainMetShare dataloader for brain metastasis segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. One of 't1_gre_post', 't1_se_pre', 't1_se_post' or 'flair'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brainmetshare_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
