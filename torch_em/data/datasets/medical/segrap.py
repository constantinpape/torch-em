"""The SegRap dataset contains annotations for organ-at-risk (OAR) segmentation in head and neck CT scans
of nasopharyngeal carcinoma patients.

It comprises the training set of the SegRap2023 challenge (https://segrap2023.grand-challenge.org): 120 patients
with a pre-aligned pair of a non-contrast and a contrast-enhanced CT scan, and annotations for 45 OARs.

NOTE: The label legend is as follows. Since some of the 45 OARs are nested (e.g. the hippocampi inside the
temporal lobes, the cochleae inside the middle ears), the challenge distributes the annotations as a single
label volume with 54 disjoint sub-parts, where each overlap of two OARs gets its own id. The ids are
(see `SUBPART_NAMES` and `LABEL_IDS`):
- 1: Brain, 2: BrainStem, 3: Chiasm, 4/5: TemporalLobe_L/R, 6/7: TemporalLobe_Hippocampus_L/R,
  8/9: Hippocampus_L/R, 10/11: Eye_L/R, 12/13: Lens_L/R, 14/15: OpticNerve_L/R, 16/17: MiddleEar_L/R,
  18/19: IAC_L/R, 20/21: MiddleEar_TympanicCavity_L/R, 22/23: TympanicCavity_L/R,
  24/25: MiddleEar_VestibulSemi_L/R, 26/27: VestibulSemi_L/R, 28/29: Cochlea_L/R,
  30/31: MiddleEar_ETbone_L/R, 32/33: ETbone_L/R, 34: Pituitary, 35: OralCavity, 36/37: Mandible_L/R,
  38/39: Submandibular_L/R, 40/41: Parotid_L/R, 42/43: Mastoid_L/R, 44/45: TMjoint_L/R, 46: SpinalCord,
  47: Esophagus, 48: Larynx, 49: Larynx_Glottic, 50: Larynx_Supraglot, 51: Larynx_PharynxConst,
  52: PharynxConst, 53: Thyroid, 54: Trachea
`OAR_TO_LABEL_IDS` maps each of the 45 OARs to the ids it consists of. It is taken from the official
post-processing code of the challenge (https://github.com/HiLab-git/SegRap2023/blob/main/Tutorial/postprocessing.py).
The ids were verified on the data: the label volumes contain the ids 0 to 54.

The data is a redistribution of the official challenge data at
https://huggingface.co/datasets/YongchengYAO/SegRap23-Lite (CC BY-NC 4.0), which holds the unchanged images
and Task001 labels of all 120 training cases, renamed after the case ids. The official release at
https://segrap2023.grand-challenge.org/dataset/ requires a signed end user agreement, so please make sure
that you are allowed to use the data for your purpose.

NOTE: The official release can also be used. Download 'SegRap2023_Training_Set_120cases.zip' and
'SegRap2023_Training_Set_120cases_OneHot_Labels.zip' as described on the dataset page and extract them into
'<path>', such that '<path>/SegRap2023_Training_Set_120cases/segrap_XXXX/image.nii.gz' (and
'image_contrast.nii.gz') and '<path>/SegRap2023_Training_Set_120cases_OneHot_Labels/Task001/segrap_XXXX.nii.gz'
exist. This dataset will then use the official data instead of the redistribution.

This dataset is from the publication https://doi.org/10.1016/j.media.2024.103447.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "ct": "https://huggingface.co/datasets/YongchengYAO/SegRap23-Lite/resolve/main/Images-CT.zip",
    "ct_contrast": "https://huggingface.co/datasets/YongchengYAO/SegRap23-Lite/resolve/main/Images-contrastCT.zip",
    "labels": "https://huggingface.co/datasets/YongchengYAO/SegRap23-Lite/resolve/main/Masks-Task1.zip",
}

CHECKSUMS = {
    "ct": "1e7b849c5f0296200e5ad9503eeeb9b2a8b3360297078d0ce016c8f8335534ef",
    "ct_contrast": "000cc8bf9041c49f7e28ad3e183f7af3f9e701b1842031fb17e21a89db0a8f6a",
    "labels": "3441d2bd56ecff4485b99a197b36251b09d2030468840f9399419b517691d297",
}

FOLDER_NAMES = {"ct": "Images-CT", "ct_contrast": "Images-contrastCT", "labels": "Masks-Task1"}

# The file names of the two scans in the official release.
OFFICIAL_FILE_NAMES = {"ct": "image.nii.gz", "ct_contrast": "image_contrast.nii.gz"}

SUBPART_NAMES = [
    "Brain", "BrainStem", "Chiasm", "TemporalLobe_L", "TemporalLobe_R", "TemporalLobe_Hippocampus_L",
    "TemporalLobe_Hippocampus_R", "Hippocampus_L", "Hippocampus_R", "Eye_L", "Eye_R", "Lens_L", "Lens_R",
    "OpticNerve_L", "OpticNerve_R", "MiddleEar_L", "MiddleEar_R", "IAC_L", "IAC_R",
    "MiddleEar_TympanicCavity_L", "MiddleEar_TympanicCavity_R", "TympanicCavity_L", "TympanicCavity_R",
    "MiddleEar_VestibulSemi_L", "MiddleEar_VestibulSemi_R", "VestibulSemi_L", "VestibulSemi_R", "Cochlea_L",
    "Cochlea_R", "MiddleEar_ETbone_L", "MiddleEar_ETbone_R", "ETbone_L", "ETbone_R", "Pituitary", "OralCavity",
    "Mandible_L", "Mandible_R", "Submandibular_L", "Submandibular_R", "Parotid_L", "Parotid_R", "Mastoid_L",
    "Mastoid_R", "TMjoint_L", "TMjoint_R", "SpinalCord", "Esophagus", "Larynx", "Larynx_Glottic",
    "Larynx_Supraglot", "Larynx_PharynxConst", "PharynxConst", "Thyroid", "Trachea",
]

LABEL_IDS = {"background": 0, **{name: i + 1 for i, name in enumerate(SUBPART_NAMES)}}

OAR_TO_LABEL_IDS = {
    "Brain": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "BrainStem": [2],
    "Chiasm": [3],
    "TemporalLobe_L": [4, 6],
    "TemporalLobe_R": [5, 7],
    "Hippocampus_L": [8, 6],
    "Hippocampus_R": [9, 7],
    "Eye_L": [10, 12],
    "Eye_R": [11, 13],
    "Lens_L": [12],
    "Lens_R": [13],
    "OpticNerve_L": [14],
    "OpticNerve_R": [15],
    "MiddleEar_L": [18, 16, 20, 24, 28, 30],
    "MiddleEar_R": [19, 17, 21, 25, 29, 31],
    "IAC_L": [18],
    "IAC_R": [19],
    "TympanicCavity_L": [22, 20],
    "TympanicCavity_R": [23, 21],
    "VestibulSemi_L": [26, 24],
    "VestibulSemi_R": [27, 25],
    "Cochlea_L": [28],
    "Cochlea_R": [29],
    "ETbone_L": [32, 30],
    "ETbone_R": [33, 31],
    "Pituitary": [34],
    "OralCavity": [35],
    "Mandible_L": [36],
    "Mandible_R": [37],
    "Submandibular_L": [38],
    "Submandibular_R": [39],
    "Parotid_L": [40],
    "Parotid_R": [41],
    "Mastoid_L": [42],
    "Mastoid_R": [43],
    "TMjoint_L": [44],
    "TMjoint_R": [45],
    "SpinalCord": [46],
    "Esophagus": [47],
    "Larynx": [48, 49, 50, 51],
    "Larynx_Glottic": [49],
    "Larynx_Supraglot": [50],
    "PharynxConst": [51, 52],
    "Thyroid": [53],
    "Trachea": [54],
}


def _get_official_case_dirs(path):
    case_dirs = glob(os.path.join(path, "**", "SegRap2023_Training_Set_120cases", "segrap_*"), recursive=True)
    return natsorted([p for p in case_dirs if os.path.isdir(p)])


def _download_component(path, name, download):
    data_dir = os.path.join(path, FOLDER_NAMES[name])
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{FOLDER_NAMES[name]}.zip")
    util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_segrap_data(
    path: Union[os.PathLike, str], modality: Literal["ct", "ct_contrast"] = "ct", download: bool = False
) -> str:
    """Download the SegRap dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The CT scan to download. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if modality not in OFFICIAL_FILE_NAMES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose one of {list(OFFICIAL_FILE_NAMES)}.")

    if len(_get_official_case_dirs(path)) > 0:  # The official data was downloaded manually.
        return path

    _download_component(path, "labels", download)
    _download_component(path, modality, download)

    return path


def get_segrap_paths(
    path: Union[os.PathLike, str], modality: Literal["ct", "ct_contrast"] = "ct", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SegRap data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The CT scan to use as input. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_segrap_data(path, modality, download)

    case_dirs = _get_official_case_dirs(data_dir)
    if len(case_dirs) > 0:  # The official layout, with one folder per case and the labels in a separate folder.
        label_dir = os.path.join(os.path.split(os.path.split(case_dirs[0])[0])[0], "Task001")
        raw_paths = [os.path.join(p, OFFICIAL_FILE_NAMES[modality]) for p in case_dirs]
        label_paths = [os.path.join(label_dir, f"{os.path.basename(p)}.nii.gz") for p in case_dirs]
    else:  # The redistributed layout, with the files named after the case ids.
        raw_paths = natsorted(glob(os.path.join(data_dir, FOLDER_NAMES[modality], "*.nii.gz")))
        label_paths = [
            os.path.join(data_dir, FOLDER_NAMES["labels"], os.path.basename(p)) for p in raw_paths
        ]

    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_segrap_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["ct", "ct_contrast"] = "ct",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegRap dataset for organ-at-risk segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The CT scan to use as input. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_segrap_paths(path, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_segrap_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["ct", "ct_contrast"] = "ct",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegRap dataloader for organ-at-risk segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The CT scan to use as input. Either 'ct' (non-contrast) or 'ct_contrast' (contrast-enhanced).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_segrap_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
