"""The DAP Atlas dataset contains annotations for 142 anatomical structures in whole-body CT scans.

The dataset consists of automatically generated (and post-processed with anatomical guidelines) dense anatomical
label volumes for 533 CT scans of the AutoPET dataset. The label volumes are distributed via the DAP Atlas repository,
the CT scans are the (original resolution) 'CT.nii.gz' volumes of the AutoPET dataset, which is downloaded via
`torch_em.data.datasets.medical.autopet`. The names of the label volumes ('AutoPET_<subject id>_<last 5 digits of
the study uid>') uniquely identify the matching AutoPET CT.
NOTE: The AutoPET download is ~300 GB, but it is shared with the AutoPET dataset (see `autopet_path`).

The label id of each structure is given in `CLASS_IDS` (see also the table in the appendix of the publication).
The label ids are: 1: left to annotate (unlabeled placeholder), 2: muscles, 3: fat, 4: abdominal tissue,
5: mediastinal tissue, 6: esophagus, 7: stomach, 8: small bowel, 9: duodenum, 10: colon, 11: rectum, 12: gallbladder,
13: liver, 14: pancreas, 15: kidney left, 16: kidney right, 17: bladder, 18: gonads, 19: prostate, 20: uterocervix,
21: uterus, 22: breast left, 23: breast right, 24: spinal canal, 25: brain, 26: spleen, 27: adrenal gland left,
28: adrenal gland right, 29: thyroid left, 30: thyroid right, 31: thymus, 32-37: gluteus maximus / medius / minimus
(left, right), 38-39: iliopsoas (left, right), 40-41: autochthon (left, right), 42: skin, 43-66: vertebrae C1-L5,
67-90: costa 1-12 (left, right), 91: rib cartilage, 92: sternum corpus, 93-94: clavicula (left, right),
95-96: scapula (left, right), 97-98: humerus (left, right), 99: skull, 100-101: hip (left, right), 102: sacrum,
103-104: femur (left, right), 105: heart, 106: heart atrium left, 107: heart tissue, 108: heart atrium right,
109: heart myocardium, 110: heart ventricle left, 111: heart ventricle right, 112-113: iliac artery (left, right),
114: aorta, 115-116: iliac vena (left, right), 117: inferior vena cava, 118: portal vein and splenic vein,
119: celiac trunk, 120-124: lung lobes, 125: bronchus, 126: trachea, 127: pulmonary artery,
128-129: cheek (left, right),
130-131: eyeball (left, right), 132: nasal cavity, 133-134: common carotid artery (right, left),
135: sternum manubrium, 136-137: internal carotid artery (right, left), 138-139: internal jugular vein (right, left),
140: brachiocephalic artery, 141-142: brachiocephalic vein (right, left), 143-144: subclavian artery (right, left).

The dataset is located at https://github.com/alexanderjaus/AtlasDataset.

This dataset is from the publication https://doi.org/10.48550/arXiv.2307.13375.
Please cite it (and the AutoPET publication https://doi.org/10.1038/s41597-022-01718-3) if you use this dataset
in your research.
"""

import os
from glob import glob
from warnings import warn
from natsort import natsorted
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .autopet import get_autopet_data


URL = "https://drive.google.com/uc?id=1ex0a9eQULLvKPDwijmijX2h49A-ockNy"
CHECKSUM = "eea8b7bf2378a5085bc4ef7014bb4d764c9e276a6d8c8a55add19e9bb2b80622"

CLASS_NAMES = [
    "left_to_annotate", "muscles", "fat", "abdominal_tissue", "mediastinal_tissue", "esophagus", "stomach",
    "small_bowel", "duodenum", "colon", "rectum", "gallbladder", "liver", "pancreas", "kidney_left", "kidney_right",
    "bladder", "gonads", "prostate", "uterocervix", "uterus", "breast_left", "breast_right", "spinal_canal", "brain",
    "spleen", "adrenal_gland_left", "adrenal_gland_right", "thyroid_left", "thyroid_right", "thymus",
    "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right", "iliopsoas_left", "iliopsoas_right", "autochthon_left",
    "autochthon_right", "skin", "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5",
    "vertebrae_C6", "vertebrae_C7", "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
    "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "costa_1_left", "costa_1_right",
    "costa_2_left", "costa_2_right", "costa_3_left", "costa_3_right", "costa_4_left", "costa_4_right", "costa_5_left",
    "costa_5_right", "costa_6_left", "costa_6_right", "costa_7_left", "costa_7_right", "costa_8_left", "costa_8_right",
    "costa_9_left", "costa_9_right", "costa_10_left", "costa_10_right", "costa_11_left", "costa_11_right",
    "costa_12_left", "costa_12_right", "rib_cartilage", "sternum_corpus", "clavicula_left", "clavicula_right",
    "scapula_left", "scapula_right", "humerus_left", "humerus_right", "skull", "hip_left", "hip_right", "sacrum",
    "femur_left", "femur_right", "heart", "heart_atrium_left", "heart_tissue", "heart_atrium_right",
    "heart_myocardium", "heart_ventricle_left", "heart_ventricle_right", "iliac_artery_left", "iliac_artery_right",
    "aorta", "iliac_vena_left", "iliac_vena_right", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "celiac_trunk", "lung_lower_lobe_left", "lung_upper_lobe_left", "lung_lower_lobe_right", "lung_middle_lobe_right",
    "lung_upper_lobe_right", "bronchus", "trachea", "pulmonary_artery", "cheek_left", "cheek_right", "eyeball_left",
    "eyeball_right", "nasal_cavity", "common_carotid_artery_right", "common_carotid_artery_left", "sternum_manubrium",
    "internal_carotid_artery_right", "internal_carotid_artery_left", "internal_jugular_vein_right",
    "internal_jugular_vein_left", "brachiocephalic_artery", "brachiocephalic_vein_right", "brachiocephalic_vein_left",
    "subclavian_artery_right", "subclavian_artery_left",
]
"""The classes of the DAP Atlas dataset. The label id of a class is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the class name to its label id in the label volumes."""


def get_dap_atlas_data(
    path: Union[os.PathLike, str], autopet_path: Optional[Union[os.PathLike, str]] = None, download: bool = False
) -> Tuple[str, str]:
    """Download the DAP Atlas dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        autopet_path: Filepath to the folder where the AutoPET dataset is (or will be) downloaded.
            By default, it is downloaded to the 'autopet' sub-folder of `path`.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the label volumes are downloaded.
        Filepath where the AutoPET data is downloaded.
    """
    if autopet_path is None:
        autopet_path = os.path.join(path, "autopet")
    get_autopet_data(autopet_path, download)

    label_dir = os.path.join(path, "Atlas_final_dataset_V1_533")
    if not os.path.exists(label_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "dap_atlas_masks.zip")
        util.download_source_gdrive(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    return label_dir, autopet_path


def get_dap_atlas_paths(
    path: Union[os.PathLike, str], autopet_path: Optional[Union[os.PathLike, str]] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DAP Atlas data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        autopet_path: Filepath to the folder where the AutoPET dataset is (or will be) downloaded.
            By default, it is downloaded to the 'autopet' sub-folder of `path`.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    label_dir, autopet_path = get_dap_atlas_data(path, autopet_path, download)

    # Match each label volume 'AutoPET_<subject id>_<study suffix>.nii.gz' to the CT in the AutoPET study folder
    # 'PETCT_<subject id>/<study description>-<study suffix>/CT.nii.gz'.
    autopet_dir = os.path.join(autopet_path, "AutoPET-II", "FDG-PET-CT-Lesions")
    raw_paths, label_paths, missing = [], [], []
    for label_path in natsorted(glob(os.path.join(label_dir, "AutoPET_*.nii.gz"))):
        _, subject_id, study_suffix = os.path.basename(label_path)[:-len(".nii.gz")].split("_")
        ct_paths = glob(os.path.join(autopet_dir, f"PETCT_{subject_id}", f"*-{study_suffix}", "CT.nii.gz"))
        if len(ct_paths) > 1:
            raise RuntimeError(f"Found multiple AutoPET CTs for '{label_path}': {ct_paths}.")
        if not ct_paths:  # The AutoPET download is incomplete, so this case is skipped.
            missing.append(os.path.basename(label_path))
            continue
        raw_paths.append(ct_paths[0])
        label_paths.append(label_path)

    if missing:
        warn(f"Could not find the AutoPET CT for {len(missing)} of {len(missing) + len(raw_paths)} label volumes. "
             f"These cases are skipped. Is the AutoPET data in '{autopet_path}' complete?")

    if not raw_paths:
        raise RuntimeError(f"Could not match any DAP Atlas label volume to an AutoPET CT in '{autopet_path}'.")

    assert len(raw_paths) == len(label_paths)
    return raw_paths, label_paths


def get_dap_atlas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    autopet_path: Optional[Union[os.PathLike, str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DAP Atlas dataset for anatomical structure segmentation in whole-body CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        autopet_path: Filepath to the folder where the AutoPET dataset is (or will be) downloaded.
            By default, it is downloaded to the 'autopet' sub-folder of `path`.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_dap_atlas_paths(path, autopet_path, download)

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


def get_dap_atlas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    autopet_path: Optional[Union[os.PathLike, str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DAP Atlas dataloader for anatomical structure segmentation in whole-body CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        autopet_path: Filepath to the folder where the AutoPET dataset is (or will be) downloaded.
            By default, it is downloaded to the 'autopet' sub-folder of `path`.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dap_atlas_dataset(path, patch_shape, autopet_path, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
