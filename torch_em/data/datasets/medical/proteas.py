"""The PROTEAS dataset contains annotations for brain metastasis segmentation in longitudinal MRI.

The dataset consists of 40 patients with metastatic brain cancer (45 archives, as a few patients are split into
'a' and 'b' courses of treatment) and 185 imaging studies (MRI and CT) over the course of radiotherapy and follow-up.
It provides manual segmentations of 65 brain metastases, for each MRI study of a patient (the 'baseline' and the
follow-ups 'fu1', 'fu2', ...), together with the radiotherapy plan and radiomics tables. The MRI studies come as
skull-stripped and co-registered T1 ('t1'), contrast-enhanced T1 ('t1c'), T2 ('t2') and FLAIR ('fla') volumes of shape
(240, 240, 155), on the same voxel grid as the segmentation masks. This loader pairs a chosen MRI sequence of each
study with its mask. The raw DICOM series, the planning CT and the dose maps are not used and not kept.

The masks distinguish three tumor regions, with the label ids 1 = necrotic core, 2 = enhancing tumor and 3 = edema
(see `LABEL_IDS`). NOTE: The dataset record does not document the label ids and they do not follow the BraTS
numbering. They were inferred from the data: label 1 is enclosed by label 2, label 2 is bright in the
contrast-enhanced T1 volumes and label 3 is the outermost region, bright in the FLAIR volumes.

NOTE: This is not the same as the BEAMSTER dataset in `torch_em.data.datasets.medical.beamster`, which has binary
metastasis masks on contrast-enhanced T1 scans from a single time point, without longitudinal follow-up.

The data is located at https://doi.org/10.5281/zenodo.17253793 (v1 of the record, released in October 2025 under a
CC BY 4.0 license). The latest version of the same record, https://doi.org/10.5281/zenodo.20025432, is access
restricted, so this loader uses the open v1 record, whose availability may change.

The dataset is from the publication https://doi.org/10.1038/s41597-025-06131-0.
Please cite it if you use this dataset for your research.
"""

import os
import uuid
import shutil
import zipfile
from glob import glob
from concurrent import futures
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/17253793/files/{patient_id}.zip/content"

CHECKSUMS = {
    "P01": "9355b7f50d3e690737c7a5e4d22046d8df206212d4b09fa31ec1ad9f469ba54b",
    "P02": "0b4978e3036e1e5c3c8f0f55168c74b612152686ce3d7f8681e933b89e7098d8",
    "P03": "6f0a8e9d66016c06ec7e186ad3dac72f620c56a74345de66c75b70f4d3a73f9a",
    "P04a": "ec12923914188ef81eed02a7f6f4aed1ae839a40b1d6e63a47af90ebff540487",
    "P04b": "83c13be378f3efe79388bf3ad909f6426b6793b54ea3333c6a57955934218a35",
    "P05": "282d3d8f2ce3fb8a79cffa69ae6e28f414770cbf2e404211597474f58e11130a",
    "P06": "0a5d2f7f9998454abcf9143ec9e29309cf97d68946c8b41eaad6e7c90fdefa14",
    "P07a": "9698954c9f277036f710affa2590cbfb5f4aff0a17041307bd4411e9d9afe36e",
    "P07b": "9ea5e9ffcfe4ff80246dca983d3ea5da3309cc96c20c51b0ac00f7b500854d35",
    "P08": "6e6f559840766dd10af8ce6f20a94747f20319537ee3f568a68eba578e093e7b",
    "P09": "14bd3371ddb19f4fb486f469f27b0db42f78d19ac4c975abf2b30b0859ef35d0",
    "P10": "b5dd8535efa50a6102e1387338750273cec39ac8beee3b38bdffd7edc5c605a9",
    "P11": "2b2ab945002116de468dd1caf50914e272c0ee473b6708b37d497d548dffd2e0",
    "P12": "c08da88ed4b56e64d3db161b0b318d0e1b98baa6c28266fd9b1181979337a2e1",
    "P13": "ab0842a95e31e69ca0c64f4cfbbc495af481a3f8ab798a5cd3f6b49c01dae2c8",
    "P14": "ed2d56706456a2aca8e26cee365349326126d0271efa97141a16e86ea95afeaa",
    "P15": "b4abed11f5b9a9eedf7df659afe0e4ca4b57f135454963af71e610e7e6a19469",
    "P16": "beb26fb1c01f77c80f22eee591c73c5657b2cc3426d19d3bd4e6cceec525be79",
    "P17a": "1a776b528c60a7a49924c7bf1c59eedc93c37e6e0083c0d8f16ddb2932daf0f4",
    "P17b": "13c396baa500ee3b2c82b72c89e15f9c044850a02c7aa3615f2cced9ab6b39b7",
    "P18": "f064cfbb7303d5c8ca6597b60ca9550152c8c6c85de7bf900269d2708dfa930d",
    "P19": "56902860f712f4bce08e20c642217d4f18c45f3297a6d8376284f121a6c2cf18",
    "P20a": "24d04ba76de099dad365951f74788f436f086e87bafcbdd4704c884fc5d82a97",
    "P20b": "0072f1d4158727aa4088b3fb32294512a78aa7772da0f7a98063ee051e7660ba",
    "P21": "124192750c623382fc4911a112e9b372e90952e75a4e9da3fa602e467bccb3d3",
    "P22": "1011e6b69a5f864ba43032ded01ff6bbde6a02fc72354d2c63f7307c96a890af",
    "P23a": "818425160a1514fb2d3f6b5eb86b439cb4cc4f56b93c1e2c3b5690fb53e2ab99",
    "P23b": "2241d9c24d03cd69583e709bf498ab6abc86e791664048cd59d3de8453dd98dc",
    "P24": "b660ea009c3604cf5f4891ce6abb7fabf25f428f34a515353df55aa67b0944b5",
    "P25": "ebfd97a5efdb798f7c9a53c0536d25125d3fc14d85d90b7ef2b7d42949ec1503",
    "P26": "b526e4a2ec8f4a77097ccf925667ee6a4a527a6d213722d9f853f19a4546b31a",
    "P27": "76017fe710787309d4aa0ceb6a590c58fc5130c0e9e08a63119bf4b9b65ec5d7",
    "P28": "6e0c811e7e6da46296b9dc6e73050e65d8abada81f1900713c3fddb728a5b989",
    "P29": "6432a22a555c15d0086b2e7ada50a14f1927af1009fd6ec4fa7bf05d5efe4f18",
    "P30": "90c8beef41f82004181d1b7cd7db390f81723e3f57040b5ffa1f55205eb5734b",
    "P31": "553a410ebe4bc0d159b93f048e38021c22cd7ee4b506d9aa72ffb0241a463628",
    "P32": "05192b28f86022c40b82b1d82192428bd8387fe3378436bb0cc4333cc062e824",
    "P33": "64ee5d2d83fa2aebc4838603b8ada018d2082af71dd7e1e680585344377f1e63",
    "P34": "60063b57524b0978942b28eec70ac8cf28af053623454e1c0fdf446168663f5d",
    "P35": "73a23c05f6a1700384eb4d7b9ccf96ff745f6e7f1627e27a678d859521006dd0",
    "P36": "f4f897589a16831fe2ff2dd5d3185d39a98ce23313e03837d9399297e0a89ace",
    "P37": "d645a3910d394d82620588a6249fa6a9ece61a0f06f042acb9c69e7e296bb947",
    "P38": "5c8c12811462024ad51ccba928bc8ce405fb397632e4b675dc072f3c4a5dbe06",
    "P39": "bb3c06bcda4125597c353eaba8949bce420efcea831bb91cc047af14eb5b1296",
    "P40": "6a09cb9a8bbdd383760efa490255f18d0f43c348630bffc59dbf5d3ea985a569",
}

SEQUENCES = ["t1", "t1c", "t2", "fla"]

LABEL_IDS = {"necrotic_core": 1, "enhancing_tumor": 2, "edema": 3}


def _extract_patient(zip_path, patient_id, patient_dir):
    tmp_dir = f"{patient_dir}.{uuid.uuid4().hex}.incomplete"
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name for name in zf.namelist()
            if name.endswith(".nii.gz") and ("/BraTS/" in name or "/tumor_segmentation/" in name)
        ]
        zf.extractall(tmp_dir, members)

    os.replace(os.path.join(tmp_dir, patient_id), patient_dir)
    shutil.rmtree(tmp_dir)


def _download_patient(patient_id, path):
    zip_path = os.path.join(path, f"{patient_id}.zip")
    util.download_source(
        path=zip_path, url=URL.format(patient_id=patient_id), download=True, checksum=CHECKSUMS[patient_id],
    )
    _extract_patient(zip_path, patient_id, os.path.join(path, "patients", patient_id))
    os.remove(zip_path)


def get_proteas_data(
    path: Union[os.PathLike, str], n_patients: Optional[int] = None, download: bool = False,
) -> str:
    """Download the PROTEAS dataset.

    NOTE: The archives of all patients are about 15 GB, as they contain the DICOM series next to the NIfTI volumes
    used here. Use `n_patients` to only download a subset for a quick start.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        n_patients: The number of patients (archives) to download, sorted by patient id. By default all 45 are
            downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the extracted data is stored.
    """
    patient_dir = os.path.join(path, "patients")
    patient_ids = sorted(CHECKSUMS)[:n_patients]
    missing = [pid for pid in patient_ids if not os.path.exists(os.path.join(patient_dir, pid))]
    if not missing:
        return patient_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    os.makedirs(patient_dir, exist_ok=True)
    with futures.ThreadPoolExecutor(4) as pool:
        tasks = [pool.submit(_download_patient, pid, path) for pid in missing]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Download PROTEAS patients"):
            task.result()

    return patient_dir


def get_proteas_paths(
    path: Union[os.PathLike, str],
    sequence: Literal["t1", "t1c", "t2", "fla"] = "t1c",
    n_patients: Optional[int] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PROTEAS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence: The choice of MRI sequence. One of 't1', 't1c' (contrast-enhanced T1), 't2' or 'fla' (FLAIR).
        n_patients: The number of patients to use, sorted by patient id. By default all 45 are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data, one per annotated MRI study.
        List of filepaths for the label data.
    """
    if sequence not in SEQUENCES:
        raise ValueError(f"'{sequence}' is not a valid sequence. Choose one of {SEQUENCES}.")

    patient_dir = get_proteas_data(path, n_patients, download)

    raw_paths, label_paths = [], []
    for patient_id in sorted(CHECKSUMS)[:n_patients]:
        for label_path in natsorted(glob(os.path.join(patient_dir, patient_id, "tumor_segmentation", "*.nii.gz"))):
            study = os.path.basename(label_path)[:-len(".nii.gz")].split("_tumor_mask_")[1]
            raw_path = os.path.join(patient_dir, patient_id, "BraTS", study, f"{sequence}.nii.gz")
            if os.path.exists(raw_path):
                raw_paths.append(raw_path)
                label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_proteas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sequence: Literal["t1", "t1c", "t2", "fla"] = "t1c",
    n_patients: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PROTEAS dataset for brain metastasis segmentation in longitudinal MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        sequence: The choice of MRI sequence. One of 't1', 't1c' (contrast-enhanced T1), 't2' or 'fla' (FLAIR).
        n_patients: The number of patients to use, sorted by patient id. By default all 45 are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_proteas_paths(path, sequence, n_patients, download)

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
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_proteas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    sequence: Literal["t1", "t1c", "t2", "fla"] = "t1c",
    n_patients: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PROTEAS dataloader for brain metastasis segmentation in longitudinal MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sequence: The choice of MRI sequence. One of 't1', 't1c' (contrast-enhanced T1), 't2' or 'fla' (FLAIR).
        n_patients: The number of patients to use, sorted by patient id. By default all 45 are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_proteas_dataset(
        path, patch_shape, sequence, n_patients, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
