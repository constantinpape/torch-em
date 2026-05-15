"""The PCNS dataset contains manual annotations for nucleus instance segmentation
in H&E stained histopathology images of fourteen cancer types from TCGA.

The dataset contains 1,365 manually annotated patches of 400x400 pixels at 40x
magnification, covering BLCA, BRCA, CESC, COAD, GBM, LUAD, LUSC, PAAD, PRAD,
READ, SKCM, STAD, UCEC and UVM cancer types. Annotations were created by three
human annotators correcting Mask R-CNN predictions.

NOTE: This dataset requires manual download. Please download
'manual_segmentation_data.tar.gz' from the TCIA collection page at
https://www.cancerimagingarchive.net/analysis-result/pan-cancer-nuclei-seg/
(direct Box link: https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019/file/586046955275)
and place it in the 'path' directory you pass to the dataset functions.

NOTE: For the automatic segmentation results of 5,060 WSIs (~665 GB) via the same TCIA
collection, use the IBM Aspera Connect plugin from the TCIA page. The Aspera manifests
cover 10 cancer types (BLCA, BRCA, CESC, GBM, LUAD, LUSC, PAAD, PRAD, SKCM, UCEC)
with per-WSI polygon CSV files under '{cancer_type}_polygon/' subdirectories.

The dataset is located at https://doi.org/10.7937/TCIA.2019.4A4DKP9U.
This dataset is from the publication https://doi.org/10.1038/s41597-020-0528-1.
Please cite it if you use this dataset in your research.
"""

import io
import json
import os
import shutil
import tarfile
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import imageio.v3 as imageio
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


CROSSWALK_URL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "Pan-Cancer-Nuclei-Seg_1365patches_to_TCGA-ID_readme.txt"
)

BOX_URL = "https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019/file/586046955275"

CANCER_TYPES = [
    "blca", "brca", "cesc", "coad", "gbm", "luad", "lusc",
    "paad", "prad", "read", "skcm", "stad", "ucec", "uvm",
]


def _load_crosswalk(path: str) -> pd.DataFrame:
    crosswalk_path = os.path.join(path, "pcns_crosswalk.txt")
    if not os.path.exists(crosswalk_path):
        util.download_source(path=crosswalk_path, url=CROSSWALK_URL, download=True)

    with open(crosswalk_path, "rb") as f:
        raw = f.read()

    if raw[:2] == b"\x1f\x8b":
        import gzip
        content = gzip.decompress(raw).decode("utf-8")
    else:
        content = raw.decode("utf-8")

    lines = content.split("\n")
    csv_start = next((i for i, line in enumerate(lines) if line.startswith("Patch-ID,")), None)
    if csv_start is None:
        raise RuntimeError("Failed to parse the PCNS crosswalk file. Re-download it and try again.")

    df = pd.read_csv(io.StringIO("\n".join(lines[csv_start:])))
    df = df.dropna(subset=["Patch-ID", "CancerType"])
    df["Patch-ID"] = df["Patch-ID"].astype(int)
    df["CancerType"] = df["CancerType"].str.lower()
    return df


def _create_split_csv(path: str, all_patch_ids: List[int], split: str) -> List[int]:
    csv_path = os.path.join(path, "pcns_split.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[split] = df[split].apply(lambda x: json.loads(x.replace("'", '"')))
        return df.iloc[0][split]

    print(f"Creating a new split file at '{csv_path}'.")
    train_ids, test_ids = train_test_split(all_patch_ids, test_size=0.2)
    split_ids = {"train": sorted(train_ids), "test": sorted(test_ids)}
    pd.DataFrame.from_dict([split_ids]).to_csv(csv_path, index=False)
    return split_ids[split]


def _create_samples(path: str, extract_dir: str, crosswalk_df: pd.DataFrame) -> str:
    preprocessed_dir = os.path.join(path, "preprocessed_data")
    if os.path.exists(preprocessed_dir):
        return preprocessed_dir
    os.makedirs(preprocessed_dir, exist_ok=True)

    crop_paths = {
        int(os.path.basename(p).split("_crop")[0]): p
        for p in glob(os.path.join(extract_dir, "**", "*_crop.png"), recursive=True)
    }

    ct_map = {int(row["Patch-ID"]): str(row["CancerType"]).lower() for _, row in crosswalk_df.iterrows()}

    valid_ids = [
        pid for pid in crop_paths
        if os.path.exists(crop_paths[pid].replace("_crop.png", "_labeled_mask_corrected.png"))
    ]

    for patch_id in tqdm(sorted(valid_ids), desc="Creating PCNS samples"):
        image_path = crop_paths[patch_id]
        mask_path = image_path.replace("_crop.png", "_labeled_mask_corrected.png")

        raw = imageio.imread(image_path)[..., :3].transpose(2, 0, 1)
        mask = imageio.imread(mask_path).astype(np.int32)
        h, w = mask.shape

        h5_path = os.path.join(preprocessed_dir, f"{patch_id}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/instances", data=mask, compression="gzip")

            has_all = True
            for k in range(3):
                common_path = image_path.replace("_crop.png", f"_labeled_mask_common{k}.png")
                if os.path.exists(common_path):
                    common_mask = imageio.imread(common_path).astype(np.int32)
                else:
                    common_mask = np.zeros((h, w), dtype=np.int32)
                    has_all = False
                f.create_dataset(f"labels/common{k}", data=common_mask, compression="gzip")

            f.attrs["cancer_type"] = ct_map.get(patch_id, "unknown")
            f.attrs["has_common"] = has_all

    return preprocessed_dir


def get_pcns_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Locate and extract the PCNS dataset, then build per-sample H5 files.

    The dataset requires manual download. Download 'manual_segmentation_data.tar.gz'
    from https://www.cancerimagingarchive.net/analysis-result/pan-cancer-nuclei-seg/
    and place it in the 'path' directory before calling this function.

    After preprocessing the final layout under 'path' is:
    - manual_segmentation_data.tar.gz
    - pcns_crosswalk.txt
    - pcns_split.csv
    - preprocessed_data/{patch_id}.h5 (one per patch)

    Each sample H5 stores:
    - raw: (3, H, W) uint8 RGB image
    - labels/instances: (H, W) int32 corrected instance mask
    - labels/common0/1/2: (H, W) int32 per-annotator masks (zero-filled if absent)
    - attrs['cancer_type']: str cancer type code
    - attrs['has_common']: bool, True for the 27 patches with per-annotator annotations

    Args:
        path: Filepath to the folder where the tarball was placed and data will be extracted.
        download: Unused. The dataset cannot be downloaded automatically.

    Returns:
        The filepath to the preprocessed_data directory containing per-sample H5 files.
    """
    path = os.path.normpath(path)
    preprocessed_dir = os.path.join(path, "preprocessed_data")

    if os.path.exists(preprocessed_dir):
        return preprocessed_dir

    tar_path = os.path.join(path, "manual_segmentation_data.tar.gz")
    if download:
        raise RuntimeError(
            "The PCNS dataset cannot be downloaded automatically. "
            f"Please download 'manual_segmentation_data.tar.gz' manually from {BOX_URL} "
            f"and place it at '{tar_path}'."
        )
    if not os.path.exists(tar_path):
        raise RuntimeError(
            "The PCNS dataset requires manual download. "
            f"Please download 'manual_segmentation_data.tar.gz' from {BOX_URL} "
            f"and place it at '{tar_path}'."
        )

    extract_dir = os.path.join(path, "_raw")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting PCNS data to '{extract_dir}'...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    crosswalk_df = _load_crosswalk(path)
    _create_samples(path, extract_dir, crosswalk_df)

    shutil.rmtree(extract_dir)

    return preprocessed_dir


def get_pcns_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    cancer_type: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get the paths to the per-sample H5 files for the requested split.

    Args:
        path: Filepath to the folder where the data is located.
        split: The data split to use. Either 'train' or 'test'.
        cancer_type: The cancer type(s) to load. If None, all fourteen types are used.
            Valid values: 'blca', 'brca', 'cesc', 'coad', 'gbm', 'luad', 'lusc',
            'paad', 'prad', 'read', 'skcm', 'stad', 'ucec', 'uvm'.
        download: Unused. The dataset cannot be downloaded automatically.

    Returns:
        List of filepaths to the per-sample H5 files for the requested split and cancer type.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose from 'train' or 'test'.")

    preprocessed_dir = get_pcns_data(path, download)
    crosswalk_df = _load_crosswalk(path)

    all_patch_ids = crosswalk_df["Patch-ID"].tolist()
    split_ids = set(_create_split_csv(path, all_patch_ids, split))

    if cancer_type is not None:
        if isinstance(cancer_type, str):
            cancer_type = [cancer_type]
        cancer_type = [ct.lower() for ct in cancer_type]
        invalid = [ct for ct in cancer_type if ct not in CANCER_TYPES]
        if invalid:
            raise ValueError(f"Invalid cancer type(s): {invalid}. Choose from {CANCER_TYPES}.")
        type_ids = set(crosswalk_df[crosswalk_df["CancerType"].isin(cancer_type)]["Patch-ID"].tolist())
        split_ids = split_ids & type_ids

    volume_paths = [
        os.path.join(preprocessed_dir, f"{pid}.h5")
        for pid in sorted(split_ids)
        if os.path.exists(os.path.join(preprocessed_dir, f"{pid}.h5"))
    ]

    if not volume_paths:
        raise RuntimeError(
            f"No samples found for split='{split}', cancer_type={cancer_type!r}. "
            "Ensure the data was extracted and preprocessed correctly."
        )

    return volume_paths


def get_pcns_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    cancer_type: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PCNS dataset for nucleus instance segmentation.

    Args:
        path: Filepath to the folder where the data is located.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        cancer_type: The cancer type(s) to load. If None, all fourteen types are used.
            Valid values: 'blca', 'brca', 'cesc', 'coad', 'gbm', 'luad', 'lusc',
            'paad', 'prad', 'read', 'skcm', 'stad', 'ucec', 'uvm'.
        download: Unused. The dataset cannot be downloaded automatically.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_pcns_paths(path, split, cancer_type, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels/instances",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_pcns_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    cancer_type: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PCNS dataloader for nucleus instance segmentation.

    Args:
        path: Filepath to the folder where the data is located.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        cancer_type: The cancer type(s) to load. If None, all fourteen types are used.
            Valid values: 'blca', 'brca', 'cesc', 'coad', 'gbm', 'luad', 'lusc',
            'paad', 'prad', 'read', 'skcm', 'stad', 'ucec', 'uvm'.
        download: Unused. The dataset cannot be downloaded automatically.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pcns_dataset(path, patch_shape, split, cancer_type, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
