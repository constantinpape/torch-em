"""PanTS (The Pancreatic Tumor Segmentation Dataset) contains annotations for pancreatic tumor and
sub-segment segmentation, together with 24 surrounding anatomical structures, in abdominal CT scans.

The dataset consists of 36,390 CT volumes from 145 medical centers in total, of which 9,901 (case ids
'PanTS_00000001' to 'PanTS_00009901') form the public release used by this module: 9,000 training cases
('PanTS_00000001'-'PanTS_00009000') and 901 public in-distribution test cases
('PanTS_00009001'-'PanTS_00009901'). The remaining cases are proprietary external test sets and are not
released. Each case has a CT scan ('ct.nii.gz') and a combined semantic label volume ('combined_labels.nii.gz')
with the label ids described in `CLASS_IDS`, covering the pancreas and its sub-segments (head, body, tail),
the pancreatic duct, the pancreatic lesion (tumor), and 24 further abdominal/thoracic organs and vascular /
skeletal structures.

The dataset is hosted in two parts:
- The CT scans are hosted, ungated, on HuggingFace at https://huggingface.co/datasets/BodyMaps/PanTSMini, split
  into 9 training shards of 1000 cases each (~30-36 GB per shard) and 1 test shard of 901 cases (~28 GB).
- The labels are hosted, ungated, as a single combined archive (~15.5 GB) at
  https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz, which covers all 9901 cases at once (unlike
  the per-range image shards).
Downloading a case therefore always downloads the whole image shard it belongs to; use `max_cases` or
`case_ids` to only fetch the shard(s) required for a small subset of cases. The label archive, however, is not
sharded, so fetching even a single case's label requires streaming through the (compressed) archive until that
case is found; the module only ever holds the matching labels in memory / on disk, but still has to receive
all bytes up to the last requested case, which in the worst case is the whole ~15.5 GB archive.
The data is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.48550/arXiv.2507.01291.
Please cite it if you use this dataset in your research.
"""

import os
import re
import tarfile
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "BodyMaps/PanTSMini"
LABEL_URL = "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"

CLASS_NAMES = [
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "bladder", "celiac_artery", "colon",
    "common_bile_duct", "duodenum", "femur_left", "femur_right", "gall_bladder", "kidney_left", "kidney_right",
    "liver", "lung_left", "lung_right", "pancreas", "pancreas_body", "pancreas_head", "pancreas_tail",
    "pancreatic_duct", "postcava", "prostate", "spleen", "stomach", "superior_mesenteric_artery", "veins",
    "pancreatic_lesion",
]
"""The anatomical structures of the PanTS dataset, in the order of their label id."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the name of a structure to its label id in the combined label volumes."""

LAST_TRAIN_CASE = 9000
LAST_TEST_CASE = 9901


def _case_name(number: int) -> str:
    return f"PanTS_{number:08d}"


def _case_number(case_name: str) -> int:
    return int(case_name.split("_")[-1])


def _find_case_dirs(image_dir):
    return natsorted(
        d for d in glob(os.path.join(image_dir, "PanTS_*")) if os.path.isfile(os.path.join(d, "ct.nii.gz"))
    )


def _shards_for_cases(files, case_numbers, split):
    tag = "ImageTr" if split == "train" else "ImageTe"
    shards = set()
    for fpath in files:
        if not fpath.endswith(".tar.gz"):
            continue
        match = re.search(rf"{tag}_(\d{{8}})_(\d{{8}})\.tar\.gz$", os.path.basename(fpath))
        if match is None:
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if case_numbers is None or any(start <= number <= end for number in case_numbers):
            shards.add(fpath)
    return shards


def _download_images(path, split, max_cases, case_ids, download):
    image_dir = os.path.join(path, "ImageTr" if split == "train" else "ImageTe")
    last_case = LAST_TRAIN_CASE if split == "train" else LAST_TEST_CASE
    first_case = 1 if split == "train" else (LAST_TRAIN_CASE + 1)

    if case_ids is not None:
        case_numbers = [_case_number(cid) for cid in case_ids]
    elif max_cases is not None:
        case_numbers = list(range(first_case, min(first_case + max_cases - 1, last_case) + 1))
    else:
        case_numbers = None

    case_dirs = _find_case_dirs(image_dir) if os.path.exists(image_dir) else []
    if case_dirs and case_numbers is not None:
        have = {_case_number(os.path.basename(d)) for d in case_dirs}
        if set(case_numbers).issubset(have):
            return image_dir
    elif case_dirs and case_numbers is None:
        return image_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {image_dir}, but download was set to False")

    from huggingface_hub import HfApi, snapshot_download

    os.makedirs(path, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    shards = _shards_for_cases(files, case_numbers, split)
    if not shards:
        raise RuntimeError(f"Could not find any PanTSMini image shards for the '{split}' split.")

    print(f"Downloading {len(shards)} PanTSMini image shard(s) for the '{split}' split.")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=path, allow_patterns=sorted(shards))

    for tar_path in natsorted(glob(os.path.join(path, "*.tar.gz"))):
        util.unzip_tarfile(tar_path=tar_path, dst=image_dir, remove=False)

    case_dirs = _find_case_dirs(image_dir)
    if not case_dirs:
        raise RuntimeError(f"Could not find any 'PanTS_XXXXXXXX' case folders under '{image_dir}'.")

    return image_dir


def _extract_labels_for_cases(path, case_numbers):
    train_dir = os.path.join(path, "LabelTr")
    test_dir = os.path.join(path, "LabelTe")

    def _missing(numbers):
        missing = []
        for number in numbers:
            dst = os.path.join(train_dir if number <= LAST_TRAIN_CASE else test_dir, _case_name(number))
            if not os.path.exists(os.path.join(dst, "combined_labels.nii.gz")):
                missing.append(number)
        return missing

    if case_numbers is not None:
        missing = _missing(case_numbers)
        if not missing:
            return train_dir, test_dir
        wanted = set(missing)
    else:
        wanted = None

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Streaming the PanTS label archive (~15.5 GB, covers all cases, cannot be sharded)...")
    with requests.get(LABEL_URL, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            found = set()
            for member in tqdm(tf, desc="Extracting PanTS labels"):
                top = member.name.split("/")[0]
                if not top.startswith("PanTS_"):
                    continue
                number = _case_number(top)
                if wanted is not None and number not in wanted:
                    continue
                dst = train_dir if number <= LAST_TRAIN_CASE else test_dir
                tf.extract(member, dst)
                if wanted is not None:
                    found.add(number)
                    if found == wanted:
                        break

    return train_dir, test_dir


def get_pants_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[str, str]:
    """Download the PanTS dataset.

    The dataset is ungated and does not require a HuggingFace account or access token. It is, however, very
    large (~300 GB for the images alone), and a case can only be downloaded together with the full image shard
    (of up to 1000 cases) it belongs to. Use `max_cases` or `case_ids` to only download the shard(s) required
    for a small subset of cases; leave both at their default (None) to download the full split.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (9000 cases) or 'test' (901 cases).
        max_cases: The maximum number of cases to download, taken in order starting from the first case of
            `split`. Only the shard(s) covering these cases are downloaded. Mutually exclusive with `case_ids`.
        case_ids: Explicit list of case ids (eg. ['PanTS_00000001']) to download. Only the shard(s) covering
            these cases are downloaded. Mutually exclusive with `max_cases`.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the image data.
        Filepath to the folder with the label data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Please choose one of 'train' or 'test'.")
    assert max_cases is None or case_ids is None, "'max_cases' and 'case_ids' are mutually exclusive."

    image_dir = _download_images(path, split, max_cases, case_ids, download)
    case_dirs = _find_case_dirs(image_dir)
    if case_ids is not None:
        case_dirs = [d for d in case_dirs if os.path.basename(d) in case_ids]
    elif max_cases is not None:
        case_dirs = case_dirs[:max_cases]
    case_numbers = [_case_number(os.path.basename(d)) for d in case_dirs]

    if not download and not os.path.exists(os.path.join(path, "LabelTr" if split == "train" else "LabelTe")):
        raise RuntimeError(f"Cannot find the label data at {path}, but download was set to False")

    train_dir, test_dir = _extract_labels_for_cases(path, case_numbers if case_ids or max_cases else None)
    label_dir = train_dir if split == "train" else test_dir

    return image_dir, label_dir


def get_pants_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PanTS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (9000 cases) or 'test' (901 cases).
        max_cases: The maximum number of cases to use. See `get_pants_data` for details.
        case_ids: Explicit list of case ids to use. See `get_pants_data` for details.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, label_dir = get_pants_data(path, split, max_cases, case_ids, download)

    case_dirs = _find_case_dirs(image_dir)
    if case_ids is not None:
        case_dirs = [d for d in case_dirs if os.path.basename(d) in case_ids]
    elif max_cases is not None:
        case_dirs = case_dirs[:max_cases]

    raw_paths, label_paths = [], []
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        label_path = os.path.join(label_dir, case_name, "combined_labels.nii.gz")
        if not os.path.exists(label_path):
            continue
        raw_paths.append(os.path.join(case_dir, "ct.nii.gz"))
        label_paths.append(label_path)

    if len(raw_paths) == 0 or len(raw_paths) != len(label_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_pants_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PanTS dataset for pancreatic tumor and abdominal anatomy segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (9000 cases) or 'test' (901 cases).
        max_cases: The maximum number of cases to use. See `get_pants_data` for details.
        case_ids: Explicit list of case ids to use. See `get_pants_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_pants_paths(path, split, max_cases, case_ids, download)

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


def get_pants_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PanTS dataloader for pancreatic tumor and abdominal anatomy segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (9000 cases) or 'test' (901 cases).
        max_cases: The maximum number of cases to use. See `get_pants_data` for details.
        case_ids: Explicit list of case ids to use. See `get_pants_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pants_dataset(
        path, patch_shape, split, max_cases, case_ids, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
