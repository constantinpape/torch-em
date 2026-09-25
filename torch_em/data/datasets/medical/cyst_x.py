"""The Cyst-X dataset contains annotations for whole-pancreas segmentation in T1-weighted and T2-weighted
abdominal MRI.

The dataset consists of 1,461 full-volume MRI scans of 764 patients from 7 centers (723 T1-weighted and 738
T2-weighted scans), stored as nifti files. Each scan is paired with a binary mask of the whole pancreas
(0 = background, 1 = pancreas). NOTE: The masks are pancreas masks, they do not contain the pancreatic cysts,
ducts, mural nodules or tumors that the dataset was built to study (IPMN risk stratification).

The data is located at https://huggingface.co/datasets/phy710/Cyst-X (also mirrored at https://osf.io/74vfs/),
released under a CC-BY-NC-4.0 license (non-commercial use only). The full collection is large, use `n_cases` to
only download a subset of it.

The dataset and its code are described at https://github.com/NUBagciLab/Cyst-X.
Please cite the Cyst-X project if you use this dataset for your research.
"""

import os
import json
from glob import glob
from concurrent import futures
from typing import Union, Tuple, List, Literal, Optional

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


API_URL = "https://huggingface.co/api/datasets/phy710/Cyst-X"
FILE_URL = "https://huggingface.co/datasets/phy710/Cyst-X/resolve/main/{name}"

SEQUENCES = ["t1", "t2"]


def _list_files(path):
    cache_path = os.path.join(path, "file_list.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    import requests

    response = requests.get(API_URL, params={"blobs": "true"}, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    files = {
        sibling["rfilename"]: sibling["lfs"]["sha256"]
        for sibling in response.json()["siblings"]
        if sibling["rfilename"].startswith("IPMN_images_masks/") and "lfs" in sibling
    }

    os.makedirs(path, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(files, f)

    return files


def _download_case(path, sequence, case_id, files):
    for kind in ("images", "masks"):
        name = f"IPMN_images_masks/{sequence}/{kind}/{case_id}.nii.gz"
        os.makedirs(os.path.join(path, sequence, kind), exist_ok=True)
        util.download_source(
            path=os.path.join(path, sequence, kind, f"{case_id}.nii.gz"), url=FILE_URL.format(name=name),
            download=True, checksum=files[name],
        )


def get_cyst_x_data(
    path: Union[os.PathLike, str],
    sequence: Literal["t1", "t2"],
    n_cases: Optional[int] = None,
    n_workers: int = 8,
    download: bool = False,
) -> str:
    """Download the Cyst-X dataset.

    NOTE: The full collection contains 1,461 scans. Use `n_cases` to only download a subset for a quick start.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence: The choice of MRI sequence. Either 't1' or 't2'.
        n_cases: The number of cases to download, sorted by case id. By default all cases are downloaded.
        n_workers: The number of parallel download workers.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if sequence not in SEQUENCES:
        raise ValueError(f"'{sequence}' is not a valid sequence. Choose one of {SEQUENCES}.")

    os.makedirs(path, exist_ok=True)
    files = _list_files(path)

    prefix = f"IPMN_images_masks/{sequence}/images/"
    case_ids = sorted(name[len(prefix):-len(".nii.gz")] for name in files if name.startswith(prefix))
    if n_cases is not None:
        case_ids = case_ids[:n_cases]

    missing = [
        case_id for case_id in case_ids
        if not all(
            os.path.exists(os.path.join(path, sequence, kind, f"{case_id}.nii.gz")) for kind in ("images", "masks")
        )
    ]
    if missing and not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    with futures.ThreadPoolExecutor(n_workers) as pool:
        tasks = [pool.submit(_download_case, path, sequence, case_id, files) for case_id in missing]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Download Cyst-X cases"):
            task.result()

    return os.path.join(path, sequence)


def get_cyst_x_paths(
    path: Union[os.PathLike, str],
    sequence: Literal["t1", "t2"],
    n_cases: Optional[int] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Cyst-X data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence: The choice of MRI sequence. Either 't1' or 't2'.
        n_cases: The number of cases to use, sorted by case id. By default all cases are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cyst_x_data(path, sequence, n_cases, download=download)

    raw_paths = sorted(glob(os.path.join(data_dir, "images", "*.nii.gz")))
    if n_cases is not None:
        raw_paths = raw_paths[:n_cases]
    label_paths = [os.path.join(data_dir, "masks", os.path.basename(p)) for p in raw_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_cyst_x_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sequence: Literal["t1", "t2"],
    n_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Cyst-X dataset for whole-pancreas segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        sequence: The choice of MRI sequence. Either 't1' or 't2'.
        n_cases: The number of cases to use, sorted by case id. By default all cases are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cyst_x_paths(path, sequence, n_cases, download)

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


def get_cyst_x_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    sequence: Literal["t1", "t2"],
    n_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Cyst-X dataloader for whole-pancreas segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sequence: The choice of MRI sequence. Either 't1' or 't2'.
        n_cases: The number of cases to use, sorted by case id. By default all cases are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cyst_x_dataset(path, patch_shape, sequence, n_cases, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
