"""The SELMA3D dataset contains annotated 3D light-sheet microscopy images of cleared brain tissue.

This loader provides the cell nucleus subset used by the model-ranking benchmark. It contains twelve
fluorescence volumes with binary nucleus annotations. The train, validation and test split follows the
benchmark split from https://github.com/kreshuklab/model_ranking.

The dataset is located at https://doi.org/10.6019/S-BIAD1196 and is available under the CC BY 4.0 license.
It is from the publication https://doi.org/10.48550/arXiv.2501.03880.
Please cite the dataset and publication if you use this dataset in your research.
"""

import os
from typing import List, Literal, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD1196"
DATA_ROOT = os.path.join("SELMA3D_training_annotated", "shannel_cells")
URL_ROOT = f"{BASE_URL}/SELMA3D_training_annotated/shannel_cells"

SPLITS = {
    "train": tuple(f"patchvolume_{sample_id:03d}" for sample_id in range(8)),
    "val": ("patchvolume_008",),
    "test": tuple(f"patchvolume_{sample_id:03d}" for sample_id in range(9, 12)),
}

RAW_CHECKSUMS = {
    "patchvolume_000": "25e1e351872db53a22f2ff9196892e4cefcc44c3d0be98186c13d3c77f8e0397",
    "patchvolume_001": "d12e43353982148b6726110ddef6331c0f057b964924accbd492c03ae2af09b7",
    "patchvolume_002": "74915c737bb470052808e3cdbeb6303fb4c64e81f0a0b0e375057116a9b10590",
    "patchvolume_003": "631a43a1f511f17ae2a2c92f2fb7841fd6f7eab683fcaf0ddaf530b7eb9a9501",
    "patchvolume_004": "f2043a0db9247d1a24349c499712223fc04050c54bbc2d09e4b854905faa5510",
    "patchvolume_005": "297e6e93418497a7f6af4814f08cd03741a21723a0e06ed7003db60ca59d8b9f",
    "patchvolume_006": "b2ec6c7d6d3a6c4d6fc99237a04c2ceb392aaed7505e33387bb10f11f1dc031a",
    "patchvolume_007": "f5e6219cc27a07772d696eb8f2c235b8f5aea96a1e6d1874a1e1254b38c90017",
    "patchvolume_008": "09bf8dac83abfeb575f76feaf4fb252258e78d7631b4c9f51a55460cfdaf9529",
    "patchvolume_009": "7f28921b85cb7a69c3937c3acfe1a4c1fa4091bd8d1d80a0c223e0f1ab82240c",
    "patchvolume_010": "acad1cdb850705d8263482c1a0c196f4ae801ddc6ef1a64e8f1b2132a93da486",
    "patchvolume_011": "acbb1c96da13958c892c9351aa0520988263640e3d247e961d93256bc2970f82",
}

LABEL_CHECKSUMS = {
    "patchvolume_000": "ad59e596aba5bb3562360fa824f9f23474facf2d80242c79edcc5cd8b71f5491",
    "patchvolume_001": "9669072ce2a4cad1a96b5393afc0f10c6f146f3452ebfe6c6e7adbe941ca4eaa",
    "patchvolume_002": "3cebf4f81169b3349971b06a673d2d948bc3dfc929379c9d6664cb35553d8a98",
    "patchvolume_003": "648b71876b83c2d65c8cc5f6ecbaf2176e39b400483ebe233d7d17c52d95fa53",
    "patchvolume_004": "3ad54ed31e99513916f1bdf931b8694e32b524c70791d1dfba8031bc279a1ffb",
    "patchvolume_005": "09b860c25291b1a99f82d790cfa2b6296f50056e863ce8004e68d5c113dab666",
    "patchvolume_006": "839cd4b5a8ca6a54ba8c805fcfbaa03b3fa2e570d4136e497aa611275980d7e4",
    "patchvolume_007": "a067cc4587290745eb8964faadb8037e8a3b48ee0f471efc08ab11fbf9205ec8",
    "patchvolume_008": "908bf1bce5ea16da4eb8275dd39cfbec03bbb330f1e0e3fd59fdb4fbdd16ebaf",
    "patchvolume_009": "2bb38e298870fd783b986e3bb953a648f07dbe50b55e242b2eb2e740984d66ca",
    "patchvolume_010": "41f240844354753118398633278733d14ddbc04f74121f4cfe8f524761e7616f",
    "patchvolume_011": "3568b78e697997995123d270f3ae217659811949d3c79f9298acb2ac93c34522",
}


def _convert_to_h5(raw_path: str, label_path: str, h5_path: str) -> None:
    import h5py
    import nibabel as nib
    import numpy as np

    if os.path.exists(h5_path):
        return

    raw = np.asarray(nib.load(raw_path).dataobj)
    labels = np.asarray(nib.load(label_path).dataobj)

    if raw.ndim != 3 or labels.ndim != 3 or raw.shape != labels.shape:
        raise RuntimeError(
            f"Invalid SELMA3D pair: raw shape {raw.shape} and label shape {labels.shape}."
        )

    # NIfTI stores the spatial axes as XYZ. Transpose them to torch-em's ZYX convention.
    raw = raw.transpose(2, 1, 0).astype("float32", copy=False)
    labels = labels.transpose(2, 1, 0).astype("uint8", copy=False)

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    tmp_path = f"{h5_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, chunks=(32, 128, 128), compression="gzip")
        f.create_dataset("label", data=labels, chunks=(32, 128, 128), compression="gzip")
    os.replace(tmp_path, h5_path)


def get_selma3d_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> str:
    """Download and preprocess the SELMA3D cell nucleus dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed HDF5 data for the selected split.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    data_root = os.path.join(path, DATA_ROOT)
    h5_dir = os.path.join(data_root, "h5", split)
    expected_h5_paths = [os.path.join(h5_dir, f"{sample}.h5") for sample in SPLITS[split]]
    if all(os.path.exists(h5_path) for h5_path in expected_h5_paths):
        return h5_dir

    raw_dir = os.path.join(data_root, "raw")
    label_dir = os.path.join(data_root, "gt")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for sample in SPLITS[split]:
        raw_path = os.path.join(raw_dir, f"{sample}_0000.nii.gz")
        label_path = os.path.join(label_dir, f"{sample}.nii.gz")
        h5_path = os.path.join(h5_dir, f"{sample}.h5")

        raw_url = f"{URL_ROOT}/raw/{sample}_0000.nii.gz"
        label_url = f"{URL_ROOT}/gt/{sample}.nii.gz"
        util.download_source(raw_path, raw_url, download, checksum=RAW_CHECKSUMS[sample])
        util.download_source(label_path, label_url, download, checksum=LABEL_CHECKSUMS[sample])
        _convert_to_h5(raw_path, label_path, h5_path)

    return h5_dir


def get_selma3d_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the SELMA3D HDF5 volumes.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the HDF5 volumes for the selected split.
    """
    h5_dir = get_selma3d_data(path, split, download)
    h5_paths = [os.path.join(h5_dir, f"{sample}.h5") for sample in SPLITS[split]]

    missing_paths = [h5_path for h5_path in h5_paths if not os.path.exists(h5_path)]
    if missing_paths:
        raise RuntimeError(f"Could not find {len(missing_paths)} SELMA3D volumes for split '{split}'.")

    return h5_paths


def get_selma3d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SELMA3D dataset for semantic nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The SELMA3D patch shape must be three-dimensional, got {patch_shape}.")

    h5_paths = get_selma3d_paths(path, split, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="label",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs,
    )


def get_selma3d_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the SELMA3D dataloader for semantic nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_selma3d_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
