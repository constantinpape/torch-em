"""The CShaper dataset contains 3D fluorescence microscopy images of Caenorhabditis
elegans early embryos with cell instance segmentation annotations.

The dataset is organised into training and evaluation splits:
- Training: Sample01, Sample02 (27 timepoints each)
- Evaluation: Sample02, Sample03, Sample04 (7 timepoints each)

Each timepoint is a separate 3D NIfTI volume (.nii.gz):
- Raw membrane images: RawMemb/{sample}_{tp}_rawMemb.nii.gz
- Cell segmentation: SegCell/{sample}_{tp}_segCell.nii.gz

NOTE: The data must be downloaded manually. Download the zip from the SharePoint link
provided at https://doi.org/10.6084/m9.figshare.12839315 and place it as
`{path}/OneDrive.zip` (or whatever filename it downloads as).

The dataset is from the publication https://doi.org/10.1093/bioinformatics/btab710.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# Root path inside the zip after extraction
_ZIP_ROOT = "CShaper Supplementary Data/DMapNet Training and Evaluation"

TRAIN_SAMPLES = ["Sample01", "Sample02"]
EVAL_SAMPLES = ["Sample02", "Sample03", "Sample04"]


def get_cshaper_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Extract the CShaper dataset zip.

    NOTE: The zip must be downloaded manually from the SharePoint link at
    https://doi.org/10.6084/m9.figshare.12839315 and placed inside `path`.
    Any zip file found in `path` will be extracted automatically.

    Args:
        path: Filepath to a folder containing the downloaded CShaper zip.
        download: Ignored (manual download required).

    Returns:
        The filepath to the extracted data root directory.
    """
    data_dir = os.path.join(path, _ZIP_ROOT)
    if os.path.exists(data_dir):
        return data_dir

    # Find any zip in path
    zips = glob(os.path.join(path, "*.zip"))
    if not zips:
        raise RuntimeError(
            f"No zip file found in {path}. "
            "Please download the CShaper data manually from the SharePoint link at "
            "https://doi.org/10.6084/m9.figshare.12839315 and place the zip in `path`."
        )

    util.unzip(zips[0], path)
    return data_dir


def _convert_to_h5(data_dir: str, split: str) -> str:
    """Convert NIfTI timepoint files to per-timepoint HDF5 files.

    Args:
        data_dir: The extracted CShaper root directory.
        split: "train" or "val".

    Returns:
        The directory containing the converted HDF5 files.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError(
            "The 'nibabel' package is required to read CShaper NIfTI files. "
            "Install with: pip install nibabel"
        )
    import h5py

    split_subdir = "TrainingData" if split == "train" else "EvaluationData"
    split_dir = os.path.join(data_dir, split_subdir)

    h5_dir = os.path.join(data_dir, f"h5_{split}")
    if os.path.exists(h5_dir) and len(glob(os.path.join(h5_dir, "*.h5"))) > 0:
        return h5_dir
    os.makedirs(h5_dir, exist_ok=True)

    sample_dirs = natsorted([
        d for d in glob(os.path.join(split_dir, "*/")) if os.path.isdir(d)
    ])

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir.rstrip("/"))
        raw_files = natsorted(glob(os.path.join(sample_dir, "RawMemb", "*.nii.gz")))
        seg_dir = os.path.join(sample_dir, "SegCell")

        for raw_path in raw_files:
            # e.g. Sample01_030_rawMemb.nii.gz → Sample01_030
            basename = os.path.basename(raw_path)
            tp_stem = basename.replace("_rawMemb.nii.gz", "")
            h5_path = os.path.join(h5_dir, f"{tp_stem}.h5")

            if os.path.exists(h5_path):
                continue

            seg_path = os.path.join(seg_dir, f"{tp_stem}_segCell.nii.gz")
            if not os.path.exists(seg_path):
                continue

            raw_vol = nib.load(raw_path).get_fdata().astype("float32")
            seg_vol = nib.load(seg_path).get_fdata().astype("int32")

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("raw", data=raw_vol, compression="gzip")
                f.create_dataset("labels", data=seg_vol, compression="gzip")

    return h5_dir


def get_cshaper_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"] = "train",
    samples: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CShaper data.

    Args:
        path: Filepath to a folder containing the downloaded CShaper zip.
        split: The data split to use. Either "train" (Sample01, Sample02) or
            "val" (Sample02, Sample03, Sample04).
        samples: Optional list of sample names to restrict to (e.g., ["Sample01"]).
            If None, all samples for the split are used.
        download: Ignored (manual download required).

    Returns:
        List of filepaths for the HDF5 image data (key: "raw").
        List of filepaths for the HDF5 label data (key: "labels").
    """
    if split not in ("train", "val"):
        raise ValueError(f"Invalid split '{split}'. Choose 'train' or 'val'.")

    data_dir = get_cshaper_data(path, download)
    h5_dir = _convert_to_h5(data_dir, split)

    h5_files = natsorted(glob(os.path.join(h5_dir, "*.h5")))

    if len(h5_files) == 0:
        raise RuntimeError(f"No HDF5 files found in {h5_dir}. Check the dataset structure.")

    if samples is not None:
        h5_files = [p for p in h5_files if any(os.path.basename(p).startswith(s) for s in samples)]

    return h5_files, h5_files


def get_cshaper_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val"] = "train",
    samples: Optional[List[str]] = None,
    raw_key: str = "raw",
    label_key: str = "labels",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CShaper dataset for C. elegans embryo cell segmentation.

    Args:
        path: Filepath to a folder containing the downloaded CShaper zip.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either "train" or "val".
        samples: Optional list of sample names to restrict to (e.g., ["Sample01"]).
        raw_key: The HDF5 key for raw image data.
        label_key: The HDF5 key for label data.
        download: Ignored (manual download required).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cshaper_paths(path, split, samples, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_cshaper_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val"] = "train",
    samples: Optional[List[str]] = None,
    raw_key: str = "raw",
    label_key: str = "labels",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CShaper dataloader for C. elegans embryo cell segmentation.

    Args:
        path: Filepath to a folder containing the downloaded CShaper zip.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either "train" or "val".
        samples: Optional list of sample names to restrict to (e.g., ["Sample01"]).
        raw_key: The HDF5 key for raw image data.
        label_key: The HDF5 key for label data.
        download: Ignored (manual download required).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cshaper_dataset(path, patch_shape, split, samples, raw_key, label_key, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
