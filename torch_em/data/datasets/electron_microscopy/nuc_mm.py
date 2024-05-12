import os
from glob import glob

import h5py
import torch_em

from .. import util

URL = "https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8"


def _extract_split(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob(os.path.join(image_folder, "*.h5")))
    label_files = sorted(glob(os.path.join(label_folder, "*.h5")))
    assert len(image_files) == len(label_files)
    for image, label in zip(image_files, label_files):
        with h5py.File(image, "r") as f:
            vol = f["main"][:]
        with h5py.File(label, "r") as f:
            seg = f["main"][:]
        assert vol.shape == seg.shape
        out_path = os.path.join(output_folder, os.path.basename(image))
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=vol, compression="gzip")
            f.create_dataset("labels", data=seg, compression="gzip")


def _require_dataset(path, sample, download):
    # downloading the dataset
    util.download_source_gdrive(path, URL, download, download_type="folder")

    if sample == "mouse":
        input_folder = os.path.join(path, "Mouse (NucMM-M)")
    else:
        input_folder = os.path.join(path, "Zebrafish (NucMM-Z)")
    assert os.path.exists(input_folder), input_folder

    sample_folder = os.path.join(path, sample)
    _extract_split(
        os.path.join(input_folder, "Image", "train"), os.path.join(input_folder, "Label", "train"),
        os.path.join(sample_folder, "train")
    )
    _extract_split(
        os.path.join(input_folder, "Image", "val"), os.path.join(input_folder, "Label", "val"),
        os.path.join(sample_folder, "val")
    )


def get_nuc_mm_dataset(path, sample, split, patch_shape, download=False, **kwargs):
    """Dataset for the segmentation of nuclei in EM and X-Ray.

    This dataset is from the publication https://doi.org/10.1007/978-3-030-87193-2_16.
    Please cite it if you use this dataset for a publication.
    """
    assert sample in ("mouse", "zebrafish")
    assert split in ("train", "val")

    sample_folder = os.path.join(path, sample)
    if not os.path.exists(sample_folder):
        _require_dataset(path, sample, download)

    split_folder = os.path.join(sample_folder, split)
    paths = sorted(glob(os.path.join(split_folder, "*.h5")))

    raw_key, label_key = "raw", "labels"
    return torch_em.default_segmentation_dataset(
        paths, raw_key, paths, label_key, patch_shape, is_seg_dataset=True, **kwargs
    )


def get_nuc_mm_loader(path, sample, split, patch_shape, batch_size, download=False, **kwargs):
    """Dataset for the segmentation of nuclei in EM and X-Ray. See 'get_nuc_mm_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_nuc_mm_dataset(path, sample, split, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
