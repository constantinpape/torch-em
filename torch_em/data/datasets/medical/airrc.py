"""AirRC contains annotations for pulmonary artery, vein and airway (lumen and wall)
segmentation in chest CT.

The dataset consists of 3D masks for 254 CT scans, manually corrected and multi-expert
verified. The masks are distributed on figshare, while the raw CT volumes are the original
scans of the LUNA16 dataset (https://luna16.grand-challenge.org/, itself a re-release of a
subset of LIDC-IDRI), which are downloaded from Zenodo and matched to the AirRC masks by their
DICOM Series Instance UID.

NOTE: The AirRC masks are semantic label volumes with 4 foreground classes, following
`LABEL_IDS`: 1 = pulmonary artery, 2 = pulmonary vein, 3 = airway lumen, 4 = airway wall. The
masks are on their own grid (a resampled, isotropic 1mm crop around the lungs), which does not
match the grid of the original LUNA16 CT. This module therefore resamples the matching LUNA16
CT volume onto the mask grid (trilinear interpolation) and stores the pair in a single hdf5
file per case.

The AirRC masks are located at https://doi.org/10.6084/m9.figshare.26878867 (figshare, CC BY
4.0). The LUNA16 CT volumes are located at https://doi.org/10.5281/zenodo.3723295 (subsets 0-6)
and https://doi.org/10.5281/zenodo.4121926 (subsets 7-9), both public domain / CC BY 3.0.

This dataset is from the publication https://doi.org/10.1038/s41597-025-06074-6. The LUNA16 CT
volumes are from https://doi.org/10.1016/j.media.2017.06.015 (via LIDC-IDRI,
https://doi.org/10.1118/1.3528204). Please cite them if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABELS_URLS = {
    "metadata": "https://ndownloader.figshare.com/files/57305609",
    "labelsTr.zip": "https://ndownloader.figshare.com/files/57334925",
    "labelsTr.z01": "https://ndownloader.figshare.com/files/57334910",
    "labelsTr.z02": "https://ndownloader.figshare.com/files/57334913",
    "labelsTr.z03": "https://ndownloader.figshare.com/files/57334916",
    "labelsTr.z04": "https://ndownloader.figshare.com/files/57334919",
    "labelsTr.z05": "https://ndownloader.figshare.com/files/57334922",
    "labelsTr.z06": "https://ndownloader.figshare.com/files/57334928",
    "labelsTr.z07": "https://ndownloader.figshare.com/files/57334931",
}

LABELS_CHECKSUMS = {
    "metadata": "77a0e10d152517d8840ff1298af18f25ce7830ff13d665f60449ed17ce0836ed",
    "labelsTr.zip": "6a3c5c39c14da117e9aea2836309cee528657cd130974e47acc5cffc6649b098",
    "labelsTr.z01": "cd36b2e58a9b37d38ef3a55ea7b333cb21516a8d55df2c80900f7f1072f945ba",
    "labelsTr.z02": "9d88bb8a6dbd1371d5bd17e8d2c48d2ddaac9c5cfdb4863d3f0a9fa162077f2e",
    "labelsTr.z03": "828a919e6f835da98615bee9f0b06ea93f310830c9988b2c350f87d10e35ea5b",
    "labelsTr.z04": "174418141b2f069c6fa4e70b79ccbdb44a3d786c60d1a29120b90f6748198f9b",
    "labelsTr.z05": "50f0441d66d6db6c92d6b43986aac8438e2f566003a5b893b590e664193fbb59",
    "labelsTr.z06": "f7fb0b38cc2746f993c508a3ba24e0f4934379efdc0fa3a9cc943fc6c5325339",
    "labelsTr.z07": "fce0ee59d6426699ebda7bfc1acb84a786c09989564f4d2479577fdd70f771df",
}

# The LUNA16 CT subsets are hosted across two Zenodo records, split because the full dataset
# exceeds the size Zenodo allows in a single record. Only sha256 checksums are recorded for
# the subsets that were used to validate this module; the Zenodo records only provide md5
# checksums for the rest, which `download_source` (sha256-only) cannot verify against.
LUNA16_URLS = {
    "subset0": "https://zenodo.org/records/3723295/files/subset0.zip",
    "subset1": "https://zenodo.org/records/3723295/files/subset1.zip",
    "subset2": "https://zenodo.org/records/3723295/files/subset2.zip",
    "subset3": "https://zenodo.org/records/3723295/files/subset3.zip",
    "subset4": "https://zenodo.org/records/3723295/files/subset4.zip",
    "subset5": "https://zenodo.org/records/3723295/files/subset5.zip",
    "subset6": "https://zenodo.org/records/3723295/files/subset6.zip",
    "subset7": "https://zenodo.org/records/4121926/files/subset7.zip",
    "subset8": "https://zenodo.org/records/4121926/files/subset8.zip",
    "subset9": "https://zenodo.org/records/4121926/files/subset9.zip",
}

LUNA16_CHECKSUMS = {f"subset{i}": None for i in range(10)}

LABEL_IDS = {"background": 0, "artery": 1, "vein": 2, "airway_lumen": 3, "airway_wall": 4}
"""The semantic label ids of the AirRC classes."""


def _get_airrc_labels(path, download):
    label_dir = os.path.join(path, "labelsTr")
    if os.path.exists(label_dir):
        return label_dir

    os.makedirs(path, exist_ok=True)

    metadata_path = os.path.join(path, "metadata.xlsx")
    util.download_source(
        path=metadata_path, url=LABELS_URLS["metadata"], download=download, checksum=LABELS_CHECKSUMS["metadata"]
    )

    # The labels are distributed as a split zip archive (labelsTr.zip + labelsTr.z01 - z07),
    # which have to be joined into a single archive before they can be extracted.
    for name in ["labelsTr.zip"] + [f"labelsTr.z{i:02d}" for i in range(1, 8)]:
        part_path = os.path.join(path, name)
        util.download_source(path=part_path, url=LABELS_URLS[name], download=download, checksum=LABELS_CHECKSUMS[name])

    combined_path = os.path.join(path, "labelsTr_combined.zip")
    if not os.path.exists(combined_path):
        import subprocess
        subprocess.run(
            ["zip", "-FF", os.path.join(path, "labelsTr.zip"), "--out", combined_path], check=True, cwd=path
        )

    util.unzip(zip_path=combined_path, dst=path, remove=False)
    return label_dir


def _get_luna16_raw(path, download, subset_ids):
    raw_dir = os.path.join(path, "luna16")
    os.makedirs(raw_dir, exist_ok=True)

    for i in subset_ids:
        name = f"subset{i}"
        marker_path = os.path.join(path, f"{name}.extracted")
        if os.path.exists(marker_path):
            continue

        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=LUNA16_URLS[name], download=download, checksum=LUNA16_CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=raw_dir, remove=False)
        open(marker_path, "w").close()

    return raw_dir


def _preprocess_airrc(label_dir, raw_dir, preprocessed_dir):
    import h5py
    import SimpleITK as sitk

    os.makedirs(preprocessed_dir, exist_ok=True)

    # Each LUNA16 subset zip unpacks into its own 'subsetN' sub-folder, so the raw volumes have
    # to be looked up recursively rather than directly under `raw_dir`.
    raw_paths_by_uid = {
        os.path.basename(p)[:-len(".mhd")]: p for p in glob(os.path.join(raw_dir, "**", "*.mhd"), recursive=True)
    }

    label_paths = natsorted(glob(os.path.join(label_dir, "*.nii.gz")))
    for label_path in tqdm(label_paths, desc="Preprocessing the AirRC scans"):
        series_uid = os.path.basename(label_path)[:-len(".nii.gz")]

        out_path = os.path.join(preprocessed_dir, f"{series_uid}.h5")
        if os.path.exists(out_path):
            continue

        raw_path = raw_paths_by_uid.get(series_uid)
        if raw_path is None:  # The matching LUNA16 subset was not downloaded.
            continue

        label_img = sitk.ReadImage(label_path)
        raw_img = sitk.ReadImage(raw_path)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(label_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)  # air / outside-scan HU value
        resampled_raw = resampler.Execute(raw_img)

        raw = sitk.GetArrayFromImage(resampled_raw).astype("float32")
        labels = sitk.GetArrayFromImage(label_img).astype("uint8")
        assert raw.shape == labels.shape, f"Shape mismatch for {series_uid}: {raw.shape} vs. {labels.shape}."

        with h5py.File(f"{out_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        os.rename(f"{out_path}.tmp", out_path)


def get_airrc_data(
    path: Union[os.PathLike, str], subset_ids: Optional[List[int]] = None, download: bool = False
) -> str:
    """Download the AirRC dataset and the matching LUNA16 CT volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset_ids: The LUNA16 subset ids (0-9) to download and match against the AirRC masks.
            By default, all ten subsets are downloaded (around 65 GB). Restricting this list
            only yields the AirRC cases whose CT volume happens to be in the requested subsets.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    if subset_ids is None:
        subset_ids = list(range(10))

    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)

    label_dir = _get_airrc_labels(path, download)
    raw_dir = _get_luna16_raw(path, download, subset_ids)
    _preprocess_airrc(label_dir, raw_dir, preprocessed_dir)

    return preprocessed_dir


def get_airrc_paths(
    path: Union[os.PathLike, str], subset_ids: Optional[List[int]] = None, download: bool = False
) -> List[str]:
    """Get paths to the AirRC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset_ids: The LUNA16 subset ids (0-9) to download and match against the AirRC masks.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the
        label data ('labels').
    """
    preprocessed_dir = get_airrc_data(path, subset_ids, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_airrc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    subset_ids: Optional[List[int]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AirRC dataset for pulmonary artery, vein and airway segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        subset_ids: The LUNA16 subset ids (0-9) to download and match against the AirRC masks.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_airrc_paths(path, subset_ids, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_airrc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    subset_ids: Optional[List[int]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AirRC dataloader for pulmonary artery, vein and airway segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset_ids: The LUNA16 subset ids (0-9) to download and match against the AirRC masks.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_airrc_dataset(path, patch_shape, subset_ids, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
