"""The LNDb dataset contains annotations for lung nodule segmentation in chest CT.

It consists of 294 chest CT that were collected at the Centro Hospitalar e Universitario de Sao Joao in Porto
between 2016 and 2018. The findings were marked by up to three radiologists out of a group of five, who segmented
every finding that they considered a nodule with a diameter of at least 3 mm. Non-nodules and nodules below 3 mm
were only marked with a centroid and are not segmented. This module uses the 236 CT of the challenge training set,
which are the only scans with public segmentations. The 58 test CT are only released in the newer version of the
dataset (https://doi.org/10.5281/zenodo.7153205) and come with nodule centroids, but without segmentations, so
they are not included here.

The CT are distributed as MetaImage ('mhd' / 'raw') volumes and the annotations as one mask per radiologist, in
which the voxel values are the finding ids of that radiologist. This module joins the per-radiologist findings
into nodules with the official matching of 'trainNodules_gt.csv' (findings of different radiologists are the same
nodule if their centroids are closer than the maximum of their radii and 3 mm) and stores the scan and the nodule
masks together in hdf5 files. Each nodule gets one instance id and the files contain the nodule masks for all
consensus levels ('labels/consensus_1' to 'labels/consensus_3'), where consensus level n contains the voxels that
at least n radiologists marked as part of the nodule (level 1 is the union of all readings). The number of
radiologists that annotated each voxel is stored in 'labels/n_readers' and the number of radiologists that read
the scan in the 'n_readers' attribute (59 scans were read by one, 110 by two and 67 by three radiologists,
so the higher consensus levels are empty for the scans with fewer readers). Findings that all readers
considered non-nodules and findings that were not segmented are not included, while the findings below the
equivalent diameter of 3 mm that the official example scripts filter out are kept.

The volumes are converted to the (Z, Y, X) axis order, which is the order the MetaImage files are stored in.

The data is located at https://doi.org/10.5281/zenodo.6613714. NOTE: The license field of the Zenodo record
states CC BY 4.0, but its description states CC BY-NC-ND 4.0, so please make sure that you are allowed to use
the data for your purpose.

This dataset is from the publication https://doi.org/10.48550/arXiv.1911.08434 and the challenge it was used for
is described in https://doi.org/10.1016/j.media.2021.102027.
Please cite them if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "data0": "https://zenodo.org/records/6613714/files/data0.rar",
    "data1": "https://zenodo.org/records/6613714/files/data1.rar",
    "data2": "https://zenodo.org/records/6613714/files/data2.rar",
    "data3": "https://zenodo.org/records/6613714/files/data3.rar",
    "data4": "https://zenodo.org/records/6613714/files/data4.rar",
    "data5": "https://zenodo.org/records/6613714/files/data5.rar",
    "masks": "https://zenodo.org/records/6613714/files/masks.rar",
    "csvs": "https://zenodo.org/records/6613714/files/trainset_csv.zip",
}

CHECKSUMS = {
    "data0": "2ae6b7b760f5286bde2f51b32fcdedbf4e1e35aee575cb971bbde5d1aef517d7",
    "data1": "f5b33500127e09c0c320561c9c72e72a7934f02214bb091d717d95ed27977057",
    "data2": "a8819b5ad36023cb85d1bf840a136c413bbac38dbd3cbc2b4bddb2ef43acf0cf",
    "data3": "05a4b817c112aec650e78e6ba6908737a26550510de5a2b62d2b8c0dbc3059e7",
    "data4": "ffa1a2de6525a1409c7d5a0d0bbdf9ce758219cca7914c08471adf3cb4405d15",
    "data5": "6e94ebcc2650879feb6928b30f8bd6f8675a1c0557d9e7c06852fba81e5cea6d",
    "masks": "cdc9cfbbb868ec6fc6e2612afc9bc12cd6e07f8d345610a4dffbf76e1bcc7980",
    "csvs": "cda2e6377344dd90b7926f2c10da29d5260ca59337bd8f1c1490639ffa8fbb94",
}

# The nodule masks are instance labels, so the label ids are the nodule ids of a scan.
LABEL_IDS = {"background": 0, "nodule": "1, 2, ... (one id per nodule)"}

MAX_CONSENSUS_LEVEL = 3

N_VOLUMES = 236

N_MASKS = 480

METAIMAGE_DTYPES = {
    "MET_CHAR": "int8",
    "MET_UCHAR": "uint8",
    "MET_SHORT": "int16",
    "MET_USHORT": "uint16",
    "MET_INT": "int32",
    "MET_UINT": "uint32",
    "MET_FLOAT": "float32",
    "MET_DOUBLE": "float64",
}


def _read_metaimage(mhd_path):
    """Read a MetaImage volume with the axis order (Z, Y, X).

    The header is a plain text file that refers to the raw data of the volume, so no additional dependency
    (such as SimpleITK) is needed to read it.
    """
    header = {}
    with open(mhd_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()

    shape = [int(v) for v in header["DimSize"].split()][::-1]
    dtype = METAIMAGE_DTYPES[header["ElementType"]]
    raw_path = os.path.join(os.path.dirname(mhd_path), os.path.basename(header["ElementDataFile"]))
    return np.fromfile(raw_path, dtype=dtype).reshape(shape)


def _read_csv(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def _get_findings_per_scan(csv_dir):
    """Group the joined findings of 'trainNodules_gt.csv' by scan.

    Each finding is a list of the (radiologist id, finding id of that radiologist) pairs that were joined into it.
    The findings that all readers considered non-nodules are skipped.
    """
    findings = defaultdict(list)
    for row in _read_csv(os.path.join(csv_dir, "trainNodules_gt.csv")):
        if int(row["Nodule"]) != 1:
            continue
        rad_ids = [int(v) for v in row["RadID"].split(",")]
        rad_finding_ids = [int(v) for v in row["RadFindingID"].split(",")]
        findings[int(row["LNDbID"])].append(list(zip(rad_ids, rad_finding_ids)))
    return findings


def _build_nodule_labels(mask_dir, scan_id, findings, shape):
    """Derive the instance labels for all consensus levels from the per-radiologist masks of one scan."""
    masks = {}
    for mask_path in glob(os.path.join(mask_dir, f"LNDb-{scan_id:04d}_rad*.mhd")):
        rad_id = int(os.path.basename(mask_path).split("_rad")[1][:-len(".mhd")])
        masks[rad_id] = _read_metaimage(mask_path)

    n_readers = np.zeros(shape, dtype="uint8")
    consensus = {level: np.zeros(shape, dtype="uint16") for level in range(1, MAX_CONSENSUS_LEVEL + 1)}

    nodule_id = 0
    for finding in findings:
        # Count how many radiologists annotated each voxel as part of this nodule.
        counts = np.zeros(shape, dtype="uint8")
        for rad_id, rad_finding_id in finding:
            if rad_id in masks:
                counts += (masks[rad_id] == rad_finding_id)

        # Findings that were only marked with a centroid are not segmented and are skipped.
        if not counts.any():
            continue

        nodule_id += 1
        n_readers = np.maximum(n_readers, counts)
        for level, labels in consensus.items():
            labels[counts >= level] = nodule_id

    return consensus, n_readers, len(masks)


def _preprocess_lndb(data_dir, mask_dir, csv_dir, preprocessed_dir):
    import h5py

    findings_per_scan = _get_findings_per_scan(csv_dir)
    scan_ids = [int(row["LNDbID"]) for row in _read_csv(os.path.join(csv_dir, "trainCTs.csv"))]
    os.makedirs(preprocessed_dir, exist_ok=True)

    for scan_id in tqdm(scan_ids, desc="Preprocessing the LNDb scans"):
        volume_path = os.path.join(preprocessed_dir, f"LNDb-{scan_id:04d}.h5")
        if os.path.exists(volume_path):
            continue

        raw = _read_metaimage(os.path.join(data_dir, f"LNDb-{scan_id:04d}.mhd"))
        consensus, n_readers, n_masks = _build_nodule_labels(
            mask_dir, scan_id, findings_per_scan[scan_id], raw.shape
        )

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.attrs["scan_id"] = scan_id
            f.attrs["n_readers"] = n_masks
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/n_readers", data=n_readers, compression="gzip")
            for level, labels in consensus.items():
                f.create_dataset(f"labels/consensus_{level}", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_lndb_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LNDb dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_VOLUMES:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    # The scans are spread over six rar archives, which all unpack into the same folder. A marker file is written
    # for every archive that was unpacked, so that an interrupted run does not download the 24 GB again.
    data_dir = os.path.join(path, "data")
    for name in [f"data{i}" for i in range(6)]:
        marker_path = os.path.join(path, f"{name}.extracted")
        if os.path.exists(marker_path):
            continue
        rar_path = os.path.join(path, f"{name}.rar")
        util.download_source(path=rar_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
        util.unzip_rarfile(rar_path=rar_path, dst=data_dir)
        open(marker_path, "w").close()

    # There is one mask per radiologist and scan, which gives 480 masks for the 236 scans.
    mask_dir = os.path.join(path, "masks")
    if len(glob(os.path.join(mask_dir, "*.mhd"))) != N_MASKS:
        rar_path = os.path.join(path, "masks.rar")
        util.download_source(path=rar_path, url=URLS["masks"], download=download, checksum=CHECKSUMS["masks"])
        util.unzip_rarfile(rar_path=rar_path, dst=path)

    csv_dir = os.path.join(path, "trainset_csv")
    if not os.path.exists(os.path.join(csv_dir, "trainNodules_gt.csv")):
        zip_path = os.path.join(path, "trainset_csv.zip")
        util.download_source(path=zip_path, url=URLS["csvs"], download=download, checksum=CHECKSUMS["csvs"])
        util.unzip(zip_path=zip_path, dst=csv_dir)

    _preprocess_lndb(data_dir, mask_dir, csv_dir, preprocessed_dir)
    return preprocessed_dir


def get_lndb_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the LNDb data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data
        ('labels/consensus_<level>' and 'labels/n_readers').
    """
    data_dir = get_lndb_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(volume_paths) > 0

    return volume_paths


def get_lndb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    consensus_level: int = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LNDb dataset for lung nodule segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        consensus_level: The minimum number of radiologists (1 to 3) that have to agree on a voxel for it to be part
            of a nodule. 1 corresponds to the union of all radiologist annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if consensus_level not in range(1, MAX_CONSENSUS_LEVEL + 1):
        raise ValueError(f"'{consensus_level}' is not a valid consensus level. Please choose a value from 1 to 3.")

    volume_paths = get_lndb_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/consensus_{consensus_level}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_lndb_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    consensus_level: int = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LNDb dataloader for lung nodule segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        consensus_level: The minimum number of radiologists (1 to 3) that have to agree on a voxel for it to be part
            of a nodule. 1 corresponds to the union of all radiologist annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lndb_dataset(path, patch_shape, consensus_level, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
