"""The Macrophage / A431 dataset contains FIB-SEM and SBF-SEM volumes of human primary macrophages and A431 cells
with semantic annotations of organelles (nucleus, lipid droplets and, for one volume, also mitochondria,
Golgi apparatus and endoplasmic reticulum).

It contains seven volumes:
- fib_tnf: FIB-SEM of a macrophage treated with TNFa for 1 h (2719 slices; nucleus, lipid droplets).
- fib_control: FIB-SEM of a macrophage in control medium (2888 slices; mitochondria, golgi, er, nucleus,
  lipid droplets).
- sbf_a431: SBF-SEM of an A431 cell (162 slices; nucleus, lipid droplets).
- sbf_oa_chol_7t, sbf_oa_chol_av1, sbf_oa_chol_av3: SBF-SEM of macrophages loaded with oleic acid / cholesterol
  (143 / 154 / 207 slices; nucleus, lipid droplets touching the nucleus).
- sbf_no_load: SBF-SEM of a macrophage in control medium (257 slices; nucleus, lipid droplets touching the nucleus).

The annotations are quick semantic segmentations (created with Microscopy Image Browser) and are not necessarily
dense: the lipid droplet labels of the SBF-SEM volumes only cover the droplets close to the nucleus.

The data is located at https://www.ebi.ac.uk/empiar/EMPIAR-13420/ (https://doi.org/10.6019/EMPIAR-13420).
Please cite the EMPIAR entry if you use this dataset in your research.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/13420/data"

# For each volume: the folder, the folders and file patterns of the image / label slices, the number of slices
# and the label ids of the organelles. NOTE: The label ids differ between the volumes. If an organelle has
# several ids they are merged into a single binary mask.
VOLUMES = {
    "fib_tnf": {
        "folder": "FIB/240729_HumanMP_TNFa_1U",
        "raw_dir": "TIFs", "raw": "240729_HumanMP_TNFa_1U_{:04d}.tif",
        "label_dir": "Labels_NucleusLDs_TIFs", "label": "Labels_240729_HumanMP_TNFa_1U_nucleus_LD_{:04d}.tif",
        "n_slices": 2719,
        "organelles": {"nucleus": [1], "lipid_droplets": [2]},
    },
    "fib_control": {
        "folder": "FIB/240801_HumanMP_Control",
        "raw_dir": "TIFs", "raw": "240801_HumanMP_Control_{:04d}.tif",
        "label_dir": "Labels_Combined_model_TIFs", "label": "Labels_240801_HumanMP_Control_combined_{:04d}.tif",
        "n_slices": 2888,
        "organelles": {"mitochondria": [1], "golgi": [2], "er": [3], "nucleus": [4], "lipid_droplets": [5]},
    },
    "sbf_a431": {
        "folder": "SBFSEM/231106_A431_CLEM_CM1_3J",
        "raw_dir": "TIFs", "raw": "231106_A431_CLEM_CM1_3J_R01_{:03d}.tif",
        "label_dir": "Labels_TIF", "label": "Labels_231106_A431_CLEM_CM1_3J_R01_LDs_localization_{:03d}.tif",
        "n_slices": 162,
        "organelles": {"nucleus": [3], "lipid_droplets": [1, 2]},
    },
    "sbf_oa_chol_7t": {
        "folder": "SBFSEM/240306_Microphages_OA-Chol_7T",
        "raw_dir": "TIFs", "raw": "240306_OA-Chol_7T_R02_{:03d}.tif",
        "label_dir": "Labels_TIF", "label": "Labels_240306_OA-Chol_7T_R02_{:03d}.tif",
        "n_slices": 143,
        "organelles": {"nucleus": [1], "lipid_droplets": [2]},
    },
    "sbf_oa_chol_av1": {
        "folder": "SBFSEM/240307_Microphages_OA-Chol_AV-1",
        "raw_dir": "TIF", "raw": "240307_OA-Chol_AV_R02_AV-1_{:03d}.tif",
        "label_dir": "Labels_TIF", "label": "Labels_240307_OA-Chol_AV_R02_AV-1_{:03d}.tif",
        "n_slices": 154,
        "organelles": {"nucleus": [1], "lipid_droplets": [2]},
    },
    "sbf_oa_chol_av3": {
        "folder": "SBFSEM/240307_Microphages_OA-Chol_AV-3",
        "raw_dir": "TIFs", "raw": "240307_OA-Chol_AV_R01_AV-3_{:03d}.tif",
        "label_dir": "Labels_TIF", "label": "Labels_240307_OA-Chol_AV_R01_AV-3_{:03d}.tif",
        "n_slices": 207,
        "organelles": {"nucleus": [1], "lipid_droplets": [2]},
    },
    "sbf_no_load": {
        "folder": "SBFSEM/240424_Microphages_no_load",
        "raw_dir": "TIFs", "raw": "240424_Microphages_no_load_R02_{:03d}.tif",
        "label_dir": "Labels_TIF", "label": "Labels_240424_Microphages_no_load_R02_Nuclei_{:03d}.tif",
        "n_slices": 257,
        "organelles": {"nucleus": [1], "lipid_droplets": [2, 3]},
    },
}
# The urls of the first image slice of each volume. The urls of all other slices follow the file patterns above.
URLS = {
    name: f"{BASE_URL}/{v['folder']}/dataset/{v['raw_dir']}/{v['raw'].format(1)}" for name, v in VOLUMES.items()
}
# NOTE: There are no checksums, since the volumes are distributed as thousands of individual tif files.
CHECKSUMS = None


def _download_slice(args):
    import tifffile

    url, tmp_path, download = args
    util.download_source(tmp_path, url, download, checksum=CHECKSUMS)
    data = tifffile.imread(tmp_path)
    os.remove(tmp_path)
    return data


def _slice_args(info, kind, tmp_dir, download):
    if kind == "raw":
        sub, folder, pattern = "dataset", info["raw_dir"], info["raw"]
    else:
        sub, folder, pattern = "labels", info["label_dir"], info["label"]
    return [
        (f"{BASE_URL}/{info['folder']}/{sub}/{folder}/{pattern.format(i + 1)}",
         os.path.join(tmp_dir, f"{kind}_{i}.tif"), download)
        for i in range(info["n_slices"])
    ]


def _download_volume(volume_path, name, download):
    import h5py

    info = VOLUMES[name]
    n_slices = info["n_slices"]
    organelles = info["organelles"]

    tmp_dir = f"{volume_path}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    raw_args = _slice_args(info, "raw", tmp_dir, download)
    label_args = _slice_args(info, "label", tmp_dir, download)

    tmp_h5 = f"{volume_path}.incomplete"
    ds_raw, ds_labels = None, {}
    # Download and convert the data in blocks of slices to keep the memory footprint low.
    step = 16
    with h5py.File(tmp_h5, "w") as f, ThreadPoolExecutor(4) as pool:
        for start in range(0, n_slices, step):
            raw = np.stack(list(pool.map(_download_slice, raw_args[start:start + step])))
            labels = np.stack(list(pool.map(_download_slice, label_args[start:start + step])))
            assert raw.shape == labels.shape, f"Shape mismatch of raw and labels: {raw.shape}, {labels.shape}"

            if ds_raw is None:
                shape = (n_slices,) + raw.shape[1:]
                chunks = (1, min(512, shape[1]), min(512, shape[2]))
                ds_raw = f.create_dataset("raw", shape=shape, dtype=raw.dtype, chunks=chunks, compression="gzip")
                for organelle in organelles:
                    ds_labels[organelle] = f.create_dataset(
                        f"labels/{organelle}", shape=shape, dtype="uint8", chunks=chunks, compression="gzip"
                    )

            stop = start + len(raw)
            ds_raw[start:stop] = raw
            for organelle, ids in organelles.items():
                ds_labels[organelle][start:stop] = np.isin(labels, ids).astype("uint8")

    os.replace(tmp_h5, volume_path)
    os.rmdir(tmp_dir)


def get_macrophage_a431_data(
    path: Union[os.PathLike, str],
    name: Literal[
        "fib_tnf", "fib_control", "sbf_a431", "sbf_oa_chol_7t", "sbf_oa_chol_av1", "sbf_oa_chol_av3", "sbf_no_load"
    ],
    download: bool = False,
) -> str:
    """Download a volume of the Macrophage / A431 dataset.

    The volumes are large (the FIB-SEM volumes have ~3 GB, the SBF-SEM volumes ~4-12 GB). They are downloaded
    slice by slice and converted to an hdf5 file with the raw data and one binary label volume per organelle.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the volume.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the hdf5 file with the volume.
    """
    if name not in VOLUMES:
        raise ValueError(f"'{name}' is not a valid volume name. Please choose from {list(VOLUMES)}.")

    volume_path = os.path.join(path, f"{name}.h5")
    if os.path.exists(volume_path):
        return volume_path

    if not download:
        raise RuntimeError(f"Cannot find the data at {volume_path}, but download was set to False")

    os.makedirs(path, exist_ok=True)
    _download_volume(volume_path, name, download)
    return volume_path


def get_macrophage_a431_paths(
    path: Union[os.PathLike, str],
    names: Union[str, List[str], None] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the Macrophage / A431 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        names: The names of the volumes. By default the 'fib_control' volume is used.
            Choose from 'fib_tnf', 'fib_control', 'sbf_a431', 'sbf_oa_chol_7t', 'sbf_oa_chol_av1',
            'sbf_oa_chol_av3' and 'sbf_no_load'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files.
    """
    if names is None:
        names = ["fib_control"]
    elif isinstance(names, str):
        names = [names]

    return [get_macrophage_a431_data(path, name, download) for name in names]


def get_macrophage_a431_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    names: Union[str, List[str], None] = None,
    organelle: Literal["nucleus", "lipid_droplets", "mitochondria", "golgi", "er"] = "nucleus",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Macrophage / A431 dataset for organelle segmentation in FIB-SEM and SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        names: The names of the volumes. By default the 'fib_control' volume is used.
            Choose from 'fib_tnf', 'fib_control', 'sbf_a431', 'sbf_oa_chol_7t', 'sbf_oa_chol_av1',
            'sbf_oa_chol_av3' and 'sbf_no_load'.
        organelle: The organelle to segment. Either 'nucleus', 'lipid_droplets', 'mitochondria', 'golgi' or 'er'.
            Mitochondria, golgi and er are only available for the 'fib_control' volume.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    all_organelles = {o for v in VOLUMES.values() for o in v["organelles"]}
    if organelle not in all_organelles:
        raise ValueError(f"'{organelle}' is not a valid organelle. Please choose from {sorted(all_organelles)}.")

    if names is None:
        names = ["fib_control"]
    elif isinstance(names, str):
        names = [names]
    for name in names:
        if name in VOLUMES and organelle not in VOLUMES[name]["organelles"]:
            raise ValueError(f"The organelle '{organelle}' is not annotated in '{name}'.")

    volume_paths = get_macrophage_a431_paths(path, names, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 3)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{organelle}",
        patch_shape=patch_shape,
        **kwargs
    )


def get_macrophage_a431_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    names: Union[str, List[str], None] = None,
    organelle: Literal["nucleus", "lipid_droplets", "mitochondria", "golgi", "er"] = "nucleus",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Macrophage / A431 dataloader for organelle segmentation in FIB-SEM and SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        names: The names of the volumes. By default the 'fib_control' volume is used.
            Choose from 'fib_tnf', 'fib_control', 'sbf_a431', 'sbf_oa_chol_7t', 'sbf_oa_chol_av1',
            'sbf_oa_chol_av3' and 'sbf_no_load'.
        organelle: The organelle to segment. Either 'nucleus', 'lipid_droplets', 'mitochondria', 'golgi' or 'er'.
            Mitochondria, golgi and er are only available for the 'fib_control' volume.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_macrophage_a431_dataset(path, patch_shape, names, organelle, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
