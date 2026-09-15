"""The Mediastinal CT datasets contain annotations for mediastinal lymph node and anatomical structure
segmentation in contrast-enhanced chest CT scans of lung cancer patients (St. Olavs University Hospital, Trondheim).

Two datasets are provided, selected via the `task` argument:
- 'lymph_nodes': The benchmark subset of Bouget et al. (2022). 15 CT volumes with 3D lymph node annotations
  (proofread by an expert radiologist). The label volume contains one instance id per lymph node; the mediastinal
  station of each node is listed in 'Benchmark/stations_sto.csv'. The archive also provides binary masks for the
  esophagus, azygos vein, subclavian / carotid arteries and brachiocephalic veins next to the CT of each patient.
- 'structures': The mediastinal CT dataset of Bouget et al. (2019). 15 CT volumes with annotations for 14 mediastinal
  anatomical structures (see `STRUCTURE_IDS`) and, in a separate file, the lymph nodes. The data is distributed as
  MetaImage (.mhd / .raw) files and is converted to nifti once by `get_mediastinal_ct_data`.
  Label ids: 1: esophagus, 2: pulmonary trunk, 3: aortic arch, 4: ascending aorta, 5: descending aorta, 6: azygos vein,
  7: heart, 8: vena cava, 9: brachiocephalic veins, 11: spine, 12: pulmonary vein, 13: subclavian and carotid arteries,
  14: lungs, 16: airways (ids 10 and 15 are not used).

The datasets are located at https://github.com/dbouget/ct_mediastinal_structures_segmentation.

The datasets are from the publications https://doi.org/10.1080/21681163.2022.2043778 (lymph nodes)
and https://doi.org/10.1007/s11548-019-01948-8 (structures).
Please cite them if you use these datasets in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "lymph_nodes": "https://drive.google.com/uc?id=1ZsFq7PslqQ5ow_dXB01kDkaKPqYDXD5d",
    "structures": "https://drive.google.com/uc?id=1YqCRcBpsFoE4JsBq5NROqIpeijnITpe1",
}

CHECKSUMS = {
    "lymph_nodes": "c6b4bab94e8e69d72a9e9707b8e1c261944784c49652772519d3155d4d052478",
    "structures": "b7e0fa1c8e242259fdc062e68f56e9e561d92e36264c556854d2d9b55e282531",
}

STRUCTURE_IDS = {
    "esophagus": 1,
    "pulmonary_trunk": 2,
    "aortic_arch": 3,
    "ascending_aorta": 4,
    "descending_aorta": 5,
    "azygos": 6,
    "heart": 7,
    "vena_cava": 8,
    "brachiocephalic_veins": 9,
    "spine": 11,
    "pulmonary_vein": 12,
    "subclavian_and_carotid_arteries": 13,
    "lungs": 14,
    "airways": 16,
}
"""The label ids of the anatomical structures in the 'structures' dataset."""

MHD_DTYPES = {
    "MET_CHAR": np.int8, "MET_UCHAR": np.uint8, "MET_SHORT": np.int16, "MET_USHORT": np.uint16,
    "MET_INT": np.int32, "MET_UINT": np.uint32, "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64,
}


def read_mhd(path: str) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Read an uncompressed MetaImage (.mhd + .raw) volume.

    Args:
        path: The filepath to the .mhd header.

    Returns:
        The volume with axis order (x, y, z), matching the axis order nibabel uses for nifti files.
        The voxel spacing in (x, y, z) order.
    """
    header = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                header[key.strip()] = value.strip()

    if header.get("CompressedData", "False") == "True":
        raise NotImplementedError(f"Compressed MetaImage data is not supported: {path}")

    shape = tuple(int(s) for s in header["DimSize"].split())
    spacing = tuple(float(s) for s in header["ElementSpacing"].split())
    raw_path = os.path.join(os.path.split(path)[0], header["ElementDataFile"])
    data = np.fromfile(raw_path, dtype=MHD_DTYPES[header["ElementType"]])

    # The raw data is stored with x as the fastest axis, i.e. in (z, y, x) order.
    data = data.reshape(shape[::-1]).transpose(2, 1, 0)
    return data, spacing


def _convert_structures_to_nifti(data_dir):
    import nibabel as nib

    for patient_dir in tqdm(natsorted(glob(os.path.join(data_dir, "pat*"))), desc="Converting MetaImage to nifti"):
        patient_id = os.path.basename(patient_dir)
        inputs = {
            "data": os.path.join(patient_dir, "Data", f"{patient_id}.mhd"),
            "structures": os.path.join(patient_dir, "Structures", f"{patient_id}_organs_gt.mhd"),
            "lymph_nodes": os.path.join(patient_dir, "LN", f"{patient_id}_LN_gt.mhd"),
        }
        for name, mhd_path in inputs.items():
            out_path = os.path.join(patient_dir, f"{patient_id}_{name}.nii.gz")
            if os.path.exists(out_path):
                continue
            data, spacing = read_mhd(mhd_path)
            if name != "data":
                data = data.astype("uint16")
            nib.save(nib.Nifti1Image(data, np.diag(list(spacing) + [1.0])), out_path)


def get_mediastinal_ct_data(
    path: Union[os.PathLike, str], task: Literal["lymph_nodes", "structures"], download: bool = False
) -> str:
    """Download the Mediastinal CT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The dataset to download. Either 'lymph_nodes' (benchmark subset) or 'structures' (mediastinal CT dataset).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if task not in URLS:
        raise ValueError(f"'{task}' is not a valid task. Choose from {list(URLS.keys())}.")

    data_dir = os.path.join(path, "Benchmark" if task == "lymph_nodes" else "mediastinal_2019")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    if task == "lymph_nodes":
        zip_path = os.path.join(path, "benchmark_subset.zip")
        util.download_source_gdrive(path=zip_path, url=URLS[task], download=download, checksum=CHECKSUMS[task])
        util.unzip(zip_path=zip_path, dst=path, remove=False)
    else:
        # NOTE: The file is shared as 'mediastinal_ct_dataset.zip', but it is a gzipped tarball.
        tar_path = os.path.join(path, "mediastinal_ct_dataset.tar.gz")
        util.download_source_gdrive(path=tar_path, url=URLS[task], download=download, checksum=CHECKSUMS[task])
        util.unzip_tarfile(tar_path=tar_path, dst=data_dir, remove=False)
        _convert_structures_to_nifti(data_dir)

    return data_dir


def get_mediastinal_ct_paths(
    path: Union[os.PathLike, str], task: Literal["lymph_nodes", "structures"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Mediastinal CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The dataset to use. Either 'lymph_nodes' (benchmark subset) or 'structures' (mediastinal CT dataset).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mediastinal_ct_data(path, task, download)

    if task == "lymph_nodes":
        raw_paths = natsorted(glob(os.path.join(data_dir, "Pat*", "*_data.nii.gz")))
        label_paths = [p.replace("_data.nii.gz", "_labels_LymphNodes.nii.gz") for p in raw_paths]
    else:
        raw_paths = natsorted(glob(os.path.join(data_dir, "pat*", "*_data.nii.gz")))
        label_paths = [p.replace("_data.nii.gz", "_structures.nii.gz") for p in raw_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_mediastinal_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    task: Literal["lymph_nodes", "structures"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Mediastinal CT dataset for lymph node or anatomical structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The dataset to use. Either 'lymph_nodes' (benchmark subset) or 'structures' (mediastinal CT dataset).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mediastinal_ct_paths(path, task, download)

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


def get_mediastinal_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    task: Literal["lymph_nodes", "structures"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Mediastinal CT dataloader for lymph node or anatomical structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The dataset to use. Either 'lymph_nodes' (benchmark subset) or 'structures' (mediastinal CT dataset).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mediastinal_ct_dataset(path, patch_shape, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
