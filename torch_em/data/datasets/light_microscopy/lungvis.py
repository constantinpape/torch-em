"""The LungVis 1.0 dataset contains light sheet fluorescence microscopy volumes of tissue-cleared mouse lungs
with annotations of the airway tree.

The dataset consists of 78 lungs: 37 lungs with nanoparticle delivery via different routes
(intranasal liquid aspiration 'INLA', intratracheal liquid instillation 'ITLI', nose-only aerosol inhalation 'NOAI'
and ventilator-assisted aerosol delivery 'VAAD') and 41 additional lungs imaged only for the airway geometry
('Lung_001' to 'Lung_041'). The raw data is the tissue autofluorescence channel (470 to 570 nm).
The airway annotations are of three kinds:
- 'MS': manually segmented airways (3 lungs: 'INLA_001', 'Lung_001', 'Lung_002').
- 'MCAI': manually corrected nnU-Net predictions (17 lungs).
- 'AI': uncorrected nnU-Net predictions (58 lungs).
Use `annotation="manual"` (default) to get the 20 lungs with human-verified airway annotations ('MS' and 'MCAI'),
`annotation="ai"` for the 58 lungs with automatic annotations and `annotation="all"` for all 78 lungs.

Every lung is downloaded as a separate archive (25 MB to 8.4 GB, 100 GB in total). The raw data and the airway
labels are converted to a hdf5 file per lung, with keys 'raw' (uint8) and 'labels' (uint8, 1 = airway).
Voxel sizes are 5.159 x 5.159 x (10, 15 or 20) um for most lungs.

NOTE: 'Lung_002' is a small region of 'Lung_001', 'Lung_010' is the same lung as 'ITLI_001'
and 'Lung_013' is the same lung as 'VAAD_001'.

The dataset is located at https://zenodo.org/records/7413818.

This dataset is from the publication https://doi.org/10.1038/s41467-024-54267-1.
Please cite it if you use this dataset in your research.
"""

import os
import re
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional, Sequence

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7413818/files/{lung_id}.zip"

ANNOTATION_TYPES = {
    "INLA_001": "MS", "INLA_002": "AI",
    "ITLI_001": "AI", "ITLI_002": "MCAI", "ITLI_003": "MCAI", "ITLI_004": "AI", "ITLI_005": "AI", "ITLI_006": "AI",
    "ITLI_007": "AI", "ITLI_008": "AI", "ITLI_009": "AI", "ITLI_010": "AI", "ITLI_011": "MCAI", "ITLI_012": "AI",
    "ITLI_013": "AI", "ITLI_014": "AI",
    "NOAI_001": "MCAI", "NOAI_002": "AI", "NOAI_003": "AI",
    "VAAD_001": "AI", "VAAD_002": "MCAI", "VAAD_003": "AI", "VAAD_004": "MCAI", "VAAD_005": "AI", "VAAD_006": "AI",
    "VAAD_007": "AI", "VAAD_008": "AI", "VAAD_009": "AI", "VAAD_010": "MCAI", "VAAD_011": "AI", "VAAD_012": "AI",
    "VAAD_013": "AI", "VAAD_014": "AI", "VAAD_015": "MCAI", "VAAD_016": "AI", "VAAD_017": "AI", "VAAD_018": "MCAI",
    "Lung_001": "MS", "Lung_002": "MS", "Lung_003": "MCAI", "Lung_004": "MCAI", "Lung_005": "MCAI",
    "Lung_006": "MCAI", "Lung_007": "MCAI", "Lung_008": "MCAI", "Lung_009": "MCAI", "Lung_010": "MCAI",
    "Lung_011": "AI", "Lung_012": "AI", "Lung_013": "AI", "Lung_014": "AI", "Lung_015": "AI", "Lung_016": "AI",
    "Lung_017": "AI", "Lung_018": "AI", "Lung_019": "AI", "Lung_020": "AI", "Lung_021": "AI", "Lung_022": "AI",
    "Lung_023": "AI", "Lung_024": "AI", "Lung_025": "AI", "Lung_026": "AI", "Lung_027": "AI", "Lung_028": "AI",
    "Lung_029": "AI", "Lung_030": "AI", "Lung_031": "AI", "Lung_032": "AI", "Lung_033": "AI", "Lung_034": "AI",
    "Lung_035": "AI", "Lung_036": "AI", "Lung_037": "AI", "Lung_038": "AI", "Lung_039": "AI", "Lung_040": "AI",
    "Lung_041": "AI",
}

LUNG_IDS = list(ANNOTATION_TYPES.keys())

# Checksums are only available for the archives with manual annotations ('MS' and 'MCAI').
CHECKSUMS = {
    "INLA_001": "5b23eb6b3d20435606352c50d8d578e6eea88fdfaeea83eb4adcc0fa6180d8a1",
    "ITLI_002": "76444796540ab3ce33d52961944c9104925e3697cc4df4a8de84519c4619433a",
    "ITLI_003": "c331642a781f91dd258e7f7470f5c554e82c8bdad7ff13e36fecebcd38ae840b",
    "ITLI_011": "5ce194f22e963cfd9bfa2251ad3240394ee7af8a9ba9931ba16a39b2d2cbd882",
    "NOAI_001": "9b3aa922cee98541c98430a137b95bcc28deb937336613407cbd40ef556ae257",
    "VAAD_002": "ea007e5f41644d2a3d6cc19fe9f1ec48c66d618bd4b0ad3676b36683be3ac7e6",
    "VAAD_004": "bff40988bcd2932ebb0579d77b641c385c79f124ad7c9d819436568e105bccc3",
    "VAAD_010": "51f3d0fff0bfb509dab3646a3fe1c961b36787e2d47436a2d5339880d7d3d421",
    "VAAD_015": "359294fb0ab4a42a6e2935255a54368c4a1b874d913f4d7e5309d27593d27589",
    "VAAD_018": "c712a57589ea467e7c066ee2c4aeae7e71baeac9b4c3792f7f56197fb0504ea8",
    "Lung_001": "589eee28e1ac75998a3fa6baf32e164d94ec58ff61aee1edeae0e3e932bd5879",
    "Lung_002": "783b794dfe98407330923b904d07c5375e8931ca7c534c07821168e1e9866749",
    "Lung_003": "80943302fcac472fd5a28e849575c843819250c0e841941ac927f5574e0c5b74",
    "Lung_004": "41c3dea4bc9e06951af778198b36d3a6ac0163c543c40ec825e464270f401b9b",
    "Lung_005": "96d16eabf3ef0060402172f3853c337a709e9c44629f221c8c8890f00c3d13ca",
    "Lung_006": "06bc7f382272362c4fea323b5b0eed730103602b383e97096df8fb915edfd397",
    "Lung_007": "ff34057643fbf7d38d669489be86f1322dff140c55ef269ebf44133ce3ff137d",
    "Lung_008": "de4d68f8ebe32837a943a09306384ab9b49c8a5cc3c4b1e180b428561aee171a",
    "Lung_009": "626fbc9d07d6c080f95f9647abd29bf9a2fd2fd9e0b03f78daa9f370776acfee",
    "Lung_010": "02a57bdcefc5c7de5f5ea49e690435e464f722f1792b301bfd97b02c0192dac4",
}


def _find_volumes(extracted_dir, lung_id):
    tif_paths = natsorted(glob(os.path.join(extracted_dir, "**", "*.tif"), recursive=True))
    tif_names = {os.path.basename(p): p for p in tif_paths}

    # The labels are the MS, MCAI or AI airway volumes, e.g. 'ITLI_003_MCAI_airway AF545nm_UID_11-30-39.tif',
    # 'Lung_003_MCAI results_UID_19-49-29.tif' or 'ITLI_002_MCAI_AF545nm_UID_10-28-58.tif'.
    label_names = [name for name in tif_names if re.search(r"_(MS|MCAI|AI)[ _]", name)]
    if len(label_names) != 1:
        raise RuntimeError(f"Expected exactly one label volume for '{lung_id}', found {label_names}.")
    label_name = label_names[0]

    # The raw data is the tissue autofluorescence channel, e.g. 'Lung_002_Raw AF545nm_UID_11-08-31.tif'.
    # Some lungs have multiple autofluorescence channels. We pick the one the airways were annotated on if the
    # label name contains the wavelength, otherwise the channel with the shortest wavelength (the primary AF channel).
    raw_names = {}
    for name in tif_names:
        match = re.search(r"[Rr]aw[ _]?AF(\d+)nm", name)
        if match is not None:
            raw_names[int(match.group(1))] = name
    if len(raw_names) == 0:
        raise RuntimeError(f"Could not find the raw volume for '{lung_id}' in {list(tif_names.keys())}.")
    annotated = [wavelength for wavelength in raw_names if str(wavelength) in label_name]
    raw_name = raw_names[annotated[0] if len(annotated) == 1 else min(raw_names)]

    return tif_names[raw_name], tif_names[label_name]


def _convert_to_hdf5(extracted_dir, output_path, lung_id):
    import h5py
    import tifffile

    raw_path, label_path = _find_volumes(extracted_dir, lung_id)

    raw = tifffile.imread(raw_path)
    labels = tifffile.imread(label_path)
    if raw.shape != labels.shape:
        raise RuntimeError(f"Shape mismatch for '{lung_id}': raw {raw.shape} vs labels {labels.shape}.")

    # The airway foreground is stored either as 1 or as 255. We normalize it to 1.
    labels = (labels > 0).astype("uint8")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(1,) + raw.shape[1:])
        f.create_dataset("labels", data=labels, compression="gzip", chunks=(1,) + labels.shape[1:])


def get_lungvis_data(path: Union[os.PathLike, str], lung_id: str, download: bool = False) -> str:
    """Download and preprocess one lung of the LungVis dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        lung_id: The lung to download. One of the ids in `LUNG_IDS`.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the hdf5 file with the raw data and airway labels of this lung.
    """
    if lung_id not in LUNG_IDS:
        raise ValueError(f"'{lung_id}' is not a valid lung id. Choose one of {LUNG_IDS}.")

    volume_path = os.path.join(path, f"{lung_id}.h5")
    if os.path.exists(volume_path):
        return volume_path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{lung_id}.zip")
    util.download_source(
        path=zip_path, url=URL.format(lung_id=lung_id), download=download, checksum=CHECKSUMS.get(lung_id)
    )
    extracted_dir = os.path.join(path, lung_id)
    util.unzip(zip_path=zip_path, dst=path)

    _convert_to_hdf5(extracted_dir, volume_path, lung_id)
    shutil.rmtree(extracted_dir)

    return volume_path


def get_lungvis_paths(
    path: Union[os.PathLike, str],
    annotation: Literal["manual", "ai", "all"] = "manual",
    lung_ids: Optional[Sequence[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the LungVis data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The type of airway annotations. Either 'manual' (manually segmented or manually corrected),
            'ai' (uncorrected nnU-Net predictions) or 'all'.
        lung_ids: The lungs to use. By default, all lungs matching the chosen annotation type are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 volumes, which contain the raw data and airway labels.
    """
    if annotation not in ("manual", "ai", "all"):
        raise ValueError(f"'{annotation}' is not a valid annotation type. Choose one of 'manual', 'ai' or 'all'.")

    if lung_ids is None:
        if annotation == "manual":
            lung_ids = [lid for lid, atype in ANNOTATION_TYPES.items() if atype in ("MS", "MCAI")]
        elif annotation == "ai":
            lung_ids = [lid for lid, atype in ANNOTATION_TYPES.items() if atype == "AI"]
        else:
            lung_ids = LUNG_IDS
    elif isinstance(lung_ids, str):
        lung_ids = [lung_ids]

    volume_paths = [get_lungvis_data(path, lung_id, download) for lung_id in lung_ids]
    return volume_paths


def get_lungvis_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation: Literal["manual", "ai", "all"] = "manual",
    lung_ids: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LungVis dataset for airway segmentation in light sheet microscopy volumes of mouse lungs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The type of airway annotations. Either 'manual' (manually segmented or manually corrected),
            'ai' (uncorrected nnU-Net predictions) or 'all'.
        lung_ids: The lungs to use. By default, all lungs matching the chosen annotation type are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_lungvis_paths(path, annotation, lung_ids, download)

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


def get_lungvis_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation: Literal["manual", "ai", "all"] = "manual",
    lung_ids: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LungVis dataloader for airway segmentation in light sheet microscopy volumes of mouse lungs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The type of airway annotations. Either 'manual' (manually segmented or manually corrected),
            'ai' (uncorrected nnU-Net predictions) or 'all'.
        lung_ids: The lungs to use. By default, all lungs matching the chosen annotation type are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lungvis_dataset(path, patch_shape, annotation, lung_ids, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
