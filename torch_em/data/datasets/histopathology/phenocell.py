"""The PhenoCell dataset contains annotations for cell phenotyping in
H&E stained histopathology images, with instance segmentation and 14 granular
cell types derived from co-registered multiplexed (CODEX) imaging.

The dataset is part of the PhenoBench (PathoCellBench) benchmark and is hosted on
HuggingFace at https://huggingface.co/datasets/Kainmueller-Lab/phenobench.
This dataset is from the publication https://doi.org/10.48550/arXiv.2507.03532.
Please cite it if you use this dataset in your research.

The data consists of 109 fields of view of 1440x1920 pixels. On the first use each
field of view is converted into a single chunked and compressed HDF5 file with the
following layout:
    - 'raw/histopathology/h&e': the (3, H, W) H&E image.
    - 'raw/codex/all': the (58, H, W) stack of co-registered CODEX channels.
    - 'raw/codex/<marker>_<target>': each individual CODEX channel (H, W), e.g.
      'raw/codex/CD20_B_cells' (see `CODEX_CHANNELS` for the full list of 58 channels).
    - 'labels/instances': the instance segmentation.
    - 'labels/semantic_coarse': the coarse 15-class cell type map (the benchmark labels).
    - 'labels/semantic_fine': the fine-grained 30-class cell type map.

The coarse semantic classes ('semantic_coarse' label choice) are:
    0: Background
    1: B cells
    2: Macrophages/Monocytes
    3: Adipocytes
    4: Dendritic cells
    5: T cells
    6: Granulocytes
    7: NK cells
    8: Nerves
    9: Plasma cells
    10: Smooth muscle
    11: Stroma
    12: Tumor cells
    13: Vasculature/Lymphatics
    14: Other cells

The 'semantic_fine' label choice has 30 granular classes that the coarse ones are
collapsed from:
    0: background
    1: B cells
    2: CD11b+ monocytes
    3: CD11b+CD68+ macrophages
    4: CD11c+ DCs
    5: CD163+ macrophages
    6: CD3+ T cells
    7: CD4+ T cells
    8: CD4+ T cells CD45RO+
    9: CD4+ T cells GATA3+
    10: CD68+ macrophages
    11: CD68+ macrophages GzmB+
    12: CD68+CD163+ macrophages
    13: CD8+ T cells
    14: NK cells
    15: Tregs
    16: adipocytes
    17: dirt
    18: granulocytes
    19: immune cells
    20: immune cells / vasculature
    21: lymphatics
    22: nerves
    23: plasma cells
    24: smooth muscle
    25: stroma
    26: tumor cells
    27: tumor cells / immune cells
    28: undefined
    29: vasculature

NOTE: Downloading requires 'huggingface_hub'. The dataset is large (each field of
view is around 350 MB), so by default only the requested split is downloaded.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import torch

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HF_REPO = "Kainmueller-Lab/phenobench"
SRC_HDF_DIR = "pathocell_hdf"
SPLIT_FILE = "data/phenocell/splits/phenocell_dataset_split.csv"

# Source label key in the downloaded HDF5 -> destination key in the converted HDF5.
SOURCE_LABELS = {
    "gt_inst": "labels/instances",
    "gt_ct_coarse": "labels/semantic_coarse",
    "gt_ct": "labels/semantic_fine",
}

LABEL_KEYS = {
    "instances": "labels/instances",
    "semantic_coarse": "labels/semantic_coarse",
    "semantic_fine": "labels/semantic_fine",
}

# The multi-channel raw inputs. Individual CODEX channels (see CODEX_CHANNELS) can also be chosen.
MODALITY_KEYS = {
    "histopathology": "raw/histopathology/h&e",
    "codex": "raw/codex/all",
}

# The 58 CODEX channels in their stored order, named '<marker>_<target>' (the keys under 'raw/codex/').
CODEX_CHANNELS = [
    "CD44_stroma", "FOXP3_regulatory_T_cells", "CDX2_intestinal_epithelia", "CD8_cytotoxic_T_cells",
    "p53_tumor_suppressor", "GATA3_Th2_helper_T_cells", "CD45_hematopoietic_cells", "T-bet_Th1_cells",
    "beta-catenin_Wnt_signaling", "HLA-DR_MHC-II", "PD-L1_checkpoint", "Ki67_proliferation",
    "CD45RA_naive_T_cells", "CD4_T_helper_cells", "CD21_DCs", "MUC-1_epithelia", "CD30_costimulator",
    "CD2_T_cells", "Vimentin_cytoplasm", "CD20_B_cells", "LAG-3_checkpoint", "Na-K-ATPase_membranes",
    "CD5_T_cells", "IDO-1_metabolism", "Cytokeratin_epithelia", "CD11b_macrophages", "CD56_NK_cells",
    "aSMA_smooth_muscle", "BCL-2_apoptosis", "CD25_IL-2_Ra", "Collagen_IV_bas._memb.", "CD11c_DCs",
    "PD-1_checkpoint", "HOCHST13", "Granzyme_B_cytotoxicity", "EGFR_signaling", "VISTA_costimulator",
    "CD15_granulocytes", "CD194_CCR4_chemokine_R", "ICOS_costimulator", "MMP9_matrix_metalloproteinase",
    "Synaptophysin_neuroendocrine", "CD71_transferrin_R", "GFAP_nerves", "CD7_T_cells", "CD3_T_cells",
    "Chromogranin_A_neuroendocrine", "CD163_macrophages", "CD57_NK_cells", "CD45RO_memory_cells",
    "CD68_macrophages", "CD31_vasculature", "Podoplanin_lymphatics", "CD34_vasculature", "CD38_multifunctional",
    "CD138_plasma_cells", "MMP12_matrix_metalloproteinase", "DRAQ5",
]


def _samples_for_split(split_csv, split):
    import pandas as pd

    df = pd.read_csv(split_csv)
    if split is not None:
        if split not in ("train", "valid", "test"):
            raise ValueError(f"'{split}' is not a valid split choice. Use 'train', 'valid' or 'test'.")
        df = df[df["train_test_val_split"] == split]

    return sorted(df["sample_name"].tolist())


def _convert_sample(src_path, output_path):
    import h5py

    with h5py.File(src_path, "r") as f:
        image = f["img"][:]
        codex = f["ifl"][:]
        labels = {dst: f[src][0] for src, dst in SOURCE_LABELS.items()}

    if codex.shape[0] != len(CODEX_CHANNELS):
        raise RuntimeError(f"Expected {len(CODEX_CHANNELS)} CODEX channels, but found {codex.shape[0]}.")

    tmp_path = output_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw/histopathology/h&e", data=image, compression="gzip", chunks=(1, 512, 512))
        f.create_dataset("raw/codex/all", data=codex, compression="gzip", chunks=(1, 512, 512))
        for i, name in enumerate(CODEX_CHANNELS):
            f.create_dataset(f"raw/codex/{name}", data=codex[i], compression="gzip", chunks=(512, 512))
        for dst, label in labels.items():
            f.create_dataset(dst, data=label, compression="gzip", chunks=(512, 512))

    os.replace(tmp_path, output_path)


def get_phenocell_data(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "valid", "test"]] = None,
    download: bool = False,
) -> str:
    """Download and preprocess the PhenoCell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. Either 'train', 'valid', 'test' or None for all fields of view.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise ImportError("'huggingface_hub' is required to download PhenoCell. Install it via conda/pip.")

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    if not os.path.exists(os.path.join(path, SPLIT_FILE)):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=SPLIT_FILE, local_dir=path)

    samples = _samples_for_split(os.path.join(path, SPLIT_FILE), split)
    to_convert = [s for s in samples if not os.path.exists(os.path.join(preprocessed_dir, f"{Path(s).stem}.h5"))]

    if to_convert:
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        patterns = [f"{SRC_HDF_DIR}/{s}" for s in to_convert]
        snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=path, allow_patterns=patterns)

        for sample in tqdm(to_convert, desc="Converting PhenoCell fields of view"):
            _convert_sample(
                os.path.join(path, SRC_HDF_DIR, sample),
                os.path.join(preprocessed_dir, f"{Path(sample).stem}.h5"),
            )

    return preprocessed_dir


def get_phenocell_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "valid", "test"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the PhenoCell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. Either 'train', 'valid', 'test' or None for all fields of view.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_phenocell_data(path, split, download)
    samples = _samples_for_split(os.path.join(path, SPLIT_FILE), split)
    volume_paths = [os.path.join(preprocessed_dir, f"{Path(s).stem}.h5") for s in samples]

    missing = [p for p in volume_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Could not find the data at {missing}.")

    return volume_paths


def get_phenocell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "valid", "test"]] = None,
    label_choice: Literal["instances", "semantic_coarse", "semantic_fine"] = "instances",
    modality: str = "histopathology",
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the PhenoCell dataset for cell phenotyping in H&E stained histopathology images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use. Either 'train', 'valid', 'test' or None for all fields of view.
        label_choice: The label type. Either 'instances', 'semantic_coarse' (15-class) or 'semantic_fine' (30-class).
        modality: The raw input. Either 'histopathology' (3-channel H&E), 'codex' (58-channel multiplexed stack)
            or the name of a single CODEX channel (see `CODEX_CHANNELS`), e.g. 'CD20_B_cells'.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_choice not in LABEL_KEYS:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose from {list(LABEL_KEYS.keys())}.")

    if modality in MODALITY_KEYS:
        raw_key, with_channels = MODALITY_KEYS[modality], True
    elif modality in CODEX_CHANNELS:
        raw_key, with_channels = f"raw/codex/{modality}", False
    else:
        raise ValueError(f"'{modality}' is not a valid modality. Use 'histopathology', 'codex' or a CODEX channel.")

    volume_paths = get_phenocell_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": modality == "histopathology"}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=raw_key,
        label_paths=volume_paths,
        label_key=LABEL_KEYS[label_choice],
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=True,
        with_channels=with_channels,
        ndim=2,
        **kwargs
    )


def get_phenocell_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Optional[Literal["train", "valid", "test"]] = None,
    label_choice: Literal["instances", "semantic_coarse", "semantic_fine"] = "instances",
    modality: str = "histopathology",
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PhenoCell dataloader for cell phenotyping in H&E stained histopathology images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The split to use. Either 'train', 'valid', 'test' or None for all fields of view.
        label_choice: The label type. Either 'instances', 'semantic_coarse' (15-class) or 'semantic_fine' (30-class).
        modality: The raw input. Either 'histopathology' (3-channel H&E), 'codex' (58-channel multiplexed stack)
            or the name of a single CODEX channel (see `CODEX_CHANNELS`), e.g. 'CD20_B_cells'.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_phenocell_dataset(
        path=path, patch_shape=patch_shape, split=split, label_choice=label_choice, modality=modality,
        download=download, label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
