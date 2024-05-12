import os
import shutil
import numpy as np
from glob import glob

import torch_em
from . import util

URLS = {
    "s_aureus": "https://zenodo.org/record/5550933/files/DeepBacs_Data_Segmentation_Staph_Aureus_dataset.zip?download=1",
    "e_coli": "https://zenodo.org/record/5550935/files/DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip?download=1",
    "b_subtilis": "https://zenodo.org/record/5639253/files/Multilabel_U-Net_dataset_B.subtilis.zip?download=1",
    "mixed": "https://zenodo.org/record/5551009/files/DeepBacs_Data_Segmentation_StarDist_MIXED_dataset.zip?download=1",
}
CHECKSUMS = {
    "s_aureus": "4047792f1248ee82fce34121d0ade84828e55db5a34656cc25beec46eacaf307",
    "e_coli": "f812a2f814c3875c78fcc1609a2e9b34c916c7a9911abbf8117f423536ef1c17",
    "b_subtilis": "1",
    "mixed": "2730e6b391637d6dc05bbc7b8c915fd8184d835ac3611e13f23ac6f10f86c2a0",
}


def _require_deebacs_dataset(path, bac_type, download):
    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{bac_type}.zip")
    if not os.path.exists(zip_path):
        util.download_source(zip_path, URLS[bac_type], download, checksum=CHECKSUMS[bac_type])
    util.unzip(zip_path, os.path.join(path, bac_type))

    # let's get a val split for the expected bacteria type
    _assort_val_set(path, bac_type)


def _assort_val_set(path, bac_type):
    image_paths = glob(os.path.join(path, bac_type, "training", "source", "*"))
    image_paths = [os.path.split(_path)[-1] for _path in image_paths]

    val_partition = 0.2
    # let's get a balanced set of bacterias, if bac_type is mixed
    if bac_type == "mixed":
        _sort_1, _sort_2, _sort_3 = [], [], []
        for _path in image_paths:
            if _path.startswith("JE2"):
                _sort_1.append(_path)
            elif _path.startswith("pos"):
                _sort_2.append(_path)
            elif _path.startswith("train_"):
                _sort_3.append(_path)

        val_image_paths = [
            *np.random.choice(_sort_1, size=int(val_partition * len(_sort_1)), replace=False),
            *np.random.choice(_sort_2, size=int(val_partition * len(_sort_2)), replace=False),
            *np.random.choice(_sort_3, size=int(val_partition * len(_sort_3)), replace=False)
        ]
    else:
        val_image_paths = np.random.choice(image_paths, size=int(val_partition * len(image_paths)), replace=False)

    val_image_dir = os.path.join(path, bac_type, "val", "source")
    val_label_dir = os.path.join(path, bac_type, "val", "target")
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    for sample_id in val_image_paths:
        src_val_image_path = os.path.join(path, bac_type, "training", "source", sample_id)
        dst_val_image_path = os.path.join(val_image_dir, sample_id)
        shutil.move(src_val_image_path, dst_val_image_path)

        src_val_label_path = os.path.join(path, bac_type, "training", "target", sample_id)
        dst_val_label_path = os.path.join(val_label_dir, sample_id)
        shutil.move(src_val_label_path, dst_val_label_path)


def _get_paths(path, bac_type, split):
    # the bacteria types other than mixed are a bit more complicated so we don't have the dataloaders for them yet
    # mixed is the combination of all other types
    if split == "train":
        dir_choice = "training"
    else:
        dir_choice = split

    if bac_type != "mixed":
        raise NotImplementedError(f"Currently only the bacteria type 'mixed' is supported, not {bac_type}")
    image_folder = os.path.join(path, bac_type, dir_choice, "source")
    label_folder = os.path.join(path, bac_type, dir_choice, "target")
    return image_folder, label_folder


def get_deepbacs_dataset(
    path, split, patch_shape, bac_type="mixed", download=False, **kwargs
):
    """Dataset for the segmentation of bacteria in light microscopy.

    This dataset is from the publication https://doi.org/10.1038/s42003-022-03634-z.
    Please cite it if you use this dataset for a publication.
    """
    assert split in ("train", "val", "test")
    bac_types = list(URLS.keys())
    assert bac_type in bac_types, f"{bac_type} is not in expected bacteria types: {bac_types}"

    data_folder = os.path.join(path, bac_type)
    if not os.path.exists(data_folder):
        _require_deebacs_dataset(path, bac_type, download)

    image_folder, label_folder = _get_paths(path, bac_type, split)

    dataset = torch_em.default_segmentation_dataset(
        image_folder, "*.tif", label_folder, "*.tif", patch_shape=patch_shape, **kwargs
    )
    return dataset


def get_deepbacs_loader(path, split, patch_shape, batch_size, bac_type="mixed", download=False, **kwargs):
    """Dataloader for the segmentation of bacteria in light microscopy. See 'get_deepbacs_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepbacs_dataset(path, split, patch_shape, bac_type=bac_type, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
