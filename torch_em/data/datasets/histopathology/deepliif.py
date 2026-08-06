"""DeepLIIF contains annotations for nucleus segmentation and classification in IHC images
of lung, bladder and breast cancer tissue.

NOTE: The cell segmentations are a bit enlarged, since they grow over the full boundary that
the mask draws around each cell.

The lung and bladder tissue shares one archive per split. The breast cancer tissue, which is
stained for Ki67, stems from an earlier release of the data and has its own archives. It only
covers the train and the val split.

Every image comes as six co-registered 512x512 panels, which are stitched next to each other:
the IHC input, the hematoxylin channel and the multiplex immunofluorescence modalities
(mpIF DAPI, mpIF Lap2 and the mpIF marker), followed by the segmentation mask.
The mask marks cells with positive protein expression in red and negative cells in blue.
A green boundary surrounds each cell, so that this module can separate touching cells
and derive instance labels from the mask.
NOTE: The labels keep every cell of the mask. A few cells are only a few pixels large,
since saving the masks as png introduced compression artifacts.

The data is hosted at https://zenodo.org/records/4751737 and licensed under CC-BY-4.0.
This dataset is from the publication https://doi.org/10.1038/s42256-022-00471-x.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components
from skimage.segmentation import watershed

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/4751737/files/{filename}?download=1"

# The lung and bladder tissue shares one archive per split, the breast cancer tissue has its own.
FILENAMES = {
    ("lung_bladder", "train"): "DeepLIIF_Training_Set.zip",
    ("lung_bladder", "val"): "DeepLIIF_Validation_Set.zip",
    ("lung_bladder", "test"): "DeepLIIF_Testing_Set.zip",
    ("breast", "train"): "BC-DeepLIIF_Training_Set.zip",
    ("breast", "val"): "BC-DeepLIIF_Validation_Set.zip",
}

CHECKSUMS = {
    ("lung_bladder", "train"): "704b75ad9d15d5c5ffda8c48a53a7037544208bbcb7a1f3bbb2b1ce9c4f63d02",
    ("lung_bladder", "val"): "e8dcec56d4cb44a7037170060e39ec503ba3b70f817abd00cb36fb86eb4e7e1a",
    ("lung_bladder", "test"): "7663611b1274049b677dd18deee8fbd1e44370884ec1f4d8d7196924d2a7acec",
    ("breast", "train"): "598bef02f1dcc54f888976ad5ed267e1672816e9326c2011a35c68d5b0167d8e",
    ("breast", "val"): "349278854a95b0a31790e81acd21f87c7afaa8f3641ce5fb4c47b09987e45b23",
}

SPLITS = ["train", "val", "test"]

TISSUES = ["lung", "bladder", "breast"]

# The lung and bladder images carry their tissue in the filename, the breast cancer images do not.
FILE_PREFIXES = {"lung": "Lung_", "bladder": "Bladder_"}

# The image panels, in the order in which they are stitched together. The mask follows them.
MODALITIES = ["ihc", "hematoxylin", "dapi", "lap2", "marker"]


def _get_archive(tissue, split):
    """Map a tissue to the archive that holds it for a split."""
    key = ("breast" if tissue == "breast" else "lung_bladder", split)
    if key not in FILENAMES:
        raise ValueError(f"The '{tissue}' tissue has no '{split}' split.")
    return key


def _get_labels(mask):
    """Derive the instance and the semantic labels from the segmentation panel.

    The panel paints cells with positive expression red and negative cells blue, and draws a
    green boundary around every cell. The cell interiors are separate already, so they seed a
    watershed that grows them back over the boundary. This recovers the full extent of a cell,
    which taking the interiors alone would shrink by the width of the boundary.

    The class of a cell follows from the color of its interior, so that the semantic labels
    and the instances agree on where a cell ends.
    """
    positive, negative = mask[..., 0] > 127, mask[..., 2] > 127
    interior = positive | negative
    foreground = interior | (mask[..., 1] > 127)

    seeds = connected_components(interior)
    instances = watershed(mask[..., 1], markers=seeds, mask=foreground).astype("uint16")

    # A cell is positive if its interior holds more red than blue pixels.
    n_labels = int(instances.max()) + 1
    n_positive = np.bincount(instances[positive], minlength=n_labels)
    n_negative = np.bincount(instances[negative], minlength=n_labels)

    classes = np.where(n_positive >= n_negative, 2, 1).astype("uint8")
    classes[(n_positive + n_negative) == 0] = 0
    classes[0] = 0
    semantic = classes[instances]

    return instances, semantic


def _preprocess_data(input_dir, data_dir):
    import h5py

    os.makedirs(data_dir, exist_ok=True)
    image_paths = natsorted(glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        raise RuntimeError(f"Could not find the images in {input_dir}.")

    for image_path in image_paths:
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(data_dir, f"{fname}.h5")
        if os.path.exists(out_path):
            continue

        panels = np.split(imageio.imread(image_path)[..., :3], len(MODALITIES) + 1, axis=1)
        instances, semantic = _get_labels(panels[-1])

        with h5py.File(out_path, "a") as f:
            for modality, panel in zip(MODALITIES, panels):
                f.create_dataset(f"raw/{modality}", data=panel.transpose(2, 0, 1), compression="gzip")

            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")


def _get_tissues(tissue, split):
    """Resolve the tissue argument to the tissues that the split actually holds."""
    if tissue is None:
        return [name for name in TISSUES if ("breast" if name == "breast" else "lung_bladder", split) in FILENAMES]

    tissues = [tissue] if isinstance(tissue, str) else list(tissue)
    for name in tissues:
        if name not in TISSUES:
            raise ValueError(f"'{name}' is not a valid tissue. Choose one of {TISSUES}.")
    return tissues


def get_deepliif_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    tissue: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Download the DeepLIIF dataset for one split.

    Args:
        path: The folder where the function stores the data.
        split: The split of the dataset. Either 'train', 'val' or 'test'.
        tissue: The tissue to download. See `TISSUES` for the valid choices. By default this
            downloads every tissue that the split holds.
        download: Whether to download the data if it is not present.

    Returns:
        The list of filepaths to the folders with the prepared data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    data_dirs = []
    for archive in dict.fromkeys(_get_archive(name, split) for name in _get_tissues(tissue, split)):
        archive_dir = os.path.join(path, "_".join(archive))
        data_dir = os.path.join(archive_dir, "data")
        data_dirs.append(data_dir)
        if glob(os.path.join(data_dir, "*.h5")):
            continue

        os.makedirs(archive_dir, exist_ok=True)
        filename = FILENAMES[archive]
        input_dir = os.path.join(archive_dir, os.path.splitext(filename)[0])
        if not os.path.exists(input_dir):
            zip_path = os.path.join(archive_dir, filename)
            util.download_source(
                path=zip_path, url=URL.format(filename=filename), download=download, checksum=CHECKSUMS[archive]
            )
            util.unzip(zip_path=zip_path, dst=archive_dir)

        _preprocess_data(input_dir, data_dir)

    return data_dirs


def get_deepliif_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    tissue: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get the paths to the DeepLIIF data.

    Args:
        path: The folder where the function stores the data.
        split: The split of the dataset. Either 'train', 'val' or 'test'.
        tissue: The tissue to use. See `TISSUES` for the valid choices. By default this uses
            every tissue that the split holds.
        download: Whether to download the data if it is not present.

    Returns:
        The list of filepaths to the input data.
    """
    tissues = _get_tissues(tissue, split)
    get_deepliif_data(path, split, tissues, download)

    volume_paths = []
    for name in tissues:
        data_dir = os.path.join(path, "_".join(_get_archive(name, split)), "data")
        prefix = FILE_PREFIXES.get(name, "")
        volume_paths.extend(natsorted(glob(os.path.join(data_dir, f"{prefix}*.h5"))))

    assert len(volume_paths) > 0, f"Could not find data for the split '{split}' and the tissue '{tissue}'."
    return natsorted(volume_paths)


def get_deepliif_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    tissue: Optional[Union[str, List[str]]] = None,
    modality: Literal["ihc", "hematoxylin", "dapi", "lap2", "marker"] = "ihc",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DeepLIIF dataset for nucleus segmentation and classification in IHC images.

    Args:
        path: The folder where the function stores the data.
        patch_shape: The patch shape to use for training.
        split: The split of the dataset. Either 'train', 'val' or 'test'.
        tissue: The tissue to use. See `TISSUES` for the valid choices. By default this uses
            every tissue that the split holds. Note that 'breast' has no test split.
        modality: The image modality to use as input. See `MODALITIES` for the valid choices.
        label_choice: The choice of labels. Either 'instances' for the nucleus instances, or
            'semantic' for the classification into background, negative cells and positive cells.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Choose one of {MODALITIES}.")

    if label_choice not in ["instances", "semantic"]:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'instances' or 'semantic'.")

    volume_paths = get_deepliif_paths(path, split, tissue, download)
    kwargs = util.update_kwargs(kwargs, "with_channels", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=f"raw/{modality}",
        label_paths=volume_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_deepliif_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    tissue: Optional[Union[str, List[str]]] = None,
    modality: Literal["ihc", "hematoxylin", "dapi", "lap2", "marker"] = "ihc",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DeepLIIF dataloader for nucleus segmentation and classification in IHC images.

    Args:
        path: The folder where the function stores the data.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split of the dataset. Either 'train', 'val' or 'test'.
        tissue: The tissue to use. See `TISSUES` for the valid choices. By default this uses
            every tissue that the split holds. Note that 'breast' has no test split.
        modality: The image modality to use as input. See `MODALITIES` for the valid choices.
        label_choice: The choice of labels. Either 'instances' for the nucleus instances, or
            'semantic' for the classification into background, negative cells and positive cells.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepliif_dataset(path, patch_shape, split, tissue, modality, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
