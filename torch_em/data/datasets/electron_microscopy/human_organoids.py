"""The Human Organoids dataset contains annotations for several organelles in EM images
(mitochondria, nuclei, actin, entotic cell, and junctions) for patient-derived colorectal cancer organoids.

This dataset is from the publication https://doi.org/10.1016/j.devcel.2023.03.001.
Please cite it if you use this dataset in your research.

The data itself can be downloaded from EMPIAR via aspera.
- You can install aspera via mamba. We recommend to do this in a separate environment
  to avoid dependency issues:
    - `$ mamba create -c conda-forge -c hcc -n aspera aspera-cli`
- After this you can run `$ mamba activate aspera` to have an environment with aspera installed.
- You can then download the data for one of the three datasets like this:
    - ascp -QT -l 200m -P33001 -i <PREFIX>/etc/asperaweb_id_dsa.openssh emp_ext2@fasp.ebi.ac.uk:/<EMPIAR_ID> <PATH>
    - Where <PREFIX> is the path to the mamba environment, <EMPIAR_ID> the id of one of the three datasets
      and <PATH> where you want to download the data.
- After this you can use the functions in this file if you use <PATH> as location for the data.

NOTE: We have implemented automatic download, but this leads to dependency
issues, so we recommend to download the data manually and then run the loaders with the correct path.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Literal, Tuple

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _prepare_dataset(data_root):
    import mrcfile
    import h5py

    raw_paths = glob(os.path.join(data_root, "*bin2.mrc"))
    for raw_path in tqdm(raw_paths, desc="Preprocessing volumes"):
        vol_path = Path(raw_path).with_suffix(".h5")
        if os.path.exists(vol_path):
            continue

        with mrcfile.open(raw_path, "r") as f:
            raw = f.data

        # Get the corresponding label paths.
        label_paths = [p for p in glob(raw_path.replace(".mrc", "*.mrc")) if p != raw_path]

        labels = {}
        for label_path in label_paths:
            label_name = Path(label_path).stem.split("_")[-1]

            if label_name == "cell":  # A simple replacement for one outlier case.
                label_name = "entotic_cell"

            with mrcfile.open(label_path, "r") as f:
                curr_label = f.data

            labels[label_name] = curr_label

        # Finally, drop them all in a single h5 file.
        with h5py.File(vol_path, "w") as f:
            f.create_dataset("raw", data=raw, chunks=(8, 128, 128), compression="gzip")
            for lname, lvol in labels.items():
                f.create_dataset(lname, data=lvol, chunks=(8, 128, 128), compression="gzip")

        # And finally, remove all other volumes.
        os.remove(raw_path)
        [os.remove(p) for p in label_paths]


def get_human_organoids_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Human Organoids data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    access_id = "11380"
    data_path = util.download_source_empiar(path, access_id, download)

    data_root = os.path.join(data_path, "data")
    assert os.path.exists(data_root)

    _prepare_dataset(data_root)

    return data_root


def get_human_organoids_paths(
    path: Union[os.PathLike, str],
    organelle: Literal["mitos", "nuclei", "actin", "entotic_cell", "junctions"],
    download: bool = False,
) -> List[str]:
    """Get the paths to Human Organoids data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        organelle: The choice of organelle from 'mitos', 'nuclei', 'actin', 'entotic_cell', 'junctions'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data (both raw images and labels included).
    """
    import h5py

    assert isinstance(organelle, str) and organelle in ["mitos", "nuclei", "actin", "entotic_cell", "junctions"], \
        f"The choice of organelle '{organelle}' does not match the available choices."

    data_path = get_human_organoids_data(path, download)
    vol_paths = glob(os.path.join(data_path, "*.h5"))

    # Filter out volumes without organelle labels.
    vol_paths = [p for p in vol_paths if organelle in h5py.File(p, "r").keys()]
    assert vol_paths, f"The provided organelle labels for '{organelle}' not found."

    return vol_paths


def get_human_organoids_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    organelle: Literal["mitos", "nuclei", "actin", "entotic_cell", "junctions"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for the Human Organoids data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        organelle: The choice of organelle from 'mitos', 'nuclei', 'actin', 'entotic_cell', 'junctions'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    vol_paths = get_human_organoids_paths(path, organelle, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=vol_paths,
        raw_key="raw",
        label_paths=vol_paths,
        label_key=organelle,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_human_organoids_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    organelle: Literal["mitos", "nuclei", "actin", "entotic_cell", "junctions"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the dataloader for the Human Organoids data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organelle: The choice of organelle from 'mitos', 'nuclei', 'actin', 'entotic_cell', 'junctions'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_human_organoids_dataset(path, patch_shape, organelle, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
