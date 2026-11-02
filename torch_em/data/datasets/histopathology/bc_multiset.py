"""The BC-MultiSet dataset contains nucleus annotations for H&E stained images of invasive breast carcinoma.

The dataset holds 5,586 patches of 256 x 256 pixels at 40x magnification from 50 patients, split into
four cross-validation folds. Two board-certified pathologists labeled the nuclei of three cell types (tumor,
fibroblast and inflammatory) and reconciled them by consensus. It also provides patient-level clinical and
molecular data (ER, PR, HER2, Ki67, TILs and more), which this module does not use.
NOTE: The patches are cut from larger crops with a stride of 192 pixels, so neighboring patches overlap.
NOTE: Not every nucleus is annotated: some nuclei in the images carry no label and become background.
NOTE: The label files are pickled numpy dicts. This module reads them with a restricted unpickler that only
accepts numpy arrays, so that no code from the files is executed.

The dataset is located at https://doi.org/10.5281/zenodo.20717567 (CC BY 4.0).
Please cite it if you use this dataset for your research.
"""

import io
import os
import pickle
import zipfile
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional, Sequence

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/20717567/files/BC-MultiSet-50.zip/content"
CHECKSUM = "1da5d93f397c80f0cf7e41adaa5e432d5164a0918588fc2629ee9f5c1fdbe208"
FOLDS = (1, 2, 3, 4)

# The only globals that the label pickles need.
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"The global '{module}.{name}' is not allowed in a label file.")
        if name == "_reconstruct":
            try:
                from numpy._core.multiarray import _reconstruct
            except ImportError:  # numpy < 2
                from numpy.core.multiarray import _reconstruct
            return _reconstruct
        return getattr(np, name)


def _read_labels(data):
    import numpy.lib.format as npy_format

    stream = io.BytesIO(data)
    version = npy_format.read_magic(stream)
    read_header = npy_format.read_array_header_1_0 if version == (1, 0) else npy_format.read_array_header_2_0
    shape, _, dtype = read_header(stream)
    assert shape == () and dtype == np.dtype("O"), (shape, dtype)
    # The pickle holds a 0-d object array, which wraps the dict of label maps.
    labels = _NumpyOnlyUnpickler(stream).load().item()
    return labels["inst_map"], labels["type_map"]


def _create_h5_files(path, zip_path):
    import h5py

    with zipfile.ZipFile(zip_path) as archive:
        for fold in FOLDS:
            out_dir = os.path.join(path, "preprocessed", f"fold{fold}")
            os.makedirs(out_dir, exist_ok=True)
            prefix = f"BC-MultiSet-50/fold{fold}/labels/"
            names = [n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".npy")]
            for name in tqdm(names, desc=f"Preprocess BC-MultiSet fold {fold}"):
                stem = os.path.splitext(os.path.basename(name))[0]
                image_name = f"BC-MultiSet-50/fold{fold}/images/{stem}.png"
                if image_name not in archive.NameToInfo:
                    continue
                image = imageio.imread(archive.read(image_name), extension=".png")[..., :3]
                instances, types = _read_labels(archive.read(name))
                assert image.shape[:2] == instances.shape == types.shape, stem

                with h5py.File(os.path.join(out_dir, f"{stem}.h5"), "w") as f:
                    f.create_dataset("raw", data=image.transpose(2, 0, 1), compression="gzip")
                    f.create_dataset("labels/instances", data=instances.astype("uint32"), compression="gzip")
                    f.create_dataset("labels/semantic", data=types.astype("uint8"), compression="gzip")


def get_bc_multiset_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BC-MultiSet dataset.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "BC-MultiSet-50.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    _create_h5_files(path, zip_path)
    return data_dir


def get_bc_multiset_paths(
    path: Union[os.PathLike, str], folds: Optional[Sequence[int]] = None, download: bool = False
) -> List[str]:
    """Get paths to the BC-MultiSet data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        folds: The cross-validation folds, out of 1, 2, 3 and 4. By default, all folds are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_bc_multiset_data(path, download)
    folds = FOLDS if folds is None else folds
    assert all(fold in FOLDS for fold in folds), folds
    return [p for fold in folds for p in natsorted(glob(os.path.join(data_dir, f"fold{fold}", "*.h5")))]


def get_bc_multiset_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    folds: Optional[Sequence[int]] = None,
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BC-MultiSet dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        patch_shape: The patch shape to use for training.
        folds: The cross-validation folds, out of 1, 2, 3 and 4. By default, all folds are used.
        label_choice: The choice of labels, 'instances' or 'semantic' (the three cell types).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert label_choice in ("instances", "semantic"), label_choice
    paths = get_bc_multiset_paths(path, folds, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_bc_multiset_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    folds: Optional[Sequence[int]] = None,
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BC-MultiSet dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        folds: The cross-validation folds, out of 1, 2, 3 and 4. By default, all folds are used.
        label_choice: The choice of labels, 'instances' or 'semantic' (the three cell types).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bc_multiset_dataset(path, patch_shape, folds, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
