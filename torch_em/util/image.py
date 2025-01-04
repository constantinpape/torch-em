import os
from typing import Optional, Sequence, Union

import imageio.v3 as imageio
import numpy as np
from elf.io import open_file
from numpy.typing import ArrayLike

try:
    import tifffile
except ImportError:
    tifffile = None

TIF_EXTS = (".tif", ".tiff")


def supports_memmap(image_path):
    """@private
    """
    if tifffile is None:
        return False
    ext = os.path.splitext(image_path)[1]
    if ext.lower() not in TIF_EXTS:
        return False
    try:
        tifffile.memmap(image_path, mode="r")
    except ValueError:
        return False
    return True


def load_image(image_path, memmap=True):
    """@private
    """
    if supports_memmap(image_path) and memmap:
        return tifffile.memmap(image_path, mode="r")
    elif tifffile is not None and os.path.splitext(image_path)[1].lower() in (".tiff", ".tif"):
        return tifffile.imread(image_path)
    elif os.path.splitext(image_path)[1].lower() == ".nrrd":
        import nrrd
        return nrrd.read(image_path)[0]
    elif os.path.splitext(image_path)[1].lower() == ".mha":
        import SimpleITK as sitk
        image = sitk.ReadImage(image_path)
        return sitk.GetArrayFromImage(image)
    else:
        return imageio.imread(image_path)


class MultiDatasetWrapper:
    """@private
    """
    def __init__(self, *file_datasets):
        # Make sure we have the same shapes.
        reference_shape = file_datasets[0].shape
        assert all(reference_shape == ds.shape for ds in file_datasets)
        self.file_datasets = file_datasets

        self.shape = (len(self.file_datasets),) + reference_shape

    def __getitem__(self, index):
        channel_index, spatial_index = index[:1], index[1:]
        data = []
        for ds in self.file_datasets:
            ds_data = ds[spatial_index]
            data.append(ds_data)
        data = np.stack(data)
        data = data[channel_index]
        return data


def load_data(
    path: Union[str, Sequence[str]],
    key: Optional[Union[str, Sequence[str]]] = None,
    mode: str = "r",
) -> ArrayLike:
    """Load data from a file or multiple files.

    Supports loading regular image formats, such as tif or jpg, or container data formats, such as hdf5, n5 or zarr.
    For the latter case, specify the name of the internal dataset to load via the `key` argument.

    Args:
        path: The file path or paths to the data.
        key: The key or keys to the internal datasets.
        mode: The mode for reading datasets.

    Returns:
        The loaded data.
    """
    have_single_file = isinstance(path, str)
    have_single_key = isinstance(key, str)

    if key is None:
        if have_single_file:
            return load_image(path)
        else:
            return np.stack([load_image(p) for p in path])
    else:
        if have_single_key and have_single_file:
            return open_file(path, mode=mode)[key]
        elif have_single_key and not have_single_file:
            return MultiDatasetWrapper(*[open_file(p, mode=mode)[key] for p in path])
        elif not have_single_key and have_single_file:
            return MultiDatasetWrapper(*[open_file(path, mode=mode)[k] for k in key])
        else:  # have multipe keys and multiple files
            return MultiDatasetWrapper(*[open_file(p, mode=mode)[k] for k in key for p in path])
