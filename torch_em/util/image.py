# TODO this should be partially refactored into elf.io before the next elf release
# and then be used in image_stack_wrapper as welll
import os
import numpy as np

from elf.io import open_file

try:
    import imageio.v3 as imageio
except ImportError:
    import imageio

try:
    import tifffile
except ImportError:
    tifffile = None


TIF_EXTS = (".tif", ".tiff")


def supports_memmap(image_path):
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


def load_data(path, key, mode="r"):
    have_single_file = isinstance(path, str)
    if key is None and have_single_file:
        return load_image(path)
    elif key is None and not have_single_file:
        return np.stack([load_image(p) for p in path])
    elif key is not None and have_single_file:
        return open_file(path, mode=mode)[key]
    elif key is not None and not have_single_file:
        return MultiDatasetWrapper(*[open_file(p, mode=mode)[key] for p in path])
