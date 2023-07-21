# TODO this should be partially refactored into elf.io before the next elf release
# and then be used in image_stack_wrapper as welll
import os

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
    else:
        return imageio.imread(image_path)


def load_data(path, key, mode="r"):
    if key is None:
        return load_image(path)
    else:
        return open_file(path, mode=mode)[key]
