# TODO this should be partially refactored into elf.io before the next elf release
# and then be used in image_stack_wrapper as welll
import os
import imageio

try:
    import tifffile
except ImportError:
    tifffile = None

TIF_EXTS = ('.tif', '.tiff')


def supports_memmap(image_path):
    if tifffile is None:
        return False
    ext = os.path.splitext(image_path)[1]
    if ext.lower() not in TIF_EXTS:
        return False
    try:
        tifffile.memmap(image_path, mode='r')
    except ValueError:
        return False
    return True


def load_image(image_path):
    if supports_memmap(image_path):
        return tifffile.memmap(image_path, mode='r')
    else:
        # TODO handle multi-channel images
        return imageio.imread(image_path)
