import os
import torch_em

from . import util

URL = "https://drive.google.com/uc?export=download&id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA"
CHECKSUM = ""


# TODO separate via organ
def _download_monuseg(path, download):
    # check if we have extracted the images and labels already
    im_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")
    if os.path.exists(im_path) and os.path.exists(label_path):
        return

    raise NotImplementedError("Download and post-processing for the monuseg data is not yet implemented.")

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "monuseg.zip")
    util.download_source_gdrive(zip_path, URL, download=download, checksum=CHECKSUM)


# TODO
def _process_monuseg():
    pass


def get_monuseg_dataset(
    path, patch_shape, download=False, offsets=None, boundaries=False, binary=False, **kwargs
):
    _download_monuseg(path, download)

    image_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    return torch_em.default_segmentation_dataset(
        image_path, "*.tif", label_path, "*.tif", patch_shape, is_seg_dataset=False, **kwargs
    )


# TODO implement selecting organ
def get_monuseg_loader(
    path, patch_shape, batch_size, download=False, offsets=None, boundaries=False, binary=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_monuseg_dataset(
        path, patch_shape, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
