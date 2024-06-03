import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal

import torch_em

from .. import util


URLS = {
    "CT": [
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part1.zip",  # inputs part 1
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part2.zip",  # inputs part 2
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_labels.zip",  # labels
    ],
    "X-Ray": ["https://zenodo.org/records/10913196/files/train.zip"]
}

CHECKSUMS = {
    "CT": [
        "e2e9f99798960607ffced1fbdeee75a626c41bf859eaf4125029a38fac6b7609",  # inputs part 1
        "19f3cdc5edd1daf9324c70f8ba683eed054f6ed8f2b1cc59dbd80724f8f0bbb2",  # inputs part 2
        "c4d3857e02d3ee5d0df6c8c918dd3cf5a7c9419135f1ec089b78215f37c6665c"  # labels
    ],
    "X-Ray": ["48d107979eb929a3c61da4e75566306a066408954cf132907bda570f2a7de725"]
}

TARGET_DIRS = {
    "CT": ["CT/images", "CT/images", "CT/labels"],
    "X-Ray": ["X-Ray"]
}

MODALITIES = ["CT", "X-Ray"]


def get_pengwin_data(path, modality, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    # if os.path.exists(data_dir):
    #     return data_dir

    urls = URLS[modality]
    checksums = CHECKSUMS[modality]
    dst_dirs = TARGET_DIRS[modality]

    for url, checksum, dst_dir in zip(urls, checksums, dst_dirs):
        zip_path = os.path.join(path, os.path.split(url)[-1])
        print(zip_path)
        breakpoint()
        util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
        util.unzip(zip_path=zip_path, dst=os.path.join(data_dir, dst_dir), remove=False)

    return data_dir


def _get_pengwin_paths(path, modality, download):
    if not isinstance(modality, str) and modality in MODALITIES:
        raise ValueError(f"Please choose a modality from {MODALITIES}.")

    data_dir = get_pengwin_data(path=path, download=download)

    if modality == "CT":
        image_paths = natsorted(glob(os.path.join(data_dir, modality, "images", "*.mha")))
        gt_paths = natsorted(glob(os.path.join(data_dir, modality, "labels", "*.mha")))
    else:
        base_dir = os.path.join(data_dir, modality, "train")
        image_paths = natsorted(glob(os.path.join(base_dir, "input", "images", "*.tif")))
        gt_paths = natsorted(glob(os.path.join(base_dir, "output", "images", "*.tif")))

    return image_paths, gt_paths


def get_pengwin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["CT", "X-Ray"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of pelvic fracture in CT and X-Ray images.

    This dataset is from the PENGWIN Challenge:
    - https://pengwin.grand-challenge.org/pengwin/
    - Related publication: https://doi.org/10.1007/978-3-031-43996-4_30

    Please cite it if you use this dataset in your publication.
    """
    image_paths, gt_paths = _get_pengwin_paths(path=path, modality=modality, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_pengwin_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    modality: Literal["CT", "X-Ray"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of pelvic fracture in CT and X-Ray images. See `get_pengwin_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pengwin_dataset(
        path=path,
        patch_shape=patch_shape,
        modality=modality,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
