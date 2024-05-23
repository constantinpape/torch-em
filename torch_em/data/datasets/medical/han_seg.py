import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util


URL = "https://zenodo.org/records/7442914/files/HaN-Seg.zip"
CHECKSUM = "20226dd717f334dc1b1afe961b3375f946fa56b64a80bf5349128f90c0bbfa5f"


def get_han_seg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "HaN-Seg")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "HaN-Seg.zip")
    util.download_source(
        path=zip_path, url=URL, download=download, checksum=CHECKSUM
    )
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return data_dir


def _get_han_seg_paths(path, download):
    data_dir = get_han_seg_data(path=path, download=download)

    all_case_dirs = natsorted(glob(os.path.join(data_dir, "set_1", "case_*")))
    for case_dir in all_case_dirs:

        all_vols = []
        all_nrrd_paths = glob(os.path.join(case_dir, "*.nrrd"))
        for nrrd_path in all_nrrd_paths:
            from pathlib import Path
            import nrrd

            image_id = Path(nrrd_path).stem
            if not image_id.endswith("_MR_T1"):
                continue

            data, header = nrrd.read(nrrd_path)
            all_vols.append(data.transpose(2, 1, 0))

        import napari
        v = napari.Viewer()
        v.add_image(all_vols[0])
        # for vol in all_vols[1:]:
        #     v.add_labels(vol)
        napari.run()

    image_paths = ...
    gt_paths = ...

    return image_paths, gt_paths


def get_han_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    image_paths, gt_paths = _get_han_seg_paths(path=path, download=download)

    dataset = ...

    return dataset


def get_han_seg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_han_seg_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
