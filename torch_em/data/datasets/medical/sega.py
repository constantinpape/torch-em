import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal

import torch_em

from .. import util


URL = {
    "kits": "https://figshare.com/ndownloader/files/30950821",
    "rider": "https://figshare.com/ndownloader/files/30950914",
    "dongyang": "https://figshare.com/ndownloader/files/30950971"
}

CHECKSUMS = {
    "kits": "6c9c2ea31e5998348acf1c4f6683ae07041bd6c8caf309dd049adc7f222de26e",
    "rider": "7244038a6a4f70ae70b9288a2ce874d32128181de2177c63a7612d9ab3c4f5fa",
    "dongyang": "0187e90038cba0564e6304ef0182969ff57a31b42c5969d2b9188a27219da541"
}

ZIPFILES = {
    "kits": "KiTS.zip",
    "rider": "Rider.zip",
    "dongyang": "Dongyang.zip"
}


def get_sega_data(path, data_choice, download):
    os.makedirs(path, exist_ok=True)

    data_choice = data_choice.lower()

    zip_fid = ZIPFILES[data_choice]

    data_dir = os.path.join(path, Path(zip_fid).stem)
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, zip_fid)
    util.download_source(
        path=zip_path, url=URL[data_choice], download=download, checksum=CHECKSUMS[data_choice],
    )
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_sega_paths(path, data_choice, download):
    if data_choice is None:
        data_choices = URL.keys()
    else:
        if isinstance(data_choice, str):
            data_choices = [data_choice]

    data_dirs = [get_sega_data(path=path, data_choice=data_choice, download=download) for data_choice in data_choices]

    image_paths, gt_paths = [], []
    for data_dir in data_dirs:
        all_volumes_paths = glob(os.path.join(data_dir, "*", "*.nrrd"))
        for volume_path in all_volumes_paths:
            if volume_path.endswith(".seg.nrrd"):
                gt_paths.append(volume_path)
            else:
                image_paths.append(volume_path)

    # now let's wrap the volumes to nifti format
    fimage_dir = os.path.join(path, "data", "images")
    fgt_dir = os.path.join(path, "data", "labels")

    os.makedirs(fimage_dir, exist_ok=True)
    os.makedirs(fgt_dir, exist_ok=True)

    fimage_paths, fgt_paths = [], []
    for image_path, gt_path in zip(natsorted(image_paths), natsorted(gt_paths)):
        fimage_path = os.path.join(fimage_dir, f"{Path(image_path).stem}.nii.gz")
        fgt_path = os.path.join(fgt_dir, f"{Path(image_path).stem}.nii.gz")

        fimage_paths.append(fimage_path)
        fgt_paths.append(fgt_path)

        if os.path.exists(fimage_path) and os.path.exists(fgt_path):
            continue

        import nrrd
        import numpy as np
        import nibabel as nib

        image = nrrd.read(image_path)[0]
        gt = nrrd.read(gt_path)[0]

        image_nifti = nib.Nifti2Image(image, np.eye(4))
        gt_nifti = nib.Nifti2Image(gt, np.eye(4))

        nib.save(image_nifti, fimage_path)
        nib.save(gt_nifti, fgt_path)

    return natsorted(fimage_paths), natsorted(fgt_paths)


def get_sega_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of aorta in computed tomography angiography  (CTA) scans.

    This dataset is from Pepe et al. - https://doi.org/10.1007/978-3-031-53241-2
    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_sega_paths(path=path, data_choice=data_choice, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs,
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )

    return dataset


def get_sega_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of aorta in CTA scans. See `get_sega_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sega_dataset(
        path=path,
        patch_shape=patch_shape,
        data_choice=data_choice,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
