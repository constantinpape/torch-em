import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple

import imageio.v3 as imageio

import torch_em

from .. import util


URL = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
CHECKSUM = "54601e952315d8f67383e9202a6e145997ade429f54f7e0af44b4e158714f424"


def get_montgomery_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "MontgomerySet")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "NLM-MontgomeryCXRSet.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_montgomery_paths(path, download):
    data_dir = get_montgomery_data(path=path, download=download)
    gt_dir = os.path.join(data_dir, "ManualMask", "gt")

    image_paths = sorted(glob(os.path.join(data_dir, "CXR_png", "*.png")))

    if os.path.exists(gt_dir):
        gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))
        if len(image_paths) == len(gt_paths):
            return image_paths, gt_paths

    else:
        os.makedirs(gt_dir, exist_ok=True)

    lmask_dir = os.path.join(data_dir, "ManualMask", "leftMask")
    rmask_dir = os.path.join(data_dir, "ManualMask", "rightMask")
    gt_paths = []
    for image_path in tqdm(image_paths, desc="Merging left and right lung halves"):
        image_id = os.path.split(image_path)[-1]

        # merge the left and right lung halves into one gt file
        gt = imageio.imread(os.path.join(lmask_dir, image_id))
        gt += imageio.imread(os.path.join(rmask_dir, image_id))
        gt = gt.astype("uint8")

        gt_path = os.path.join(gt_dir, image_id)

        imageio.imwrite(gt_path, gt)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_montgomery_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = True,
    download: bool = False,
    **kwargs
):
    """Dataset for the segmentation of lungs in x-ray.

    This dataset is from the publication:
    - https://doi.org/10.1109/TMI.2013.2284099
    - https://doi.org/10.1109/tmi.2013.2290491

    The database is located at
    https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html.

    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_montgomery_paths(path=path, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )
    return dataset


def get_montgomery_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = True,
    download: bool = False,
    **kwargs
):
    """Dataloader for the segmentation of lungs in x-ray. See 'get_montgomery_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_montgomery_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
