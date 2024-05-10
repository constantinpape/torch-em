import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple

import imageio.v3 as imageio
from skimage.transform import resize

import torch_em
from torch_em.transform import get_raw_transform

from .. import util
from ... import ImageCollectionDataset


URL = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
CHECKSUM = "54601e952315d8f67383e9202a6e145997ade429f54f7e0af44b4e158714f424"


def _download_montgomery_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "MontgomerySet")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "NLM-MontgomeryCXRSet.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


class _ResizeInputs:
    def __init__(self, target_shape, is_label=False):
        self.target_shape = target_shape
        self.is_label = is_label

    def __call__(self, inputs):
        print(inputs.shape)
        if self.is_label:
            anti_aliasing = True
        else:
            anti_aliasing = False

        inputs = resize(
            image=inputs,
            output_shape=self.target_shape,
            order=3,
            anti_aliasing=anti_aliasing,
            preserve_range=True,
        )

        return inputs


def _get_montgomery_paths(path, download):
    data_dir = _download_montgomery_data(path=path, download=download)
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
    download: bool = False,
    resize_inputs: bool = True,
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
        raw_transform = get_raw_transform(augmentation1=_ResizeInputs(target_shape=patch_shape))
        label_transform = _ResizeInputs(target_shape=patch_shape, is_label=True)
    else:
        raw_transform, label_transform = None, None

    dataset = ImageCollectionDataset(
        raw_image_paths=image_paths,
        label_image_paths=gt_paths,
        patch_shape=(5000, 5000),
        raw_transform=raw_transform,
        label_transform=label_transform,
        **kwargs
    )
    return dataset


def get_montgomery_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    resize_inputs: bool = True,
    **kwargs
):
    """Dataloader for the segmentation of lungs in x-ray. See 'get_montgomery_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_montgomery_dataset(
        path=path,
        patch_shape=patch_shape,
        download=download,
        resize_inputs=resize_inputs,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
