import os
import shutil
from glob import glob
from typing import Tuple, Union
from urllib.parse import urljoin
from urllib3.exceptions import ProtocolError

import torch_em

from .. import util


BASE_URL = "https://uwaterloo.ca/vision-image-processing-lab/sites/ca.vision-image-processing-lab/files/uploads/files/"


ZIPFILES = {
    "set1": "skin_image_data_set-1.zip",  # patients with melanoma
    "set2": "skin_image_data_set-2.zip"  # patients without melanoma
}

CHECKSUMS = {
    "set1": "1788cd3eb7a4744012aad9a154e514fc5b82b9f3b19e31cc1b6ded5fc6bed297",
    "set2": "108a818baf20b36ef4544ebda10a8075dad99e335f0535c9533bb14cb02b5c53"
}


def get_uwaterloo_skin_data(path, chosen_set, download):
    os.makedirs(path, exist_ok=True)

    assert chosen_set in ZIPFILES.keys(), f"'{chosen_set}' is not a valid set."

    data_dir = os.path.join(path, f"{chosen_set}_Data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, ZIPFILES[chosen_set])
    url = urljoin(BASE_URL, ZIPFILES[chosen_set])

    try:
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[chosen_set])
    except ProtocolError:  # the 'uwaterloo.ca' quite randomly times out of connections, pretty weird.
        msg = "The server seems to be unreachable at the moment. "
        msg += f"We recommend downloading the data from {url} at '{path}'. "
        print(msg)
        quit()

    util.unzip(zip_path=zip_path, dst=path)

    setnum = chosen_set[-1]
    tmp_dir = os.path.join(path, fr"Skin Image Data Set-{setnum}")
    shutil.move(src=tmp_dir, dst=data_dir)

    return data_dir


def _get_uwaterloo_skin_paths(path, download):
    data_dir = get_uwaterloo_skin_data(path=path, chosen_set="set1", download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "skin_data", "melanoma", "*", "*_orig.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, "skin_data", "melanoma", "*", "*_contour.png")))

    data_dir = get_uwaterloo_skin_data(path=path, chosen_set="set2", download=download)

    image_paths.extend(sorted(glob(os.path.join(data_dir, "skin_data", "notmelanoma", "*", "*_orig.jpg"))))
    gt_paths.extend(sorted(glob(os.path.join(data_dir, "skin_data", "notmelanoma", "*", "*_contour.png"))))

    return image_paths, gt_paths


def get_uwaterloo_skin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for skin lesion segmentation in dermoscopy images.

    The database is located at https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection.
    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_uwaterloo_skin_paths(path=path, download=download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_uwaterloo_skin_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for skin lesion segmentation in dermoscopy images. See `get_uwaterloo_skin_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_uwaterloo_skin_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
