import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal

import torch_em

from .. import util


def get_piccolo_data(path, download):
    """The database is located at:
    - https://www.biobancovasco.bioef.eus/en/Sample-and-data-e-catalog/Databases/PD178-PICCOLO-EN1.html

    Follow the instructions below to get access to the dataset.
    - Visit the attached website above
    - Fill up the access request form: https://labur.eus/EzJUN
    - Send an email to Basque Biobank at solicitudes.biobancovasco@bioef.eus, requesting access to the dataset.
    - The team will request you to follow-up with some formalities.
    - Then, you will gain access to the ".rar" file.
    - Finally, provide the path where the rar file is stored, and you should be good to go.
    """
    if download:
        raise NotImplementedError(
            "Automatic download is not possible for this dataset. See 'get_piccolo_data' for details."
        )

    data_dir = os.path.join(path, r"piccolo dataset-release0.1")
    if os.path.exists(data_dir):
        return data_dir

    rar_file = os.path.join(path, r"piccolo dataset_widefield-release0.1.rar")
    if not os.path.exists(rar_file):
        raise FileNotFoundError(
            "You must download the PICCOLO dataset from the Basque Biobank, see 'get_piccolo_data' for details."
        )

    util.unzip_rarfile(rar_path=rar_file, dst=path, remove=False)
    return data_dir


def _get_piccolo_paths(path, split, download):
    data_dir = get_piccolo_data(path=path, download=download)

    split_dir = os.path.join(data_dir, split)

    image_paths = natsorted(glob(os.path.join(split_dir, "polyps", "*")))
    gt_paths = natsorted(glob(os.path.join(split_dir, "masks", "*")))

    return image_paths, gt_paths


def get_piccolo_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for polyp segmentation in narrow band imaging colonoscopy images.

    This dataset is from Sánchez-Peralta et al. - https://doi.org/10.3390/app10238501.
    To access the dataset, see `get_piccolo_data` for details.

    Please cite it if you use this data in a publication.
    """
    image_paths, gt_paths = _get_piccolo_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )
    return dataset


def get_piccolo_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "validation", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for polyp segmentation in narrow band imaging colonoscopy images.
    See `get_piccolo_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_piccolo_dataset(
        path=path, patch_shape=patch_shape, split=split, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
