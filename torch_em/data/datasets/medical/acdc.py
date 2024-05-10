import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util
from ... import ConcatDataset


URL = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download"
CHECKSUM = "2787e08b0d3525cbac710fc3bdf69ee7c5fd7446472e49db8bc78548802f6b5e"


def _download_acdc_dataset(path, download):
    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ACDC.zip")
    trg_dir = os.path.join(path, "ACDC")
    if os.path.exists(trg_dir):
        return trg_dir

    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)
    return trg_dir


def _get_acdc_paths(path, split, download):
    root_dir = _download_acdc_dataset(path=path, download=download)

    if split == "train":
        input_dir = os.path.join(root_dir, "database", "training")
    else:
        input_dir = os.path.join(root_dir, "database", "testing")

    all_patient_dirs = natsorted(glob(os.path.join(input_dir, "patient*")))

    image_paths, gt_paths = [], []
    for per_patient_dir in all_patient_dirs:
        # the volumes with frames are for particular time frames (end diastole (ED) and end systole (ES))
        # the "frames" denote - ED and ES phase instances, which have manual segmentations.
        all_volumes = glob(os.path.join(per_patient_dir, "*frame*.nii.gz"))
        for vol_path in all_volumes:
            sres = vol_path.find("gt")
            if sres == -1:  # this means the search was invalid, hence it's the  mri volume
                image_paths.append(vol_path)
            else:  # this means that the search went through, hence it's the ground truth volume
                gt_paths.append(vol_path)

    return natsorted(image_paths), natsorted(gt_paths)


def get_acdc_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
):
    """Dataset fir multi-structure segmentation in cardiac MRI.

    The labels have the following mapping:
    - 0 (background), 1 (right ventricle cavity),2 (myocardium), 3 (left ventricle cavity)

    The database is located at
    https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb

    The dataset is from the publication https://doi.org/10.1109/TMI.2018.2837502

    Please cite it if you use this dataset for a publication.
    """
    assert split in ["train", "test"], f"{split} is not a valid split."

    image_paths, gt_paths = _get_acdc_paths(path=path, split=split, download=download)

    all_datasets = []
    for image_path, gt_path in zip(image_paths, gt_paths):
        per_vol_ds = torch_em.default_segmentation_dataset(
            raw_paths=image_path,
            raw_key="data",
            label_paths=gt_path,
            label_key="data",
            patch_shape=patch_shape,
            is_seg_dataset=True,
            **kwargs
        )
        all_datasets.append(per_vol_ds)

    return ConcatDataset(*all_datasets)


def get_acdc_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    """Dataloader for multi-structure segmentation in cardiac MRI, See `get_acdc_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_acdc_dataset(path=path, split=split, patch_shape=patch_shape, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
