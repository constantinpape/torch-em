import os
import shutil
from tqdm import tqdm
from glob import glob
from typing import List, Optional

import imageio.v3 as imageio

import torch_em
from torch_em.data.datasets import util


URL = {
    "train": "https://drive.google.com/uc?export=download&id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
    "test": "https://drive.google.com/uc?export=download&id=1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw"
}

CHECKSUM = {
    "train": "25d3d3185bb2970b397cafa72eb664c9b4d24294aee382e7e3df9885affce742",
    "test": "13e522387ae8b1bcc0530e13ff9c7b4d91ec74959ef6f6e57747368d7ee6f88a"
}

# here's the description: https://drive.google.com/file/d/1xYyQ31CHFRnvTCTuuHdconlJCMk2SK7Z/view?usp=sharing
ORGAN_SPLITS = {
    "breast": ["TCGA-A7-A13E-01Z-00-DX1", "TCGA-A7-A13F-01Z-00-DX1", "TCGA-AR-A1AK-01Z-00-DX1",
               "TCGA-AR-A1AS-01Z-00-DX1", "TCGA-E2-A1B5-01Z-00-DX1", "TCGA-E2-A14V-01Z-00-DX1"],
    "kidney": ["TCGA-B0-5711-01Z-00-DX1", "TCGA-HE-7128-01Z-00-DX1", "TCGA-HE-7129-01Z-00-DX1",
               "TCGA-HE-7130-01Z-00-DX1", "TCGA-B0-5710-01Z-00-DX1", "TCGA-B0-5698-01Z-00-DX1"],
    "liver": ["TCGA-18-5592-01Z-00-DX1", "TCGA-38-6178-01Z-00-DX1", "TCGA-49-4488-01Z-00-DX1",
              "TCGA-50-5931-01Z-00-DX1", "TCGA-21-5784-01Z-00-DX1", "TCGA-21-5786-01Z-00-DX1"],
    "prostate": ["TCGA-G9-6336-01Z-00-DX1", "TCGA-G9-6348-01Z-00-DX1", "TCGA-G9-6356-01Z-00-DX1",
                 "TCGA-G9-6363-01Z-00-DX1", "TCGA-CH-5767-01Z-00-DX1", "TCGA-G9-6362-01Z-00-DX1"],
    "bladder": ["TCGA-DK-A2I6-01A-01-TS1", "TCGA-G2-A2EK-01A-02-TSB"],
    "colon": ["TCGA-AY-A8YK-01A-01-TS1", "TCGA-NH-A8F7-01A-01-TS1"],
    "stomach": ["TCGA-KB-A93J-01A-01-TS1", "TCGA-RD-A8N9-01A-01-TS1"]
}


def _download_monuseg(path, download, split):
    assert split in ["train", "test"], "The split choices in MoNuSeg datset are train/test, please choose from them"

    # check if we have extracted the images and labels already
    im_path = os.path.join(path, "images", split)
    label_path = os.path.join(path, "labels", split)
    if os.path.exists(im_path) and os.path.exists(label_path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"monuseg_{split}.zip")
    util.download_source_gdrive(zip_path, URL[split], download=download, checksum=CHECKSUM[split])

    _process_monuseg(path, split)


def _process_monuseg(path, split):
    util.unzip(os.path.join(path, f"monuseg_{split}.zip"), path)

    # assorting the images into expected dir;
    # converting the label xml files to numpy arrays (of same dimension as input images) in the expected dir
    root_img_save_dir = os.path.join(path, "images", split)
    root_label_save_dir = os.path.join(path, "labels", split)

    os.makedirs(root_img_save_dir, exist_ok=True)
    os.makedirs(root_label_save_dir, exist_ok=True)

    if split == "train":
        all_img_dir = sorted(glob(os.path.join(path, "*", "Tissue*", "*")))
        all_xml_label_dir = sorted(glob(os.path.join(path, "*", "Annotations", "*")))
    else:
        all_img_dir = sorted(glob(os.path.join(path, "MoNuSegTestData", "*.tif")))
        all_xml_label_dir = sorted(glob(os.path.join(path, "MoNuSegTestData", "*.xml")))

    assert len(all_img_dir) == len(all_xml_label_dir)

    for img_path, xml_label_path in tqdm(zip(all_img_dir, all_xml_label_dir),
                                         desc=f"Converting {split} split to the expected format",
                                         total=len(all_img_dir)):
        desired_label_shape = imageio.imread(img_path).shape[:-1]

        img_id = os.path.split(img_path)[-1]
        dst = os.path.join(root_img_save_dir, img_id)
        shutil.move(src=img_path, dst=dst)

        _label = util.generate_labeled_array_from_xml(shape=desired_label_shape, xml_file=xml_label_path)
        _fileid = img_id.split(".")[0]
        imageio.imwrite(os.path.join(root_label_save_dir, f"{_fileid}.tif"), _label)

    shutil.rmtree(glob(os.path.join(path, "MoNuSeg*"))[0])
    if split == "train":
        shutil.rmtree(glob(os.path.join(path, "__MACOSX"))[0])


def get_monuseg_dataset(
    path, patch_shape, split, organ_type: Optional[List[str]] = None, download=False,
    offsets=None, boundaries=False, binary=False, **kwargs
):
    """Dataset from https://monuseg.grand-challenge.org/Data/
    """
    _download_monuseg(path, download, split)

    image_paths = sorted(glob(os.path.join(path, "images", split, "*")))
    label_paths = sorted(glob(os.path.join(path, "labels", split, "*")))

    if split == "train" and organ_type is not None:
        # get all patients for multiple organ selection
        all_organ_splits = sum([ORGAN_SPLITS[o][:] for o in organ_type], [])

        image_paths = [
            _path for _path in image_paths if os.path.split(_path)[-1].split(".")[0] in all_organ_splits
        ]
        label_paths = [
            _path for _path in label_paths if os.path.split(_path)[-1].split(".")[0] in all_organ_splits
        ]

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    return torch_em.default_segmentation_dataset(
        image_paths, None, label_paths, None, patch_shape, is_seg_dataset=False, **kwargs
    )


def get_monuseg_loader(
    path, patch_shape, batch_size, split, organ_type=None, download=False, offsets=None, boundaries=False, binary=False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_monuseg_dataset(
        path, patch_shape, split, organ_type=organ_type, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
