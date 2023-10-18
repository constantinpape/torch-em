import os
import shutil
from glob import glob
from tqdm import tqdm

import imageio.v2 as imageio

import torch_em
from torch_em.data.datasets import util


URL = {
    "train": "https://drive.google.com/uc?export=download&id=1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq",
    "test": "https://drive.google.com/uc?export=download&id=1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ"
}


CHECKSUM = {
    "train": "5b7cbeb34817a8f880d3fddc28391e48d3329a91bf3adcbd131ea149a725cd92",
    "test": "bcbc38f6bf8b149230c90c29f3428cc7b2b76f8acd7766ce9fc908fc896c2674"
}


# TODO separate via organ
def _download_monusac(path, download, split):
    assert split in ["train", "test"], "Please choose from train/test"

    # check if we have extracted the images and labels already
    im_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")
    if os.path.exists(im_path) and os.path.exists(label_path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"monusac_{split}.zip")
    util.download_source_gdrive(zip_path, URL[split], download=download, checksum=CHECKSUM[split])

    _process_monusac(path, split)


def _process_monusac(path, split):
    util.unzip(os.path.join(path, f"monusac_{split}.zip"), path)

    # assorting the images into expected dir;
    # converting the label xml files to numpy arrays (of same dimension as input images) in the expected dir
    root_img_save_dir = os.path.join(path, "images", split)
    root_label_save_dir = os.path.join(path, "labels", split)

    os.makedirs(root_img_save_dir, exist_ok=True)
    os.makedirs(root_label_save_dir, exist_ok=True)

    all_patient_dir = sorted(glob(os.path.join(path, "MoNuSAC_images_and_annotations", "*")))

    for patient_dir in tqdm(all_patient_dir, desc=f"Converting {split} inputs for all patients"):
        all_img_dir = sorted(glob(os.path.join(patient_dir, "*.tif")))
        all_xml_label_dir = sorted(glob(os.path.join(patient_dir, "*.xml")))

        assert len(all_img_dir) == len(all_xml_label_dir)

        for img_path, xml_label_path in zip(all_img_dir, all_xml_label_dir):
            desired_label_shape = imageio.imread(img_path).shape[:-1]

            img_id = os.path.split(img_path)[-1]
            dst = os.path.join(root_img_save_dir, img_id)
            shutil.move(src=img_path, dst=dst)

            _label = util.generate_labeled_array_from_xml(shape=desired_label_shape, xml_file=xml_label_path)
            _fileid = img_id.split(".")[0]
            imageio.imwrite(os.path.join(root_label_save_dir, f"{_fileid}.tif"), _label)

    shutil.rmtree(glob(os.path.join(path, "MoNuSAC*"))[0])


def get_monusac_dataset(
    path, patch_shape, split, download=False, offsets=None, boundaries=False, binary=False, **kwargs
):
    """Dataset from https://monusac-2020.grand-challenge.org/Data/
    """
    _download_monusac(path, download, split)

    image_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    return torch_em.default_segmentation_dataset(
        image_path, "*.tif", label_path, "*.tif", patch_shape, is_seg_dataset=False, **kwargs
    )


# TODO implement selecting organ
def get_monusac_loader(
    path, patch_shape, split, batch_size, download=False, offsets=None, boundaries=False, binary=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_monusac_dataset(
        path, patch_shape, split, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader


def main():
    path = "/scratch/usr/nimanwai/data/monusac"
    loader = get_monusac_loader(
        path=path,
        download=True,
        patch_shape=(512, 512),
        batch_size=2,
        split="test"
    )

    print("Length of loader: ", len(loader))


if __name__ == "__main__":
    main()
