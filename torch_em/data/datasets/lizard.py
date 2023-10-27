import os
import warnings
from glob import glob
from shutil import rmtree

import h5py
import imageio.v3 as imageio
import torch_em

from scipy.io import loadmat
from tqdm import tqdm
from .import util

# TODO: the links don't work anymore (?)
# workaround to still make this work (kaggle still has the dataset in the same structure):
#   - download the zip files manually from here - https://www.kaggle.com/datasets/aadimator/lizard-dataset
#   - Kaggle API (TODO) - `kaggle datasets download -d aadimator/lizard-dataset`
URL1 = "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images1.zip"
URL2 = "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images2.zip"
LABEL_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_labels.zip"

CHECKSUM1 = "d2c4e7c83dff634624c9c14d4a1a0b821d4e9ac41e05e3b36303d8f0c510113d"
CHECKSUM2 = "9f529f30d9de66587167991a8bf75aaad07ce1d518b72e825c868ac7c33015ed"
LABEL_CHECKSUM = "79f22ca83ca535682fba340cbc8bb66b74abd1ead4151ffc8593f204fcb97dec"


def _extract_images(image_folder, label_folder, output_dir):
    image_files = glob(os.path.join(image_folder, "*.png"))
    for image_file in tqdm(image_files, desc=f"Extract images from {image_folder}"):
        fname = os.path.basename(image_file)
        label_file = os.path.join(label_folder, fname.replace(".png", ".mat"))
        assert os.path.exists(label_file), label_file

        image = imageio.imread(image_file)
        assert image.ndim == 3 and image.shape[-1] == 3

        labels = loadmat(label_file)
        segmentation = labels["inst_map"]
        assert image.shape[:-1] == segmentation.shape
        classes = labels["class"]

        image = image.transpose((2, 0, 1))
        assert image.shape[1:] == segmentation.shape

        output_file = os.path.join(output_dir, fname.replace(".png", ".h5"))
        with h5py.File(output_file, "a") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("labels/segmentation", data=segmentation, compression="gzip")
            f.create_dataset("labels/classes", data=classes, compression="gzip")


def _require_lizard_data(path, download):
    image_files = glob(os.path.join(path, "*.h5"))
    if len(image_files) > 0:
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "lizard_images1.zip")
    util.download_source(zip_path, URL1, download=download, checksum=CHECKSUM1)
    util.unzip(zip_path, path, remove=True)

    zip_path = os.path.join(path, "lizard_images2.zip")
    util.download_source(zip_path, URL2, download=download, checksum=CHECKSUM2)
    util.unzip(zip_path, path, remove=True)

    zip_path = os.path.join(path, "lizard_labels.zip")
    util.download_source(zip_path, LABEL_URL, download=download, checksum=LABEL_CHECKSUM)
    util.unzip(zip_path, path, remove=True)

    image_folder1 = os.path.join(path, "Lizard_Images1")
    image_folder2 = os.path.join(path, "Lizard_Images2")
    label_folder = os.path.join(path, "Lizard_Labels")

    assert os.path.exists(image_folder1), image_folder1
    assert os.path.exists(image_folder2), image_folder2
    assert os.path.exists(label_folder), label_folder

    _extract_images(image_folder1, os.path.join(label_folder, "Labels"), path)
    _extract_images(image_folder2, os.path.join(label_folder, "Labels"), path)

    rmtree(image_folder1)
    rmtree(image_folder2)
    rmtree(label_folder)


def get_lizard_dataset(path, patch_shape, download=False, **kwargs):
    """Dataset for the segmentation of nuclei in histopathology.

    This dataset is from the publication https://doi.org/10.48550/arXiv.2108.11195.
    Please cite it if you use this dataset for a publication.
    """
    if download:
        warnings.warn(f"The download link does not work right now. Please manually download the zip files to {path} from https://www.kaggle.com/datasets/aadimator/lizard-dataset")

    _require_lizard_data(path, download)

    data_paths = glob(os.path.join(path, "*.h5"))
    data_paths.sort()

    raw_key = "image"
    label_key = "labels/segmentation"
    return torch_em.default_segmentation_dataset(
        data_paths, raw_key, data_paths, label_key, patch_shape, ndim=2, with_channels=True, **kwargs
    )


# TODO implement loading the classification labels
# TODO implement selecting different tissue types
# TODO implement train / val / test split (is pre-defined in a csv)
def get_lizard_loader(path, patch_shape, batch_size, download=False, **kwargs):
    """Dataloader for the segmentation of nuclei in histopathology. See 'get_lizard_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_lizard_dataset(path, patch_shape, download=download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
