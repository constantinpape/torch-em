import os
import tempfile
import unittest
from glob import glob
from shutil import rmtree

import tifffile
import numpy as np
from torch_em.util.test import create_image_collection_test_data


class TestImageCollectionDataset(unittest.TestCase):
    folder = "./data"
    n_images = 10

    def setUp(self):
        create_image_collection_test_data(self.folder,
                                          n_images=self.n_images,
                                          min_shape=(256, 256),
                                          max_shape=(512, 512))

    def tearDown(self):
        rmtree(self.folder, ignore_errors=True)

    def test_dataset(self):
        from torch_em.data import ImageCollectionDataset
        patch_shape = (256, 256)

        raw_paths = glob(os.path.join(self.folder, "images", "*.tif"))
        label_paths = glob(os.path.join(self.folder, "labels", "*.tif"))
        ds = ImageCollectionDataset(raw_paths, label_paths,
                                    patch_shape=patch_shape)
        self.assertEqual(len(ds), self.n_images)
        self.assertEqual(ds._ndim, 2)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)

    def test_dataset_with_sampler(self):
        from torch_em.data import ImageCollectionDataset
        from torch_em.data.sampler import MinIntensitySampler
        patch_shape = (256, 256)

        raw_paths = glob(os.path.join(self.folder, "images", "*.tif"))
        label_paths = glob(os.path.join(self.folder, "labels", "*.tif"))
        sampler = MinIntensitySampler(min_intensity=0.51, p_reject=0.9)
        ds = ImageCollectionDataset(raw_paths, label_paths,
                                    patch_shape=patch_shape,
                                    sampler=sampler)
        self.assertEqual(len(ds), self.n_images)
        self.assertEqual(ds._ndim, 2)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)


def generate_sample_data(folder, n_images, image_shape, label_shape):
    im_folder = os.path.join(folder, "images")
    label_folder = os.path.join(folder, "labels")
    os.makedirs(im_folder)
    os.makedirs(label_folder)
    for i in range(n_images):
        raw = np.empty(image_shape, dtype=np.uint8)
        label = np.ones(label_shape, dtype=np.float32)
        tifffile.imwrite(os.path.join(im_folder, f"test_{i}.tif"), raw)
        tifffile.imwrite(os.path.join(label_folder, f"test_{i}.tif"), label)


class TestImageCollectionDatasetWithChannels(unittest.TestCase):
    def test_channel_end(self):
        from torch_em.data import ImageCollectionDataset

        patch_shape = (256, 256)

        with tempfile.TemporaryDirectory() as td:
            generate_sample_data(td, 10, (256, 256, 2), (256, 256))
            raw_paths = glob(os.path.join(td, "images", "*.tif"))
            label_paths = glob(os.path.join(td, "labels", "*.tif"))

            ds = ImageCollectionDataset(raw_paths, label_paths,
                                        patch_shape=patch_shape)
            self.assertEqual(len(ds), 10)
            self.assertEqual(ds._get_sample(0)[0].shape[0], 2)

    def test_channel_begin(self):
        from torch_em.data import ImageCollectionDataset

        patch_shape = (256, 256)

        with tempfile.TemporaryDirectory() as td:
            generate_sample_data(td, 10, (2, 256, 256), (256, 256))
            raw_paths = glob(os.path.join(td, "images", "*.tif"))
            label_paths = glob(os.path.join(td, "labels", "*.tif"))

            ds = ImageCollectionDataset(raw_paths, label_paths,
                                        patch_shape=patch_shape)
            self.assertEqual(len(ds), 10)
            self.assertEqual(ds._get_sample(0)[0].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
