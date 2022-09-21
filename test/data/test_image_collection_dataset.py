import os
import tempfile
import unittest
from glob import glob
from shutil import rmtree

import tifffile
import numpy as np
from torch_em.util.test import create_image_collection_test_data


class TestSegmentationDataset(unittest.TestCase):
    folder = './data'
    n_images = 10

    def setUp(self):
        create_image_collection_test_data(self.folder,
                                          n_images=self.n_images,
                                          min_shape=(256, 256),
                                          max_shape=(512, 512))

    def tearDown(self):
        rmtree(self.folder)

    def test_dataset(self):
        from torch_em.data import ImageCollectionDataset
        patch_shape = (256, 256)

        raw_paths = glob(os.path.join(self.folder, 'images', '*.tif'))
        label_paths = glob(os.path.join(self.folder, 'labels', '*.tif'))
        ds = ImageCollectionDataset(raw_paths, label_paths,
                                    patch_shape=patch_shape)
        self.assertEqual(len(ds), self.n_images)
        self.assertEqual(ds._ndim, 2)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)


def generate_sample_data(folder, n_images, shape):
    im_folder = os.path.join(folder, "images")
    label_folder = os.path.join(folder, "labels")
    os.makedirs(im_folder)
    os.makedirs(label_folder)
    for i in range(n_images):
        raw = np.empty(shape, dtype=np.uint8)
        label = np.ones(shape, dtype=np.float32)
        tifffile.imwrite(os.path.join(im_folder, f"test_{i}.tif"), raw)
        tifffile.imwrite(os.path.join(label_folder, f"test_{i}.tif"), label)


class TestChannelsDataset(unittest.TestCase):
    def test_channel_end(self):
        from torch_em.data import ImageCollectionDataset
        
        patch_shape = (256, 256)

        with tempfile.TemporaryDirectory() as td:
            raw_paths = glob(os.path.join(td, "images", "*.tif"))
            label_paths = glob(os.path.join(td, "labels", "*.tif"))

            generate_sample_data(td, 10, (64, 64, 2))
            ds = ImageCollectionDataset(raw_paths, label_paths,
                                    patch_shape=patch_shape)
            self.assertEqual(len(ds), 10)
            self.assertEqual(ds._get_sample(0)[0].shape[0], 2)
    
    def test_channel_begin(self):
        from torch_em.data import ImageCollectionDataset
        
        patch_shape = (256, 256)

        with tempfile.TemporaryDirectory() as td:
            raw_paths = glob(os.path.join(td, "images", "*.tif"))
            label_paths = glob(os.path.join(td, "labels", "*.tif"))

            generate_sample_data(td, 10, (2, 64, 64))
            ds = ImageCollectionDataset(raw_paths, label_paths,
                                    patch_shape=patch_shape)
            self.assertEqual(len(ds), 10)
            self.assertEqual(ds._get_sample(0)[0].shape[0], 2)


if __name__ == '__main__':
    unittest.main()
