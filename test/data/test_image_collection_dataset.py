import os
import unittest
from glob import glob
from shutil import rmtree

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


if __name__ == '__main__':
    unittest.main()
