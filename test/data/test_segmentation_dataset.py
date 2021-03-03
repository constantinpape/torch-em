import os
import unittest
import numpy as np
from torch_em.util.test import create_segmentation_test_data


class TestSegmentationDataset(unittest.TestCase):
    path = './data.h5'
    raw_key = 'raw'
    label_key = 'labels'
    shape = (128,) * 3

    def setUp(self):
        create_segmentation_test_data(self.path, self.raw_key, self.label_key,
                                      shape=self.shape, chunks=(32,) * 3)

    def tearDown(self):
        os.remove(self.path)

    def test_dataset(self):
        from torch_em.data import SegmentationDataset
        patch_shape = (32, 32, 32)
        ds = SegmentationDataset(self.path, self.raw_key,
                                 self.path, self.label_key,
                                 patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, self.shape)
        self.assertEqual(ds.labels.shape, self.shape)
        self.assertEqual(ds._ndim, 3)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)

    def test_roi(self):
        from torch_em.data import SegmentationDataset
        patch_shape = (32, 32, 32)
        roi = np.s_[32:96, 32:96, 32:96]
        ds = SegmentationDataset(self.path, self.raw_key,
                                 self.path, self.label_key,
                                 patch_shape=patch_shape,
                                 roi=roi)

        roi_shape = 3 * (64,)
        self.assertEqual(ds.raw.shape, roi_shape)
        self.assertEqual(ds.labels.shape, roi_shape)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
