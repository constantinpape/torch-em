import os
import unittest

import h5py
import numpy as np
from torch_em.util.test import create_segmentation_test_data


class TestSegmentationDataset(unittest.TestCase):
    path = "./data.h5"
    raw_key = "raw"
    channel_key = "raw_with_channels"
    label_key = "labels"
    shape = (128,) * 3

    def setUp(self):
        chunks = (32,) * 3
        create_segmentation_test_data(self.path, self.raw_key, self.label_key,
                                      shape=self.shape, chunks=chunks)
        with h5py.File(self.path, "a") as f:
            shape = (3,) + self.shape
            chunks = (1,) + chunks
            data_with_channels = np.random.rand(*shape)
            f.create_dataset(self.channel_key, data=data_with_channels, chunks=chunks)

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

    def test_with_channels(self):
        from torch_em.data import SegmentationDataset
        patch_shape = (32, 32, 32)
        ds = SegmentationDataset(self.path, self.channel_key,
                                 self.path, self.label_key,
                                 patch_shape=patch_shape, ndim=3)
        self.assertEqual(ds._ndim, 3)
        expected_raw_shape = (3,) + patch_shape
        expected_label_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_raw_shape)
            self.assertEqual(y.shape, expected_label_shape)


if __name__ == "__main__":
    unittest.main()
