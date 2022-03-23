import os
import unittest

import h5py
import numpy as np


class TestRawDataset(unittest.TestCase):
    path = "./data.h5"

    def tearDown(self):
        os.remove(self.path)

    def create_default_data(self, key, shape=None, chunks=None):
        shape = (128,) * 3 if shape is None else shape
        chunks = tuple(min(32, sh) for sh in shape) if chunks is None else chunks
        with h5py.File(self.path, "a") as f:
            f.create_dataset(key, data=np.random.rand(*shape), chunks=chunks)
        return shape, chunks

    def test_dataset_3d(self):
        from torch_em.data import RawDataset
        raw_key = "raw"
        shape, chunks = self.create_default_data(raw_key)
        patch_shape = chunks
        ds = RawDataset(self.path, raw_key, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds._ndim, 3)
        expected_shape = (1,) + patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_shape)

    def test_dataset_2d(self):
        from torch_em.data import RawDataset
        raw_key = "raw"
        shape, chunks = self.create_default_data(raw_key)
        patch_shape = (1, 32, 32)
        ds = RawDataset(self.path, raw_key, patch_shape=patch_shape, ndim=2)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds._ndim, 2)
        expected_shape = patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_shape)

    def _test_dataset_4d(self):
        from torch_em.data import RawDataset
        raw_key = "raw"
        shape = (12, 64, 64, 64)
        self.create_default_data(raw_key, shape=shape)
        patch_shape = (3, 32, 32, 32)
        ds = RawDataset(self.path, raw_key, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds._ndim, 4)
        expected_shape = (1,) + patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_shape)

    def test_roi(self):
        from torch_em.data import RawDataset
        raw_key = "raw"
        self.create_default_data(raw_key)
        patch_shape = (32, 32, 32)
        roi = np.s_[32:96, 32:96, 32:96]
        ds = RawDataset(self.path, raw_key, patch_shape=patch_shape, roi=roi)
        roi_shape = 3 * (64,)
        self.assertEqual(ds.raw.shape, roi_shape)
        expected_shape = (1,) + patch_shape
        for i in range(10):
            self.assertEqual(tuple(ds[i].shape), expected_shape)

    def test_with_channels(self):
        from torch_em.data import RawDataset
        raw_key = "raw"
        self.create_default_data(raw_key, shape=(3, 128, 128, 128), chunks=(1, 32, 32, 32))
        patch_shape = (32, 32, 32)
        ds = RawDataset(self.path, raw_key, patch_shape=patch_shape, ndim=3, with_channels=True)
        self.assertEqual(ds._ndim, 3)
        expected_raw_shape = (3,) + patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_raw_shape)


if __name__ == "__main__":
    unittest.main()
