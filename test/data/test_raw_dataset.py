import os
import unittest

import h5py
import numpy as np

from torch_em.data.raw_dataset import RawDataset, RawDatasetWithMasks


class TestRawDatasetBase:
    """Base class to hold shared tests for RawDataset and subclasses."""

    path = "./data.h5"
    dataset_class = None

    def tearDown(self):
        os.remove(self.path)

    def create_default_data(self, key, shape=None, chunks=None):
        shape = (128,) * 3 if shape is None else shape
        chunks = tuple(min(32, sh) for sh in shape) if chunks is None else chunks
        with h5py.File(self.path, "a") as f:
            f.create_dataset(key, data=np.random.rand(*shape), chunks=chunks)
        return shape, chunks

    def test_dataset_3d(self):
        raw_key = "raw"
        shape, chunks = self.create_default_data(raw_key)
        patch_shape = chunks
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds._ndim, 3)
        expected_shape = (1,) + patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_shape)

    def test_dataset_2d(self):
        raw_key = "raw"
        shape, chunks = self.create_default_data(raw_key)
        patch_shape = (1, 32, 32)
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape, ndim=2)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds._ndim, 2)
        expected_shape = patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_shape)

    def test_dataset_4d(self):
        raw_key = "raw"
        shape = (12, 64, 64, 64)
        self.create_default_data(raw_key, shape=shape)
        patch_shape = (3, 32, 32, 32)
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds._ndim, 4)
        expected_shape = patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_shape)

    def test_roi(self):
        raw_key = "raw"
        self.create_default_data(raw_key)
        patch_shape = (32, 32, 32)
        roi = np.s_[32:96, 32:96, 32:96]
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape, roi=roi)
        roi_shape = 3 * (64,)
        self.assertEqual(ds.raw.shape, roi_shape)
        expected_shape = (1,) + patch_shape
        for i in range(10):
            self.assertEqual(tuple(ds[i].shape), expected_shape)

    def test_with_channels(self):
        raw_key = "raw"
        self.create_default_data(raw_key, shape=(3, 128, 128, 128), chunks=(1, 32, 32, 32))
        patch_shape = (32, 32, 32)
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape, ndim=3, with_channels=True)
        self.assertEqual(ds._ndim, 3)
        expected_raw_shape = (3,) + patch_shape
        for i in range(10):
            self.assertEqual(ds[i].shape, expected_raw_shape)


class TestRawDataset(TestRawDatasetBase, unittest.TestCase):
    dataset_class = RawDataset


class TestRawDatasetWithMasks(TestRawDatasetBase, unittest.TestCase):
    dataset_class = RawDatasetWithMasks
    mrc_path = "./mask.mrc"
    
    def tearDown(self):
        super().tearDown()

        if os.path.exists(self.mrc_path):
            os.remove(self.mrc_path)
    
    def create_default_mrc_data(self, data):
        import mrcfile
        with mrcfile.new(self.mrc_path, overwrite=True) as f:
            f.set_data(data.astype(np.float32))

    # subclass-specific tests, designed for a 3d dataset
    def test_sample_mask(self):
        from torch_em.data.sampler import MinForegroundSampler

        raw_key, mask_key = "raw", "mask"
        shape = (128, 128, 128)
        chunks = (32, 32, 32)
        
        # define central z slices 40:80 as the sample ROI, above and below is empty
        sample_mask = np.zeros(shape, dtype=np.float32)
        sample_mask[40:80, :, :] = 1.0

        with h5py.File(self.path, "a") as f:
            f.create_dataset(raw_key, data=sample_mask, chunks=chunks)
            f.create_dataset(mask_key, data=sample_mask, chunks=chunks)

        patch_shape = chunks

        min_fraction = 0.95
        sampler = MinForegroundSampler(min_fraction=min_fraction)

        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape, sampler=sampler, 
                                sample_mask_path=self.path, sample_mask_key=mask_key)

        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds.sample_mask.shape, shape)
        self.assertEqual(ds._ndim, 3)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            patch = ds[i]
            self.assertEqual(patch.shape, expected_shape)

            # check that patches are sampled inside the ROI defined by sample mask
            self.assertGreaterEqual(patch.mean().item(), min_fraction)

    def test_bg_mask(self):
        raw_key, mask_key = "raw", "mask"

        shape, chunks = self.create_default_data(raw_key)
        self.create_default_data(mask_key)

        patch_shape = chunks
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape, 
                                bg_mask_path=self.path, bg_mask_key=mask_key)

        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds.bg_mask.shape, shape)
        self.assertEqual(ds._ndim, 3)
        expected_shape = (2,) + patch_shape

        for i in range(10):

            patch = ds[i]
            self.assertEqual(patch.shape, expected_shape)

    def test_bg_mask_mrc(self):
        raw_key, mask_key = "raw", None

        shape, chunks = self.create_default_data(raw_key)
        self.create_default_mrc_data(np.zeros(shape))

        patch_shape = chunks
        ds = self.dataset_class(self.path, raw_key, patch_shape=patch_shape, 
                                bg_mask_path=self.mrc_path, bg_mask_key=None)
        
        self.assertEqual(ds.bg_mask.shape, shape)
        
if __name__ == "__main__":
    unittest.main()
