import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
from torch_em.util.test import create_segmentation_test_data


class TestSegmentationDataset(unittest.TestCase):
    tmp_folder = "./tmp"
    path = "./tmp/data.h5"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)

    def create_default_data(self, raw_key, label_key):
        shape = (128,) * 3
        chunks = (32,) * 3
        create_segmentation_test_data(self.path, raw_key, label_key, shape=shape, chunks=chunks)
        return shape, chunks

    def test_dataset_3d(self):
        from torch_em.data import SegmentationDataset
        raw_key, label_key = "raw", "labels"
        shape, chunks = self.create_default_data(raw_key, label_key)

        patch_shape = chunks
        ds = SegmentationDataset(self.path, raw_key, self.path, label_key, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds.labels.shape, shape)
        self.assertEqual(ds._ndim, 3)

        expected_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)

    def test_dataset_2d(self):
        from torch_em.data import SegmentationDataset
        raw_key, label_key = "raw", "labels"
        shape, _ = self.create_default_data(raw_key, label_key)

        patch_shape = (1, 32, 32)
        ds = SegmentationDataset(self.path, raw_key, self.path, label_key, patch_shape=patch_shape, ndim=2)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds.labels.shape, shape)
        self.assertEqual(ds._ndim, 2)

        expected_shape = patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)

    def test_dataset_4d(self):
        from torch_em.data import SegmentationDataset
        raw_key, label_key = "raw", "labels"
        shape = (4, 64, 64, 64)
        chunks = (1, 32, 32, 32)
        create_segmentation_test_data(self.path, raw_key, label_key, shape=shape, chunks=chunks)

        patch_shape = (2, 32, 32, 32)
        ds = SegmentationDataset(self.path, raw_key, self.path, label_key, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds.labels.shape, shape)
        self.assertEqual(ds._ndim, 4)

        expected_shape = patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)

    def test_dataset_3d4d(self):
        from torch_em.data import SegmentationDataset
        raw_key, label_key = "raw", "labels"
        shape = (4, 128, 128, 128)
        chunks = (1, 32, 32, 32)
        create_segmentation_test_data(self.path, raw_key, label_key, shape=shape, chunks=chunks)

        patch_shape = (1, 32, 32, 32)
        ds = SegmentationDataset(self.path, raw_key, self.path, label_key, ndim=3, patch_shape=patch_shape)
        self.assertEqual(ds.raw.shape, shape)
        self.assertEqual(ds.labels.shape, shape)
        self.assertEqual(ds._ndim, 3)

        expected_shape = patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(y.shape, expected_shape)

    def test_roi(self):
        from torch_em.data import SegmentationDataset
        raw_key, label_key = "raw", "labels"
        self.create_default_data(raw_key, label_key)

        patch_shape = (32, 32, 32)
        roi = np.s_[32:96, 32:96, 32:96]
        ds = SegmentationDataset(self.path, raw_key, self.path, label_key, patch_shape=patch_shape, roi=roi)

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

        raw_key, label_key = "raw", "labels"
        shape = (3, 128, 128, 128)
        chunks = (1, 32, 32, 32)
        with h5py.File(self.path, "a") as f:
            data_with_channels = np.random.rand(*shape)
            f.create_dataset(raw_key, data=data_with_channels, chunks=chunks)
            f.create_dataset(label_key, data=np.random.randint(0, 4, size=shape[1:]), chunks=chunks[1:])

        patch_shape = (32, 32, 32)
        ds = SegmentationDataset(
            self.path, raw_key, self.path, label_key, patch_shape=patch_shape, with_channels=True
        )
        self.assertEqual(ds._ndim, 3)
        expected_raw_shape = (3,) + patch_shape
        expected_label_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_raw_shape)
            self.assertEqual(y.shape, expected_label_shape)

    def test_with_label_channels(self):
        from torch_em.data import SegmentationDataset

        raw_key, label_key = "raw", "labels"
        shape = (3, 128, 128, 128)
        chunks = (1, 32, 32, 32)
        with h5py.File(self.path, "a") as f:
            f.create_dataset(label_key, data=np.random.rand(*shape), chunks=chunks)
            f.create_dataset(raw_key, data=np.random.rand(*shape[1:]), chunks=chunks[1:])

        patch_shape = (32, 32, 32)
        ds = SegmentationDataset(
            self.path, raw_key, self.path, label_key, patch_shape=patch_shape, with_label_channels=True
        )
        self.assertEqual(ds._ndim, 3)
        expected_raw_shape = (1,) + patch_shape
        expected_label_shape = (3,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_raw_shape)
            self.assertEqual(y.shape, expected_label_shape)

    def test_with_raw_and_label_channels(self):
        from torch_em.data import SegmentationDataset

        raw_key, label_key = "raw", "labels"
        raw_shape = (3, 128, 128, 128)
        label_shape = (2, 128, 128, 128)
        chunks = (1, 32, 32, 32)
        with h5py.File(self.path, "a") as f:
            f.create_dataset(raw_key, data=np.random.rand(*raw_shape), chunks=chunks)
            f.create_dataset(label_key, data=np.random.rand(*label_shape), chunks=chunks)

        patch_shape = (32, 32, 32)
        ds = SegmentationDataset(
            self.path, raw_key, self.path, label_key,
            patch_shape=patch_shape, with_channels=True, with_label_channels=True
        )
        self.assertEqual(ds._ndim, 3)
        expected_raw_shape = (3,) + patch_shape
        expected_label_shape = (2,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_raw_shape)
            self.assertEqual(y.shape, expected_label_shape)

    def test_tif(self):
        import imageio.v3 as imageio
        from torch_em.data import SegmentationDataset

        raw_path = os.path.join(self.tmp_folder, "raw.tif")
        label_path = os.path.join(self.tmp_folder, "labels.tif")
        shape = (128, 128, 128)
        imageio.imwrite(raw_path, np.random.rand(*shape).astype("float32"))
        imageio.imwrite(label_path, np.random.rand(*shape).astype("float32"))

        patch_shape = (32, 32, 32)
        raw_key, label_key = None, None
        ds = SegmentationDataset(
            raw_path, raw_key, label_path, label_key, patch_shape=patch_shape
        )

        expected_patch_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_patch_shape)
            self.assertEqual(y.shape, expected_patch_shape)


if __name__ == "__main__":
    unittest.main()
