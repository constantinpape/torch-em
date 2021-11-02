import os
import unittest

from torch_em.data import SegmentationDataset
from torch_em.util.test import create_segmentation_test_data


class TestDatasetWrapper(unittest.TestCase):
    path = "./data.h5"
    raw_key = "raw"
    label_key = "labels"
    shape = (32,) * 3

    def setUp(self):
        create_segmentation_test_data(self.path, self.raw_key, self.label_key, shape=self.shape, chunks=(32,) * 3)

    def tearDown(self):
        os.remove(self.path)

    def test_wrap_dataset(self):
        from torch_em.data import DatasetWrapper

        patch_shape = (32, 32, 32)
        ds = SegmentationDataset(self.path, self.raw_key, self.path, self.label_key, patch_shape=patch_shape)
        wrapped_ds = DatasetWrapper(ds, lambda xy: (xy[0][:, :10, :10, :10], xy[1][:, :20, :20, :20]))

        expected_shape_x = (1, 10, 10, 10)
        expected_shape_y = (1, 20, 20, 20)
        for i in range(3):
            x, y = wrapped_ds[i]
            self.assertEqual(x.shape, expected_shape_x)
            self.assertEqual(y.shape, expected_shape_y)


if __name__ == "__main__":
    unittest.main()
