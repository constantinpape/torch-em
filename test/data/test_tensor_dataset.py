import unittest

import numpy as np


class TestTensorDataset(unittest.TestCase):
    n_images = 10

    def _get_data(self, ndim=2, n_channels=None):
        images, labels = [], []
        if ndim == 2:
            shape_start, shape_stop = 128, 512
        else:
            shape_start, shape_stop = 32, 96
        for _ in range(self.n_images):
            shape = tuple(np.random.randint(shape_start, shape_stop) for _ in range(ndim))
            im_shape = shape if n_channels is None else (n_channels,) + shape
            images.append(np.random.rand(*im_shape))
            labels.append(np.random.rand(*shape))
        return images, labels

    def _check_dataset(self, ds, patch_shape, n_channels=None):
        ndim = len(patch_shape)
        self.assertEqual(len(ds), self.n_images)
        self.assertEqual(ds._ndim, ndim)

        expected_im_shape = (1 if n_channels is None else n_channels,) + patch_shape
        expected_label_shape = (1,) + patch_shape
        for i in range(10):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_im_shape)
            self.assertEqual(y.shape, expected_label_shape)

    def test_tensor_dataset(self):
        from torch_em.data import TensorDataset

        patch_shape = (256, 256)
        images, labels = self._get_data()
        ds = TensorDataset(images, labels, patch_shape=patch_shape)
        self._check_dataset(ds, patch_shape)

    def test_tensor_dataset_with_channels(self):
        from torch_em.data import TensorDataset

        patch_shape = (256, 256)
        images, labels = self._get_data(n_channels=3)
        ds = TensorDataset(images, labels, patch_shape=patch_shape, with_channels=True)
        self._check_dataset(ds, patch_shape, n_channels=3)

    def test_tensor_dataset_3d(self):
        from torch_em.data import TensorDataset

        patch_shape = (32, 64, 64)
        images, labels = self._get_data(ndim=3)
        ds = TensorDataset(images, labels, patch_shape=patch_shape)
        self._check_dataset(ds, patch_shape)

    def test_tensor_dataset_3d_with_channels(self):
        from torch_em.data import TensorDataset

        patch_shape = (32, 64, 64)
        images, labels = self._get_data(ndim=3, n_channels=3)
        ds = TensorDataset(images, labels, patch_shape=patch_shape, with_channels=True)
        self._check_dataset(ds, patch_shape, n_channels=3)


if __name__ == "__main__":
    unittest.main()
