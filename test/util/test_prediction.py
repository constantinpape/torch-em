import unittest

import numpy as np


class TestPrediction(unittest.TestCase):
    def test_predict_with_halo_2d(self):
        from torch_em.model import UNet2d
        from torch_em.util import predict_with_halo

        model = UNet2d(in_channels=1, out_channels=3,
                       initial_features=8, depth=3)

        shape = (1024, 1024)
        data = np.random.rand(*shape).astype('float32')

        out = predict_with_halo(data, model, gpu_ids=['cpu'],
                                block_shape=(256, 256), halo=(16, 16))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)

    def test_predict_with_halo_3d(self):
        from torch_em.model import UNet3d
        from torch_em.util import predict_with_halo

        model = UNet3d(in_channels=1, out_channels=3,
                       initial_features=8, depth=3)

        shape = (128,) * 3
        data = np.random.rand(*shape).astype('float32')

        out = predict_with_halo(data, model, gpu_ids=['cpu'],
                                block_shape=(32, 32, 32), halo=(8, 8, 8))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
