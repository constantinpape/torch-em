import unittest

import numpy as np


class TestPrediction(unittest.TestCase):
    def test_predict_with_halo_2d(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_predict_with_halo_3d(self):
        from torch_em.model import UNet3d
        from torch_em.util.prediction import predict_with_halo
        model = UNet3d(in_channels=1, out_channels=3, initial_features=8, depth=3)

        shape = (128,) * 3
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(32, 32, 32), halo=(8, 8, 8))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_prediction_with_halo_multiple_outputs(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        outputs = [
            (np.zeros(shape, dtype="float32"), np.s_[0]),
            (np.zeros((2,) + shape, dtype="float32"), np.s_[1:3])
        ]
        predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16), output=outputs)

        self.assertEqual(outputs[0][0].shape, shape)
        self.assertFalse(np.allclose(outputs[0][0], 0))

        self.assertEqual(outputs[1][0].shape, (2,) + shape)
        self.assertFalse(np.allclose(outputs[1][0], 0))

    def test_predict_with_halo_channels(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo

        model = UNet2d(in_channels=2, out_channels=3, initial_features=8, depth=3)
        shape = (2, 512, 512)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(64, 64), halo=(8, 8), with_channels=True)
        expected_shape = (3,) + shape[1:]
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))


if __name__ == "__main__":
    unittest.main()
