import unittest
import numpy as np


class TestSampler(unittest.TestCase):
    def test_min_foreground_sampler(self):
        from torch_em.data import MinForegroundSampler
        threshold = 0.5
        sampler = MinForegroundSampler(min_fraction=threshold)

        shape = (32, 32)
        n_samples = 100
        for _ in range(n_samples):
            data = np.random.rand(*shape)
            labels = (np.random.rand(*shape) > 0.5).astype("float32")
            accept = sampler(data, labels)
            expected = (labels.sum() / labels.size) > threshold
            self.assertEqual(accept, expected)

    def test_min_intensity_sampler(self):
        from torch_em.data import MinIntensitySampler
        threshold = 0.5
        sampler = MinIntensitySampler(min_intensity=threshold)
        shape = (32, 32)

        n_samples = 100
        for _ in range(n_samples):
            data = np.random.rand(*shape)
            accept = sampler(data)
            expected = np.median(data) > threshold
            self.assertEqual(accept, expected)


if __name__ == "__main__":
    unittest.main()
