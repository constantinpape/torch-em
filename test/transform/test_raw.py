import unittest
from copy import deepcopy

import numpy as np
import torch


class TestRaw(unittest.TestCase):
    def _test_standardize(self, input_):
        from torch_em.transform.raw import standardize

        def check_out(out):
            self.assertEqual(out.shape, input_.shape)
            if torch.is_tensor(out):
                mean, std = out.mean().numpy(), out.std().numpy()
            else:
                mean, std = out.mean(), out.std()
            self.assertLess(mean, 0.001)
            self.assertLess(np.abs(1.0 - std), 0.001)

        # test standardize without arguments
        out = standardize(deepcopy(input_))
        check_out(out)

        # test standardize with axis
        out = standardize(deepcopy(input_), axis=(1, 2))
        check_out(out)

        # test standardize with fixed mean and std
        mean, std = input_.mean(), input_.std()
        out = standardize(deepcopy(input_), mean=mean, std=std)
        check_out(out)

    def test_standardize_torch(self):
        input_ = torch.rand(3, 128, 128)
        self._test_standardize(input_)

    def test_standardize_numpy(self):
        input_ = np.random.rand(3, 128, 128)
        self._test_standardize(input_)

    def _test_normalize(self, input_):
        from torch_em.transform.raw import normalize

        def check_out(out):
            self.assertEqual(out.shape, input_.shape)
            if torch.is_tensor(out):
                min_, max_ = out.min().numpy(), out.max().numpy()
            else:
                min_, max_ = out.min(), out.max()
            self.assertLess(min_, 0.001)
            self.assertLess(np.abs(1.0 - max_), 0.001)

        # test normalize without arguments
        out = normalize(deepcopy(input_))
        check_out(out)

        # test normalize with axis
        out = normalize(deepcopy(input_), axis=(1, 2))
        check_out(out)

        # test normalize with fixed min, max
        min_, max_ = input_.min(), input_.max() - input_.min()
        out = normalize(deepcopy(input_), minval=min_, maxval=max_)
        check_out(out)

    def test_normalize_torch(self):
        input_ = torch.randn(3, 128, 128)
        self._test_normalize(input_)

    def test_normalize_numpy(self):
        input_ = np.random.randn(3, 128, 128)
        self._test_normalize(input_)

    def _test_normalize_percentile(self, input_):
        from torch_em.transform.raw import normalize_percentile

        def check_out(out):
            self.assertEqual(out.shape, input_.shape)

        # test normalize without arguments
        out = normalize_percentile(deepcopy(input_))
        check_out(out)

        # test normalize with axis
        out = normalize_percentile(deepcopy(input_), axis=(1, 2))
        check_out(out)

        # test normalize with percentile arguments
        out = normalize_percentile(deepcopy(input_), lower=5.0, upper=95.0)
        check_out(out)

    def test_normalize_percentile_torch(self):
        input_ = torch.randn(3, 128, 128)
        self._test_normalize_percentile(input_)

    def test_normalize_percentile_numpy(self):
        input_ = np.random.randn(3, 128, 128)
        self._test_normalize_percentile(input_)


if __name__ == "__main__":
    unittest.main()
