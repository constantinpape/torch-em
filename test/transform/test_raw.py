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

    def test_random_percentile_normalization_uniform_sampling(self):
        from unittest.mock import patch

        from torch_em.transform.raw import RandomPercentileNormalization

        transform = RandomPercentileNormalization()
        with patch("torch_em.transform.raw.np.random.uniform", side_effect=(3.26, 96.74)) as sample:
            lower, upper = transform.sample_percentiles()

        self.assertEqual(sample.call_args_list[0].args, (0.0, 5.0))
        self.assertEqual(sample.call_args_list[1].args, (95.0, 100.0))
        self.assertEqual((lower, upper), (3.3, 96.7))

        asymmetric = RandomPercentileNormalization(upper_percentile_bounds=(80.0, 90.0))
        with patch("torch_em.transform.raw.np.random.uniform", side_effect=(3.26, 84.44)):
            self.assertEqual(asymmetric.sample_percentiles(), (3.3, 84.4))

        with patch("torch_em.transform.raw.np.random.uniform", side_effect=(-10.0, 110.0)):
            self.assertEqual(transform.sample_percentiles(), (0.0, 100.0))
        with patch("torch_em.transform.raw.np.random.uniform", side_effect=(70.0, 0.0)):
            self.assertEqual(transform.sample_percentiles(), (5.0, 95.0))

    def test_random_percentile_normalization_normal_sampling(self):
        from unittest.mock import patch

        from torch_em.transform.raw import RandomPercentileNormalization

        transform = RandomPercentileNormalization(
            upper_percentile_bounds=(90.0, 95.0),
            distribution="normal",
            distribution_kwargs={"mean": 2.0, "std": 1.0},
        )
        with patch("torch_em.transform.raw.np.random.normal", side_effect=(2.74, 7.26)) as sample:
            self.assertEqual(transform.sample_percentiles(), (2.7, 92.7))
        self.assertEqual([call.args for call in sample.call_args_list], [(2.0, 1.0), (2.0, 1.0)])

        deterministic = RandomPercentileNormalization(
            distribution="normal",
            distribution_kwargs={"mean": 4.0, "std": 0.0},
        )
        self.assertEqual(deterministic.sample_percentiles(), (4.0, 96.0))

    def test_random_percentile_normalization_configuration(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from torch_em.transform.raw import RandomPercentileNormalization

        transform1 = RandomPercentileNormalization(
            lower_percentile_bounds=(1.0, 10.0),
            rounding_decimals=2,
            seed=42,
        )
        transform2 = RandomPercentileNormalization(
            lower_percentile_bounds=(1.0, 10.0),
            rounding_decimals=2,
            seed=42,
        )
        samples1 = [transform1.sample_percentiles() for _ in range(3)]
        samples2 = [transform2.sample_percentiles() for _ in range(3)]
        self.assertEqual(samples1, samples2)

        worker_transform = RandomPercentileNormalization(seed=42)
        with patch("torch_em.transform.raw.torch.utils.data.get_worker_info", return_value=SimpleNamespace(id=1)):
            worker_sample = worker_transform.sample_percentiles()
        expected_generator = np.random.default_rng(np.random.SeedSequence([42, 1]))
        expected_lower = round(expected_generator.uniform(0.0, 5.0), 1)
        expected_upper = round(expected_generator.uniform(95.0, 100.0), 1)
        self.assertEqual(worker_sample, (expected_lower, expected_upper))

    def test_random_percentile_normalization_values(self):
        from torch_em.transform.raw import RandomPercentileNormalization, normalize_percentile

        input_ = np.arange(200, dtype="uint16").reshape(2, 10, 10) * 100
        transform = RandomPercentileNormalization(
            distribution="normal",
            distribution_kwargs={"mean": 4.0, "std": 0.0},
            axis=(1, 2),
        )

        output = transform(input_)
        expected = np.clip(normalize_percentile(input_, lower=4.0, upper=96.0, axis=(1, 2)), 0.0, 1.0)
        self.assertTrue(np.allclose(output, expected))
        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(output.min(), 0.0)
        self.assertLessEqual(output.max(), 1.0)

    def test_random_percentile_normalization_preprocessing_composition(self):
        from torch_em.transform.raw import RandomPercentileNormalization, RawTransform

        input_ = np.arange(16, dtype="uint16").reshape(4, 4) * 4000
        seen = {}

        def preprocessing(raw):
            seen["dtype"] = raw.dtype
            seen["maximum"] = raw.max()
            return np.stack([raw] * 3)

        normalizer = RandomPercentileNormalization(
            lower_percentile_bounds=(0.0, 0.0),
            axis=(1, 2),
        )
        transform = RawTransform(normalizer=normalizer, augmentation1=preprocessing)
        output = transform(input_)

        self.assertEqual(seen, {"dtype": np.dtype("uint16"), "maximum": np.uint16(60000)})
        self.assertEqual(output.shape, (3, 4, 4))
        self.assertTrue(np.isfinite(output).all())

    def test_random_percentile_normalization_invalid_arguments(self):
        from torch_em.transform.raw import RandomPercentileNormalization

        invalid_kwargs = [
            {"distribution": "beta"},
            {"distribution": "normal"},
            {"distribution": "normal", "distribution_kwargs": {"mean": 2.0}},
            {"distribution": "normal", "distribution_kwargs": {"mean": -1.0, "std": 1.0}},
            {"distribution": "normal", "distribution_kwargs": {"mean": 2.0, "std": -1.0}},
            {"distribution": "uniform", "distribution_kwargs": {"mean": 2.0, "std": 1.0}},
            {"lower_percentile_bounds": (0.0, 50.0)},
            {"upper_percentile_bounds": (50.0, 100.0)},
            {"upper_percentile_bounds": (90.0, 101.0)},
            {"rounding_decimals": -1},
            {"seed": -1},
            {"eps": 0.0},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RandomPercentileNormalization(**kwargs)

        with self.assertRaises(TypeError):
            RandomPercentileNormalization(seed=1.5)

        self.assertEqual(RandomPercentileNormalization(seed=np.int64(42)).seed, 42)


if __name__ == "__main__":
    unittest.main()
