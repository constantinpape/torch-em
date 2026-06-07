import unittest

import torch


class TestMemory(unittest.TestCase):
    #
    # GPU-free tests for the core search / classification / resolution logic.
    #

    def test_search_max_int_basic(self):
        from torch_em.util.memory import _search_max_int

        # Exercises bracketing past a power of two and binary-search convergence.
        self.assertEqual(_search_max_int(lambda n: n <= 37, upper_bound=1024), 37)
        self.assertEqual(_search_max_int(lambda n: n <= 1, upper_bound=1024), 1)
        self.assertEqual(_search_max_int(lambda n: n <= 1000, upper_bound=1024), 1000)

    def test_search_max_int_at_powers_of_two(self):
        from torch_em.util.memory import _search_max_int

        for budget in (2, 16, 64, 512):
            self.assertEqual(_search_max_int(lambda n, b=budget: n <= b, upper_bound=1024), budget)

    def test_search_max_int_hits_upper_bound(self):
        from torch_em.util.memory import _search_max_int

        # Everything fits -> the upper bound is returned exactly (cap path).
        self.assertEqual(_search_max_int(lambda n: True, upper_bound=8), 8)
        self.assertEqual(_search_max_int(lambda n: True, upper_bound=1), 1)

    def test_search_max_int_raises_when_one_fails(self):
        from torch_em.util.memory import _search_max_int

        with self.assertRaises(RuntimeError):
            _search_max_int(lambda n: False, upper_bound=1024)

    def test_search_max_int_matches_bruteforce(self):
        from torch_em.util.memory import _search_max_int

        # Cross-check the search against a brute-force maximum for many thresholds.
        upper_bound = 200
        for threshold in range(1, upper_bound + 5):
            expected = min(threshold, upper_bound)
            self.assertEqual(_search_max_int(lambda n, t=threshold: n <= t, upper_bound), expected)

    def test_is_oom_error(self):
        from torch_em.util.memory import _is_oom_error

        self.assertTrue(_is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertTrue(_is_oom_error(RuntimeError("CUDA OUT OF MEMORY")))
        self.assertFalse(_is_oom_error(RuntimeError("some other runtime error")))
        self.assertFalse(_is_oom_error(ValueError("invalid shape")))

    def test_resolve_in_channels(self):
        from torch_em.util.memory import _resolve_in_channels

        class _Dummy:
            in_channels = 4

        self.assertEqual(_resolve_in_channels(_Dummy(), None), 4)
        self.assertEqual(_resolve_in_channels(_Dummy(), 7), 7)  # Explicit override wins.

        class _NoChannels:
            pass

        with self.assertRaises(ValueError):
            _resolve_in_channels(_NoChannels(), None)

    def test_resolve_min_divisible(self):
        from torch_em.model import UNet2d, UNet3d
        from torch_em.util.memory import _resolve_min_divisible

        model_2d = UNet2d(in_channels=1, out_channels=1, depth=4, initial_features=4)
        self.assertEqual(_resolve_min_divisible(model_2d, 2, None), (16, 16))

        model_3d = UNet3d(in_channels=1, out_channels=1, depth=3, initial_features=4)
        self.assertEqual(_resolve_min_divisible(model_3d, 3, None), (8, 8, 8))

        # Explicit (e.g. anisotropic) value is passed through.
        self.assertEqual(_resolve_min_divisible(model_3d, 3, (1, 16, 16)), (1, 16, 16))

        with self.assertRaises(ValueError):
            _resolve_min_divisible(model_3d, 3, (16, 16))  # Length mismatch.

    def test_cpu_device_raises(self):
        from torch_em.model import UNet2d
        from torch_em.util import compute_max_batch_size, compute_max_patch_shape

        model = UNet2d(in_channels=1, out_channels=1, depth=3, initial_features=4)
        with self.assertRaises(RuntimeError):
            compute_max_batch_size(model, patch_shape=(64, 64), device="cpu")
        with self.assertRaises(RuntimeError):
            compute_max_patch_shape(model, ndim=2, device="cpu")

    #
    # CUDA-guarded integration tests (skipped on CPU-only machines / CI).
    #

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA device.")
    def test_compute_max_batch_size_2d(self):
        from torch_em.model import UNet2d
        from torch_em.util import compute_max_batch_size
        from torch_em.util.memory import _attempt_forward

        model = UNet2d(in_channels=1, out_channels=1, depth=3, initial_features=8).cuda()
        patch_shape = (64, 64)
        batch_size = compute_max_batch_size(model, patch_shape=patch_shape, max_batch_size=8)
        self.assertIsInstance(batch_size, int)
        self.assertGreaterEqual(batch_size, 1)
        # The returned batch size must actually fit.
        device = next(model.parameters()).device
        self.assertTrue(
            _attempt_forward(model, device, torch.float32, 1, batch_size, patch_shape, None)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA device.")
    def test_compute_max_patch_shape_2d(self):
        from torch_em.model import UNet2d
        from torch_em.util import compute_max_patch_shape

        model = UNet2d(in_channels=1, out_channels=1, depth=3, initial_features=8).cuda()
        patch_shape = compute_max_patch_shape(model, ndim=2, max_scale_factor=4)
        self.assertEqual(len(patch_shape), 2)
        for axis in patch_shape:
            self.assertGreaterEqual(axis, 8)  # min_divisible == 2 ** depth == 8.
            self.assertEqual(axis % 8, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA device.")
    def test_compute_max_batch_size_3d(self):
        from torch_em.model import UNet3d
        from torch_em.util import compute_max_batch_size

        model = UNet3d(in_channels=1, out_channels=1, depth=3, initial_features=8).cuda()
        batch_size = compute_max_batch_size(model, patch_shape=(32, 32, 32), max_batch_size=4)
        self.assertGreaterEqual(batch_size, 1)


if __name__ == "__main__":
    unittest.main()
