import unittest
import itertools

import numpy as np

import torch

from torch_em.transform import Tile, generic


class TestTile(unittest.TestCase):
    def test_tile(self):
        for ndim, reps in [(1, (4, 2)), (2, (4, 2)), (3, (4, 2))]:
            with self.subTest():
                self._test_tile_impl(ndim, reps)

    @staticmethod
    def _test_tile_impl(ndim, reps):
        tile_aug = Tile(reps, match_shape_exactly=len(reps) == ndim)
        test_shape = [2, 3, 4][:ndim]
        data = np.random.random(test_shape)

        x = torch.tensor(data)

        expected = np.tile(x.numpy(), reps)
        if len(reps) == ndim:
            expected_torch = x.repeat(*reps)
            assert expected.shape == expected_torch.shape

        actual = tile_aug(x)

        assert actual.shape == expected.shape

        a = np.array(data)

        actual = tile_aug(a)
        assert actual.shape == expected.shape

    def test_resize_longest_inputs(self):
        input_shapes = [(520, 704), (256, 384), (1040, 1200)]
        target_shapes = [(256, 256), (512, 512), (1024, 1024)]

        for (input_shape, target_shape) in itertools.product(input_shapes, target_shapes):
            test_image = np.zeros(input_shape, dtype=np.float32)

            raw_transform = generic.ResizeLongestSideInputs(target_shape=target_shape)
            resized_image = raw_transform(inputs=test_image)

            assert resized_image.shape == target_shape
            assert resized_image.dtype == test_image.dtype


if __name__ == "__main__":
    unittest.main()
