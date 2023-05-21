import unittest

import numpy as np
import torch

from torch_em.transform import Tile


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


if __name__ == "__main__":
    unittest.main()
