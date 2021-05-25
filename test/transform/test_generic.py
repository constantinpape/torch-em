import numpy
import torch
import pytest

from torch_em.transform import Tile


@pytest.mark.parametrize("ndim,reps", [(1, (4, 2)), (2, (4, 2)), (3, (4, 2))])
def test_tile(ndim, reps):
    tile_aug = Tile(reps, match_shape_exactly=len(reps) == ndim)
    test_shape = [2, 3, 4][:ndim]
    data = numpy.random.random(test_shape)

    x = torch.tensor(data)

    expected = numpy.tile(x.numpy(), reps)
    if len(reps) == ndim:
        expected_torch = x.repeat(*reps)
        assert expected.shape == expected_torch.shape

    actual = tile_aug(x)

    assert actual.shape == expected.shape

    a = numpy.array(data)

    actual = tile_aug(a)
    assert actual.shape == expected.shape
