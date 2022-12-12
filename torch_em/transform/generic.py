from typing import Any, Dict, Optional, Sequence, Union

import numpy
import torch

from skimage.transform import rescale


class Tile(torch.nn.Module):
    _params = None

    def __init__(self, reps: Sequence[int] = (2,), match_shape_exactly: bool = True):
        super().__init__()
        self.reps = reps
        self.match_shape_exactly = match_shape_exactly

    def forward(self, input: Union[torch.Tensor, numpy.ndarray], params: Optional[Dict[str, Any]] = None):
        assert not self.match_shape_exactly or len(input.shape) == len(self.reps), (input.shape, self.reps)
        if isinstance(input, torch.Tensor):
            # return torch.tile(input, self.reps)  # todo: use torch.tile (for pytorch >=1.8?)
            reps = list(self.reps)
            for _ in range(max(0, len(input.shape) - len(reps))):
                reps.insert(0, 1)

            for _ in range(max(0, len(reps) - len(input.shape))):
                input = input.unsqueeze(0)

            return input.repeat(*reps)
        elif isinstance(input, numpy.ndarray):
            return numpy.tile(input, self.reps)
        else:
            raise NotImplementedError(type(input))


# a simple way to compose transforms
class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        outputs = self.transforms[0](*inputs)
        for trafo in self.transforms[1:]:
            outputs = trafo(*outputs)
        return outputs


class Rescale:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, *inputs):
        outputs = tuple(rescale(inp, scale=self.scale, preserve_range=True) for inp in inputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
