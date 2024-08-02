from math import ceil, floor
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from skimage.transform import rescale, resize

import torch


class Tile(torch.nn.Module):
    _params = None

    def __init__(self, reps: Sequence[int] = (2,), match_shape_exactly: bool = True):
        super().__init__()
        self.reps = reps
        self.match_shape_exactly = match_shape_exactly

    def forward(self, input: Union[torch.Tensor, np.ndarray], params: Optional[Dict[str, Any]] = None):
        assert not self.match_shape_exactly or len(input.shape) == len(self.reps), (input.shape, self.reps)
        if isinstance(input, torch.Tensor):
            # return torch.tile(input, self.reps)  # todo: use torch.tile (for pytorch >=1.8?)
            reps = list(self.reps)
            for _ in range(max(0, len(input.shape) - len(reps))):
                reps.insert(0, 1)

            for _ in range(max(0, len(reps) - len(input.shape))):
                input = input.unsqueeze(0)

            return input.repeat(*reps)
        elif isinstance(input, np.ndarray):
            return np.tile(input, self.reps)
        else:
            raise NotImplementedError(type(input))


# a simple way to compose transforms
class Compose:
    def __init__(self, *transforms, is_multi_tensor=True):
        self.transforms = transforms
        self.is_multi_tensor = is_multi_tensor

    def __call__(self, *inputs):
        outputs = self.transforms[0](*inputs)
        for trafo in self.transforms[1:]:
            if self.is_multi_tensor:
                outputs = trafo(*outputs)
            else:
                outputs = trafo(outputs)

        return outputs


class Rescale:
    def __init__(self, scale, with_channels=None):
        self.scale = scale
        self.with_channels = with_channels

    def _rescale_with_channels(self, input_, **kwargs):
        out = [rescale(inp, **kwargs)[None] for inp in input_]
        return np.concatenate(out, axis=0)

    def __call__(self, *inputs):
        if self.with_channels is None:
            outputs = tuple(rescale(inp, scale=self.scale, preserve_range=True) for inp in inputs)
        else:
            if isinstance(self.with_channels, (tuple, list)):
                assert len(self.with_channels) == len(inputs)
                with_channels = self.with_channels
            else:
                with_channels = [self.with_channels] * len(inputs)
            outputs = tuple(
                self._rescale_with_channels(inp, scale=self.scale, preserve_range=True) if wc else
                rescale(inp, scale=self.scale, preserve_range=True)
                for inp, wc in zip(inputs, with_channels)
            )
        if len(outputs) == 1:
            return outputs[0]
        return outputs


class ResizeInputs:
    def __init__(self, target_shape, is_label=False, is_rgb=False):
        self.target_shape = target_shape
        self.is_label = is_label
        self.is_rgb = is_rgb

    def __call__(self, inputs):
        if self.is_label:  # kwargs needed for int data
            kwargs = {"order": 0,  "anti_aliasing": False}
        else:  # we use the default settings for float data
            kwargs = {}

        if self.is_rgb:
            assert inputs.ndim == 3 and inputs.shape[0] == 3
            patch_shape = (3, *self.target_shape)
        else:
            patch_shape = self.target_shape

        inputs = resize(
            image=inputs,
            output_shape=patch_shape,
            preserve_range=True,
            **kwargs
        ).astype(inputs.dtype)

        return inputs


class ResizeLongestSideInputs:
    def __init__(self, target_shape, is_label=False, is_rgb=False):
        self.target_shape = target_shape
        self.is_label = is_label
        self.is_rgb = is_rgb

        h, w = self.target_shape[-2], self.target_shape[-1]
        if h != w:  # We currently support resize feature for square-shaped target shape only.
            raise ValueError("'ResizeLongestSideInputs' does not support non-square shaped target shapes.")

        self.target_length = self.target_shape[-1]

    def _get_preprocess_shape(self, oldh, oldw):
        """Inspired from Segment Anything.

        - https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/transforms.py
        """
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __call__(self, inputs):
        if self.is_label:  # kwargs needed for int data
            kwargs = {"order": 0,  "anti_aliasing": False}
        else:  # we use the default settings for float data
            kwargs = {}

        # Let's get the new shape with the longest side equal to the target length.
        new_shape = self._get_preprocess_shape(inputs.shape[-2], inputs.shape[-1])

        if self.is_rgb:  # for rgb inputs, we assume channels first
            assert inputs.ndim == 3 and inputs.shape[0] == 3
            patch_shape = (3, *new_shape)
        elif inputs.ndim == 3:  # for 3d inputs, we assume channels first
            patch_shape = (inputs.shape[0], *patch_shape)
        else:
            patch_shape = new_shape

        # Next, we resize the input image along the longest side.
        inputs = resize(
            image=inputs, output_shape=patch_shape, preserve_range=True, **kwargs
        ).astype(inputs.dtype)

        # Finally, we pad the remaining height to match the expected target shape.
        pad_width = [(sh - dsh) / 2 for sh, dsh in zip(self.target_shape, new_shape)]
        pad_width = (
            (ceil(pad_width[0]), floor(pad_width[0])), (ceil(pad_width[1]), floor(pad_width[1]))
        )
        if self.is_rgb or inputs.ndim == 3:  # in either cases, we assume channels first.
            pad_width = ((0, 0), *pad_width)

        inputs = np.pad(array=inputs, pad_width=pad_width, mode="constant")
        return inputs


class PadIfNecessary:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def _pad_if_necessary(self, data):
        if data.ndim == len(self.shape):
            pad_shape = self.shape
        else:
            dim_diff = data.ndim - len(self.shape)
            pad_shape = data.shape[:dim_diff] + self.shape
            assert len(pad_shape) == data.ndim

        data_shape = data.shape
        if all(dsh == sh for dsh, sh in zip(data_shape, pad_shape)):
            return data

        pad_width = [sh - dsh for dsh, sh in zip(data_shape, pad_shape)]
        assert all(pw >= 0 for pw in pad_width)
        pad_width = [(0, pw) for pw in pad_width]
        return np.pad(data, pad_width, mode="reflect")

    def __call__(self, *inputs):
        outputs = tuple(self._pad_if_necessary(input_) for input_ in inputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
