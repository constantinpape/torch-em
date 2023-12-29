import torch
import numpy as np
import kornia
from skimage.transform import resize

from ..util import ensure_tensor


class RandomElasticDeformationStacked(kornia.augmentation.AugmentationBase3D):
    def __init__(self,
                 control_point_spacing=1,
                 sigma=(32.0, 32.0),
                 alpha=(4.0, 4.0),
                 interpolation=kornia.constants.Resample.BILINEAR,
                 p=0.5,
                 keepdim=False,
                 same_on_batch=True):
        super().__init__(p=p,  # keepdim=keepdim,
                         same_on_batch=same_on_batch)
        if isinstance(control_point_spacing, int):
            self.control_point_spacing = [control_point_spacing] * 2
        else:
            self.control_point_spacing = control_point_spacing
        assert len(self.control_point_spacing) == 2
        self.interpolation = interpolation
        self.flags = dict(
            interpolation=torch.tensor(self.interpolation.value),
            sigma=sigma,
            alpha=alpha
        )

    # The same transformation applied to all samples in a batch
    def generate_parameters(self, batch_shape):
        assert len(batch_shape) == 5
        shape = batch_shape[3:]
        control_shape = tuple(
            sh // spacing for sh, spacing in zip(shape, self.control_point_spacing)
        )
        deformation_fields = [
            np.random.uniform(-1, 1, control_shape),
            np.random.uniform(-1, 1, control_shape)
        ]
        deformation_fields = [
            resize(df, shape, order=3)[None] for df in deformation_fields
        ]
        noise = np.concatenate(deformation_fields, axis=0)[None].astype("float32")
        noise = torch.from_numpy(noise)
        return {"noise": noise}

    def __call__(self, input, params=None):
        assert len(input.shape) == 5
        if params is None:
            params = self.generate_parameters(input.shape)
            self._params = params

        noise = params["noise"]
        mode = "bilinear" if (self.flags["interpolation"] == 1).all() else "nearest"
        noise_ch = noise.expand(input.shape[1], -1, -1, -1)
        input_transformed = []
        for i, x in enumerate(torch.unbind(input, dim=0)):
            x_transformed = kornia.geometry.transform.elastic_transform2d(
                            x, noise_ch, sigma=self.flags["sigma"],
                            alpha=self.flags["alpha"], mode=mode,
                            padding_mode="reflection"
                            )
            input_transformed.append(x_transformed)
        input_transformed = torch.stack(input_transformed)
        return input_transformed


class RandomElasticDeformation(kornia.augmentation.AugmentationBase2D):
    def __init__(self,
                 control_point_spacing=1,
                 sigma=(4.0, 4.0),
                 alpha=(32.0, 32.0),
                 interpolation=kornia.constants.Resample.BILINEAR,
                 p=0.5,
                 keepdim=False,
                 same_on_batch=False):
        super().__init__(p=p,  # keepdim=keepdim,
                         same_on_batch=same_on_batch)
        if isinstance(control_point_spacing, int):
            self.control_point_spacing = [control_point_spacing] * 2
        else:
            self.control_point_spacing = control_point_spacing
        assert len(self.control_point_spacing) == 2
        self.interpolation = interpolation
        self.flags = dict(
            interpolation=torch.tensor(self.interpolation.value),
            sigma=sigma,
            alpha=alpha
        )

    # TODO do we need special treatment for batches, channels > 1?
    def generate_parameters(self, batch_shape):
        assert len(batch_shape) == 4, f"{len(batch_shape)}"
        shape = batch_shape[2:]
        control_shape = tuple(
            sh // spacing for sh, spacing in zip(shape, self.control_point_spacing)
        )
        deformation_fields = [
            np.random.uniform(-1, 1, control_shape),
            np.random.uniform(-1, 1, control_shape)
        ]
        deformation_fields = [
            resize(df, shape, order=3)[None] for df in deformation_fields
        ]
        noise = np.concatenate(deformation_fields, axis=0)[None].astype("float32")
        noise = torch.from_numpy(noise)
        return {"noise": noise}

    def __call__(self, input, params=None):
        if params is None:
            params = self.generate_parameters(input.shape)
            self._params = params
        noise = params["noise"]
        mode = "bilinear" if (self.flags["interpolation"] == 1).all() else "nearest"
        return kornia.geometry.transform.elastic_transform2d(
            input, noise, sigma=self.flags["sigma"], alpha=self.flags["alpha"], mode=mode,
            padding_mode="reflection"
        )


class RandomResizeAndPad(kornia.augmentation.AugmentationBase2D):
    """Bring inputs to output shape by randomly resizing and padding.
    """
    def __init__(self, output_shape, padding_mode="constant", same_on_batch=False):
        super().__init__(
            p=1.0, same_on_batch=same_on_batch
        )
        if len(output_shape) != 2:
            raise ValueError(f"Can only resize 2d shape, got {len(output_shape)}")
        self.output_shape = output_shape
        self.padding_mode = padding_mode
        self.flags = dict(interpolation=torch.tensor(kornia.constants.Resample.BILINEAR.value))

    def generate_parameters(self, shape):
        assert len(shape) == len(self.output_shape)
        resize_shape = []
        for ims, outs in zip(shape, self.output_shape):
            diff = outs - ims

            # The output shape is bigger than the input shape.
            # We resize to a random size in between the two (the rest will get padded).
            if diff > 0:
                res = np.random.randint(ims, outs + 1)

            # The output shape is smaller than or equal to the input shape.
            # We just resize to the output shape.
            else:
                res = outs

            resize_shape.append(res)
        return {"resize_shape": tuple(resize_shape)}

    def resize_and_pad(self, data, resize_shape):
        # shapes match already, we don't have to do anything
        if tuple(data[-2:].shape) == self.output_shape:
            return data

        # interpolation mode
        mode = "bilinear" if (self.flags["interpolation"] == 1).all() else "nearest"
        antialias = True if (self.flags["interpolation"] == 1).all() else False

        print("Resizing", data.shape, "to", resize_shape, "with", mode)
        out = kornia.geometry.transform.resize(
            data, resize_shape, interpolation=mode, antialias=antialias
        )

        # pad the rest
        if tuple(out.shape[-2:]) != self.output_shape:
            pad_shape = tuple(
                outsh - sh for outsh, sh in zip(self.output_shape, out.shape[-2:])
            )
            pad_shape = (pad_shape[1], 0, pad_shape[0], 0)
            assert all(ps >= 0 for ps in pad_shape), f"{pad_shape}"
            out = torch.nn.functional.pad(out, pad_shape, mode=self.padding_mode)

        assert tuple(out.shape[-2:]) == self.output_shape, f"{out.shape}, {self.output_shape}"
        return out

    def __call__(self, input, params=None):
        if params is None:
            params = self.generate_parameters(input.shape[-2:])
            self._params = params
        return self.resize_and_pad(input, params["resize_shape"])


# TODO implement 'require_halo', and estimate the halo from the transformations
# so that we can load a bigger block and cut it away
class KorniaAugmentationPipeline(torch.nn.Module):
    interpolatable_torch_types = [torch.float16, torch.float32, torch.float64]
    interpolatable_numpy_types = [np.dtype("float32"), np.dtype("float64")]

    def __init__(self, *kornia_augmentations, dtype=torch.float32):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(kornia_augmentations)
        self.dtype = dtype
        self.halo = self.compute_halo()

    # for now we only add a halo for the random rotation trafos and
    # also don't compute the halo dynamically based on the input shape
    def compute_halo(self):
        halo = None
        for aug in self.augmentations:
            if isinstance(aug, kornia.augmentation.RandomRotation):
                halo = [32, 32]
            if isinstance(aug, kornia.augmentation.RandomRotation3D):
                halo = [32, 32, 32]
        return halo

    def is_interpolatable(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.dtype in self.interpolatable_torch_types
        else:
            return tensor.dtype in self.interpolatable_numpy_types

    def transform_tensor(self, augmentation, tensor, interpolatable, params=None):
        interpolating = "interpolation" in getattr(augmentation, "flags", [])
        if interpolating:
            resampler = kornia.constants.Resample.get("BILINEAR" if interpolatable else "NEAREST")
            augmentation.flags["interpolation"] = torch.tensor(resampler.value)
        transformed = augmentation(tensor, params)
        return transformed, augmentation._params

    def forward(self, *tensors):
        interpolatable = [self.is_interpolatable(tensor) for tensor in tensors]
        tensors = [ensure_tensor(tensor, self.dtype) for tensor in tensors]
        for aug in self.augmentations:

            t0, params = self.transform_tensor(aug, tensors[0], interpolatable[0])
            transformed_tensors = [t0]
            for tensor, interpolate in zip(tensors[1:], interpolatable[1:]):
                tensor, _ = self.transform_tensor(aug, tensor, interpolate, params=params)
                transformed_tensors.append(tensor)

            tensors = transformed_tensors
        return tensors

    def halo(self, shape):
        return self.halo


# TODO elastic deformation
# Try out:
# - RandomPerspective
AUGMENTATIONS = {
    "RandomAffine": {"degrees": 90, "scale": (0.9, 1.1)},
    "RandomAffine3D": {"degrees": (90, 90, 90), "scale": (0.0, 1.1)},
    "RandomDepthicalFlip3D": {},
    "RandomHorizontalFlip": {},
    "RandomHorizontalFlip3D": {},
    "RandomRotation": {"degrees": 90},
    "RandomRotation3D": {"degrees": (90, 90, 90)},
    "RandomVerticalFlip": {},
    "RandomVerticalFlip3D": {},
    "RandomElasticDeformation3D": {"alpha": [5, 5], "sigma": [30, 30]}
}


DEFAULT_2D_AUGMENTATIONS = [
    "RandomHorizontalFlip",
    "RandomVerticalFlip"
]
DEFAULT_3D_AUGMENTATIONS = [
    "RandomHorizontalFlip3D",
    "RandomVerticalFlip3D",
    "RandomDepthicalFlip3D",
]
DEFAULT_ANISOTROPIC_AUGMENTATIONS = [
    "RandomHorizontalFlip3D",
    "RandomVerticalFlip3D",
    "RandomDepthicalFlip3D",
]


def create_augmentation(trafo):
    assert trafo in dir(kornia.augmentation) or trafo in globals().keys(), f"Transformation {trafo} not defined"
    if trafo in dir(kornia.augmentation):
        return getattr(kornia.augmentation, trafo)(**AUGMENTATIONS[trafo])

    return globals()[trafo](**AUGMENTATIONS[trafo])


def get_augmentations(ndim=2,
                      transforms=None,
                      dtype=torch.float32):
    if transforms is None:
        assert ndim in (2, 3, "anisotropic"), f"Expect ndim to be one of (2, 3, 'anisotropic'), got {ndim}"
        if ndim == 2:
            transforms = DEFAULT_2D_AUGMENTATIONS
        elif ndim == 3:
            transforms = DEFAULT_3D_AUGMENTATIONS
        else:
            transforms = DEFAULT_ANISOTROPIC_AUGMENTATIONS
    transforms = [create_augmentation(trafo) for trafo in transforms]

    assert all(isinstance(trafo, kornia.augmentation.base._AugmentationBase)
               for trafo in transforms)
    augmentations = KorniaAugmentationPipeline(
        *transforms,
        dtype=dtype
    )
    return augmentations
