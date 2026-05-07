from typing import Dict, List, Optional, Sequence, Tuple, Union

import kornia
import numpy as np
import torch
from skimage.transform import resize

from ..util import ensure_tensor


class RandomElasticDeformationStacked(kornia.augmentation.AugmentationBase3D):
    """Random elastic deformations implemented with kornia.

    This transformation can be applied to 3D data, the same deformation is applied to each plane.

    Args:
        control_point_spacing: The control point spacing for the deformation field.
        sigma: Sigma for smoothing the deformation field.
        alpha: Alpha value.
        interpolation: Interpolation order for applying the transformation to the data.
        p: Probability for applying the transformation.
        keepdim:
        same_on_batch:
    """
    def __init__(
        self,
        control_point_spacing: Union[int, Sequence[int]] = 1,
        sigma: Tuple[float, float] = (32.0, 32.0),
        alpha: Tuple[float, float] = (4.0, 4.0),
        interpolation=kornia.constants.Resample.BILINEAR,
        p: float = 0.5,
        keepdim: bool = False,
        same_on_batch: bool = True,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch)
        if isinstance(control_point_spacing, int):
            self.control_point_spacing = [control_point_spacing] * 2
        else:
            self.control_point_spacing = control_point_spacing
        assert len(self.control_point_spacing) == 2
        self.interpolation = interpolation
        self.flags = dict(interpolation=torch.tensor(self.interpolation.value), sigma=sigma, alpha=alpha)

    def generate_parameters(self, batch_shape):
        """@private
        """
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

    def __call__(self, input: torch.Tensor, params: Optional[Dict] = None) -> torch.Tensor:
        """Apply the augmentation to a tensor.

        Args:
            input: The input tensor.
            params: The transformation parameters.

        Returns:
            The transformed tensor.
        """
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
                x, noise_ch, sigma=self.flags["sigma"], alpha=self.flags["alpha"], mode=mode, padding_mode="reflection"
            )
            input_transformed.append(x_transformed)
        input_transformed = torch.stack(input_transformed)
        return input_transformed


class RandomElasticDeformation(kornia.augmentation.AugmentationBase2D):
    """Random elastic deformations implemented with kornia.

    Args:
        control_point_spacing: The control point spacing for the deformation field.
        sigma: Sigma for smoothing the deformation field.
        alpha: Alpha value.
        resample: Interpolation order for applying the transformation to the data.
        p: Probability for applying the transformation.
        keepdim:
        same_on_batch:
    """
    def __init__(
        self,
        control_point_spacing: Union[int, Sequence[int]] = 1,
        sigma: Tuple[float, float] = (32.0, 32.0),
        alpha: Tuple[float, float] = (4.0, 4.0),
        resample=kornia.constants.Resample.BILINEAR,
        p: float = 0.5,
        keepdim: bool = False,
        same_on_batch: bool = True,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch)
        if isinstance(control_point_spacing, int):
            self.control_point_spacing = [control_point_spacing] * 2
        else:
            self.control_point_spacing = control_point_spacing
        assert len(self.control_point_spacing) == 2
        self.resample = resample
        self.flags = dict(resample=torch.tensor(self.resample.value), sigma=sigma, alpha=alpha)

    def generate_parameters(self, batch_shape):
        """@private
        """
        assert len(batch_shape) == 4, f"{len(batch_shape)}"
        shape = batch_shape[2:]
        control_shape = tuple(sh // spacing for sh, spacing in zip(shape, self.control_point_spacing))
        deformation_fields = [np.random.uniform(-1, 1, control_shape), np.random.uniform(-1, 1, control_shape)]
        deformation_fields = [resize(df, shape, order=3)[None] for df in deformation_fields]
        noise = np.concatenate(deformation_fields, axis=0)[None].astype("float32")
        noise = torch.from_numpy(noise)
        return {"noise": noise}

    def __call__(self, input: torch.Tensor, params: Optional[Dict] = None) -> torch.Tensor:
        """Apply the augmentation to a tensor.

        Args:
            input: The input tensor.
            params: The transformation parameters.

        Returns:
            The transformed tensor.
        """
        if params is None:
            params = self.generate_parameters(input.shape)
            self._params = params
        noise = params["noise"]
        mode = "bilinear" if (self.flags["resample"] == 1).all() else "nearest"
        return kornia.geometry.transform.elastic_transform2d(
            input, noise, sigma=self.flags["sigma"], alpha=self.flags["alpha"], mode=mode, padding_mode="reflection"
        )


# TODO: Implement 'require_halo', and estimate the halo from the transformations
# so that we can load a bigger block and cut it away.
class KorniaAugmentationPipeline(torch.nn.Module):
    """Pipeline to apply multiple kornia augmentations to data.

    Args:
        kornia_augmentations: Augmentations implemented with kornia.
        dtype: The data type of the return data.
    """
    interpolatable_torch_types = [torch.float16, torch.float32, torch.float64]
    interpolatable_numpy_types = [np.dtype("float32"), np.dtype("float64")]

    def __init__(self, *kornia_augmentations, dtype: Union[str, torch.dtype] = torch.float32):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(kornia_augmentations)
        self.dtype = dtype
        self.halo = self.compute_halo()

    # for now we only add a halo for the random rotation trafos and
    # also don't compute the halo dynamically based on the input shape
    def compute_halo(self):
        """@private
        """
        halo = None
        for aug in self.augmentations:
            if isinstance(aug, kornia.augmentation.RandomRotation):
                halo = [32, 32]
            if isinstance(aug, kornia.augmentation.RandomRotation3D):
                halo = [32, 32, 32]
        return halo

    def is_interpolatable(self, tensor):
        """@private
        """
        if torch.is_tensor(tensor):
            return tensor.dtype in self.interpolatable_torch_types
        else:
            return tensor.dtype in self.interpolatable_numpy_types

    def transform_tensor(self, augmentation, tensor, interpolatable, params=None):
        """@private
        """
        interpolating = "interpolation" in getattr(augmentation, "flags", [])
        if interpolating:
            resampler = kornia.constants.Resample.get("BILINEAR" if interpolatable else "NEAREST")
            augmentation.flags["interpolation"] = torch.tensor(resampler.value)
        transformed = augmentation(tensor, params)
        return transformed, augmentation._params

    def forward(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        """Apply augmentations to a list of tensors.

        Args:
            tensors: The input tensors.

        Returns:
            List of transformed tensors.
        """
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
        """@private
        """
        return self.halo


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
"""All available augmentations and their default parameters.
"""

DEFAULT_2D_AUGMENTATIONS = [
    "RandomHorizontalFlip",
    "RandomVerticalFlip"
]
"""The default parameters for 2D data.
"""
DEFAULT_3D_AUGMENTATIONS = [
    "RandomHorizontalFlip3D",
    "RandomVerticalFlip3D",
    "RandomDepthicalFlip3D",
]
"""The default parameters for 3D data.
"""
DEFAULT_ANISOTROPIC_AUGMENTATIONS = [
    "RandomHorizontalFlip3D",
    "RandomVerticalFlip3D",
    "RandomDepthicalFlip3D",
]
"""The default parameters for anisotropic 3D data.
"""


def create_augmentation(trafo):
    """@private
    """
    assert trafo in dir(kornia.augmentation) or trafo in globals().keys(), f"Transformation {trafo} not defined"
    if trafo in dir(kornia.augmentation):
        return getattr(kornia.augmentation, trafo)(**AUGMENTATIONS[trafo])
    return globals()[trafo](**AUGMENTATIONS[trafo])


def get_augmentations(ndim: Union[int, str] = 2, transforms=None, dtype: Union[str, torch.dtype] = torch.float32):
    """Get augmentation pipeline.

    Args:
        ndim: The dimensionality for the augmentations. One of 2, 3 or "anisotropic".
        transforms: The transformations to use for the augmentations.
            If None, the default augmentations for the given data dimensionality will be used.
        dtype: The data type of the output data of the augmentation.

    Returns:
        The augmentation pipeline.
    """
    if transforms is None:
        assert ndim in (2, 3, "anisotropic"), f"Expect ndim to be one of (2, 3, 'anisotropic'), got {ndim}"
        if ndim == 2:
            transforms = DEFAULT_2D_AUGMENTATIONS
        elif ndim == 3:
            transforms = DEFAULT_3D_AUGMENTATIONS
        else:
            transforms = DEFAULT_ANISOTROPIC_AUGMENTATIONS
    transforms = [create_augmentation(trafo) for trafo in transforms]
    assert all(isinstance(trafo, kornia.augmentation.base._AugmentationBase) for trafo in transforms)
    augmentations = KorniaAugmentationPipeline(*transforms, dtype=dtype)
    return augmentations
