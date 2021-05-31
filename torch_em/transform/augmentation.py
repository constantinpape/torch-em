from typing import List, Optional, Sequence, Tuple, Union

import kornia.augmentation.utils
import numpy as np
import torch
from skimage.transform import resize

from ..util import ensure_tensor


# TODO RandomElastic3D ?


class RandomElasticDeformation(kornia.augmentation.AugmentationBase2D):
    def __init__(self,
                 control_point_spacing: Union[int, Sequence[int]] = 1,
                 sigma=(4.0, 4.0),
                 alpha=(32.0, 32.0),
                 resample=kornia.constants.Resample.BILINEAR,
                 p=0.5,
                 keepdim=False,
                 same_on_batch=False):
        super().__init__(p=p,  # keepdim=keepdim,
                         same_on_batch=same_on_batch,
                         return_transform=False)
        if isinstance(control_point_spacing, int):
            self.control_point_spacing = [control_point_spacing] * 2
        else:
            self.control_point_spacing = control_point_spacing
        assert len(self.control_point_spacing) == 2
        self.resample = resample
        self.flags = dict(
            resample=torch.tensor(self.resample.value),
            sigma=sigma,
            alpha=alpha
        )

    # TODO do we need special treatment for batches, channels > 1?
    def generate_parameters(self, batch_shape):
        assert len(batch_shape) == 4
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
        noise = np.concatenate(deformation_fields, axis=0)[None].astype('float32')
        noise = torch.from_numpy(noise)
        return {'noise': noise}

    def apply_transform(self, input, params):
        noise = params['noise']
        mode = 'bilinear' if (self.flags['resample'] == 1).all() else 'nearest'
        # NOTE mode is currently only available on my fork, need kornia PR:
        # https://github.com/kornia/kornia/pull/883
        return kornia.geometry.transform.elastic_transform2d(
            input, noise, sigma=self.flags['sigma'], alpha=self.flags['alpha'], mode=mode
        )


KorniaAugmentation = Union[kornia.augmentation.AugmentationBase2D, kornia.augmentation.AugmentationBase3D]

# TODO implement 'require_halo', and estimate the halo from the transformations
# so that we can load a bigger block and cut it away
class KorniaAugmentationPipeline(torch.nn.Module):
    interpolatable_torch_types = [torch.float16, torch.float32, torch.float64]
    interpolatable_numpy_types = [np.dtype("float32"), np.dtype("float64")]

    def __init__(self, *augmentations: KorniaAugmentation, return_transform: bool = False, dtype=torch.float32):
        super().__init__()
        self.return_transform = return_transform
        self.is3D = any(isinstance(aug, kornia.augmentation.AugmentationBase3D) for aug in augmentations)
        for aug in augmentations:
            aug.return_transform = return_transform

        self.augmentations: Sequence[KorniaAugmentation] = torch.nn.ModuleList(augmentations)  # type: ignore
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

    @staticmethod
    def _configure_augmentation(augmentation: KorniaAugmentation, interpolatable):
        interpolating = "interpolation" in getattr(augmentation, "flags", [])
        if interpolating:
            resampler = kornia.constants.Resample.get("BILINEAR" if interpolatable else "NEAREST")
            augmentation.flags["interpolation"] = torch.tensor(resampler.value)

    def _get_eye(self, tensor: torch.Tensor):
        return kornia.eye_like(4 if self.is3D else 3, tensor)

    def ensure_batch_tensor(self, tensor: torch.Tensor):
        ensure_batch = (
            kornia.augmentation.utils.helpers._transform_input3d
            if self.is3D
            else kornia.augmentation.utils.helpers._transform_input
        )
        return ensure_batch(ensure_tensor(tensor, self.dtype))

    def forward(
        self, *tensors, trans_matrices: Optional[Sequence[torch.Tensor]] = None
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        interpolatable = [self.is_interpolatable(tensor) for tensor in tensors]
        tensors = [self.ensure_batch_tensor(tensor) for tensor in tensors]

        if trans_matrices is None:
            trans_matrices = [self._get_eye(t) for t in tensors]
        else:
            trans_matrices = list(trans_matrices)

        assert len(trans_matrices) == len(tensors)
        for ti, tensor in enumerate(tensors):
            if trans_matrices[ti] is None:
                trans_matrices[ti] = self._get_eye(tensor)

        transformed_tensors = []
        all_params = [None] * len(self.augmentations)
        for ti, (tensor, interpolate) in enumerate(zip(tensors, interpolatable)):
            for ai, aug in enumerate(self.augmentations):
                self._configure_augmentation(aug, interpolatable)
                tensor, trans_matrices[ti] = aug((tensor, trans_matrices[ti]), all_params[ai])

                if all_params[ai] is None:
                    all_params[ai] = aug._params
                else:
                    assert all_params[ai] == aug._params

            transformed_tensors.append(tensor)

        return (transformed_tensors, trans_matrices) if self.return_transform else transformed_tensors

    def halo(self, shape):
        return self.halo

    def apply_inverse_affine(
        self, *tensors: torch.Tensor, forward_transforms: Sequence[torch.Tensor], padding_mode="border"
    ):
        assert len(tensors) == len(forward_transforms)
        trans_matrices = torch.linalg.inv(torch.stack(list(forward_transforms)))
        return [
            kornia.warp_affine3d(
                src=t,
                M=m,
                dsize=t.shape[-3:],
                flags="bliinear" if self.is_interpolatable(t) else "nearest",
                padding_mode=padding_mode,
            )
            for t, m in zip(tensors, trans_matrices)
        ]


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
}


DEFAULT_2D_AUGMENTATIONS = ["RandomHorizontalFlip", "RandomVerticalFlip"]
DEFAULT_3D_AUGMENTATIONS = ["RandomHorizontalFlip3D", "RandomVerticalFlip3D", "RandomDepthicalFlip3D"]
DEFAULT_ANISOTROPIC_AUGMENTATIONS = ["RandomHorizontalFlip3D", "RandomVerticalFlip3D", "RandomDepthicalFlip3D"]


def get_augmentations(ndim=2, transforms=None, dtype=torch.float32, return_transforms: bool = False):
    if transforms is None:
        assert ndim in (2, 3, "anisotropic"), f"Expect ndim to be one of (2, 3, 'anisotropic'), got {ndim}"
        if ndim == 2:
            transforms = DEFAULT_2D_AUGMENTATIONS
        elif ndim == 3:
            transforms = DEFAULT_3D_AUGMENTATIONS
        else:
            transforms = DEFAULT_ANISOTROPIC_AUGMENTATIONS
        transforms = [getattr(kornia.augmentation, trafo)(**AUGMENTATIONS[trafo]) for trafo in transforms]

    return KorniaAugmentationPipeline(*transforms, dtype=dtype, return_transform=return_transforms)
