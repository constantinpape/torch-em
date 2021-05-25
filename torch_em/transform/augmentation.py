import torch
import numpy as np
import kornia
from skimage.transform import resize

from ..util import ensure_tensor


# TODO RandomElastic3D ?


class RandomElasticDeformation(kornia.augmentation.AugmentationBase2D):
    def __init__(self,
                 control_point_spacing=1,
                 sigma=(4., 4.),
                 alpha=(32., 32.),
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


# TODO implement 'require_halo', and estimate the halo from the transformations
# so that we can load a bigger block and cut it away
class KorniaAugmentationPipeline(torch.nn.Module):
    interpolatable_torch_tpyes = [torch.float16, torch.float32, torch.float64]
    interpolatable_numpy_types = [np.dtype('float32'), np.dtype('float64')]

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
        interpolating = 'interpolation' in getattr(augmentation, 'flags', [])
        if interpolating:
            resampler = kornia.constants.Resample.get('BILINEAR' if interpolatable else 'NEAREST')
            augmentation.flags['interpolation'] = torch.tensor(resampler.value)

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
    "RandomAffine": {"degrees": 90,"scale": (0.9, 1.1)},
    "RandomAffine3D": {"degrees": (90, 90, 90), "scale": (0.0, 1.1)},
    "RandomDepthicalFlip3D": {},
    "RandomHorizontalFlip": {},
    "RandomHorizontalFlip3D": {},
    "RandomRotation": {"degrees": 90},
    "RandomRotation3D": {"degrees": (90, 90, 90)},
    "RandomVerticalFlip": {},
    "RandomVerticalFlip3D": {},
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
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomDepthicalFlip3D",
]


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
        transforms = [
            getattr(kornia.augmentation, trafo)(**AUGMENTATIONS[trafo])
            for trafo in transforms
        ]

    assert all(isinstance(trafo, kornia.augmentation.base._AugmentationBase)
               for trafo in transforms)
    augmentations = KorniaAugmentationPipeline(
        *transforms,
        dtype=dtype
    )
    return augmentations
