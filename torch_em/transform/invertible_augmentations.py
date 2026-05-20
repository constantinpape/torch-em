from typing import Callable
import torch
import kornia.augmentation as K


DEFAULT_WEAK_AUGMENTATIONS = {
    "intensity": {},
    "geometrical": {
        "RandomHorizontalFlip": {},
        "RandomVerticalFlip": {},
        "RandomRotation90": {"times": (-1, 2)},
    }
}

DEFAULT_STRONG_AUGMENTATIONS = {
    "intensity": {
        "RandomGaussianBlur": {"kernel_size": (3, 3), "sigma": (0.1, 1.0)},
        "RandomGaussianNoise": {"mean": (0.0), "std": (0.1)},
    },
    "geometrical": {
        "RandomHorizontalFlip": {},
        "RandomVerticalFlip": {},
        "RandomRotation90": {"times": (-1, 2)},
    }
}


def get_intensity_augmentations(aug_name, ndim, p: float = 0.75) -> callable:
    if aug_name == "weak":
        aug_dict = DEFAULT_WEAK_AUGMENTATIONS["intensity"]
    elif aug_name == "strong":
        aug_dict = DEFAULT_STRONG_AUGMENTATIONS["intensity"]
    else:
        raise ValueError(f"Number of dimensions must be 2 or 3, got ndim={ndim}")

    transforms_list = []
    for trafo, kwargs in aug_dict.items():
        assert trafo in dir(K), f"{trafo} not found in kornia.augmentation"
        transforms_list.append(getattr(K, trafo)(p=p, **kwargs))

    if ndim == 2:
        return K.AugmentationSequential(*transforms_list, data_keys=["input"], same_on_batch=False)
    elif ndim == 3:
        return AugmentationSequential3D(*transforms_list)


def get_geometrical_augmentations(aug_name, ndim, p: float = 0.75) -> callable:
    if aug_name == "weak":
        aug_dict = DEFAULT_WEAK_AUGMENTATIONS["geometrical"]
    elif aug_name == "strong":
        aug_dict = DEFAULT_STRONG_AUGMENTATIONS["geometrical"]
    else:
        raise ValueError(f"Number of dimensions must be 2 or 3, got ndim={ndim}")

    transforms_list = []
    for trafo, kwargs in aug_dict.items():
        assert trafo in dir(K), f"{trafo} not found in kornia.augmentation"
        transforms_list.append(getattr(K, trafo)(p=p, **kwargs))

    if ndim == 2:
        return K.AugmentationSequential(*transforms_list, data_keys=["input"], same_on_batch=False)
    elif ndim == 3:
        return AugmentationSequential3D(*transforms_list)


def get_augmentations(aug_name: str, ndim: int, p: float = 0.75):
    if aug_name == "weak":
        intensity_transforms = get_intensity_augmentations(aug_name, ndim=ndim, p=p)
        geometrical_transforms = get_geometrical_augmentations(aug_name, ndim=ndim, p=p)
    elif aug_name == "strong":
        intensity_transforms = get_intensity_augmentations(aug_name, ndim=ndim, p=p)
        geometrical_transforms = get_geometrical_augmentations(aug_name, ndim=ndim, p=p)
    else:
        raise ValueError(f"aug_name must be 'weak' or 'strong', got {aug_name}")

    return intensity_transforms, geometrical_transforms


class AugmentationSequential3D(torch.nn.Module):
    def __init__(self, *augmentations: torch.nn.Module):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)
        self._params = None

    @staticmethod
    def _flatten(x):
        """
        (B, C, D, H, W) -> (B, C*D, H, W)
        """
        if x.ndim != 5:
            raise RuntimeError(f"Expected 5D tensor, got {x.shape}")
        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        return x, (b, c, d, h, w)

    @staticmethod
    def _unflatten(x, shape):
        """
        (B, C*D, H, W) -> (B, C, D, H, W)
        """
        b, c, d, h, w = shape
        x = x.reshape(b, c, d, h, w)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params_all = []

        flat_x, shape = self._flatten(x)
        for aug in self.augmentations:
            flat_x = aug(flat_x)
            params_all.append(aug._params)
        out = self._unflatten(flat_x, shape)
        self._params = params_all
        return out

    def inverse(self, x: torch.Tensor, params) -> torch.Tensor:

        flat_x, shape = self._flatten(x)
        for aug, p in reversed(list(zip(self.augmentations, params))):
            flat_x = aug.inverse(flat_x, params=p)
        out = self._unflatten(flat_x, shape)

        return out


class InvertibleAugmenter(torch.nn.Module):

    def __init__(
        self,
        intensity_transforms: Callable[[torch.Tensor], torch.Tensor],
        geometrical_transforms: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intensity_transforms = intensity_transforms
        self.geometrical_transforms = geometrical_transforms

    def reset(self):
        self.params = None

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intensity_transforms(x)
        x = self.geometrical_transforms(x)

        self.params = self.geometrical_transforms._params

        return x

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        x_inv = self.geometrical_transforms.inverse(x, params=self.params)
        return x_inv


class MeanTeacherAugmenters:
    def __init__(
        self,
        ndim: int,
        teacher=None,
        student=None,
    ):
        self.teacher = teacher or InvertibleAugmenter(*get_augmentations("weak", ndim=ndim))
        self.student = student or InvertibleAugmenter(*get_augmentations("weak", ndim=ndim))

    def reset_all(self):
        self.teacher.reset()
        self.student.reset()


class FixMatchAugmenters:
    def __init__(
        self,
        ndim: int,
        teacher=None,
        student=None,
    ):
        self.teacher = teacher or InvertibleAugmenter(*get_augmentations("weak", ndim=ndim))
        self.student = student or InvertibleAugmenter(*get_augmentations("strong", ndim=ndim))

    def reset_all(self):
        self.teacher.reset()
        self.student.reset()


class UniMatchv2Augmenters:
    def __init__(
        self,
        ndim: int,
        weak=None,
        strong1=None,
        strong2=None,
    ):
        self.weak = weak or InvertibleAugmenter(*get_augmentations("weak", ndim=ndim))
        self.strong1 = strong1 or InvertibleAugmenter(*get_augmentations("strong", ndim=ndim))
        self.strong2 = strong2 or InvertibleAugmenter(*get_augmentations("strong", ndim=ndim))

    def reset_all(self):
        self.weak.reset()
        self.strong1.reset()
        self.strong2.reset()
