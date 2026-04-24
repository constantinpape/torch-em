import unittest
import numpy as np
import torch
import kornia.augmentation as K

from torch_em.transform.invertible_augmentations import AugmentationSequential3D, InvertibleAugmenter


def _identity(x):
    return x


class TestInvertibleAugmenter(unittest.TestCase):

    def _make_input(self, shape=(1, 1, 64, 64)):
        return torch.rand(*shape)

    def _make_input_3d(self, shape=(1, 1, 16, 64, 64)):
        return torch.rand(*shape)

    def test_flip_only(self):
        geo = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=1.0),
            data_keys=["input"],
            same_on_batch=False,
        )
        augmenter = InvertibleAugmenter(intensity_transforms=_identity, geometrical_transforms=geo)

        x = self._make_input()
        x_aug = augmenter.transform(x)
        x_inv = augmenter.reverse_transform(x_aug)

        self.assertTrue(np.allclose(x.numpy(), x_inv.numpy(), atol=1e-4))

    def test_rotation_only(self):
        geo = K.AugmentationSequential(
            K.RandomRotation90(times=(-1, 2), p=1.0),
            data_keys=["input"],
            same_on_batch=False,
        )
        augmenter = InvertibleAugmenter(intensity_transforms=_identity, geometrical_transforms=geo)

        x = self._make_input()
        x_aug = augmenter.transform(x)
        x_inv = augmenter.reverse_transform(x_aug)

        self.assertTrue(np.allclose(x.numpy(), x_inv.numpy(), atol=1e-4))

    def test_flip_and_rotation(self):
        geo = K.AugmentationSequential(
            K.RandomVerticalFlip(p=1.0),
            K.RandomRotation90(times=(-1, 2), p=1.0),
            data_keys=["input"],
            same_on_batch=False,
        )
        augmenter = InvertibleAugmenter(intensity_transforms=_identity, geometrical_transforms=geo)

        x = self._make_input()
        x_aug = augmenter.transform(x)
        x_inv = augmenter.reverse_transform(x_aug)

        self.assertTrue(np.allclose(x.numpy(), x_inv.numpy(), atol=1e-4))

    def test_intensity_augmentations(self):
        intensity = K.AugmentationSequential(
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=1.0),
            K.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
            data_keys=["input"],
            same_on_batch=False,
        )
        geo = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=1.0),
            data_keys=["input"],
            same_on_batch=False,
        )
        augmenter = InvertibleAugmenter(intensity_transforms=intensity, geometrical_transforms=geo)

        x = self._make_input()
        x_aug = augmenter.transform(x)

        self.assertEqual(x.shape, x_aug.shape)


    def test_flip_only_3d(self):
        geo = AugmentationSequential3D(
            K.RandomHorizontalFlip(p=1.0),
        )
        augmenter = InvertibleAugmenter(intensity_transforms=_identity, geometrical_transforms=geo)

        x = self._make_input_3d()
        x_aug = augmenter.transform(x)
        x_inv = augmenter.reverse_transform(x_aug)

        self.assertTrue(np.allclose(x.numpy(), x_inv.numpy(), atol=1e-4))

    def test_rotation_only_3d(self):
        geo = AugmentationSequential3D(
            K.RandomRotation90(times=(-1, 2), p=1.0),
        )
        augmenter = InvertibleAugmenter(intensity_transforms=_identity, geometrical_transforms=geo)

        x = self._make_input_3d()
        x_aug = augmenter.transform(x)
        x_inv = augmenter.reverse_transform(x_aug)

        self.assertTrue(np.allclose(x.numpy(), x_inv.numpy(), atol=1e-4))

    def test_flip_and_rotation_3d(self):
        geo = AugmentationSequential3D(
            K.RandomVerticalFlip(p=1.0),
            K.RandomRotation90(times=(-1, 2), p=1.0),
        )
        augmenter = InvertibleAugmenter(intensity_transforms=_identity, geometrical_transforms=geo)

        x = self._make_input_3d()
        x_aug = augmenter.transform(x)
        x_inv = augmenter.reverse_transform(x_aug)

        self.assertTrue(np.allclose(x.numpy(), x_inv.numpy(), atol=1e-4))

    def test_intensity_augmentations_3d(self):
        intensity = AugmentationSequential3D(
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=1.0),
            K.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
        )
        geo = AugmentationSequential3D(
            K.RandomHorizontalFlip(p=1.0),
        )
        augmenter = InvertibleAugmenter(intensity_transforms=intensity, geometrical_transforms=geo)

        x = self._make_input_3d()
        x_aug = augmenter.transform(x)

        self.assertEqual(x.shape, x_aug.shape)


if __name__ == "__main__":
    unittest.main()
