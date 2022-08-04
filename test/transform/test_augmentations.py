import unittest
import numpy as np


class TestElasticDeformations(unittest.TestCase):
    def test_default_augmentations(self):
        import torch_em.transform.augmentation as augmentation
        trafo = augmentation.get_augmentations()
        x = np.random.rand(12, 64, 64).astype("float32")
        xt = trafo(x)[0]
        self.assertEqual(xt.shape, (1,) + x.shape)

    # simple test for elastic deformations to check that the 2d elastic deformations run
    def test_elastic_deformation_2d(self):
        import torch_em.transform.augmentation as augmentation
        deform = augmentation.RandomElasticDeformation(alpha=(1.0, 1.0), p=1)
        trafo = augmentation.KorniaAugmentationPipeline(deform)
        x = np.random.rand(1, 1, 64, 64).astype("float32")
        xt = trafo(x)[0]
        self.assertEqual(xt.shape, x.shape)

    # TODO expand the tests to
    # - 3d elastic deformations
    # - testing that the interpolation mechanism works
    # - testing that same augmentations are used for tensors


if __name__ == "__main__":
    unittest.main()
