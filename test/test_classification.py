import os
import unittest
from shutil import rmtree

import numpy as np
import torch

from torch_em.util import model_is_equal


class TestClassification(unittest.TestCase):
    def tearDown(self):
        if os.path.exists("./checkpoints"):
            rmtree("./checkpoints")
        if os.path.exists("./logs"):
            rmtree("./logs")

    def _check_checkpoint(self, path, expected_iterations, expected_model, model_class, **model_kwargs):
        self.assertTrue(os.path.exists(path))
        checkpoint = torch.load(path)

        self.assertIn("optimizer_state", checkpoint)
        self.assertIn("model_state", checkpoint)

        loaded_model = model_class(**model_kwargs)
        loaded_model.load_state_dict(checkpoint["model_state"])
        self.assertTrue(model_is_equal(expected_model, loaded_model))

        self.assertEqual(checkpoint["iteration"], expected_iterations)

    def test_classification_2d(self):
        from torch_em.classification import default_classification_loader, default_classification_trainer
        from torchvision.models.resnet import resnet18

        shape = (3, 256, 256)
        image_shape = (128, 128)

        n_samples = 15
        data = [np.random.rand(*shape) for _ in range(n_samples)]

        n_classes = 8
        target = np.random.randint(0, n_classes, size=n_samples)

        loader = default_classification_loader(data, target, batch_size=1, image_shape=image_shape)

        model = resnet18(num_classes=n_classes)
        trainer = default_classification_trainer(
            name="test-model-2d", model=model, train_loader=loader, val_loader=loader,
        )
        n_iterations = 18
        trainer.fit(n_iterations)

        self._check_checkpoint(
            "./checkpoints/test-model-2d/latest.pt", 18, trainer.model, resnet18, num_classes=n_classes
        )

    def test_classification_3d(self):
        from torch_em.classification import default_classification_loader, default_classification_trainer
        from torch_em.model.resnet3d import resnet3d_18

        shape = (1, 128, 128, 128)
        image_shape = (64, 64, 64)

        n_samples = 10
        data = [np.random.rand(*shape) for _ in range(n_samples)]

        n_classes = 8
        target = np.random.randint(0, n_classes, size=n_samples)

        loader = default_classification_loader(data, target, batch_size=1, image_shape=image_shape)

        model = resnet3d_18(in_channels=1, out_channels=n_classes)
        trainer = default_classification_trainer(
            name="test-model-3d", model=model, train_loader=loader, val_loader=loader,
        )
        trainer.fit(12)

        self._check_checkpoint(
            "./checkpoints/test-model-3d/latest.pt", 12, trainer.model, resnet3d_18,
            in_channels=1, out_channels=n_classes
        )


if __name__ == "__main__":
    unittest.main()
