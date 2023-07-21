import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch_em
from torch_em.model import UNet2d


class TestUtil(unittest.TestCase):
    name = "test"
    data_path = "./data.h5"
    checkpoint_folder = "./checkpoints"
    log_folder = "./logs"

    def setUp(self):
        shape = (8, 128, 128)
        chunks = (1, 128, 128)
        with h5py.File(self.data_path, "w") as f:
            f.create_dataset("raw", data=np.random.rand(*shape), chunks=chunks)
            f.create_dataset("labels", data=np.random.rand(*shape), chunks=chunks)

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        rmtree(self.checkpoint_folder, ignore_errors=True)
        rmtree(self.log_folder, ignore_errors=True)

    def _get_model(self):
        return UNet2d(in_channels=1, out_channels=1, depth=3, initial_features=4)

    def _train_network(self):
        model = self._get_model()
        loader = torch_em.default_segmentation_loader(
            self.data_path, "raw", self.data_path, "labels",
            batch_size=1, patch_shape=(1, 64, 64), ndim=2
        )
        trainer = torch_em.default_segmentation_trainer(
            name=self.name,
            model=model,
            train_loader=loader,
            val_loader=loader,
            logger=None,
        )
        trainer.fit(5)
        return trainer

    def test_load_model(self):
        from torch_em.util import load_model

        trainer = self._train_network()

        model1 = load_model(os.path.join(self.checkpoint_folder, self.name))
        self.assertTrue(torch_em.util.model_is_equal(trainer.model, model1))

        model2 = self._get_model()
        model2 = load_model(os.path.join(self.checkpoint_folder, self.name))
        self.assertTrue(torch_em.util.model_is_equal(trainer.model, model2))


if __name__ == "__main__":
    unittest.main()
