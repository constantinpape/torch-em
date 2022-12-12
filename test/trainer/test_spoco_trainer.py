import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch

import torch_em
from torch_em.model import UNet2d


class DummySpocoLoss(torch.nn.Module):
    def forward(self, predictions, target):
        assert isinstance(predictions, tuple), type(predictions)
        assert len(predictions) == 2
        return (predictions[0] * predictions[1]).sum(dim=tuple(range(1, predictions[0].ndim)))


class DummySpocoMetric(torch.nn.Module):
    def forward(self, prediction, target):
        return prediction.sum(dim=tuple(range(1, prediction.ndim)))


class TestSpocoTrainer(unittest.TestCase):
    data_path = "./data.h5"
    checkpoint_folder = "./checkpoints"
    log_folder = "./logs"
    name = "test"

    def setUp(self):
        shape = (8, 128, 128)
        chunks = (1, 128, 128)
        with h5py.File(self.data_path, "w") as f:
            f.create_dataset("raw", data=np.random.rand(*shape), chunks=chunks)
            f.create_dataset("labels", data=np.random.randint(0, 32, size=shape), chunks=chunks)

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        if os.path.exists(self.checkpoint_folder):
            rmtree(self.checkpoint_folder)
        if os.path.exists(self.log_folder):
            rmtree(self.log_folder)

    def _get_kwargs(self, with_roi=False):
        roi = np.s_[:6, :, :] if with_roi else None
        loader = torch_em.default_segmentation_loader(
            raw_paths=self.data_path, raw_key="raw",
            label_paths=self.data_path, label_key="labels",
            batch_size=1, patch_shape=(1, 128, 128), ndim=2,
            rois=roi,
        )
        model = UNet2d(in_channels=1, out_channels=1,
                       depth=2, initial_features=4)
        # NOTE we use a dummy lss because torch_scatter is required for the full spoco loss
        # and it is not available in the CI
        kwargs = {
            "name": self.name,
            "train_loader": loader,
            "val_loader": loader,
            "model": model,
            "loss": DummySpocoLoss(),
            "metric": DummySpocoMetric(),
            "optimizer": torch.optim.Adam(model.parameters(), lr=1e-5),
            "device": torch.device("cpu"),
            "mixed_precision": False,
            "momentum": 0.95,
        }
        return kwargs

    def test_fit(self):
        from torch_em.trainer.spoco_trainer import SPOCOTrainer
        trainer = SPOCOTrainer(**self._get_kwargs())

        trainer.fit(10)

        save_folder = os.path.join(self.checkpoint_folder, self.name)
        self.assertTrue(os.path.exists(save_folder))
        best_ckpt = os.path.join(save_folder, "best.pt")
        self.assertTrue(os.path.exists(best_ckpt))
        latest_ckpt = os.path.join(save_folder, "latest.pt")
        self.assertTrue(os.path.exists(latest_ckpt))
        self.assertEqual(trainer.iteration, 10)

        trainer.fit(2)
        self.assertEqual(trainer.iteration, 12)

        trainer = SPOCOTrainer(**self._get_kwargs())
        trainer.fit(8, load_from_checkpoint="latest")
        self.assertEqual(trainer.iteration, 20)

    def test_from_checkpoint(self):
        from torch_em.trainer.spoco_trainer import SPOCOTrainer
        init_kwargs = self._get_kwargs(with_roi=True)
        trainer = SPOCOTrainer(**init_kwargs)
        trainer.fit(10)
        exp_data_shape = trainer.train_loader.dataset.raw.shape

        trainer2 = SPOCOTrainer.from_checkpoint(
            os.path.join(self.checkpoint_folder, self.name),
            name="latest"
        )
        self.assertEqual(trainer.iteration, trainer2.iteration)
        self.assertEqual(trainer2.train_loader.dataset.raw.shape, exp_data_shape)
        self.assertEqual(trainer.momentum, trainer2.momentum)
        self.assertTrue(torch_em.util.model_is_equal(trainer.model, trainer2.model))
        self.assertTrue(torch_em.util.model_is_equal(trainer.model2, trainer2.model2))

        # make sure that the optimizer was loaded properly
        lr1 = [pm["lr"] for pm in trainer.optimizer.param_groups][0]
        lr2 = [pm["lr"] for pm in trainer2.optimizer.param_groups][0]
        self.assertEqual(lr1, lr2)

        trainer2.fit(10)
        self.assertEqual(trainer2.iteration, 20)


if __name__ == "__main__":
    unittest.main()
