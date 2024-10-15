import os
import sys
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch

import torch_em
from torch_em.model import UNet2d


class TestDefaultTrainer(unittest.TestCase):
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
        rmtree(self.checkpoint_folder, ignore_errors=True)
        rmtree(self.log_folder, ignore_errors=True)

    def _get_kwargs(self, with_roi=False, compile_model=False):
        roi = np.s_[:6, :, :] if with_roi else None
        loader = torch_em.default_segmentation_loader(
            raw_paths=self.data_path, raw_key="raw",
            label_paths=self.data_path, label_key="labels",
            batch_size=1, patch_shape=(1, 128, 128), ndim=2,
            rois=roi,
        )
        model = UNet2d(in_channels=1, out_channels=1,
                       depth=2, initial_features=4)
        kwargs = {
            "name": self.name,
            "train_loader": loader,
            "val_loader": loader,
            "model": model,
            "loss": torch_em.loss.DiceLoss(),
            "metric": torch_em.loss.DiceLoss(),
            "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-5),
            "device": torch.device("cpu"),
            "mixed_precision": True,
            "compile_model": compile_model,
        }
        return kwargs

    def test_fit(self):
        from torch_em.trainer import DefaultTrainer

        trainer = DefaultTrainer(**self._get_kwargs())
        trainer.fit(10)
        train_time = trainer.train_time
        self.assertGreater(train_time, 0.0)

        save_folder = os.path.join(self.checkpoint_folder, self.name)
        self.assertTrue(os.path.exists(save_folder))
        best_ckpt = os.path.join(save_folder, "best.pt")
        self.assertTrue(os.path.exists(best_ckpt))
        latest_ckpt = os.path.join(save_folder, "latest.pt")
        self.assertTrue(os.path.exists(latest_ckpt))
        self.assertEqual(trainer.iteration, 10)

        trainer.fit(2)
        self.assertEqual(trainer.iteration, 12)
        self.assertGreater(trainer.train_time, train_time)

        trainer = DefaultTrainer(**self._get_kwargs())
        trainer.fit(8, load_from_checkpoint="latest")
        self.assertEqual(trainer.iteration, 20)

    def test_from_checkpoint(self):
        from torch_em.trainer import DefaultTrainer

        trainer = DefaultTrainer(**self._get_kwargs(with_roi=True))
        trainer.fit(10)
        exp_model = trainer.model
        exp_data_shape = trainer.train_loader.dataset.raw.shape

        trainer2 = DefaultTrainer.from_checkpoint(
            os.path.join(self.checkpoint_folder, self.name),
            name="latest"
        )
        self.assertEqual(trainer.iteration, trainer2.iteration)
        self.assertEqual(trainer.train_time, trainer2.train_time)
        self.assertEqual(trainer2.train_loader.dataset.raw.shape, exp_data_shape)
        self.assertTrue(torch_em.util.model_is_equal(exp_model, trainer2.model))

        # make sure that the optimizer was loaded properly
        lr1 = [pm["lr"] for pm in trainer.optimizer.param_groups][0]
        lr2 = [pm["lr"] for pm in trainer2.optimizer.param_groups][0]
        self.assertEqual(lr1, lr2)

        trainer2.fit(10)
        self.assertEqual(trainer2.iteration, 20)

    @unittest.skipIf(sys.version_info.minor > 10, "Not supported for python > 3.10")
    def test_compiled_model(self):
        from torch_em.trainer import DefaultTrainer

        trainer = DefaultTrainer(**self._get_kwargs(compile_model=True))
        trainer.fit(10)
        exp_model = trainer.model
        exp_data_shape = trainer.train_loader.dataset.raw.shape

        trainer2 = DefaultTrainer.from_checkpoint(
            os.path.join(self.checkpoint_folder, self.name),
            name="latest"
        )
        self.assertEqual(trainer.iteration, trainer2.iteration)
        self.assertEqual(trainer2.train_loader.dataset.raw.shape, exp_data_shape)
        self.assertTrue(torch_em.util.model_is_equal(exp_model, trainer2.model))

        trainer2.fit(10)
        self.assertEqual(trainer2.iteration, 20)


if __name__ == "__main__":
    unittest.main()
