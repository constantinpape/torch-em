import os
import unittest
from shutil import rmtree

import torch
import torch_em
import torch_em.self_training as self_training
from torch_em.model import UNet2d
from torch_em.util.test import create_segmentation_test_data


class TestMeanTeacher(unittest.TestCase):
    tmp_folder = "./tmp"
    data_path = "./tmp/data.h5"
    raw_key = "raw"
    label_key = "labels"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        create_segmentation_test_data(self.data_path, self.raw_key, self.label_key, shape=(128,) * 3, chunks=(32,) * 3)

    def tearDown(self):

        def _remove(folder):
            try:
                rmtree(folder)
            except OSError:
                pass

        _remove(self.tmp_folder)
        _remove("./logs")
        _remove("./checkpoints")

    def _test_mean_teacher(
        self,
        unsupervised_train_loader,
        supervised_train_loader=None,
        unsupervised_val_loader=None,
        supervised_val_loader=None,
        supervised_loss=None,
        supervised_loss_and_metric=None,
        unsupervised_loss_and_metric=None,
    ):
        model = UNet2d(in_channels=1, out_channels=1, initial_features=8, depth=3)
        optimizer = torch.optim.Adam(model.parameters())

        name = "mt-test"
        trainer = self_training.MeanTeacherTrainer(
            name=name,
            model=model,
            optimizer=optimizer,
            pseudo_labeler=self_training.DefaultPseudoLabeler(),
            unsupervised_loss=self_training.DefaultSelfTrainingLoss(),
            unsupervised_loss_and_metric=unsupervised_loss_and_metric,
            unsupervised_train_loader=unsupervised_train_loader,
            supervised_train_loader=supervised_train_loader,
            unsupervised_val_loader=unsupervised_val_loader,
            supervised_val_loader=supervised_val_loader,
            supervised_loss=supervised_loss,
            supervised_loss_and_metric=supervised_loss_and_metric,
            mixed_precision=False,
            device=torch.device("cpu"),
            compile_model=False,
        )
        trainer.fit(53)
        self.assertTrue(os.path.exists(f"./checkpoints/{name}/best.pt"))
        self.assertTrue(os.path.exists(f"./checkpoints/{name}/latest.pt"))

        # make sure that the trainer can be deserialized from the checkpoint
        trainer2 = self_training.MeanTeacherTrainer.from_checkpoint(os.path.join("./checkpoints", name), name="latest")
        self.assertEqual(trainer.iteration, trainer2.iteration)
        self.assertTrue(torch_em.util.model_is_equal(trainer.model, trainer2.model))
        self.assertTrue(torch_em.util.model_is_equal(trainer.teacher, trainer2.teacher))
        self.assertEqual(len(trainer.unsupervised_train_loader), len(trainer2.unsupervised_train_loader))
        if supervised_train_loader is not None:
            self.assertEqual(len(trainer.supervised_train_loader), len(trainer2.supervised_train_loader))

        # and that it can be trained further
        trainer2.fit(10)
        self.assertEqual(trainer2.iteration, 63)

        # and that we can deserialize it with get_trainer
        trainer3 = torch_em.util.get_trainer(f"./checkpoints/{name}")
        self.assertEqual(trainer3.iterations, 63)

    def get_unsupervised_loader(self, n_samples):
        augmentations = (
            torch_em.transform.raw.GaussianBlur(),
            torch_em.transform.raw.GaussianBlur(),
        )
        ds = torch_em.data.RawDataset(
            raw_path=self.data_path,
            raw_key=self.raw_key,
            patch_shape=(1, 64, 64),
            n_samples=n_samples,
            ndim=2,
            augmentations=augmentations,
        )
        loader = torch_em.segmentation.get_data_loader(ds, batch_size=1, shuffle=True)
        return loader

    def get_supervised_loader(self, n_samples):
        ds = torch_em.data.SegmentationDataset(
            raw_path=self.data_path, raw_key=self.raw_key,
            label_path=self.data_path, label_key=self.label_key,
            patch_shape=(1, 64, 64), ndim=2,
            n_samples=n_samples,
        )
        loader = torch_em.segmentation.get_data_loader(ds, batch_size=1, shuffle=True)
        return loader

    def test_mean_teacher_unsupervised(self):
        unsupervised_train_loader = self.get_unsupervised_loader(n_samples=50)
        unsupervised_val_loader = self.get_unsupervised_loader(n_samples=4)
        self._test_mean_teacher(
            unsupervised_train_loader=unsupervised_train_loader,
            unsupervised_val_loader=unsupervised_val_loader,
            unsupervised_loss_and_metric=self_training.DefaultSelfTrainingLossAndMetric(),
        )

    def test_mean_teacher_semisupervised(self):
        unsupervised_train_loader = self.get_unsupervised_loader(n_samples=50)
        supervised_train_loader = self.get_supervised_loader(n_samples=51)
        supervised_val_loader = self.get_supervised_loader(n_samples=4)
        self._test_mean_teacher(
            unsupervised_train_loader=unsupervised_train_loader,
            supervised_train_loader=supervised_train_loader,
            supervised_val_loader=supervised_val_loader,
            supervised_loss=self_training.DefaultSelfTrainingLoss(),
            supervised_loss_and_metric=self_training.DefaultSelfTrainingLossAndMetric(),
        )


if __name__ == "__main__":
    unittest.main()
