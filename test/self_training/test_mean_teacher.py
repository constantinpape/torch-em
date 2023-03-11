import os
import unittest
from shutil import rmtree

import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.util.test import create_segmentation_test_data


def simple_pseudo_labeler(teacher, input_):
    pseudo_labels = teacher(input_)
    return pseudo_labels, None


def simple_unsupservised_loss(model, model_input, pseudo_labels, label_filter):
    assert label_filter is None
    pred = model(model_input)
    loss = torch_em.loss.dice_score(pred, pseudo_labels, invert=True)
    return loss


def simple_unsupervised_loss_and_metric(model, model_input, pseudo_labels, label_filter):
    assert label_filter is None
    pred = model(model_input)
    loss = torch_em.loss.dice_score(pred, pseudo_labels, invert=True)
    return loss, loss


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
    ):
        from torch_em.self_training import MeanTeacherTrainer

        model = UNet2d(in_channels=1, out_channels=1, initial_features=8, depth=3)
        optimizer = torch.optim.Adam(model.parameters())

        trainer = MeanTeacherTrainer(
            name="mt-test",
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            unsupervised_train_loader=unsupervised_train_loader,
            supervised_train_loader=supervised_train_loader,
            unsupervised_val_loader=unsupervised_val_loader,
            supervised_val_loader=supervised_val_loader,
            pseudo_labeler=simple_pseudo_labeler,
            unsupervised_loss=simple_unsupservised_loss,
            unsupervised_loss_and_metric=simple_unsupervised_loss_and_metric,
            supervised_loss=supervised_loss,
            supervised_loss_and_metric=supervised_loss_and_metric,
            logger=None,
            mixed_precision=False,
        )
        trainer.fit(53)
        self.assertTrue("./checkpoints/best.pt")
        self.assertTrue("./checkpoints/latest.pt")
        # TODO check that deserializing and continuing to train works

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

    def test_mean_teacher_unsupervised(self):
        unsupervised_train_loader = self.get_unsupervised_loader(n_samples=50)
        unsupervised_val_loader = self.get_unsupervised_loader(n_samples=4)
        self._test_mean_teacher(
            unsupervised_train_loader,
            unsupervised_val_loader=unsupervised_val_loader
        )


if __name__ == "__main__":
    unittest.main()
