import os
import unittest
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np
import torch_em
from torch_em.model import UNet2d


class TestShallow2DeepTraining(unittest.TestCase):
    tmp_folder = "./tmp"

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)
        rmtree("./checkpoints", ignore_errors=True)
        rmtree("./logs", ignore_errors=True)

    def _create_seg_dataset(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        path = os.path.join(self.tmp_folder, "data.h5")
        raw_key = "raw"
        label_key = "label"
        with h5py.File(path, "w") as f:
            f.create_dataset(raw_key, data=np.random.rand(128, 128).astype("float32"))
            f.create_dataset(label_key, data=(np.random.rand(128, 128) > 0.5).astype("uint8"))
        return path, raw_key, label_key

    def _create_collection_dataset(self):
        n_images = 4

        im_folder = os.path.join(self.tmp_folder, "images")
        os.makedirs(im_folder)
        im_paths = []
        for i in range(n_images):
            path = os.path.join(im_folder, f"{i}.png")
            imageio.imwrite(path, np.random.randint(0, 255, size=(96, 96)).astype("uint8"))
            im_paths.append(path)

        label_folder = os.path.join(self.tmp_folder, "labels")
        os.makedirs(label_folder)
        label_paths = []
        for i in range(n_images):
            path = os.path.join(label_folder, f"{i}.png")
            imageio.imwrite(path, (np.random.rand(96, 96) > 0.5).astype("uint8"))
            label_paths.append(path)

        return im_folder, label_folder

    def test_shallow2deep_training_seg_ds(self):
        from torch_em.shallow2deep import prepare_shallow2deep, get_shallow2deep_loader
        path, raw_key, label_key = self._create_seg_dataset()
        name = "s2d-seg"
        rf_folder = os.path.join(self.tmp_folder, "rfs")
        prepare_shallow2deep(path, raw_key, path, label_key,
                             patch_shape_min=(48, 48),
                             patch_shape_max=(96, 96),
                             n_forests=12, n_threads=6,
                             output_folder=rf_folder, ndim=2)
        rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
        loader = get_shallow2deep_loader(path, raw_key, path, label_key,
                                         rf_paths, batch_size=1, patch_shape=(64, 64),
                                         n_samples=20)
        net = UNet2d(
            in_channels=1, out_channels=1, initial_features=4, gain=2, depth=2,
            final_activation="Sigmoid"
        )
        trainer = torch_em.default_segmentation_trainer(name, net, loader, loader)
        trainer.fit(40)
        self.assertTrue(os.path.exists(os.path.join(
            "./checkpoints", name, "latest.pt"
        )))

    def test_shallow2deep_training_image_ds(self):
        from torch_em.shallow2deep import prepare_shallow2deep, get_shallow2deep_loader
        im_folder, label_folder = self._create_collection_dataset()
        name = "s2d-im"
        rf_folder = os.path.join(self.tmp_folder, "rfs")
        prepare_shallow2deep(im_folder, "*.png", label_folder, "*.png",
                             patch_shape_min=(48, 48),
                             patch_shape_max=(96, 96),
                             n_forests=12, n_threads=6,
                             output_folder=rf_folder, ndim=2,
                             is_seg_dataset=False)
        rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
        loader = get_shallow2deep_loader(im_folder, "*.png", label_folder, "*.png",
                                         rf_paths, batch_size=1, patch_shape=(64, 64),
                                         n_samples=20, is_seg_dataset=False)
        net = UNet2d(
            in_channels=1, out_channels=1, initial_features=4, gain=2, depth=2,
            final_activation="Sigmoid"
        )
        trainer = torch_em.default_segmentation_trainer(name, net, loader, loader)
        trainer.fit(40)
        self.assertTrue(os.path.exists(os.path.join(
            "./checkpoints", name, "latest.pt"
        )))


if __name__ == "__main__":
    unittest.main()
