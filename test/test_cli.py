import os
import unittest
from shutil import rmtree
from subprocess import run

import imageio
import h5py
from torch_em.util.test import create_segmentation_test_data


class TestCLI(unittest.TestCase):
    tmp_folder = "./tmp_data"
    data_path = "./tmp_data/data.h5"
    image_path = "./tmp_data/input.tif"
    raw_key = "raw"
    label_key = "labels"

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)
        rmtree("./checkpoints", ignore_errors=True)
        rmtree("./logs", ignore_errors=True)

    def _create_data(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        create_segmentation_test_data(
            self.data_path,
            self.raw_key,
            self.label_key,
            shape=(32, 64, 64),
            chunks=(32, 32, 32)
        )
        with h5py.File(self.data_path, "r") as f:
            data = f[self.raw_key][0]
        imageio.v2.imwrite(self.image_path, data)

    def _train_2d(self):
        run(["torch_em.train_2d_unet",
             "-i", self.data_path, "-k", self.raw_key,
             "-l", self.data_path, "--training_label_key", self.label_key,
             "-b", "1", "-p", "1", "64", "64", "--name", "2d_unet",
             "--n_iterations", "35", "--label_mode", "foreground"])

    def test_train_2d_unet(self):
        self._create_data()
        self._train_2d()
        self.assertTrue(os.path.exists("./checkpoints/2d_unet/best.pt"))
        self.assertTrue(os.path.exists("./checkpoints/2d_unet/latest.pt"))

    def _train_3d(self):
        run(["torch_em.train_3d_unet",
             "-i", self.data_path, "-k", self.raw_key,
             "-l", self.data_path, "--training_label_key", self.label_key,
             "-b", "1", "-p", "16", "32", "32", "--name", "3d_unet",
             "--n_iterations", "10", "--label_mode", "foreground",
             "-s", "[[1,2,2],[2,2,2]]"])

    def test_train_3d_unet(self):
        self._create_data()
        self._train_3d()
        self.assertTrue(os.path.exists("./checkpoints/3d_unet/best.pt"))
        self.assertTrue(os.path.exists("./checkpoints/3d_unet/latest.pt"))

    def test_predict(self):
        self._create_data()
        self._train_2d()
        out_path = os.path.join(self.tmp_folder, "pred.h5")
        out_key = "pred"
        run(["torch_em.predict", "-c", "./checkpoints/2d_unet",
             "-i", self.image_path, "-o", out_path, "--output_key", out_key])
        self.assertTrue(os.path.exists(out_path))
        with h5py.File(out_path, "r") as f:
            self.assertIn(out_key, f)
            shape = f[out_key].shape
        expected_shape = (64, 64)
        self.assertEqual(shape, expected_shape)

    def test_predict_with_tiling(self):
        self._create_data()
        self._train_2d()
        out_path = os.path.join(self.tmp_folder, "pred.h5")
        out_key = "pred"
        run(["torch_em.predict_with_tiling", "-c", "./checkpoints/2d_unet",
             "-i", self.data_path, "-k", self.raw_key,
             "-o", out_path, "--output_key", out_key,
             "-b", "1", "64", "64"])
        with h5py.File(out_path, "r") as f:
            self.assertIn(out_key, f)
            shape = f[out_key].shape
        expected_shape = (1, 32, 64, 64)
        self.assertEqual(shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
