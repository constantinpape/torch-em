import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch

from torch_em.util.test import (create_image_collection_test_data,
                                create_segmentation_test_data)


# TODO test for instance segmentation training
# TODO all tests for 2d and 3d training
class TestSegmentation(unittest.TestCase):
    tmp_folder = "./tmp"
    data_path = "./tmp/data.h5"
    data_with_channel_path = "./tmp/data_with_channel.h5"
    raw_key = "raw"
    semantic_label_key = "semantic_labels"
    n_images = 10

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        create_segmentation_test_data(self.data_path, self.raw_key, self.semantic_label_key,
                                      shape=(128,) * 3, chunks=(32,) * 3)
        create_image_collection_test_data(self.tmp_folder,
                                          n_images=self.n_images,
                                          min_shape=(256, 256),
                                          max_shape=(512, 512))
        shape_with_channel = (3, 256, 256)
        chunks = (1, 128, 128)
        with h5py.File(self.data_with_channel_path, "a") as f:
            f.create_dataset(self.raw_key, data=np.random.rand(*shape_with_channel), chunks=chunks)
            f.create_dataset(
                self.semantic_label_key, data=np.random.randint(0, 4, size=shape_with_channel[1:]), chunks=chunks[1:]
            )

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)
        rmtree("./logs", ignore_errors=True)
        rmtree("./checkpoints", ignore_errors=True)

    def _test_training(self, model_class, model_kwargs,
                       train_loader, val_loader, n_iterations):
        from torch_em.segmentation import default_segmentation_trainer
        model = model_class(**model_kwargs)
        trainer = default_segmentation_trainer("test", model,
                                               train_loader, val_loader,
                                               mixed_precision=False,
                                               device=torch.device("cpu"),
                                               logger=None, compile_model=False)
        trainer.fit(n_iterations)

        def _test_checkpoint(cp_path, check_progress):
            self.assertTrue(os.path.exists(cp_path))
            checkpoint = torch.load(cp_path, weights_only=False)

            self.assertIn("optimizer_state", checkpoint)
            self.assertIn("model_state", checkpoint)

            loaded_model = model_class(**model_kwargs)
            loaded_model.load_state_dict(checkpoint["model_state"])

            if check_progress:
                self.assertEqual(checkpoint["iteration"], n_iterations)
                self.assertEqual(checkpoint["epoch"], 2)

        _test_checkpoint("./checkpoints/test/latest.pt", True)

        # we might not have a best checkpoint, depending on the validation error for the random data
        best = "./checkpoints/test/best.pt"
        if os.path.exists(best):
            _test_checkpoint(best, False)

    def _test_semantic_training_3d(self, model_class, model_kwargs, patch_shape):
        from torch_em.segmentation import default_segmentation_loader
        from torch_em.transform import labels_to_binary
        from torch_em.data import SegmentationDataset

        batch_size = 1

        label_trafo = labels_to_binary

        train_loader = default_segmentation_loader(self.data_path, self.raw_key,
                                                   self.data_path, self.semantic_label_key,
                                                   batch_size, patch_shape,
                                                   label_transform=label_trafo,
                                                   n_samples=25)
        self.assertIsInstance(train_loader.dataset, SegmentationDataset)
        val_loader = default_segmentation_loader(self.data_path, self.raw_key,
                                                 self.data_path, self.semantic_label_key,
                                                 batch_size, patch_shape,
                                                 label_transform=label_trafo,
                                                 n_samples=2)
        self.assertIsInstance(val_loader.dataset, SegmentationDataset)

        self._test_training(model_class, model_kwargs, train_loader, val_loader, n_iterations=51)

    def test_semantic_training_3d(self):
        from torch_em.model import UNet3d
        model_kwargs = dict(in_channels=1, out_channels=1, initial_features=4, depth=3)
        patch_shape = (64,) * 3
        self._test_semantic_training_3d(UNet3d, model_kwargs, patch_shape)

    def test_semantic_training_anisotropic(self):
        from torch_em.model import AnisotropicUNet
        model_kwargs = dict(in_channels=1, out_channels=1, initial_features=4,
                            scale_factors=[[1, 2, 2],
                                           [2, 2, 2]])
        patch_shape = (32, 64, 64)
        self._test_semantic_training_3d(AnisotropicUNet, model_kwargs, patch_shape)

    def test_semantic_training_2d(self):
        from torch_em.segmentation import default_segmentation_loader
        from torch_em.transform import labels_to_binary
        from torch_em.data import ImageCollectionDataset
        from torch_em.model import UNet2d
        model_kwargs = dict(in_channels=1, out_channels=1, initial_features=8, depth=3)

        batch_size = 1
        patch_shape = (256,) * 2

        label_trafo = labels_to_binary

        raw_paths = os.path.join(self.tmp_folder, "images")
        label_paths = os.path.join(self.tmp_folder, "labels")
        train_loader = default_segmentation_loader(raw_paths, "*.tif",
                                                   label_paths, "*.tif",
                                                   batch_size, patch_shape,
                                                   label_transform=label_trafo,
                                                   n_samples=25)
        self.assertIsInstance(train_loader.dataset, ImageCollectionDataset)
        val_loader = default_segmentation_loader(raw_paths, "*.tif",
                                                 label_paths, "*.tif",
                                                 batch_size, patch_shape,
                                                 label_transform=label_trafo,
                                                 n_samples=5)
        self.assertIsInstance(val_loader.dataset, ImageCollectionDataset)

        self._test_training(UNet2d, model_kwargs, train_loader, val_loader, n_iterations=51)

    def test_semantic_training_with_channel(self):
        from torch_em.segmentation import default_segmentation_loader
        from torch_em.transform import labels_to_binary
        from torch_em.model import UNet2d

        batch_size = 1
        patch_shape = (256, 256)
        label_trafo = labels_to_binary
        train_loader = default_segmentation_loader(self.data_with_channel_path, self.raw_key,
                                                   self.data_with_channel_path, self.semantic_label_key,
                                                   batch_size, patch_shape,
                                                   label_transform=label_trafo,
                                                   ndim=2, with_channels=True,
                                                   n_samples=25)
        val_loader = default_segmentation_loader(self.data_with_channel_path, self.raw_key,
                                                 self.data_with_channel_path, self.semantic_label_key,
                                                 batch_size, patch_shape,
                                                 label_transform=label_trafo,
                                                 ndim=2, with_channels=True,
                                                 n_samples=5)

        model_kwargs = dict(in_channels=3, out_channels=1, initial_features=8, depth=3)
        self._test_training(UNet2d, model_kwargs, train_loader, val_loader, n_iterations=51)


if __name__ == "__main__":
    unittest.main()
