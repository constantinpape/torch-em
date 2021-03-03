import os
import unittest
from shutil import rmtree

import torch

from torch_em.util.test import (create_image_collection_test_data,
                                create_segmentation_test_data)


# TODO test for instance segmentation training
# TODO all tests for 2d and 3d training
class TestSegmentation(unittest.TestCase):
    tmp_folder = './tmp'
    data_path = './tmp/data.h5'
    raw_key = 'raw'
    semantic_label_key = 'semantic_labels'
    n_images = 10

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        create_segmentation_test_data(self.data_path, self.raw_key, self.semantic_label_key,
                                      shape=(128,) * 3, chunks=(32,) * 3)
        create_image_collection_test_data(self.tmp_folder,
                                          n_images=self.n_images,
                                          min_shape=(256, 256),
                                          max_shape=(512, 512))

    def tearDown(self):

        def _remove(folder):
            try:
                rmtree(folder)
            except OSError:
                pass

        _remove(self.tmp_folder)
        _remove('./logs')
        _remove('./checkpoints')

    def test_semantic_training_3d(self):
        from torch_em.segmentation import (default_segmentation_loader,
                                           default_segmentation_trainer)
        from torch_em.transform import labels_to_binary
        from torch_em.data import SegmentationDataset
        from torch_em.model import UNet3d
        model = UNet3d(in_channels=1, out_channels=1, initial_features=8, depth=3)

        batch_size = 1
        patch_shape = (64,) * 3

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
                                                 n_samples=5)
        self.assertIsInstance(val_loader.dataset, SegmentationDataset)

        trainer = default_segmentation_trainer('test', model,
                                               train_loader, val_loader,
                                               mixed_precision=False,
                                               device=torch.device('cpu'),
                                               logger=None)
        train_iters = 51
        trainer.fit(train_iters)

        cp_path = './checkpoints/test/latest.pt'
        self.assertTrue(os.path.exists(cp_path))
        checkpoint = torch.load(cp_path)

        self.assertIn('optimizer_state', checkpoint)
        self.assertIn('model_state', checkpoint)
        self.assertEqual(checkpoint['iteration'], train_iters)
        self.assertEqual(checkpoint['epoch'], 2)

    def test_semantic_training_2d(self):
        from torch_em.segmentation import (default_segmentation_loader,
                                           default_segmentation_trainer)
        from torch_em.transform import labels_to_binary
        from torch_em.data import ImageCollectionDataset
        from torch_em.model import UNet2d
        model = UNet2d(in_channels=1, out_channels=1, initial_features=8, depth=3)

        batch_size = 1
        patch_shape = (256,) * 2

        label_trafo = labels_to_binary

        raw_paths = os.path.join(self.tmp_folder, 'images')
        label_paths = os.path.join(self.tmp_folder, 'labels')
        train_loader = default_segmentation_loader(raw_paths, '*.tif',
                                                   label_paths, '*.tif',
                                                   batch_size, patch_shape,
                                                   label_transform=label_trafo,
                                                   n_samples=25)
        self.assertIsInstance(train_loader.dataset, ImageCollectionDataset)
        val_loader = default_segmentation_loader(raw_paths, '*.tif',
                                                 label_paths, '*.tif',
                                                 batch_size, patch_shape,
                                                 label_transform=label_trafo,
                                                 n_samples=5)
        self.assertIsInstance(val_loader.dataset, ImageCollectionDataset)

        trainer = default_segmentation_trainer('test', model,
                                               train_loader, val_loader,
                                               mixed_precision=False,
                                               device=torch.device('cpu'),
                                               logger=None)
        train_iters = 51
        trainer.fit(train_iters)

        cp_path = './checkpoints/test/latest.pt'
        self.assertTrue(os.path.exists(cp_path))
        checkpoint = torch.load(cp_path)

        self.assertIn('optimizer_state', checkpoint)
        self.assertIn('model_state', checkpoint)
        self.assertEqual(checkpoint['iteration'], train_iters)
        self.assertEqual(checkpoint['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
