import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch


# TODO test for instance segmentation training
# TODO all tests for 2d and 3d training
class TestSegmentation(unittest.TestCase):
    tmp_folder = './tmp'
    data_path = './tmp/data.h5'
    raw_key = 'raw'
    semantic_labels_key = 'semantic_labels'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        shape = (128,) * 3
        chunks = (32,) * 3
        with h5py.File(self.data_path, 'a') as f:
            f.create_dataset(self.raw_key, data=np.random.rand(*shape), chunks=chunks)
            f.create_dataset(self.semantic_labels_key, data=np.random.randint(0, 4, size=shape), chunks=chunks)

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
        from torch_em.model import UNet3d
        model = UNet3d(in_channels=1, out_channels=1, initial_features=8)

        batch_size = 1
        patch_shape = (64,) * 3

        label_trafo = labels_to_binary

        train_loader = default_segmentation_loader(self.data_path, self.raw_key,
                                                   self.data_path, self.semantic_labels_key,
                                                   batch_size, patch_shape,
                                                   label_transform=label_trafo,
                                                   n_samples=25)
        val_loader = default_segmentation_loader(self.data_path, self.raw_key,
                                                 self.data_path, self.semantic_labels_key,
                                                 batch_size, patch_shape,
                                                 label_transform=label_trafo,
                                                 n_samples=5)
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
