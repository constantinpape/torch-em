import os
import unittest
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np


class TestPrepareShallow2Deep(unittest.TestCase):
    tmp_folder = "./tmp"
    rf_folder = "./tmp/rfs"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)

    def _create_seg_dataset(self):
        path = os.path.join(self.tmp_folder, "data.h5")
        raw_key = "raw"
        label_key = "label"
        with h5py.File(path, "w") as f:
            f.create_dataset(raw_key, data=np.random.rand(128, 128).astype("float32"))
            f.create_dataset(label_key, data=(np.random.rand(128, 128) > 0.5).astype("uint8"))
        return path, raw_key, label_key

    def test_prepare_shallow2deep_seg_dataset(self):
        from torch_em.shallow2deep import prepare_shallow2deep
        path, raw_key, label_key = self._create_seg_dataset()
        patch_shape_min = (48, 48)
        patch_shape_max = (64, 64)
        n_forests = 12
        n_threads = 6
        prepare_shallow2deep(
            path, raw_key, path, label_key, patch_shape_min, patch_shape_max,
            n_forests, n_threads, self.rf_folder, ndim=2, is_seg_dataset=True
        )
        self.assertTrue(os.path.exists(self.rf_folder))
        n_rfs = len(glob(os.path.join(self.rf_folder, "*.pkl")))
        self.assertEqual(n_rfs, n_forests)

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

    def test_prepare_shallow2deep_image_dataset(self):
        from torch_em.shallow2deep import prepare_shallow2deep
        im_folder, label_folder = self._create_collection_dataset()
        patch_shape_min = (48, 48)
        patch_shape_max = (64, 64)
        n_forests = 12
        n_threads = 6
        prepare_shallow2deep(
            im_folder, "*.png", label_folder, "*.png", patch_shape_min, patch_shape_max,
            n_forests, n_threads, self.rf_folder, ndim=2, is_seg_dataset=False
        )
        self.assertTrue(os.path.exists(self.rf_folder))
        n_rfs = len(glob(os.path.join(self.rf_folder, "*.pkl")))
        self.assertEqual(n_rfs, n_forests)

    def test_prepare_shallow2deep_advanced(self):
        from torch_em.shallow2deep import prepare_shallow2deep_advanced
        from torch_em.shallow2deep.prepare_shallow2deep import SAMPLING_STRATEGIES
        path, raw_key, label_key = self._create_seg_dataset()
        patch_shape_min = (48, 48)
        patch_shape_max = (64, 64)
        n_forests = 12
        n_threads = 6
        for sampling_strategy in SAMPLING_STRATEGIES:
            rf_folder = os.path.join(self.tmp_folder, f"rfs-{sampling_strategy}")
            prepare_shallow2deep_advanced(
                path, raw_key, path, label_key, patch_shape_min, patch_shape_max,
                n_forests, n_threads, rf_folder,
                forests_per_stage=4, sample_fraction_per_stage=0.10,
                ndim=2, is_seg_dataset=True
            )
            self.assertTrue(os.path.exists(rf_folder))
            n_rfs = len(glob(os.path.join(rf_folder, "*.pkl")))
            self.assertEqual(n_rfs, n_forests)

    def test_filter_machinery(self):
        from torch_em.shallow2deep.prepare_shallow2deep import (
            _get_filters, _apply_filters, _calculate_response,
        )

        # Default config: 5 filters x 4 sigmas. Per sigma there are 3 scalar filters (1 column each)
        # and 2 eigenvalue filters (ndim columns each), i.e. (3 + 2 * ndim) columns per sigma.
        for ndim, shape in [(2, (48, 48)), (3, (16, 24, 24))]:
            raw = np.random.rand(*shape).astype("float32")
            filters_and_sigmas = _get_filters(ndim, None)
            self.assertEqual(len(filters_and_sigmas), 20)

            features = _apply_filters(raw, filters_and_sigmas)
            expected_columns = 4 * (3 + 2 * ndim)
            self.assertEqual(features.shape, (raw.size, expected_columns))

            # The string-based path must resolve CamelCase names and handle the structure tensor.
            scalar = _calculate_response(raw, "gaussianSmoothing", 1.6)
            self.assertEqual(scalar.shape, shape)
            for sigma in (1.6, tuple([1.6] * ndim)):
                eigvals = _calculate_response(raw, "structureTensorEigenvalues", sigma)
                self.assertEqual(eigvals.shape, shape + (ndim,))


if __name__ == "__main__":
    unittest.main()
