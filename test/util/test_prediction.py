import unittest

import numpy as np


class TestPrediction(unittest.TestCase):
    def test_predict_with_halo_2d(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_predict_with_halo_3d(self):
        from torch_em.model import UNet3d
        from torch_em.util.prediction import predict_with_halo
        model = UNet3d(in_channels=1, out_channels=3, initial_features=8, depth=3)

        shape = (128,) * 3
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(32, 32, 32), halo=(8, 8, 8))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_prediction_with_halo_multiple_outputs(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        outputs = [
            (np.zeros(shape, dtype="float32"), np.s_[0]),
            (np.zeros((2,) + shape, dtype="float32"), np.s_[1:3])
        ]
        predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16), output=outputs)

        self.assertEqual(outputs[0][0].shape, shape)
        self.assertFalse(np.allclose(outputs[0][0], 0))

        self.assertEqual(outputs[1][0].shape, (2,) + shape)
        self.assertFalse(np.allclose(outputs[1][0], 0))

    def test_predict_with_halo_channels(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo

        model = UNet2d(in_channels=2, out_channels=3, initial_features=8, depth=3)
        shape = (2, 512, 512)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(64, 64), halo=(8, 8), with_channels=True)
        expected_shape = (3,) + shape[1:]
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_pipelined_2d(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo_pipelined(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_pipelined_3d(self):
        from torch_em.model import UNet3d
        from torch_em.util.prediction import predict_with_halo_pipelined
        model = UNet3d(in_channels=1, out_channels=3, initial_features=8, depth=3)

        shape = (128,) * 3
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo_pipelined(data, model, gpu_ids=["cpu"], block_shape=(32, 32, 32), halo=(8, 8, 8))
        expected_shape = (3,) + shape
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_pipelined_multiple_outputs(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        outputs = [
            (np.zeros(shape, dtype="float32"), np.s_[0]),
            (np.zeros((2,) + shape, dtype="float32"), np.s_[1:3])
        ]
        predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16), output=outputs, batch_size=2
        )

        self.assertEqual(outputs[0][0].shape, shape)
        self.assertFalse(np.allclose(outputs[0][0], 0))

        self.assertEqual(outputs[1][0].shape, (2,) + shape)
        self.assertFalse(np.allclose(outputs[1][0], 0))

    def test_pipelined_with_channels(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=2, out_channels=3, initial_features=8, depth=3)
        shape = (2, 512, 512)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(64, 64), halo=(8, 8), with_channels=True, batch_size=2
        )
        expected_shape = (3,) + shape[1:]
        self.assertEqual(out.shape, expected_shape)
        self.assertFalse(np.allclose(out, 0))

    def test_pipelined_batch_size(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)  # -> 16 blocks of (256, 256)
        data = np.random.rand(*shape).astype("float32")

        # exercise batch sizes that do and do not divide the number of blocks
        for batch_size in (1, 2, 3, 5):
            out = predict_with_halo_pipelined(
                data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16), batch_size=batch_size
            )
            self.assertEqual(out.shape, (3,) + shape)
            self.assertFalse(np.allclose(out, 0))

    def test_pipelined_num_workers(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        out = predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16),
            batch_size=2, num_prefetch_workers=4, num_write_workers=2, queue_size=8,
        )
        self.assertEqual(out.shape, (3,) + shape)
        self.assertFalse(np.allclose(out, 0))

    def test_pipelined_mask(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (512, 512)
        data = np.random.rand(*shape).astype("float32")

        # partial mask: only the top-left quadrant is active
        mask = np.zeros(shape, dtype="bool")
        mask[:256, :256] = True
        out = predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16), mask=mask, batch_size=2
        )
        self.assertEqual(out.shape, (3,) + shape)
        # masked-out region must be exactly zero, unmasked region must be non-zero
        self.assertTrue(np.allclose(out[:, ~mask], 0))
        self.assertFalse(np.allclose(out[:, mask], 0))

        # fully empty mask must terminate without hanging and return all zeros
        empty_mask = np.zeros(shape, dtype="bool")
        out = predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16), mask=empty_mask
        )
        self.assertTrue(np.allclose(out, 0))

    def test_pipelined_skip_block(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        shape = (512, 512)
        data = np.random.rand(*shape).astype("float32")

        # skip every other block based on its mean
        def skip_block(inp):
            return float(inp.mean()) > 0.5

        out = predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16),
            skip_block=skip_block, batch_size=2,
        )
        self.assertEqual(out.shape, (3,) + shape)

    def test_pipelined_grid_shift_raises(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo_pipelined

        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        data = np.random.rand(256, 256).astype("float32")
        with self.assertRaises(NotImplementedError):
            predict_with_halo_pipelined(
                data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16), grid_shift=(0.0, 0.25)
            )

    def test_concurrent_write_safe_helper(self):
        from torch_em.util.prediction import _concurrent_write_safe

        block_shape = (128, 128)
        start = (0, 0)

        # numpy is always safe
        self.assertTrue(_concurrent_write_safe(np.zeros((3, 512, 512), dtype="float32"), block_shape, start))

        try:
            import zarr
        except ImportError:
            zarr = None
        if zarr is not None:
            # chunk-aligned zarr (with channel axis) -> safe
            aligned = zarr.zeros(shape=(3, 512, 512), chunks=(3, 64, 64), dtype="float32")
            self.assertTrue(_concurrent_write_safe(aligned, block_shape, start))
            # chunk-aligned, no channel axis -> safe
            aligned_nc = zarr.zeros(shape=(512, 512), chunks=(128, 128), dtype="float32")
            self.assertTrue(_concurrent_write_safe(aligned_nc, block_shape, start))
            # misaligned chunks (block_shape not a multiple of chunk) -> unsafe
            misaligned = zarr.zeros(shape=(3, 512, 512), chunks=(3, 100, 100), dtype="float32")
            self.assertFalse(_concurrent_write_safe(misaligned, block_shape, start))
            # chunk larger than block (a chunk would be shared by several blocks) -> unsafe
            big_chunk = zarr.zeros(shape=(3, 512, 512), chunks=(3, 256, 256), dtype="float32")
            self.assertFalse(_concurrent_write_safe(big_chunk, block_shape, start))
            # aligned chunks but a roi start that is not chunk-aligned -> unsafe
            self.assertFalse(_concurrent_write_safe(aligned, block_shape, (32, 0)))

        try:
            import h5py
        except ImportError:
            h5py = None
        if h5py is not None:
            f = h5py.File("test_concurrent_write_safe.h5", "w", driver="core", backing_store=False)
            try:
                ds = f.create_dataset("pred", shape=(3, 512, 512), dtype="float32", chunks=(3, 64, 64))
                # hdf5 is never safe for concurrent writes, even with aligned chunks
                self.assertFalse(_concurrent_write_safe(ds, block_shape, start))
            finally:
                f.close()

    def test_pipelined_zarr_aligned_multiwriter(self):
        import warnings
        import torch
        try:
            import zarr
        except ImportError:
            self.skipTest("zarr is not available")
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (512, 512)
        data = np.random.rand(*shape).astype("float32")

        ref = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16))

        out = zarr.zeros(shape=(3,) + shape, chunks=(3, 64, 64), dtype="float32")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predict_with_halo_pipelined(
                data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16),
                output=out, num_write_workers=2,
            )
        # chunk-aligned zarr must NOT fall back to a single writer
        self.assertFalse(any("falling back" in str(w.message) for w in caught))
        np.testing.assert_allclose(np.array(out), ref, rtol=1e-5, atol=1e-6)

    def test_pipelined_zarr_misaligned_warns(self):
        import torch
        try:
            import zarr
        except ImportError:
            self.skipTest("zarr is not available")
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (512, 512)
        data = np.random.rand(*shape).astype("float32")

        ref = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16))

        out = zarr.zeros(shape=(3,) + shape, chunks=(3, 100, 100), dtype="float32")
        with self.assertWarns(UserWarning):
            predict_with_halo_pipelined(
                data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16),
                output=out, num_write_workers=2,
            )
        # output is still written correctly (clamped to a single writer)
        np.testing.assert_allclose(np.array(out), ref, rtol=1e-5, atol=1e-6)

    def test_pipelined_hdf5_warns(self):
        import torch
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not available")
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (512, 512)
        data = np.random.rand(*shape).astype("float32")

        ref = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16))

        f = h5py.File("test_pipelined_hdf5.h5", "w", driver="core", backing_store=False)
        try:
            out = f.create_dataset("pred", shape=(3,) + shape, dtype="float32", chunks=(3, 64, 64))
            with self.assertWarns(UserWarning):
                predict_with_halo_pipelined(
                    data, model, gpu_ids=["cpu"], block_shape=(128, 128), halo=(16, 16),
                    output=out, num_write_workers=2,
                )
            np.testing.assert_allclose(out[:], ref, rtol=1e-5, atol=1e-6)
        finally:
            f.close()

    def test_pipelined_consistency_2d(self):
        import torch
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        ref = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16))
        for batch_size in (1, 2, 4):
            got = predict_with_halo_pipelined(
                data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16),
                batch_size=batch_size, num_prefetch_workers=3,
            )
            # batch_size == 1 is a batch-of-one forward pass -> bit-identical to the reference;
            # batch_size > 1 differs only at floating-point ULP level (batched conv accumulation).
            atol = 1e-6 if batch_size == 1 else 1e-4
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=atol)

    def test_pipelined_consistency_3d(self):
        import torch
        from torch_em.model import UNet3d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet3d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (128,) * 3
        data = np.random.rand(*shape).astype("float32")

        ref = predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(32, 32, 32), halo=(8, 8, 8))
        for batch_size in (1, 3):
            got = predict_with_halo_pipelined(
                data, model, gpu_ids=["cpu"], block_shape=(32, 32, 32), halo=(8, 8, 8), batch_size=batch_size
            )
            # batch_size == 1 is a batch-of-one forward pass -> bit-identical to the reference;
            # batch_size > 1 differs only at floating-point ULP level (batched conv accumulation).
            atol = 1e-6 if batch_size == 1 else 1e-4
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=atol)

    def test_pipelined_consistency_with_channels(self):
        import torch
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet2d(in_channels=2, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (2, 512, 512)
        data = np.random.rand(*shape).astype("float32")

        ref = predict_with_halo(
            data, model, gpu_ids=["cpu"], block_shape=(64, 64), halo=(8, 8), with_channels=True
        )
        got = predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(64, 64), halo=(8, 8), with_channels=True, batch_size=4
        )
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_pipelined_consistency_multiple_outputs(self):
        import torch
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_halo, predict_with_halo_pipelined

        torch.manual_seed(0)
        model = UNet2d(in_channels=1, out_channels=3, initial_features=8, depth=3)
        model.eval()
        shape = (1024, 1024)
        data = np.random.rand(*shape).astype("float32")

        def make_outputs():
            return [
                (np.zeros(shape, dtype="float32"), np.s_[0]),
                (np.zeros((2,) + shape, dtype="float32"), np.s_[1:3]),
            ]

        ref = make_outputs()
        predict_with_halo(data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16), output=ref)

        got = make_outputs()
        predict_with_halo_pipelined(
            data, model, gpu_ids=["cpu"], block_shape=(256, 256), halo=(16, 16), output=got, batch_size=4
        )

        for (ref_arr, _), (got_arr, _) in zip(ref, got):
            np.testing.assert_allclose(got_arr, ref_arr, rtol=1e-4, atol=1e-4)

    def test_predict_with_padding(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_padding

        model = UNet2d(in_channels=1, out_channels=3, initial_features=4, depth=3)
        shapes = [(128, 128), (133, 33), (64, 49), (27, 97)]
        for shape in shapes:
            input_ = np.random.rand(*shape).astype("float32")
            out = predict_with_padding(model, input_, min_divisible=(8, 8), device="cpu")
            self.assertEqual(out.shape[2:], shape)

    def test_predict_with_padding_and_channels(self):
        from torch_em.model import UNet2d
        from torch_em.util.prediction import predict_with_padding

        model = UNet2d(in_channels=3, out_channels=3, initial_features=4, depth=3)
        shapes = [(3, 128, 128), (3, 133, 33), (3, 64, 49), (3, 27, 97)]
        for shape in shapes:
            input_ = np.random.rand(*shape).astype("float32")
            out = predict_with_padding(model, input_, min_divisible=(8, 8), device="cpu", with_channels=True)
            self.assertEqual(out.shape[1:], shape)


if __name__ == "__main__":
    unittest.main()
