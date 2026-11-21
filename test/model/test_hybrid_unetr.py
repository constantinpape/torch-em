import json
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist

from torch_em import default_segmentation_trainer
from torch_em.trainer.hybrid_unetr_trainer import HybridUNETRDeepSupervisionTrainer
from torch_em.model.hybrid_unetr import HYBRID_DECODERS, HybridUNETR, ZAttention, ZResidual

try:
    import sam2
except ImportError:
    sam2 = None


class _Encoder(nn.Module):
    """A small multi-scale encoder, which follows the return convention of `torch_em.model.vit.ViT_Sam2`."""
    def __init__(self, img_size=64, channels=256):
        super().__init__()
        self.img_size = img_size
        self.conv4 = nn.Conv2d(3, channels, 4, stride=4)
        self.conv8 = nn.Conv2d(channels, channels, 2, stride=2)
        self.conv16 = nn.Conv2d(channels, channels, 2, stride=2)

    def forward(self, x):
        s4 = self.conv4(x)
        s8 = self.conv8(s4)
        s16 = self.conv16(s8)
        return s16, [s4, s8, s16]


# Small decoder configurations to keep the tests fast.
SMALL_DECODERS = {
    "hiera_u": {"widths": (48, 32, 16, 8, 8), "stem_widths": (8, 16)},
    "c3f2": {"widths": (64, 32, 16), "head_width": 16, "stem_widths": (8, 16)},
    "cagd_3d": {"widths": (16, 32, 80, 128), "stem_widths": (8, 16)},
}


class TestHybridUNETR(unittest.TestCase):
    def _get_model(self, decoder, out_channels=5, img_size=64, decoder_kwargs=None, **kwargs):
        torch.manual_seed(0)
        decoder_kwargs = {**SMALL_DECODERS[decoder], **(decoder_kwargs or {})}
        model = HybridUNETR(
            img_size=img_size, encoder=_Encoder(img_size), decoder=decoder, decoder_kwargs=decoder_kwargs,
            out_channels=out_channels, **kwargs
        )
        return model.eval()

    def _randomize_z_branches(self, module):
        with torch.no_grad():
            for mod in module.modules():
                if isinstance(mod, ZResidual):
                    mod.pointwise.weight.normal_()
                if isinstance(mod, ZAttention):
                    mod.proj.weight.normal_()

    def test_output_shapes(self):
        for decoder in SMALL_DECODERS:
            model = self._get_model(decoder, out_channels=4)
            for shape in [(2, 3, 1, 64, 64), (1, 3, 3, 64, 48), (1, 3, 2, 37, 50)]:
                with self.subTest(decoder=decoder, shape=shape), torch.no_grad():
                    y = model(torch.rand(*shape))
                    self.assertEqual(tuple(y.shape), (shape[0], 4, *shape[2:]))

    def test_group_norm_with_custom_stem_widths(self):
        x = torch.rand(1, 3, 48, 64)
        for decoder in ("hiera_u", "c3f2"):
            with self.subTest(decoder=decoder), torch.no_grad():
                model = self._get_model(decoder, decoder_kwargs={"stem_widths": (12, 24)})
                self.assertEqual(model(x).shape, (1, model.out_channels, 48, 64))

    def test_image_equals_single_slice_volume(self):
        for decoder in SMALL_DECODERS:
            model = self._get_model(decoder)
            for shape in [(48, 64), (100, 128)]:  # Smaller and larger than the image size of the model.
                x = torch.rand(2, 3, *shape)
                with self.subTest(decoder=decoder, shape=shape), torch.no_grad():
                    y = model(x)
                    self.assertEqual(tuple(y.shape), (2, 5, *shape))
                    self.assertTrue(torch.equal(y, model(x[:, :, None])[:, :, 0]))

    def test_image_final_activation(self):
        model = self._get_model("hiera_u", final_activation=nn.Softmax2d())
        x = torch.rand(2, 3, 37, 50)

        with torch.no_grad():
            prediction = model(x)
            model.final_activation = nn.Softmax(dim=1)
            volume_prediction = model(x[:, :, None])

        torch.testing.assert_close(prediction, volume_prediction.squeeze(2))
        channel_sum = prediction.sum(dim=1)
        torch.testing.assert_close(channel_sum, torch.ones_like(channel_sum))

    def test_postprocess_uses_encoder_grid(self):
        reference = self._get_model("hiera_u", out_channels=1)
        model = HybridUNETR(
            img_size=128, encoder=_Encoder(64), decoder="hiera_u", decoder_kwargs=SMALL_DECODERS["hiera_u"],
        ).eval()
        model.load_state_dict(reference.state_dict())

        for shape in [(1, 3, 37, 50), (1, 3, 2, 37, 50)]:
            with self.subTest(shape=shape), torch.no_grad():
                x = torch.rand(*shape)
                torch.testing.assert_close(model(x), reference(x))

    def test_decoding_ignores_padding(self):
        # The same input is padded to 64 or 128. Both are cropped to 64 before decoding, so the outputs match.
        for decoder in SMALL_DECODERS:
            reference = self._get_model(decoder, resize_input=False)
            model = HybridUNETR(
                img_size=128, encoder=_Encoder(128), decoder=decoder, decoder_kwargs=SMALL_DECODERS[decoder],
                out_channels=reference.out_channels, resize_input=False,
            ).eval()
            self._randomize_z_branches(reference)  # So that the z context also has to match.
            model.load_state_dict(reference.state_dict())
            # Double precision, since float32 rounding depends on the memory layout of the cropped tensors.
            model, reference = model.double(), reference.double()

            for shape in [(1, 3, 40, 50), (1, 3, 2, 40, 50)]:
                with self.subTest(decoder=decoder, shape=shape), torch.no_grad():
                    x = torch.rand(*shape, dtype=torch.float64)
                    torch.testing.assert_close(model(x), reference(x), atol=1e-10, rtol=0)

    def test_input_size_without_resizing(self):
        model = HybridUNETR(
            img_size=128, encoder=_Encoder(64), decoder="hiera_u", decoder_kwargs=SMALL_DECODERS["hiera_u"],
            resize_input=False,
        ).eval()

        for shape in [(1, 3, 80, 64), (1, 3, 2, 64, 80)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "resize_input=False"):
                model(torch.rand(*shape))

        for shape in [(1, 3, 48, 64), (1, 3, 2, 64, 64)]:
            with self.subTest(shape=shape), torch.no_grad():
                prediction = model(torch.rand(*shape))
                self.assertEqual(tuple(prediction.shape), (shape[0], model.out_channels, *shape[2:]))

    def test_z_residual_kernel_size(self):
        for kernel_size in (-1, 0, 2):
            with self.subTest(kernel_size=kernel_size), self.assertRaisesRegex(ValueError, "positive odd kernel size"):
                ZResidual(channels=4, kernel_size=kernel_size)

        residual = ZResidual(channels=4, kernel_size=5)
        x = torch.randn(6, 4, 8, 8)  # Two volumes with three slices each, folded into the batch.
        with torch.no_grad():
            torch.testing.assert_close(residual(x, n_z=3), x)

    def test_volume_equals_stacked_slices_at_init(self):
        for name, kwargs in SMALL_DECODERS.items():
            torch.manual_seed(0)
            decoder = HYBRID_DECODERS[name](out_channels=5, **kwargs).double().eval()
            n_z, size = 3, 64
            features = [
                torch.randn(n_z, 256, size // stride, size // stride, dtype=torch.float64) for stride in (4, 8, 16)
            ]
            image = torch.randn(n_z, 3, size, size, dtype=torch.float64)
            with self.subTest(decoder=name), torch.no_grad():
                volume = decoder(features, image, n_z)
                slices = torch.cat([decoder([f[i:i + 1] for f in features], image[i:i + 1], 1) for i in range(n_z)])
                self.assertTrue(torch.allclose(volume, slices, atol=1e-12, rtol=0))

                # Once the z branches are trained, the slices interact.
                self._randomize_z_branches(decoder)
                volume = decoder(features, image, n_z)
                self.assertFalse(torch.allclose(volume, slices, atol=1e-6, rtol=0))

    def test_decoders_without_skip_projections(self):
        features = [torch.randn(2, c, 64 // s, 64 // s) for c, s in ((32, 4), (64, 8), (256, 16))]
        for name, kwargs in SMALL_DECODERS.items():
            with self.subTest(decoder=name), torch.no_grad():
                decoder = HYBRID_DECODERS[name](in_channels=(32, 64, 256), skip_projections=None, **kwargs)
                self.assertFalse(any(key.startswith("proj_") for key in decoder.state_dict()))
                self.assertEqual(tuple(decoder(features, torch.randn(2, 3, 64, 64), 2).shape), (2, 5, 64, 64))

    def test_state_and_config(self):
        with self.assertRaises(ValueError):
            HybridUNETR(img_size=64, encoder=_Encoder(64), decoder="invalid")

        for decoder in SMALL_DECODERS:
            with self.subTest(decoder=decoder):
                model = self._get_model(decoder, final_activation="Sigmoid")
                self._randomize_z_branches(model)
                state = model.state_dict()
                self.assertEqual({key.split(".")[0] for key in state}, {"encoder", "decoder"})

                x = torch.rand(1, 3, 2, 64, 64)
                with torch.no_grad():
                    expected = model(x)
                    features, input_shape = model.encode(x)
                    self.assertTrue(torch.equal(model.decode(features, input_shape, x.shape[2:]), expected))

                # Rebuild the model from the decoder config, after serialization.
                config = json.loads(json.dumps(model.decoder_config))
                self.assertEqual(set(config), {"decoder", "decoder_kwargs", "out_channels"})
                rebuilt = HybridUNETR(img_size=64, encoder=_Encoder(64), final_activation="Sigmoid", **config).eval()
                rebuilt.load_state_dict(state, strict=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(rebuilt(x), expected))

    def test_decoder_config_snapshot(self):
        decoder_kwargs = {**SMALL_DECODERS["hiera_u"], "widths": list(SMALL_DECODERS["hiera_u"]["widths"])}
        model = HybridUNETR(img_size=64, encoder=_Encoder(), decoder="hiera_u", decoder_kwargs=decoder_kwargs)
        expected_config = deepcopy(model.decoder_config)

        decoder_kwargs["widths"][0] = 64
        decoder_kwargs["stem_widths"] = (16, 32)
        exported_config = model.decoder_config
        exported_config["decoder_kwargs"]["widths"][-1] = 16

        self.assertEqual(model.decoder_config, expected_config)
        self.assertEqual(model.init_kwargs["decoder_kwargs"], expected_config["decoder_kwargs"])
        rebuilt = HybridUNETR(img_size=64, encoder=_Encoder(), **model.decoder_config)
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_deep_supervision(self):
        x = torch.rand(1, 3, 2, 48, 64)

        model = self._get_model("cagd_3d").train()
        self.assertIsInstance(model(x), torch.Tensor)

        model = self._get_model("cagd_3d", final_activation="Sigmoid", decoder_kwargs={"deep_supervision": True})
        with torch.no_grad():
            self.assertIsInstance(model(x), torch.Tensor)

        model.train()
        y, auxiliary_outputs = model(x)
        self.assertEqual(len(auxiliary_outputs), 3)
        for aux in auxiliary_outputs:
            self.assertEqual(aux.shape, y.shape)
            self.assertTrue(0 <= aux.min() and aux.max() <= 1)  # The activation is applied.
        (y.mean() + sum(aux.mean() for aux in auxiliary_outputs)).backward()
        self.assertTrue(all(head.weight.grad is not None for head in model.decoder.aux_heads))

        # Copying in training mode verifies that the model holds no auxiliary tensors from the last forward.
        deepcopy(model)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Needs the Gloo backend")
    def test_deep_supervision_ddp(self):
        model = self._get_model("cagd_3d", decoder_kwargs={"deep_supervision": True}).train()
        x = torch.rand(1, 3, 2, 48, 64)
        y = torch.rand(1, model.out_channels, 2, 48, 64)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y))
        criterion = nn.MSELoss()

        with tempfile.TemporaryDirectory() as tmpdir:
            init_method = (Path(tmpdir) / "rendezvous").as_uri()
            dist.init_process_group("gloo", init_method=init_method, rank=0, world_size=1)
            try:
                ddp_model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
                trainer = default_segmentation_trainer(
                    name="hybrid-unetr-ddp-test", model=ddp_model, train_loader=loader, val_loader=loader,
                    loss=criterion, device="cpu", logger=None, trainer_class=HybridUNETRDeepSupervisionTrainer,
                )
                outputs = []
                with ddp_model.register_forward_hook(lambda _module, _inputs, output: outputs.append(output)):
                    prediction, loss = trainer._forward_and_loss(x, y)
                returned_prediction, (aux16, aux8, aux4) = outputs[0]
                self.assertIs(prediction, returned_prediction)
                with torch.no_grad():
                    expected_loss = criterion(prediction, y) + (
                        0.4 * criterion(aux16, y) + 0.3 * criterion(aux8, y) + 0.2 * criterion(aux4, y)
                    )
                torch.testing.assert_close(loss, expected_loss)
                loss.backward()
                self.assertIsNotNone(prediction.grad)
                self.assertTrue(all(head.weight.grad is not None for head in model.decoder.aux_heads))

                ddp_model.eval()
                with torch.no_grad():
                    prediction, loss = trainer._forward_and_loss(x, y)
                    torch.testing.assert_close(loss, criterion(prediction, y))
            finally:
                dist.destroy_process_group()

    def test_init_skip_projections(self):
        model = self._get_model("hiera_u")
        conv_s0, conv_s1 = nn.Conv2d(256, 32, 1), nn.Conv2d(256, 64, 1)
        model.init_skip_projections(conv_s0, conv_s1)
        self.assertTrue(torch.equal(model.decoder.proj_s4.weight, conv_s0.weight))
        self.assertTrue(torch.equal(model.decoder.proj_s8.bias, conv_s1.bias))

    @unittest.skipIf(sam2 is None, "Needs sam2")
    def test_sam2_encoder(self):
        for decoder in SMALL_DECODERS:
            model = HybridUNETR(
                img_size=256, backbone="sam2", encoder="hvit_t", decoder=decoder, out_channels=3,
                decoder_kwargs=SMALL_DECODERS[decoder], use_sam_stats=True,
            ).eval()
            self.assertEqual({key.split(".")[0] for key in model.state_dict()}, {"encoder", "decoder"})
            for n_z in (1, 2):
                with self.subTest(decoder=decoder, n_z=n_z), torch.no_grad():
                    x = torch.rand(1, 3, n_z, 256, 256)
                    self.assertEqual(tuple(model(x).shape), (1, 3, n_z, 256, 256))


if __name__ == "__main__":
    unittest.main()
