import json
import unittest

import torch
import torch.nn as nn

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


class _SingleScaleEncoder(_Encoder):
    def forward(self, x):
        return super().forward(x)[0]


# Small decoder configurations to keep the tests fast.
SMALL_DECODERS = {
    "hiera_u": {"widths": (48, 32, 16, 8, 8), "stem_widths": (8, 16)},
    "c3f2": {"widths": (64, 32, 16), "head_width": 16, "stem_widths": (8, 16)},
    "cagd_3d": {"widths": (16, 32, 80, 128), "stem_widths": (8, 16)},
}


class TestHybridUNETR(unittest.TestCase):
    def _get_model(self, decoder, out_channels=5, img_size=64, **kwargs):
        from torch_em.model import HybridUNETR

        torch.manual_seed(0)
        model = HybridUNETR(
            img_size=img_size, encoder=_Encoder(img_size), decoder=decoder, decoder_kwargs=SMALL_DECODERS[decoder],
            out_channels=out_channels, **kwargs
        )
        return model.eval()

    def _randomize_z_branches(self, module):
        from torch_em.model.hybrid_unetr import ZAttention, ZResidual

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

    def test_image_equals_single_slice_volume(self):
        for decoder in SMALL_DECODERS:
            model = self._get_model(decoder)
            x = torch.rand(2, 3, 48, 64)
            with self.subTest(decoder=decoder), torch.no_grad():
                y = model(x)
                self.assertEqual(tuple(y.shape), (2, 5, 48, 64))
                self.assertTrue(torch.equal(y, model(x[:, :, None])[:, :, 0]))

    def test_volume_equals_stacked_slices_at_init(self):
        from torch_em.model.hybrid_unetr import HYBRID_DECODERS

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

    def test_z_attention(self):
        from torch_em.model.hybrid_unetr import ZAttention

        torch.manual_seed(0)
        attention = ZAttention(16, n_heads=2).double()
        x = torch.randn(2 * 4, 16, 5, 6, dtype=torch.float64)
        with torch.no_grad():
            self.assertTrue(torch.equal(attention(x, 4), x))  # The output projection is zero-initialized.
            attention.proj.weight.normal_()
            y = attention(x, 4)
            # Changing one slice changes the output of the other slices of the same volume,
            # but not of the other volume.
            x2 = x.clone()
            x2[1] += torch.randn_like(x2[1])  # Not a constant, which the LayerNorm would remove.
            y2 = attention(x2, 4)
        self.assertFalse(torch.allclose(y[0], y2[0]))
        self.assertTrue(torch.allclose(y[4:], y2[4:], atol=1e-12, rtol=0))

        with self.assertRaises(ValueError):
            ZAttention(10, n_heads=3)

    def test_default_decoders(self):
        from torch_em.model.hybrid_unetr import HYBRID_DECODERS

        # The parameter counts of the default configurations for five output channels, without the projections.
        expected = {"hiera_u": 1555445, "c3f2": 2149533, "cagd_3d": 8510816}
        n_projection_params = (256 * 32 + 32) + (256 * 64 + 64)
        for name, n_expected in expected.items():
            with self.subTest(decoder=name):
                decoder = HYBRID_DECODERS[name]()
                self.assertEqual(sum(p.numel() for p in decoder.parameters()), n_expected + n_projection_params)

                decoder = HYBRID_DECODERS[name](in_channels=(32, 64, 256), skip_projections=None)
                self.assertEqual(sum(p.numel() for p in decoder.parameters()), n_expected)
                self.assertFalse(any(key.startswith("proj_") for key in decoder.state_dict()))
                features = [torch.randn(2, c, 64 // s, 64 // s) for c, s in ((32, 4), (64, 8), (256, 16))]
                with torch.no_grad():
                    self.assertEqual(tuple(decoder(features, torch.randn(2, 3, 64, 64), 2).shape), (2, 5, 64, 64))

    def test_state_and_config(self):
        from torch_em.model import HybridUNETR

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
                    self.assertEqual(len(features["features"]), 3)
                    self.assertEqual(features["n_z"], 2)
                    self.assertTrue(torch.equal(model.decode(features, input_shape, x.shape[2:]), expected))

                # Rebuild the model from the decoder config, after serialization.
                config = json.loads(json.dumps(model.decoder_config))
                self.assertEqual(set(config), {"decoder", "decoder_kwargs", "out_channels"})
                rebuilt = HybridUNETR(img_size=64, encoder=_Encoder(64), final_activation="Sigmoid", **config).eval()
                rebuilt.load_state_dict(state, strict=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(rebuilt(x), expected))

                # Rebuild the model from the init kwargs.
                rebuilt = HybridUNETR(**model.init_kwargs).eval()
                rebuilt.load_state_dict(state, strict=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(rebuilt(x), expected))

    def test_deep_supervision(self):
        model = self._get_model("cagd_3d", final_activation="Sigmoid")
        x = torch.rand(1, 3, 2, 48, 64)

        with torch.no_grad():
            model(x)
        self.assertIsNone(model.auxiliary_outputs)  # Only in training mode.

        model.train()
        y = model(x)
        self.assertEqual(len(model.auxiliary_outputs), 3)
        for aux in model.auxiliary_outputs:
            self.assertEqual(aux.shape, y.shape)
            self.assertTrue(0 <= aux.min() and aux.max() <= 1)  # The activation is applied.
        # The auxiliary heads receive gradients from the auxiliary outputs.
        sum(aux.sum() for aux in model.auxiliary_outputs).backward()
        self.assertTrue(all(head.weight.grad is not None for head in model.decoder.aux_heads))

        # 2D inputs yield 2D auxiliary outputs.
        y = model(x[:, :, 0])
        self.assertEqual(model.auxiliary_outputs[0].shape, y.shape)

        # No auxiliary outputs without deep supervision and for the other decoders.
        from torch_em.model import HybridUNETR
        from torch_em.model.hybrid_unetr import CAGD3DDecoder

        decoder = CAGD3DDecoder(deep_supervision=False, **SMALL_DECODERS["cagd_3d"])
        self.assertFalse(any(key.startswith("aux_heads") for key in decoder.state_dict()))
        model = HybridUNETR(img_size=64, encoder=_Encoder(64), decoder=decoder).train()
        model(x)
        self.assertIsNone(model.auxiliary_outputs)
        model = self._get_model("hiera_u").train()
        model(x)
        self.assertIsNone(model.auxiliary_outputs)

    def test_init_skip_projections(self):
        model = self._get_model("hiera_u")
        conv_s0, conv_s1 = nn.Conv2d(256, 32, 1), nn.Conv2d(256, 64, 1)
        model.init_skip_projections(conv_s0, conv_s1)
        for projection, conv in ((model.decoder.proj_s4, conv_s0), (model.decoder.proj_s8, conv_s1)):
            self.assertTrue(torch.equal(projection.weight, conv.weight))
            self.assertTrue(torch.equal(projection.bias, conv.bias))
            self.assertIsNot(projection.weight, conv.weight)
        # The weights are copied, not shared.
        with torch.no_grad():
            conv_s0.weight.zero_()
        self.assertFalse(torch.equal(model.decoder.proj_s4.weight, conv_s0.weight))

    def test_init_from_sam2_checkpoint(self):
        from torch_em.model import HybridUNETR

        # A checkpoint with the layout of SAM2.
        torch.manual_seed(1)
        encoder, conv_s0, conv_s1 = _Encoder(64), nn.Conv2d(256, 32, 1), nn.Conv2d(256, 64, 1)
        state = {f"image_encoder.{key}": value for key, value in encoder.state_dict().items()}
        state.update({f"sam_mask_decoder.conv_s0.{key}": value for key, value in conv_s0.state_dict().items()})
        state.update({f"sam_mask_decoder.conv_s1.{key}": value for key, value in conv_s1.state_dict().items()})

        model = HybridUNETR(
            img_size=64, encoder=_Encoder(64), decoder="c3f2", decoder_kwargs=SMALL_DECODERS["c3f2"],
            encoder_checkpoint={"model": state},
        )
        self.assertTrue(torch.equal(model.encoder.conv4.weight, encoder.conv4.weight))
        self.assertTrue(torch.equal(model.decoder.proj_s4.weight, conv_s0.weight))
        self.assertTrue(torch.equal(model.decoder.proj_s8.bias, conv_s1.bias))

        # Projections with a different shape are not initialized from the checkpoint.
        model = HybridUNETR(
            img_size=64, encoder=_Encoder(64), decoder="c3f2", encoder_checkpoint={"model": state},
            decoder_kwargs={"skip_projections": (16, 64), **SMALL_DECODERS["c3f2"]},
        )
        self.assertEqual(model.decoder.proj_s4.weight.shape[0], 16)
        self.assertTrue(torch.equal(model.decoder.proj_s8.weight, conv_s1.weight))

    def test_invalid_arguments(self):
        from torch_em.model import HybridUNETR
        from torch_em.model.hybrid_unetr import HieraUDecoder

        with self.assertRaises(ValueError):
            HybridUNETR(img_size=64, encoder=_Encoder(64), decoder="invalid")
        with self.assertRaises(ValueError):
            HybridUNETR(img_size=64, encoder=_Encoder(64), decoder=HieraUDecoder(), decoder_kwargs={"widths": (8,)})

        model = HybridUNETR(img_size=64, encoder=_SingleScaleEncoder(64), decoder="hiera_u")
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 1, 64, 64))

        model = HybridUNETR(img_size=64, encoder=_Encoder(64), decoder=HieraUDecoder(skip_projections=None))
        self.assertEqual(model.out_channels, 5)
        with self.assertRaises(RuntimeError):
            model.decoder_config
        with self.assertRaises(RuntimeError):
            model.init_skip_projections(nn.Conv2d(256, 32, 1), nn.Conv2d(256, 64, 1))

    @unittest.skipIf(sam2 is None, "Needs sam2")
    def test_sam2_encoder(self):
        from torch_em.model import HybridUNETR

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
