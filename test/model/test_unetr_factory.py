import unittest
import tempfile
from pathlib import Path
from functools import partial
from unittest.mock import patch

import torch

from torch_em.model import vit
from torch_em.model.util import VIT_CONFIGS
from torch_em.model.unetr import get_unetr_model
from torch_em.transform.raw import get_raw_transform_for_pretrained_vit


ENCODER_CASES = {
    backbone: (f"_{config.dependency}_import_success", config.models) for backbone, config in VIT_CONFIGS.items()
}


class CustomEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = 32
        self.embed_dim = 8
        self.in_chans = 3
        self.projection = torch.nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.projection(x)


class TestUnetrFactory(unittest.TestCase):
    def _get_image_size(self, backbone, name):
        if backbone == "sam2":
            return 64
        if backbone in ("dinov2", "sam3") or name == "vit_h_14":
            return 28
        if backbone in ("mae", "scalemae") and name == "vit_h":
            return 28
        return 64 if name.endswith("_32") else 32

    def _get_kwargs(self, backbone):
        return {"pretrained": False} if backbone == "torchvision" else {}

    def _check_forward(self, model, ndim, img_size, clip_max):
        shape = (1, 1, img_size, img_size // 2) if ndim == 2 else (1, 1, 2, img_size, img_size // 2)
        x = torch.rand(shape) * clip_max
        with torch.inference_mode():
            y = model.eval()(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_backbones(self):
        for backbone, (flag, names) in ENCODER_CASES.items():
            names = names if backbone == "sam2" else names[:1]
            for name in names:
                img_size = self._get_image_size(backbone, name)
                for ndim in (2, 3):
                    with self.subTest(backbone=backbone, encoder=name, ndim=ndim):
                        if not getattr(vit, flag):
                            self.skipTest(f"The {backbone} backend is unavailable.")
                        kwargs = self._get_kwargs(backbone)
                        encoder = name
                        if ndim == 3:
                            encoder = vit.get_vision_transformer(backbone, name, img_size=img_size, **kwargs)
                        model = get_unetr_model(
                            ndim=ndim, backbone=backbone, model_type=encoder, img_size=img_size,
                            initial_features=4, **kwargs
                        )
                        _, clip_max = get_raw_transform_for_pretrained_vit(backbone)
                        self._check_forward(model, ndim, img_size, clip_max)
                        del model, encoder

    def test_variant_shapes(self):
        for backbone, (flag, names) in ENCODER_CASES.items():
            if backbone in ("sam2", "sam3"):
                continue
            for name in names:
                for ndim in (2, 3):
                    with self.subTest(backbone=backbone, encoder=name, ndim=ndim):
                        if not getattr(vit, flag):
                            self.skipTest(f"The {backbone} backend is unavailable.")
                        img_size = self._get_image_size(backbone, name)
                        # DINOv2 reads stochastic-depth rates as Python scalars.
                        with patch("torch.linspace", partial(torch.linspace, device="cpu")), torch.device("meta"):
                            model = get_unetr_model(
                                ndim=ndim, backbone=backbone, model_type=name, img_size=img_size,
                                initial_features=4, perform_range_checks=False, **self._get_kwargs(backbone)
                            ).eval()
                            if backbone == "scalemae":
                                expected_depth = {"vit_b": 12, "vit_l": 24, "vit_h": 32}[name]
                                self.assertEqual(len(model.encoder.blocks), expected_depth)
                                patch_size = 14 if name == "vit_h" else 16
                                self.assertEqual(model.encoder.patch_embed.patch_size, (patch_size, patch_size))
                            else:
                                shape = (1, 1, img_size, img_size) if ndim == 2 else (1, 1, 2, img_size, img_size)
                                with torch.inference_mode():
                                    y = model(torch.rand(shape))
                                self.assertEqual(y.shape, shape)
                                self.assertTrue(y.is_meta)
                        del model

    @unittest.skipUnless(vit._timm_import_success, "Needs timm")
    def test_backward(self):
        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                model = get_unetr_model(ndim, "mae", "vit_b", img_size=32, initial_features=4)
                shape = (1, 1, 32, 32) if ndim == 2 else (1, 1, 2, 32, 32)
                y = model(torch.rand(shape))
                y.square().mean().backward()
                for parameter in (model.encoder.patch_embed.proj.weight, model.out_conv.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_invalid_arguments(self):
        for ndim in (0, 1, 4):
            with self.subTest(ndim=ndim), self.assertRaisesRegex(ValueError, "ndim"):
                get_unetr_model(ndim, "sam", "vit_b")
        with self.assertRaisesRegex(ValueError, "init_decoder"):
            get_unetr_model(3, "sam", "vit_b", init_decoder=lambda model: model)
        with self.assertRaisesRegex(ValueError, "Unsupported backbone"):
            get_unetr_model(2, "unknown", "vit_b")
        with self.assertRaisesRegex(TypeError, "model_type"):
            get_unetr_model(2, "sam", None)

    def test_encoder_checkpoints(self):
        prefixes = {
            "sam": "image_encoder.", "sam2": "image_encoder.", "sam3": "detector.backbone.vision_backbone.trunk.",
            "cellpose_sam": "encoder.", "mae": "", "scalemae": "", "dinov2": "", "dinov3": "", "torchvision": "",
        }
        for backbone, (flag, names) in ENCODER_CASES.items():
            with self.subTest(backbone=backbone):
                if not getattr(vit, flag):
                    self.skipTest(f"The {backbone} backend is unavailable.")
                name = names[0]
                img_size = self._get_image_size(backbone, name)
                kwargs = self._get_kwargs(backbone)
                source = vit.get_vision_transformer(backbone, name, img_size=img_size, **kwargs)
                state = source.state_dict()
                del source
                prefixed = {f"module.{prefixes[backbone]}{key}": value for key, value in state.items()}
                if backbone in ("sam", "sam2", "sam3", "cellpose_sam"):
                    prefixed["module.mask_decoder.weight"] = torch.ones(1)
                elif backbone in ("mae", "scalemae"):
                    prefixed["module.decoder.weight"] = torch.ones(1)
                    prefixed["module.mask_token"] = torch.ones(1)
                elif backbone == "torchvision":
                    prefixed["module.heads.head.weight"] = torch.ones(1)
                checkpoint = {"model": prefixed}
                model = get_unetr_model(
                    2, backbone, name, encoder_checkpoint=checkpoint, img_size=img_size, initial_features=4, **kwargs
                )
                for key, parameter in model.encoder.state_dict().items():
                    torch.testing.assert_close(parameter, state[key])
                _, clip_max = get_raw_transform_for_pretrained_vit(backbone)
                self._check_forward(model, 2, img_size, clip_max)
                self.assertIn("model", checkpoint)
                del model, state, prefixed, checkpoint

    @unittest.skipUnless(vit._timm_import_success, "Needs timm")
    def test_checkpoint_paths(self):
        for backbone in ("mae", "scalemae"):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as tmpdir:
                encoder = vit.get_vision_transformer(backbone, "vit_b", img_size=32)
                expected = encoder.patch_embed.proj.weight.detach().clone()
                path = Path(tmpdir) / "encoder.pt"
                torch.save({"model": encoder.state_dict()}, path)
                del encoder
                model = get_unetr_model(3, backbone, "vit_b", encoder_checkpoint=path, img_size=32, initial_features=4)
                torch.testing.assert_close(model.encoder.patch_embed.proj.weight, expected)
                self._check_forward(model, 3, 32, 1)

    @unittest.skipUnless(vit._torchvision_import_success, "Needs torchvision")
    def test_prebuilt_encoder_checkpoint(self):
        encoder = vit.get_vision_transformer("torchvision", "vit_b_32", img_size=64, pretrained=False)
        checkpoint = encoder.state_dict()
        model = get_unetr_model(3, "torchvision", encoder, encoder_checkpoint=checkpoint, initial_features=4)
        self.assertIs(model.encoder, encoder)
        self._check_forward(model, 3, 64, 1)

    def test_custom_encoder(self):
        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                encoder = CustomEncoder()
                model = get_unetr_model(ndim, "custom", encoder, out_channels=2, initial_features=4)
                shape = (2, 1, 32, 16) if ndim == 2 else (2, 1, 2, 32, 16)
                with torch.inference_mode():
                    y = model.eval()(torch.rand(shape))
                self.assertEqual(y.shape, (2, 2, *shape[2:]))
                self.assertTrue(torch.isfinite(y).all())

    @unittest.skipUnless(vit._sam_import_success, "Needs segment_anything")
    def test_default_3d_encoder(self):
        from torch_em.model.unetr import UNETR3D

        model = UNETR3D(img_size=32, initial_features=4)
        self._check_forward(model, 3, 32, 1)

    @unittest.skipUnless(vit._timm_import_success, "Needs timm")
    def test_loaded_model_reconstruction(self):
        from torch_em.util import get_constructor_arguments

        for backbone in ("mae", "scalemae"):
            for ndim in (2, 3):
                with self.subTest(backbone=backbone, ndim=ndim):
                    encoder = vit.get_vision_transformer(backbone, "vit_b", img_size=32)
                    checkpoint = {"model": encoder.state_dict()}
                    del encoder
                    model = get_unetr_model(
                        ndim, backbone, "vit_b", encoder_checkpoint=checkpoint, img_size=32, initial_features=4
                    ).eval()
                    del checkpoint
                    restored = type(model)(**get_constructor_arguments(model)).eval()
                    restored.load_state_dict(model.state_dict())
                    shape = (1, 1, 32, 32) if ndim == 2 else (1, 1, 2, 32, 32)
                    x = torch.rand(shape)
                    with torch.inference_mode():
                        torch.testing.assert_close(restored(x), model(x))

    def test_encoder_default_sizes(self):
        for backbone, (flag, names) in ENCODER_CASES.items():
            if backbone in ("sam2", "sam3"):
                continue
            with self.subTest(backbone=backbone):
                if not getattr(vit, flag):
                    self.skipTest(f"The {backbone} backend is unavailable.")
                with patch("torch.linspace", partial(torch.linspace, device="cpu")), torch.device("meta"):
                    encoder = vit.get_vision_transformer(backbone, names[0], **self._get_kwargs(backbone))
                self.assertEqual(encoder.img_size, VIT_CONFIGS[backbone].img_size)

    @unittest.skipUnless(vit._dinov2_import_success, "Needs dinov2")
    def test_dinov2_swiglu_on_cpu(self):
        from dinov2.layers.swiglu_ffn import SwiGLUFFNFused

        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                encoder = vit.ViT_DINOv2(
                    img_size=28, patch_size=14, embed_dim=24, depth=4, num_heads=4, mlp_ratio=4,
                    block_fn=vit._get_dinov2_block, ffn_layer="swiglu", block_chunks=0
                )
                upstream_mlp = SwiGLUFFNFused(in_features=24, hidden_features=96)
                encoder.blocks[0].mlp.load_state_dict(upstream_mlp.state_dict())
                model = get_unetr_model(ndim, "dinov2", encoder, initial_features=4)
                self._check_forward(model, ndim, 28, 1)

    @unittest.skipUnless(vit._sam3_import_success, "Needs sam3")
    def test_sam3_backward(self):
        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                encoder = vit.ViT_Sam3(
                    img_size=28, pretrain_img_size=28, patch_size=14, embed_dim=32, depth=4, num_heads=4,
                    global_att_blocks=(0, 1, 2, 3), window_size=2, norm_layer="LayerNorm",
                    rel_pos_blocks=(), retain_cls_token=False
                )
                state = encoder.state_dict()
                model = get_unetr_model(ndim, "sam3", encoder, encoder_checkpoint=state, initial_features=4)
                shape = (1, 1, 28, 28) if ndim == 2 else (1, 1, 2, 28, 28)
                y = model(torch.rand(shape))
                self.assertEqual(y.shape, shape)
                self.assertEqual(y.dtype, torch.float32)
                y.square().mean().backward()
                for parameter in (encoder.blocks[0].mlp.fc1.weight, model.out_conv.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    @unittest.skipUnless(vit._sam3_import_success, "Needs sam3")
    def test_sam3_checkpoint_input_size(self):
        kwargs = {
            "pretrain_img_size": 28, "patch_size": 14, "embed_dim": 32, "depth": 4, "num_heads": 4,
            "global_att_blocks": (0, 1, 2, 3), "window_size": 2, "norm_layer": "LayerNorm",
            "rel_pos_blocks": (), "retain_cls_token": False, "use_rope": True, "use_interp_rope": True,
        }
        source = vit.ViT_Sam3(img_size=56, **kwargs)
        checkpoint = source.state_dict()
        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                encoder = vit.ViT_Sam3(img_size=28, **kwargs)
                expected_freqs = encoder.blocks[0].attn.freqs_cis.clone()
                model = get_unetr_model(ndim, "sam3", encoder, encoder_checkpoint=checkpoint, initial_features=4)
                torch.testing.assert_close(encoder.blocks[0].attn.freqs_cis, expected_freqs)
                torch.testing.assert_close(encoder.blocks[0].mlp.fc1.weight, source.blocks[0].mlp.fc1.weight)
                self._check_forward(model, ndim, 28, 1)


if __name__ == "__main__":
    unittest.main()
