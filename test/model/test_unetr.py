import unittest
import numpy as np
import torch

try:
    import segment_anything
except ImportError:
    segment_anything = None

try:
    import micro_sam
except ImportError:
    micro_sam = None


class TestVitRawTransform(unittest.TestCase):
    def _get_data(self):
        # Data in an arbitrary range [0, 1000] with some outliers, as in EM data.
        data = np.random.rand(64, 64).astype("float32") * 1000
        data[0, 0], data[0, 1] = -1e4, 1e4
        return data

    def test_raw_transform_ranges(self):
        from torch_em.model.unetr import get_vit_raw_transform

        for backbone, expected_max in [("sam", 255), ("sam2", 1), ("dinov2", 1), ("dinov3", 1)]:
            raw_transform, clip_max = get_vit_raw_transform(backbone)
            self.assertEqual(clip_max, expected_max)
            out = raw_transform(self._get_data())
            self.assertEqual(out.min(), 0)
            self.assertEqual(out.max(), expected_max)

    def test_no_backbone(self):
        from torch_em.model.unetr import get_vit_raw_transform

        self.assertEqual(get_vit_raw_transform(backbone=None), (None, None))

    def test_unsupported_backbone(self):
        from torch_em.model.unetr import get_vit_raw_transform

        with self.assertRaises(ValueError):
            get_vit_raw_transform(backbone="mae")


class TestUnetrNormalizationRange(unittest.TestCase):
    def test_input_normalization_range_check(self):
        from torch_em.model.unetr import _check_input_normalization_range

        _check_input_normalization_range(torch.tensor([0.0, 0.5, 1.0]), (0.0, 1.0))
        _check_input_normalization_range(torch.tensor([0.0, 128.0, 255.0]), (0.0, 255.0))
        _check_input_normalization_range(torch.tensor([0, 1], dtype=torch.uint8), (0.0, 255.0))

        with self.assertRaises(ValueError):
            _check_input_normalization_range(torch.tensor([0.0, 2.0]), (0.0, 1.0))

        with self.assertRaises(ValueError):
            _check_input_normalization_range(torch.tensor([0.0, float("nan")]), (0.0, 1.0))

        # SAM1: [0, 1]-scaled inputs must be rejected when expected range is [0, 255].
        with self.assertRaises(ValueError):
            _check_input_normalization_range(torch.tensor([0.0, 0.5, 1.0]), (0.0, 255.0), unit_scale_max=1.0)

        # SAM1: valid [0, 255]-scaled input should pass.
        _check_input_normalization_range(torch.tensor([0.0, 128.0, 255.0]), (0.0, 255.0), unit_scale_max=1.0)

        # Out-of-range input is silently accepted when perform_range_checks=False.
        from torch_em.model.unetr import preprocess_vit_inputs
        preprocess_vit_inputs(
            torch.ones(1, 3, 64, 64) * 2.0,
            use_dino_stats=True,
            perform_range_checks=False,
        )


@unittest.skipIf(segment_anything is None, "Needs segment_anything")
class TestUnetr(unittest.TestCase):
    def _test_net(self, net, shape):
        x = torch.rand(*shape, requires_grad=True)
        y = net(x)
        expected_shape = shape[:1] + (net.out_channels,) + shape[2:]
        self.assertEqual(y.shape, expected_shape)
        loss = y.sum()
        loss.backward()

    def test_unetr(self):
        from torch_em.model import UNETR

        model = UNETR()
        self._test_net(model, (1, 3, 512, 512))

    def test_unetr_no_resize(self):
        from torch_em.model import UNETR

        model = UNETR(resize_input=False)
        self._test_net(model, (1, 3, 512, 512))

    @unittest.skipIf(micro_sam is None, "Needs micro_sam")
    def test_unetr_from_sam(self):
        from torch_em.model import UNETR
        from micro_sam.util import models

        model_registry = models()
        checkpoint = model_registry.fetch("vit_b")

        model = UNETR(encoder_checkpoint=checkpoint, use_skip_connection=False)
        self._test_net(model, (1, 3, 512, 512))

    def test_unetr_with_conv_transpose_decoder(self):
        "NOTE: Checking for this exclusively as `use_conv_transpose`, by default, is set to `False`."
        from torch_em.model import UNETR

        model = UNETR(use_conv_transpose=True)
        self._test_net(model, (1, 3, 512, 512))

    def test_unetr3d(self):
        from torch_em.model.unetr import UNETR3D

        # Too memory hungry to run without a GPU so just a dummy test.
        model = UNETR3D(backbone="sam", encoder="vit_b")
        self.assertEqual(model.out_channels, 1)


if __name__ == "__main__":
    unittest.main()
