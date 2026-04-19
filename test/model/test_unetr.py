import unittest
import torch

try:
    import segment_anything
except ImportError:
    segment_anything = None

try:
    import micro_sam
except ImportError:
    micro_sam = None


class TestUnetrNormalizationRange(unittest.TestCase):
    def test_input_normalization_range_check(self):
        from torch_em.model.unetr import UNETRBase

        model = object.__new__(UNETRBase)
        model._check_input_normalization_range(torch.tensor([0.0, 0.5, 1.0]), (0.0, 1.0))
        model._check_input_normalization_range(torch.tensor([0.0, 128.0, 255.0]), (0.0, 255.0))
        model._check_input_normalization_range(torch.tensor([0, 1], dtype=torch.uint8), (0.0, 255.0))

        with self.assertRaises(ValueError):
            model._check_input_normalization_range(torch.tensor([0.0, 2.0]), (0.0, 1.0))

        with self.assertRaises(ValueError):
            model._check_input_normalization_range(torch.tensor([0.0, float("nan")]), (0.0, 1.0))


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
