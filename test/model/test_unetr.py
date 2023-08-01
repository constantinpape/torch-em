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
        self._test_net(model, (1, 3, 256, 256))

    @unittest.skipIf(micro_sam is None, "Needs micro_sam")
    def test_unetr_from_sam(self):
        from torch_em.model import build_unetr_with_sam_intialization

        model = build_unetr_with_sam_intialization()
        self._test_net(model, (1, 3, 256, 256))


if __name__ == "__main__":
    unittest.main()
