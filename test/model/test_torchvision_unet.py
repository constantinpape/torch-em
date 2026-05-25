import unittest
import torch


class TestTorchvisionUNet2d(unittest.TestCase):
    def _make(self, backbone="resnet18", depth=3, **kwargs):
        from torch_em.model.torchvision_unet import TorchvisionUNet2d
        return TorchvisionUNet2d(backbone, out_channels=2, depth=depth, initial_features=8, pretrained=False, **kwargs)

    def _run(self, net, shape):
        x = torch.rand(*shape, requires_grad=True)
        y = net(x)
        expected = shape[:1] + (net.out_channels,) + shape[2:]
        self.assertEqual(y.shape, expected)
        y.sum().backward()

    def test_forward_resnet(self):
        net = self._make("resnet18", depth=3)
        self._run(net, (1, 3, 64, 64))

    def test_forward_convnext(self):
        # pre_skip_factor=4 backbone
        net = self._make("convnext_tiny", depth=3)
        self._run(net, (1, 3, 64, 64))

    def test_invalid_backbone(self):
        with self.assertRaises(ValueError):
            self._make("not_a_backbone")

    def test_pretrained_requires_3_channels(self):
        from torch_em.model.torchvision_unet import TorchvisionUNet2d
        with self.assertRaises(ValueError):
            TorchvisionUNet2d("resnet18", out_channels=1, in_channels=1, pretrained=True)

    def test_custom_in_channels_scratch(self):
        from torch_em.model.torchvision_unet import TorchvisionUNet2d
        net = TorchvisionUNet2d("resnet18", out_channels=1, in_channels=1, depth=3, initial_features=8, pretrained=False)
        self._run(net, (1, 1, 64, 64))

    def test_invalid_shape(self):
        net = self._make("resnet18", depth=3)
        with self.assertRaises(ValueError):
            self._run(net, (1, 3, 65, 65))

    def test_range_check(self):
        from torch_em.model.torchvision_unet import TorchvisionUNet2d
        net = TorchvisionUNet2d("resnet18", out_channels=1, depth=3, initial_features=8, pretrained=False)
        net.register_buffer("norm_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        net.register_buffer("norm_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        with self.assertRaises(ValueError):
            net(torch.ones(1, 3, 64, 64) * 2.0)


class TestTorchvisionUNet3d(unittest.TestCase):
    def test_forward(self):
        from torch_em.model.torchvision_unet import TorchvisionUNet3d
        net = TorchvisionUNet3d("r3d_18", out_channels=2, depth=3, initial_features=8, pretrained=False)
        x = torch.rand(1, 3, 8, 32, 32, requires_grad=True)
        y = net(x)
        self.assertEqual(y.shape, (1, 2, 8, 32, 32))
        y.sum().backward()

    def test_pretrained_requires_3_channels(self):
        from torch_em.model.torchvision_unet import TorchvisionUNet3d
        with self.assertRaises(ValueError):
            TorchvisionUNet3d("r3d_18", out_channels=1, in_channels=1, pretrained=True)

    def test_invalid_backbone(self):
        from torch_em.model.torchvision_unet import TorchvisionUNet3d
        with self.assertRaises(ValueError):
            TorchvisionUNet3d("not_a_backbone", out_channels=1)


if __name__ == "__main__":
    unittest.main()
