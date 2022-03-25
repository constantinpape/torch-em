import unittest
import torch


class TestUnet(unittest.TestCase):
    def _test_net(self, net, shape):
        x = torch.rand(*shape, requires_grad=True)
        y = net(x)
        expected_shape = shape[:1] + (net.out_channels,) + shape[2:]
        self.assertEqual(y.shape, expected_shape)
        loss = y.sum()
        loss.backward()

    def test_unet2d(self):
        from torch_em.model import UNet2d
        net = UNet2d(1, 1, depth=3, initial_features=8)
        self._test_net(net, (1, 1, 64, 64))

    def test_unet_invalid_shape(self):
        from torch_em.model import UNet2d
        net = UNet2d(1, 1, depth=3, initial_features=8)
        with self.assertRaises(ValueError):
            self._test_net(net, (1, 1, 67, 67))

    def test_norms(self):
        from torch_em.model import UNet2d
        for norm in ("InstanceNorm", "GroupNorm", "BatchNorm", None):
            net = UNet2d(1, 1, depth=3, initial_features=8,
                         norm=norm)
            self._test_net(net, (1, 1, 64, 64))

    def test_side_outputs(self):
        from torch_em.model import UNet2d
        net = UNet2d(1, 1, depth=3, initial_features=8, return_side_outputs=True)
        shape = (1, 1, 64, 64)
        x = torch.rand(*shape)
        outputs = net(x)
        self.assertEqual(len(outputs), 3)
        spatial_shape = shape[2:]
        for output in outputs:
            expected_shape = (1, 1) + spatial_shape
            self.assertEqual(output.shape, expected_shape)
            spatial_shape = tuple(sh // 2 for sh in spatial_shape)

    def test_unet3d(self):
        from torch_em.model import UNet3d
        net = UNet3d(1, 1, depth=3, initial_features=4)
        self._test_net(net, (1, 1, 32, 32, 32))

    def test_anisotropic_unet(self):
        from torch_em.model import AnisotropicUNet
        scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2]]
        for anisotropic_kernel in (False, True):
            net = AnisotropicUNet(1, 1, scale_factors,
                                  initial_features=4,
                                  anisotropic_kernel=anisotropic_kernel)
            self._test_net(net, (1, 1, 8, 32, 32))

    def test_to_device(self):
        from torch_em.model import UNet2d
        net = UNet2d(1, 1, depth=3, initial_features=4)
        net.to(torch.device("cpu"))

    def test_postprocessing(self):
        from torch_em.model import UNet2d

        shape = (1, 1, 64, 64)
        x = torch.rand(*shape)

        net = UNet2d(1, 6, depth=3, initial_features=4,
                     postprocessing="affinities_to_boundaries2d")
        out = net(x)
        self.assertEqual(tuple(out.shape), shape)

        net = UNet2d(1, 6, depth=3, initial_features=4,
                     postprocessing="affinities_with_foreground_to_boundaries2d")
        out = net(x)
        self.assertEqual(tuple(out.shape), (1, 2, 64, 64))


if __name__ == "__main__":
    unittest.main()
