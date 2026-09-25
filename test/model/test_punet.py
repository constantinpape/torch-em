import unittest
import torch


class TestPUnet(unittest.TestCase):
    def _test_net(self, net, shape):
        x = torch.rand(*shape, requires_grad=True)
        net.forward(x)
        y = net.sample(x)
        expected_shape = shape[:1] + (net.num_classes,) + shape[2:]
        self.assertEqual(y.shape, expected_shape)
        loss = y.sum()
        loss.backward()

    def test_punet2d(self):
        from torch_em.model import ProbabilisticUNet
        net = ProbabilisticUNet(
                    input_channels=1,
                    num_classes=1,
                    num_filters=[64, 128, 256, 512],
                    latent_dim=6,
                    no_convs_fcomb=3,
                    beta=1.0,
                    device="cpu"
                )
        self._test_net(net, (1, 1, 128, 128))

    def test_punet_invalid_shape(self):
        from torch_em.model import ProbabilisticUNet
        net = ProbabilisticUNet(
                    input_channels=1,
                    num_classes=1,
                    num_filters=[64, 128, 256, 512],
                    latent_dim=6,
                    no_convs_fcomb=3,
                    beta=1.0,
                    device="cpu"
                )
        with self.assertRaises(ValueError):
            self._test_net(net, (1, 1, 67, 67))

    def test_to_device(self):
        from torch_em.model import ProbabilisticUNet
        net = ProbabilisticUNet(
                    input_channels=1,
                    num_classes=1,
                    num_filters=[64, 128, 256, 512],
                    latent_dim=6,
                    no_convs_fcomb=3,
                    beta=1.0
                )
        net.to(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
