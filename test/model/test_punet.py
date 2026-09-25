import unittest

import torch


class TestPUnet(unittest.TestCase):
    def _test_net(self, net, shape):
        x = torch.rand(*shape, requires_grad=True)
        net.forward(x)
        y = net.sample()
        expected_shape = shape[:1] + (net.output_channels,) + shape[2:]
        self.assertEqual(y.shape, expected_shape)
        loss = y.sum()
        loss.backward()

    def test_punet2d(self):
        from torch_em.model import ProbabilisticUNet
        net = ProbabilisticUNet(
                    input_channels=1,
                    num_raters=1,
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
                    num_raters=1,
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
                    num_raters=1,
                    num_filters=[64, 128, 256, 512],
                    latent_dim=6,
                    no_convs_fcomb=3,
                    beta=1.0
                )
        net.to(torch.device("cpu"))

    def test_multiple_rater_channels(self):
        from torch_em.model import ProbabilisticUNet

        for channels in (1, 3):
            with self.subTest(channels=channels):
                net = ProbabilisticUNet(
                    output_channels=channels, num_raters=3, num_filters=[8, 16], device="cpu"
                )
                image = torch.rand(2, 1, 64, 64)
                labels = torch.randint(0, max(channels, 2), (2, 3, 64, 64))
                captured = []
                handle = net.posterior.encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
                try:
                    net(image, labels)
                finally:
                    handle.remove()
                if channels == 1:
                    expected = torch.cat([image, labels.float() - 0.5], dim=1)
                else:
                    masks = [
                        (labels[:, rater:rater + 1] == class_id).float() - 0.5
                        for rater in range(3) for class_id in range(channels)
                    ]
                    expected = torch.cat([image] + masks, dim=1)
                torch.testing.assert_close(captured[0], expected)
                self.assertEqual(net.posterior_latent_space.mean.shape, (2, net.latent_dim))
                self.assertEqual(net.reconstruct(use_posterior_mean=True).shape, (2, channels, 64, 64))
                self._test_net(net, (2, 1, 64, 64))

    def test_multiple_rater_shapes(self):
        from torch_em.model import ProbabilisticUNet

        net = ProbabilisticUNet(num_raters=3, num_filters=[8, 16], consensus_masking=True, device="cpu")
        image = torch.rand(2, 1, 64, 64)
        labels = torch.zeros(2, 3, 64, 64)
        with self.assertRaisesRegex(ValueError, "Expected labels with shape"):
            net(image, labels[:, :1])
        net(image, labels)
        with self.assertRaisesRegex(ValueError, "Expected labels with shape"):
            net.elbo(labels[:, :1])
        with self.assertRaisesRegex(ValueError, "Expected consensus mask with shape"):
            net.elbo(labels, torch.ones(2, 2, 64, 64))
        for kwargs in ({"num_raters": 0}, {"output_channels": 0}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                ProbabilisticUNet(**kwargs)


if __name__ == "__main__":
    unittest.main()
