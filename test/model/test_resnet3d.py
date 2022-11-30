import unittest
import torch


class TestResNet3d(unittest.TestCase):
    def _test_net(self, net, shape):
        x = torch.rand(*shape, requires_grad=True)
        y = net(x)
        expected_shape = (shape[0], net.out_channels)
        self.assertEqual(y.shape, expected_shape)
        loss = y.sum()
        loss.backward()

    def test_resnet18(self):
        from torch_em.model.resnet3d import resnet3d_18

        # test model with 1 input channel and 1 output channels
        net = resnet3d_18(in_channels=1, out_channels=1)
        self._test_net(net, (1, 1, 64, 64, 64))

        # test model with 2 input channel and 9 output channels
        net = resnet3d_18(in_channels=2, out_channels=9)
        self._test_net(net, (4, 2, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
