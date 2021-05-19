import unittest
import numpy as np
import torch

from torch_em.loss import ContrastiveLoss


class TestContrastiveLoss(unittest.TestCase):
    def _test_random(self, impl):
        loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl=impl)

        target_shape = (1, 1, 32, 32)
        pred_shape = (1, 8, 32, 32)
        x = torch.rand(*pred_shape)
        x.requires_grad = True
        x.retain_grad = True
        y = torch.randint(low=0, high=5, size=target_shape)

        lval = loss(x, y)
        self.assertNotEqual(lval.item(), 0.)

        lval.backward()
        grads = x.grad
        self.assertEqual(grads.shape, x.shape)
        self.assertFalse(np.allclose(grads.numpy(), 0))

    def test_expand_random(self):
        self._test_random('expand')

    @unittest.skipUnless(ContrastiveLoss.has_torch_scatter(), "Need pytorch_scatter")
    def test_scatter_random(self):
        self._test_random('scatter')

    def _test_deterministic(self, impl):
        loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl=impl)

        target_shape = (1, 1, 32, 32)
        pred_shape = (1, 8, 32, 32)

        # this should give a small loss
        y = torch.randint(low=0, high=5, size=target_shape)
        x = 2 * y.expand(pred_shape).to(torch.float32)

        lval = loss(x, y)
        self.assertLess(lval.item(), 0.2)

        # this should give a large loss
        y = torch.randint(low=0, high=5, size=target_shape)
        x = torch.rand(*pred_shape)

        lval = loss(x, y)
        self.assertGreater(lval.item(), 1.)

    def test_expand_deterministic(self):
        self._test_deterministic('expand')

    @unittest.skipUnless(ContrastiveLoss.has_torch_scatter(), "Need pytorch_scatter")
    def test_scatter_deterministic(self):
        self._test_deterministic('scatter')

    # TODO
    # def _test_ignore(self, impl):
    #     loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl=impl, ignore_label=0)

    #     target_shape = (1, 1, 32, 32)
    #     pred_shape = (1, 8, 32, 32)
    #     x = torch.rand(*pred_shape)
    #     x.requires_grad = True
    #     x.retain_grad = True
    #     y = torch.randint(low=0, high=5, size=target_shape)

    #     lval = loss(x, y)
    #     self.assertNotEqual(lval.item(), 0.)

    #     lval.backward()
    #     grads = x.grad
    #     self.assertEqual(grads.shape, x.shape)
    #     grads = grads.numpy()
    #     self.assertFalse(np.allclose(grads, 0))

    #     ignore_mask = y.numpy() == 0
    #     self.assertTrue(np.allclose(grads[:, ignore_mask]), 0)

    # def test_expand_ignore(self):
    #     self._test_ignore('expand')

    # @unittest.skipUnless(ContrastiveLoss.has_torch_scatter(), "Need pytorch_scatter")
    # def test_scatter_ignore(self):
    #     self._test_ignore('scatter')

    def _test_impls(self, device):
        target_shape = (1, 1, 32, 32)
        pred_shape = (1, 8, 32, 32)
        x = torch.rand(*pred_shape).to(device)
        x.requires_grad = True
        x.retain_grad = True
        y = torch.randint(low=0, high=5, size=target_shape).to(device)

        # compute the loss for expand implementation
        loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl='expand')
        lval1 = loss(x, y)
        lval1.backward()
        grad1 = x.grad.detach().cpu()
        self.assertEqual(grad1.shape, x.shape)
        self.assertFalse(np.allclose(grad1, 0))

        # compute the loss for the scatter implementation
        x.grad = None  # clear the gradients
        loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl='scatter')
        lval2 = loss(x, y)
        lval2.backward()

        # compare the results
        self.assertAlmostEqual(lval1.item(), lval2.item(), places=5)
        grad2 = x.grad.detach().cpu()
        self.assertTrue(np.allclose(grad1, grad2, atol=1e-6))

    @unittest.skipUnless(ContrastiveLoss.has_torch_scatter(), "Need pytorch_scatter")
    def test_impls(self):
        self._test_impls(torch.device('cpu'))
        if torch.cuda.is_available():
            self._test_impls(torch.device('cuda'))


if __name__ == '__main__':
    unittest.main()
