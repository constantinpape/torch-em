import unittest
import numpy as np
import torch


class TestContrastiveLoss(unittest.TestCase):
    def test_contrastive_random(self):
        from torch_em.loss import ContrastiveLoss
        loss = ContrastiveLoss(delta_var=1., delta_dist=2.)

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

    def test_contrastive(self):
        from torch_em.loss import ContrastiveLoss
        loss = ContrastiveLoss(delta_var=1., delta_dist=2.)

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

    def test_contrastive_impls(self):
        from torch_em.loss import ContrastiveLoss
        target_shape = (1, 1, 32, 32)
        pred_shape = (1, 8, 32, 32)
        x = torch.rand(*pred_shape)
        y = torch.randint(low=0, high=5, size=target_shape)

        loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl='expand')
        lval1 = loss(x, y).item()

        if ContrastiveLoss.has_torch_scatter():
            loss = ContrastiveLoss(delta_var=1., delta_dist=2., impl='scatter')
            lval2 = loss(x, y).item()
            self.assertAlmostEqual(lval1, lval2, places=5)


if __name__ == '__main__':
    unittest.main()
