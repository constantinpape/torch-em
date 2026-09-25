import unittest
import numpy as np
import torch


class TestDiceLoss(unittest.TestCase):
    def test_dice_random(self):
        from torch_em.loss import DiceLoss
        loss = DiceLoss()

        shape = (1, 1, 32, 32)
        x = torch.rand(*shape)
        x.requires_grad = True
        x.retain_grad = True
        y = torch.rand(*shape)

        lval = loss(x, y)
        self.assertNotEqual(lval.item(), 0.0)

        lval.backward()
        grads = x.grad
        self.assertEqual(grads.shape, x.shape)
        self.assertFalse(np.allclose(grads.numpy(), 0))

    def test_dice(self):
        from torch_em.loss import DiceLoss
        loss = DiceLoss()

        shape = (1, 1, 32, 32)
        x = torch.ones(*shape)
        y = torch.ones(*shape)
        lval = loss(x, y)
        self.assertAlmostEqual(lval.item(), 0.0)

        x = torch.ones(*shape)
        y = torch.zeros(*shape)
        lval = loss(x, y)
        self.assertAlmostEqual(lval.item(), 1.0)

    def test_dice_invalid(self):
        from torch_em.loss import DiceLoss
        loss = DiceLoss()

        shape1 = (1, 1, 32, 32)
        shape2 = (1, 2, 32, 32)
        x = torch.rand(*shape1)
        y = torch.rand(*shape2)
        with self.assertRaises(ValueError):
            loss(x, y)

    def test_dice_reduction(self):
        from torch_em.loss import DiceLoss

        shape = (1, 3, 32, 32)
        x = torch.rand(*shape)
        y = torch.rand(*shape)

        for reduction in (None, "mean", "min", "max", "sum"):
            loss = DiceLoss(reduce_channel=reduction)
            lval = loss(x, y)
            if reduction is None:
                self.assertEqual(tuple(lval.shape), (3,))
            else:
                self.assertEqual(tuple(lval.shape), tuple())
                self.assertEqual(lval.numel(), 1)


if __name__ == '__main__':
    unittest.main()
