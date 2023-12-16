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

    def test_bce_dice_with_logits(self):
        from torch_em.loss.dice import BCEDiceLossWithLogits
        loss = BCEDiceLossWithLogits()

        shape = (1, 1, 32, 32)
        x = 18 * torch.ones(*shape)
        y = torch.ones(*shape)
        lval = loss(x, y)
        self.assertAlmostEqual(lval.item(), 0.0)

        x = 18 * torch.ones(*shape)
        y = torch.zeros(*shape)
        lval = loss(x, y)
        self.assertGreater(lval.item(), 1.0)


if __name__ == '__main__':
    unittest.main()
